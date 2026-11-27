from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch_geometric.data import Batch

from malgraph.config import load_config, model_params_from_config, resolved_paths
from malgraph.hierarchical_model import HierarchicalGraphNeuralNetwork
from malgraph.preprocess import json_item_to_data
from malgraph.real_batch import create_real_batch_data
from malgraph.vocabulary import Vocab


class MalgraphScorer:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        threshold: Optional[float] = None,
    ):
        self.config = config or load_config()
        paths = resolved_paths(self.config)
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.threshold = threshold if threshold is not None else float(self.config["model"].get("threshold", 0.5))
        self.ida_path = str(paths["ida_path"])
        self.ida_script_path = paths["ida_script"]
        self.tmp_sample_root = paths["tmp_sample_root"]
        self.tmp_sample_root.mkdir(parents=True, exist_ok=True)
        # Prefer the user's real IDA license directory on a normal install.
        # Fall back to a private copy only when spawning concurrent workers.
        home_idapro = Path.home() / ".idapro"
        private_idausr = self.tmp_sample_root.parent / "idapro" / str(os.getpid())
        if os.environ.get("IDAUSR"):
            self._idausr = Path(os.environ["IDAUSR"])
        elif home_idapro.is_dir():
            self._idausr = home_idapro
        else:
            self._idausr = private_idausr
            if not self._idausr.exists():
                try:
                    shutil.copytree(home_idapro, self._idausr)
                except (FileNotFoundError, FileExistsError):
                    self._idausr.mkdir(parents=True, exist_ok=True)

        max_vocab_size = int(self.config["model"]["max_vocab_size"])
        self.vocab = Vocab(freq_file=str(paths["vocab_path"]), max_vocab_size=max_vocab_size)
        params = model_params_from_config(self.config)
        self.model = HierarchicalGraphNeuralNetwork(
            model_params=params,
            external_vocab=self.vocab,
            global_log=None,
            grad_output=False,
        ).to(self.device)
        state = torch.load(paths["model_path"], map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _extract_json(self, sample_path: str) -> str:
        script_path = self.ida_script_path.resolve()
        cmd = [self.ida_path, "-c", "-A", f"-S{script_path}", sample_path]
        env = os.environ.copy()
        env["MALGRAPH_ROOT"] = str(resolved_paths(self.config)["root"])
        env["TVHEADLESS"] = "1"
        env["IDAUSR"] = str(self._idausr)
        # Optional cluster-only library path. Unset on a normal Linux IDA install.
        ida_i686_libs = os.environ.get("IDA_I686_LIBS", "")
        usr_lib = os.path.join(ida_i686_libs, "usr", "lib") if ida_i686_libs else ""
        lib_dir = os.path.join(ida_i686_libs, "lib") if ida_i686_libs else ""
        if usr_lib and os.path.isdir(usr_lib):
            ld = env.get("LD_LIBRARY_PATH", "")
            extra = f"{usr_lib}:{lib_dir}"
            env["LD_LIBRARY_PATH"] = f"{extra}:{ld}" if ld else extra
        ida_env_script = os.environ.get("IDA_ENV_SCRIPT", "")
        if ida_env_script and os.path.isfile(ida_env_script):
            # Source optional cluster setup, then exec IDA. On normal machines
            # IDA_ENV_SCRIPT is unset and we call IDA directly.
            bash_cmd = (
                "source {script} >/dev/null 2>&1 || true; exec \"$@\""
            ).format(script=shlex.quote(ida_env_script))
            cmd = ["bash", "-lc", bash_cmd, "ida"] + cmd
        subprocess.run(
            cmd,
            check=True,
            cwd=str(script_path.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return os.path.splitext(sample_path)[0] + ".json"

    def score_json_item(self, item: Dict[str, Any], label: int = 1) -> Optional[float]:
        data_pt = json_item_to_data(item, self.vocab, label=label)
        if data_pt is None:
            return None
        return self._score_batch(Batch.from_data_list([data_pt]))

    def score_json_file(self, json_path: Union[str, Path], label: int = 1) -> Optional[float]:
        with Path(json_path).open("r", encoding="utf-8") as f:
            item = json.load(f)
        return self.score_json_item(item, label=label)

    def score_pe_bytes(self, bytez: bytes, data_hash: str = "sample") -> Optional[float]:
        uid = uuid.uuid4().hex[:8]
        sample_name = f"{data_hash}_{uid}"
        sample_path = self.tmp_sample_root / sample_name
        sample_path.write_bytes(bytez)
        json_path = self._extract_json(str(sample_path))
        return self.score_json_file(json_path)

    def score_pe_file(self, pe_path: Union[str, Path]) -> Optional[float]:
        path = Path(pe_path)
        return self.score_pe_bytes(path.read_bytes(), data_hash=path.stem)

    def classify(self, score: float) -> str:
        return "malware" if score >= self.threshold else "benign"

    def _score_batch(self, one_batch: Batch) -> Optional[float]:
        real_batch, positions, _, external_list, function_edges, _targets = create_real_batch_data(one_batch=one_batch)
        if real_batch is None:
            return None
        real_batch = real_batch.to(self.device)
        positions = torch.tensor(positions, dtype=torch.long).to(self.device)
        with torch.no_grad():
            pred = self.model(
                real_local_batch=real_batch,
                real_bt_positions=positions,
                bt_external_names=external_list,
                bt_all_function_edges=function_edges,
                local_device=self.device,
            ).squeeze(-1).detach().cpu()
        return float(pred[0].item())
