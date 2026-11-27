from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch_geometric.data import Data

from malgraph.vocabulary import Vocab


def json_item_to_data(item: Dict[str, Any], vocab: Vocab, label: int = 1) -> Optional[Data]:
    acfg_list = []
    for one_acfg in item.get("acfg_list", []):
        block_features = one_acfg.get("block_features", [])
        block_edges = one_acfg.get("block_edges", [[], []])
        if not block_features:
            continue
        acfg_list.append(
            Data(
                x=torch.tensor(block_features, dtype=torch.float),
                edge_index=torch.tensor(block_edges, dtype=torch.long),
            )
        )
    if not acfg_list:
        return None

    item_function_names = item.get("function_names", [])
    item_function_edges = item.get("function_edges", [[], []])
    local_function_name_list = item_function_names[: len(acfg_list)]
    if len(acfg_list) != len(local_function_name_list):
        return None

    external_function_name_list = item_function_names[len(acfg_list) :]
    external_function_index_list = [vocab[f_name] for f_name in external_function_name_list]
    if not external_function_index_list:
        external_function_index_list = [vocab.pad_idx]

    return Data(
        hash=item.get("hash", "unknown"),
        local_acfgs=acfg_list,
        external_list=external_function_index_list,
        function_edges=item_function_edges,
        targets=label,
    )
