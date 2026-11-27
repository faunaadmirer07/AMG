# coding:utf-8
"""
Patched IDA extraction script for malgraph.  Python 2.7 compatible (IDA Pro).

Key changes vs the original ida_extractor/graph_handle_acfg.py:
  - JSON is written next to the input binary (same directory), not to
    tmp_sample_root, so MalgraphScorer.score_pe_bytes() finds it via
    os.path.splitext(sample_path)[0] + ".json".
  - gdl and log files go to /tmp/malgraph_work/ (writable path).
  - sys.path is patched to find func.py / raw_graphs.py from the original
    ida_extractor/ directory.
  - os.makedirs uses try/except instead of exist_ok=True (Python 2 compat).
"""
import sys
import os

_ROOT = os.environ.get('MALGRAPH_ROOT', '')
_IDA_EXTRACTOR = os.path.join(_ROOT, 'ida_extractor')
if _IDA_EXTRACTOR not in sys.path:
    sys.path.insert(0, _IDA_EXTRACTOR)

from func import *       # noqa: F401, F403
from raw_graphs import * # noqa: F401, F403
import idc
import idaapi
import time
import jsonlines
import omegaconf

cfg_path = os.path.join(_ROOT, 'configs/malgraph.yaml')
config = omegaconf.OmegaConf.load(cfg_path)

_WORK_DIR = '/tmp/malgraph_work'
for _d in [_WORK_DIR, os.path.join(_WORK_DIR, 'gdl')]:
    if not os.path.exists(_d):
        os.makedirs(_d)


def _write_json(filename, data):
    with jsonlines.open(filename, mode='w') as writer:
        writer.write(data)


def _get_acfg():
    start_time = int(time.time())
    cfgs, flag = get_func_cfgs_c(start_time)
    if flag is True:
        with open(os.path.join(_WORK_DIR, 'quit.log'), 'a') as f:
            f.write(GetInputFile() + ' timeout\n')
        idc.Exit(0)
    output = {}
    for func in cfgs.raw_graph_list:
        bs, be = [], []
        for item in func.discovre_features[0]:
            bs.append(item[0])
            be.append(item[1])
        a_fea = []
        for i in range(len(func.g.node)):
            v = func.g.node[i]['v']
            a_fea.append([len(v[0]), len(v[1]), v[2], v[3], v[4],
                          v[5], v[6], v[7], v[8], v[9], v[10]])
        output[func.funcname] = {
            'block_edges': [bs, be],
            'block_number': len(a_fea),
            'block_features': a_fea,
        }
    return output


def _parse_gdl(filename, function_acfg):
    edge_list, ext_fn, int_fn, acfg_list, order = [], [], [], [], []
    with open(filename, 'r') as fh:
        for line in fh:
            d = line.strip('\n')
            if d.startswith('node:'):
                parts = d.split(' ')
                fn = parts[5].replace('"', '')
                if fn in function_acfg and parts[7] != '80' and fn[0] != '.':
                    int_fn.append(fn)
                    acfg_list.append(function_acfg[fn])
                else:
                    ext_fn.append(fn)
                order.append(fn)
            elif d.startswith('edge:'):
                parts = d.split(' ')
                src = order[int(parts[3].replace('"', ''))]
                tgt = order[int(parts[5].replace('"', ''))]
                edge_list.append({'source': src, 'target': tgt})
    total_function = int_fn + ext_fn
    return edge_list, total_function, acfg_list


if __name__ == '__main__':
    idaapi.autoWait()

    input_path  = GetInputFilePath()
    input_dir   = os.path.dirname(input_path)
    pe_filename = os.path.splitext(os.path.basename(input_path))[0]

    # JSON goes next to the binary so MalgraphScorer._extract_json() finds it
    json_path    = os.path.join(input_dir, pe_filename + '.json')
    gdl_filename = os.path.join(_WORK_DIR, 'gdl', pe_filename)

    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    output = _get_acfg()

    res = idaapi.gen_simple_call_chart(gdl_filename, '', 'title', 0)
    if res is not True:
        with open(os.path.join(_WORK_DIR, 'error.log'), 'a') as f:
            f.write(pe_filename + ' get call graph failed\n')
        idc.Exit(0)

    edge_list, total_function, acfg_list = _parse_gdl(gdl_filename + '.gdl', output)

    fn_start = [total_function.index(e['source']) for e in edge_list]
    fn_end   = [total_function.index(e['target']) for e in edge_list]

    data = {
        'hash': pe_filename,
        'function_number': len(total_function),
        'function_edges': [fn_start, fn_end],
        'function_names': total_function,
        'acfg_list': acfg_list,
    }
    _write_json(json_path, data)

    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    with open(os.path.join(_WORK_DIR, 'time.log'), 'a') as f:
        f.write(pe_filename + ' start: ' + start_time + '; end: ' + end_time + '\n')

    idc.Exit(0)
