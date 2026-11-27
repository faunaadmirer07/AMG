# -*- coding: utf-8 -*-
# Comprehensive BB feature extraction for RL malware evasion.
#
# Cache JSON schema per binary:
# {
#   "binary_name": str,
#   "basic_blocks": [
#     {
#       "bb_start":      "0x...",   # VA hex string
#       "bb_end":        "0x...",
#       "func_addr":     "0x...",
#       "file_offset":   int,       # PE file offset of bb_start (-1 if unknown)
#       "section":       str,       # section name e.g. ".text"
#       "section_exec":  0|1,
#       "section_write": 0|1,
#       "call_sites": [             # eligible for semnop injection (sub_ targets)
#         { "call_addr": "0x...", "next_addr": "0x...", "target": "0x...",
#           "have_logic": bool, "have_cmp": bool }
#       ],
#       "import_sites": [           # calls to named imports (not eligible but useful)
#         { "call_addr": "0x...", "import_name": str }
#       ],
#       "features": {
#         // ── instruction counts ──────────────────────────────────────────
#         "num_ins":              int,
#         "num_calls_total":      int,   # all call instructions
#         "num_calls_eligible":   int,   # sub_* targets only (patchable)
#         "num_calls_import":     int,   # calls to named imports
#         "num_indirect_calls":   int,   # call [reg/mem]
#         "num_indirect_jmps":    int,   # jmp [reg/mem]
#         "num_arith":            int,
#         "num_logic":            int,
#         "num_cmp":              int,
#         "num_transfer":         int,   # jmp/jcc/call/ret
#         "num_mov":              int,
#         "num_push":             int,
#         "num_pop":              int,
#         "num_float":            int,   # FPU/SSE instructions
#         "num_string_ops":       int,   # rep movs, scas, etc.
#         "num_nop":              int,
#         "bb_size":              int,   # bytes
#         // ── CFG topology ─────────────────────────────────────────────────
#         "n_predecessors":       int,
#         "n_successors":         int,
#         "is_loop_header":       0|1,
#         "is_terminal":          0|1,   # ends with ret / no successors
#         "is_conditional":       0|1,   # n_successors == 2
#         "is_entry_block":       0|1,   # bb_start == func_addr
#         // ── function-level context ───────────────────────────────────────
#         "func_num_bbs":         int,
#         "func_num_calls":       int,
#         "func_size":            int,   # bytes
#         "func_bb_rank":         int,   # 0-indexed ordinal within func
#         // ── byte-level ───────────────────────────────────────────────────
#         "byte_entropy":         float, # Shannon entropy of BB bytes
#         // ── opcode histogram ─────────────────────────────────────────────
#         "opcodes":              {mnem: count, ...}
#       }
#     }
#   ],
#   "flat_call_sites":    [...],    # backward compat
#   "num_bbs_with_calls": int,
#   "num_call_sites":     int,
#   "total_bbs":          int,      # ALL BBs including those without calls
#   "num_functions":      int
# }
#
# NOTE: ALL BBs are stored (not only those with call sites).
#       RL env should filter on features['num_calls_eligible'] > 0.
#       Storing all BBs enables richer analysis without re-running IDA.

import os
import sys
import json
import math
import idc
import idaapi
import idautils

SAVED_PATH = os.environ.get(
    'IDA_CACHE_DIR',
    './data/malguise_latest/MalGuise_0320/rl_malware/cache/bb_data'
)

# ── Instruction category sets ──────────────────────────────────────────────

LogicInstructions = {
    'and', 'andn', 'andnpd', 'andpd', 'andps', 'andnps',
    'test', 'xor', 'xorpd', 'pslld', 'not', 'or', 'ornot', 'xnor', 'nor'
}
CmpInstructions = {
    'cmp', 'cmps', 'cmpxchg', 'cmpsb', 'cmpsw', 'cmpsd',
    'cmppd', 'cmpps', 'fcom', 'fcomp', 'fcompp', 'fcomi', 'fcomip',
    'pcmpeqb', 'pcmpeqd', 'pcmpeqw', 'pcmpgtb', 'pcmpgtd', 'pcmpgtw'
}
ArithInstructions = {
    'add', 'sub', 'div', 'imul', 'idiv', 'mul', 'shl', 'dec', 'inc',
    'adc', 'sbb', 'neg', 'shr', 'sar', 'rol', 'ror', 'rcl', 'rcr'
}
TransferInstructions = {
    'jmp', 'je', 'jne', 'jz', 'jnz', 'ja', 'jae', 'jb', 'jbe',
    'jg', 'jge', 'jl', 'jle', 'jo', 'jno', 'js', 'jns', 'jp', 'jnp',
    'jcxz', 'jecxz', 'loop', 'loope', 'loopne', 'call', 'ret', 'retn', 'retf'
}
MovInstructions = {
    'mov', 'cmova', 'cmovae', 'cmovb', 'cmovbe', 'cmove', 'cmovg',
    'cmovge', 'cmovl', 'cmovle', 'cmovne', 'cmovno', 'cmovnp', 'cmovns',
    'cmovo', 'cmovp', 'cmovs', 'movzx', 'movsx', 'lea', 'movd', 'movdqa',
    'movaps', 'movups', 'movdqu', 'movss', 'movsd', 'movq'
}
FloatInstructions = {
    'fld', 'fst', 'fstp', 'fild', 'fist', 'fistp', 'fadd', 'fsub', 'fmul',
    'fdiv', 'fabs', 'fchs', 'fsqrt', 'fsin', 'fcos', 'fptan', 'fpatan',
    'addss', 'addsd', 'addps', 'addpd', 'subss', 'subsd', 'subps', 'subpd',
    'mulss', 'mulsd', 'mulps', 'mulpd', 'divss', 'divsd', 'divps', 'divpd',
    'sqrtss', 'sqrtsd', 'sqrtps', 'sqrtpd', 'cvtsi2ss', 'cvtsi2sd',
    'cvtss2sd', 'cvtsd2ss', 'cvttss2si', 'cvttsd2si'
}
StringOpsInstructions = {
    'movs', 'movsb', 'movsw', 'movsd', 'movsq',
    'cmps', 'cmpsb', 'cmpsw', 'cmpsd', 'cmpsq',
    'scas', 'scasb', 'scasw', 'scasd', 'scasq',
    'stos', 'stosb', 'stosw', 'stosd', 'stosq',
    'lods', 'lodsb', 'lodsw', 'lodsd', 'lodsq',
    'rep', 'repe', 'repne', 'repz', 'repnz'
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_raw_bytes(bb_start, bb_size):
    """Return (raw_bytes_hex, entropy). Single IDA read shared by both."""
    if bb_size <= 0:
        return '', 0.0
    try:
        raw = idc.GetManyBytes(bb_start, bb_size)
        if raw is None or len(raw) == 0:
            return '', 0.0
        ba = bytearray(raw)
        hex_str = ba.hex()
        freq = {}
        for b in ba:
            freq[b] = freq.get(b, 0) + 1
        total = float(len(ba))
        entropy = -sum((c / total) * math.log(c / total, 2) for c in freq.values())
        return hex_str, round(entropy, 4)
    except Exception:
        return '', 0.0


def _file_offset(ea):
    """Convert VA to PE file offset. Returns -1 on failure."""
    try:
        off = idaapi.get_fileregion_offset(ea)
        return int(off) if off != idc.BADADDR and off >= 0 else -1
    except Exception:
        return -1


def _section_info(ea):
    """Return (section_name, is_exec, is_write) for the segment containing ea."""
    try:
        seg = idaapi.getseg(ea)
        if seg is None:
            return '', 0, 0
        name = idc.SegName(ea) or ''
        perm = seg.perm  # SEGPERM_EXEC=4, SEGPERM_WRITE=2, SEGPERM_READ=1
        return name, int(bool(perm & 4)), int(bool(perm & 2))
    except Exception:
        return '', 0, 0


def _build_import_map():
    """Return dict {va: name} for all named imports."""
    imports = {}
    try:
        for _ord, addr, name in idautils.Names():
            if name and addr != idc.BADADDR:
                imports[int(addr)] = name
    except Exception:
        pass
    return imports


def _is_indirect(ea, op_idx=0):
    """True if operand is indirect (register/memory, not direct near/far)."""
    try:
        op_type = idc.GetOpType(ea, op_idx)
        # o_near=7, o_far=6 are direct; everything else (mem, reg) is indirect
        return op_type not in (idc.o_near, idc.o_far, idc.o_imm)
    except Exception:
        return False


def _precompute_func_info():
    """
    Build per-function metadata dict keyed by func_ea (int).
    Returns {func_ea: {num_bbs, func_size, num_calls, bb_starts_sorted}}
    """
    func_info = {}
    for seg_ea in idautils.Segments():
        for func_ea in idautils.Functions(seg_ea, idc.SegEnd(seg_ea)):
            func = idaapi.get_func(func_ea)
            if func is None:
                continue
            try:
                fc = idaapi.FlowChart(func)
                bb_list = list(fc)
                num_calls = 0
                for bb in bb_list:
                    ea = int(bb.startEA)
                    while ea < int(bb.endEA) and ea != idc.BADADDR:
                        if idc.GetMnem(ea).lower() == 'call':
                            num_calls += 1
                        ea = int(ea + idc.ItemSize(ea))
                func_info[int(func_ea)] = {
                    'num_bbs':   len(bb_list),
                    'func_size': int(func.size()),
                    'num_calls': num_calls,
                    'bb_starts': sorted(int(b.startEA) for b in bb_list),
                }
            except Exception:
                func_info[int(func_ea)] = {
                    'num_bbs': 0, 'func_size': 0,
                    'num_calls': 0, 'bb_starts': []
                }
    return func_info


# ── Main extraction ────────────────────────────────────────────────────────

def extract_bb_grouped():
    import_map = _build_import_map()
    func_info  = _precompute_func_info()

    all_blocks = []   # all BBs (including those without eligible calls)
    total_bbs  = 0
    num_funcs  = 0

    for seg_ea in idautils.Segments():
        for func_ea in idautils.Functions(seg_ea, idc.SegEnd(seg_ea)):
            func = idaapi.get_func(func_ea)
            if func is None:
                continue
            try:
                fc = idaapi.FlowChart(func)
            except Exception:
                continue
            num_funcs += 1

            fc_blocks = list(fc)
            fi = func_info.get(int(func_ea), {})
            bb_starts_sorted = fi.get('bb_starts', [])

            # Detect loop headers (back-edge heuristic)
            loop_headers = set()
            for b in fc_blocks:
                b_start = int(b.startEA)
                for pred in b.preds():
                    if int(pred.startEA) >= b_start:
                        loop_headers.add(b_start)
                        break

            for bl in fc_blocks:
                total_bbs += 1
                bb_start = int(bl.startEA)
                bb_end   = int(bl.endEA)
                bb_size  = bb_end - bb_start

                # CFG topology
                n_preds       = len(list(bl.preds()))
                n_succs       = len(list(bl.succs()))
                is_loop_hdr   = 1 if bb_start in loop_headers else 0
                is_terminal   = 1 if n_succs == 0 else 0
                is_conditional = 1 if n_succs == 2 else 0
                is_entry      = 1 if bb_start == int(func_ea) else 0

                # Function-level context
                func_bb_rank  = bb_starts_sorted.index(bb_start) \
                                if bb_start in bb_starts_sorted else -1

                # Section info
                sec_name, sec_exec, sec_write = _section_info(bb_start)

                # File offset
                file_off = _file_offset(bb_start)

                # Raw bytes + entropy (single IDA read)
                raw_bytes_hex, entropy = _get_raw_bytes(bb_start, bb_size)

                # Instruction pass
                call_sites   = []   # eligible (sub_* targets)
                import_sites = []   # calls to named imports
                opcodes_seq  = []   # ordered [[mnem, operands], ...]

                num_ins = num_calls = num_arith = num_logic = 0
                num_cmp = num_transfer = num_mov = num_push = num_pop = 0
                num_float = num_string_ops = num_nop = 0
                num_indirect_calls = num_indirect_jmps = num_import_calls = 0
                opcodes  = {}
                have_logic = have_cmp = False

                ea = bb_start
                while ea < bb_end and ea != idc.BADADDR:
                    try:
                        mnem = idc.GetMnem(ea).lower()
                    except Exception:
                        ea = int(ea + idc.ItemSize(ea))
                        continue
                    num_ins += 1

                    if mnem:
                        opcodes[mnem] = opcodes.get(mnem, 0) + 1
                        try:
                            op_str = idc.GetOpnd(ea, 0)
                            op1    = idc.GetOpnd(ea, 1)
                            if op1:
                                op_str = op_str + ', ' + op1
                        except Exception:
                            op_str = ''
                        opcodes_seq.append([mnem, op_str])

                    if mnem in ArithInstructions:     num_arith += 1
                    if mnem in LogicInstructions:
                        have_logic = True;            num_logic += 1
                    if mnem in CmpInstructions:
                        have_cmp = True;              num_cmp += 1
                    if mnem in TransferInstructions:  num_transfer += 1
                    if mnem in MovInstructions:       num_mov += 1
                    if mnem in FloatInstructions:     num_float += 1
                    if mnem in StringOpsInstructions: num_string_ops += 1
                    if mnem == 'push':                num_push += 1
                    if mnem == 'pop':                 num_pop += 1
                    if mnem == 'nop':                 num_nop += 1

                    if mnem == 'call':
                        num_calls += 1
                        op_type = idc.GetOpType(ea, 0)

                        if op_type == idc.o_near:
                            operand = idc.GetOpnd(ea, 0)
                            target_va_str = idc.GetOpnd(ea, 0)
                            # Try to resolve numeric target
                            try:
                                target_va = idc.GetOperandValue(ea, 0)
                            except Exception:
                                target_va = idc.BADADDR

                            # Named import call
                            if target_va != idc.BADADDR and int(target_va) in import_map:
                                num_import_calls += 1
                                import_sites.append({
                                    'call_addr':   str(hex(int(ea))),
                                    'import_name': import_map[int(target_va)],
                                })

                            # Eligible (sub_* → patchable)
                            elif operand.startswith('sub_'):
                                try:
                                    target_addr = '0x' + operand.split('_')[1]
                                    next_ea = int(ea + idc.ItemSize(ea))
                                    call_sites.append({
                                        'call_addr':  str(hex(int(ea))),
                                        'next_addr':  str(hex(next_ea)),
                                        'target':     target_addr,
                                        'have_logic': have_logic,
                                        'have_cmp':   have_cmp,
                                    })
                                    have_logic = False
                                    have_cmp   = False
                                except Exception:
                                    pass

                        elif _is_indirect(ea, 0):
                            num_indirect_calls += 1

                    elif mnem in ('jmp',) and _is_indirect(ea, 0):
                        num_indirect_jmps += 1

                    ea = int(ea + idc.ItemSize(ea))

                all_blocks.append({
                    'bb_start':      str(hex(bb_start)),
                    'bb_end':        str(hex(bb_end)),
                    'func_addr':     str(hex(int(func_ea))),
                    'file_offset':   file_off,
                    'section':       sec_name,
                    'section_exec':  sec_exec,
                    'section_write': sec_write,
                    'raw_bytes_hex': raw_bytes_hex,   # hex string of BB bytes
                    'opcodes_seq':   opcodes_seq,     # [[mnem, operands], ...]
                    'call_sites':    call_sites,
                    'import_sites':  import_sites,
                    'features': {
                        'num_ins':            num_ins,
                        'num_calls_total':    num_calls,
                        'num_calls_eligible': len(call_sites),
                        'num_calls_import':   num_import_calls,
                        'num_indirect_calls': num_indirect_calls,
                        'num_indirect_jmps':  num_indirect_jmps,
                        'num_arith':          num_arith,
                        'num_logic':          num_logic,
                        'num_cmp':            num_cmp,
                        'num_transfer':       num_transfer,
                        'num_mov':            num_mov,
                        'num_push':           num_push,
                        'num_pop':            num_pop,
                        'num_float':          num_float,
                        'num_string_ops':     num_string_ops,
                        'num_nop':            num_nop,
                        'bb_size':            bb_size,
                        'n_predecessors':     n_preds,
                        'n_successors':       n_succs,
                        'is_loop_header':     is_loop_hdr,
                        'is_terminal':        is_terminal,
                        'is_conditional':     is_conditional,
                        'is_entry_block':     is_entry,
                        'func_num_bbs':       fi.get('num_bbs', 0),
                        'func_num_calls':     fi.get('num_calls', 0),
                        'func_size':          fi.get('func_size', 0),
                        'func_bb_rank':       func_bb_rank,
                        'byte_entropy':       round(entropy, 4),
                        'opcodes':            opcodes,
                    }
                })

    bbs_with_calls = [b for b in all_blocks if b['features']['num_calls_eligible'] > 0]
    flat_call_sites = []
    for bb in bbs_with_calls:
        for cs in bb['call_sites']:
            flat_call_sites.append([
                cs['call_addr'], cs['next_addr'], cs['target'],
                cs['have_logic'], cs['have_cmp']
            ])

    return {
        'basic_blocks':       all_blocks,    # ALL BBs
        'flat_call_sites':    flat_call_sites,
        'num_bbs_with_calls': len(bbs_with_calls),
        'num_call_sites':     len(flat_call_sites),
        'total_bbs':          total_bbs,
        'num_functions':      num_funcs,
    }


def main():
    pe_filename = idc.GetInputFile()
    output_path = os.path.join(SAVED_PATH, pe_filename + '_bb.json')
    result = extract_bb_grouped()
    with open(output_path, 'w') as f:
        json.dump(result, f)

    # Backward-compat flat list
    flat_path = os.path.join(SAVED_PATH, pe_filename + '.txt')
    with open(flat_path, 'w') as f:
        json.dump(result['flat_call_sites'], f)


if __name__ == '__main__':
    try:
        idaapi.autoWait()
        main()
    except Exception as e:
        with open('/tmp/ida_bb_extract_error.log', 'w') as f:
            import traceback
            traceback.print_exc(file=f)
    idc.Exit(0)
