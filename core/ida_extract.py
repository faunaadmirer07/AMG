"""
IDA Call Site Extraction with Caching.

Checks if BB-grouped JSON exists in cache directory.
If yes: loads and returns it.
If no: runs IDA Pro, saves JSON, returns it.

Usage:
    from core.ida_extract import extract_bb_data
    bb_data = extract_bb_data(pe_path, config)
    # bb_data = {
    #   'basic_blocks': [ {bb_start, bb_end, func_addr, call_sites, features}, ... ],
    #   'flat_call_sites': [ [call_addr, next_addr, target, have_logic, have_cmp], ... ],
    #   'num_bbs_with_calls': int,
    #   'num_call_sites': int
    # }
"""
import os
import json
import shlex
import tempfile


def _get_binary_hash(pe_path):
    """Extract the filename (hash) from the PE path."""
    return os.path.basename(pe_path)


def _get_cache_path(pe_path, cache_dir):
    """Get the cache file path, checking subdir structure (new) then flat (legacy)."""
    binary_name = os.path.basename(pe_path)
    # New layout: cache_dir/<subdir>/<binary_name>_bb.json
    # where subdir matches the binary's own parent directory name (e.g. console_x86).
    subdir = os.path.basename(os.path.dirname(pe_path))
    subdir_path = os.path.join(cache_dir, subdir, binary_name + '_bb.json')
    if os.path.exists(subdir_path):
        return subdir_path
    # Legacy flat layout: cache_dir/<binary_name>_bb.json
    return os.path.join(cache_dir, binary_name + '_bb.json')


def _default_idausr():
    """
    Prefer a real IDA license directory on a normal Linux install.

    IDA batch mode requires an accepted license under IDAUSR (usually
    ~/.idapro). Optional env overrides remain available for clusters.
    """
    if os.environ.get('IDAUSR'):
        return os.environ['IDAUSR']
    home_idapro = os.path.join(os.path.expanduser('~'), '.idapro')
    if os.path.isdir(home_idapro):
        return home_idapro
    return os.path.join(tempfile.gettempdir(), 'ida_user')


def _run_ida(pe_path, ida_binary, ida_script, cache_dir, timeout=300):
    """
    Run IDA Pro to extract BB-grouped call sites.

    Portable defaults for a normal Linux IDA install:
      - IDAUSR defaults to ~/.idapro when present
      - TMPDIR defaults to the system temp directory
      - Optional cluster helpers via IDA_ENV_SCRIPT / IDA_I686_LIBS only
        when those environment variables are set (no hardcoded cluster paths)
    """
    import subprocess
    binary_hash = _get_binary_hash(pe_path)
    os.makedirs(cache_dir, exist_ok=True)
    expected_output = os.path.join(cache_dir, binary_hash + '_bb.json')

    idausr = _default_idausr()
    tmpdir = os.environ.get('TMPDIR', tempfile.gettempdir())
    os.makedirs(idausr, exist_ok=True)
    os.makedirs(tmpdir, exist_ok=True)
    ida_log = os.environ.get(
        'IDA_RL_LOG',
        os.path.join(tmpdir, 'ida_rl_{}.log'.format(os.getpid())),
    )
    os.makedirs(os.path.dirname(ida_log), exist_ok=True)
    env_prefix = (
        'IDAUSR={idausr} TMPDIR={tmpdir} IDA_CACHE_DIR={cache} TVHEADLESS=1'
    ).format(
        idausr=shlex.quote(idausr),
        tmpdir=shlex.quote(tmpdir),
        cache=shlex.quote(cache_dir),
    )
    # Optional cluster-only helpers. On a normal desktop/laptop IDA install
    # these environment variables are unset and are simply skipped.
    setup = ''
    ida_env_script = os.environ.get('IDA_ENV_SCRIPT', '')
    if ida_env_script and os.path.exists(ida_env_script):
        setup = 'source {} >/dev/null 2>&1 || true; '.format(shlex.quote(ida_env_script))
    ida_i686_libs = os.environ.get('IDA_I686_LIBS', '')
    usr_lib = os.path.join(ida_i686_libs, 'usr', 'lib') if ida_i686_libs else ''
    lib_dir = os.path.join(ida_i686_libs, 'lib') if ida_i686_libs else ''
    if usr_lib and os.path.isdir(usr_lib):
        setup += (
            'export LD_LIBRARY_PATH="{usr}:{lib}:${{LD_LIBRARY_PATH:-}}"; '
        ).format(usr=shlex.quote(usr_lib), lib=shlex.quote(lib_dir))
    ida_cmd = '{} {} -A -c -L{} -S{} {} > /dev/null 2>&1'.format(
        env_prefix,
        shlex.quote(ida_binary),
        shlex.quote(ida_log),
        shlex.quote(ida_script),
        shlex.quote(pe_path),
    )
    cmd = 'bash -lc {}'.format(shlex.quote(setup + ida_cmd))

    import signal
    proc = subprocess.Popen(cmd, shell=True, start_new_session=True)
    try:
        proc.wait(timeout=timeout)
        ret = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        proc.wait()
        raise RuntimeError(
            "IDA timed out after {}s for {}".format(timeout, pe_path)
        )

    if not os.path.exists(expected_output):
        raise RuntimeError(
            "IDA did not produce output for {}. "
            "Expected: {}. Exit code: {}. "
            "On a normal Linux install: accept the IDA license once "
            "(creates ~/.idapro), set paths.ida_binary to your idal64, "
            "and ensure IDAUSR points at that license directory.".format(
                pe_path, expected_output, ret)
        )

    # Clean up IDA temp files
    for ext in ['.id0', '.id1', '.id2', '.nam', '.til', '.i64']:
        tmp = pe_path + ext
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except:
                pass

    with open(expected_output, 'r') as f:
        return json.load(f)


def extract_bb_data(pe_path, ida_binary, ida_script, cache_dir):
    """
    Extract BB-grouped call sites for a binary, with caching.

    First checks if cached _bb.json exists. If yes, loads it.
    If no, runs IDA Pro to extract and cache it.

    Args:
        pe_path: path to PE binary
        ida_binary: path to IDA Pro (idal64)
        ida_script: path to get_call_ida_bb_grouped.py
        cache_dir: directory for cached _bb.json files

    Returns:
        dict with keys: basic_blocks, flat_call_sites, num_bbs_with_calls, num_call_sites
    """
    cache_path = _get_cache_path(pe_path, cache_dir)

    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    # Cache miss — run IDA
    return _run_ida(pe_path, ida_binary, ida_script, cache_dir)


# ========== QUICK TEST ==========
if __name__ == '__main__':
    # Example: use a cached sample from the bundled demo cache.
    pe_path = './malware_samples_10/console_x86/05f761870247f9553290e3b615921460'
    ida_binary = '__SET_YOUR_IDA_PATH_HERE__'
    ida_script = './core/step1/get_call_ida_bb_grouped.py'
    cache_dir = './sample_cache_10'

    print("Testing with cached sample...")
    data = extract_bb_data(pe_path, ida_binary, ida_script, cache_dir)
    print("BBs: {}, Call sites: {}".format(data['num_bbs_with_calls'], data['num_call_sites']))
    print("First BB: {} ({} calls)".format(
        data['basic_blocks'][0]['bb_start'],
        len(data['basic_blocks'][0]['call_sites'])
    ))
    print("Cache hit — no IDA needed")
