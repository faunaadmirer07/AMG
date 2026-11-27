"""Minimal jsonlines-compatible shim for the MalGraph IDA extractor."""

from __future__ import absolute_import

import json


class Reader(object):
    def __init__(self, fp):
        self._fp = fp

    def __iter__(self):
        for line in self._fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class Writer(object):
    def __init__(self, fp):
        self._fp = fp

    def write(self, item):
        self._fp.write(json.dumps(item))
        self._fp.write("\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fp.close()


class _JsonLinesFile(object):
    def __init__(self, fp):
        self._fp = fp

    def __enter__(self):
        return Writer(self._fp)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fp.close()


def open(filename, mode="r"):
    if "b" not in mode:
        mode = mode.replace("t", "")
    fp = __builtins__["open"](filename, mode)
    return _JsonLinesFile(fp)