"""Minimal pathlib-compatible shim for the MalGraph IDA extractor."""

from __future__ import absolute_import

import os


class Path(object):
    def __init__(self, path):
        self._path = os.fspath(path) if hasattr(os, 'fspath') else str(path)

    def __str__(self):
        return self._path

    def __repr__(self):
        return "Path(%r)" % self._path

    def __unicode__(self):
        return self._path

    def __div__(self, other):
        return Path(os.path.join(self._path, str(other)))

    def __truediv__(self, other):
        return self.__div__(other)

    def resolve(self):
        return Path(os.path.abspath(self._path))

    @property
    def parent(self):
        return Path(os.path.dirname(self._path))

    @property
    def parents(self):
        parents = []
        current = os.path.abspath(self._path)
        while True:
            current = os.path.dirname(current)
            parents.append(Path(current))
            if current == os.path.dirname(current):
                break
        return parents