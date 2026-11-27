"""Minimal OmegaConf-compatible shim for the MalGraph IDA extractor.

Only the tiny subset used by the extractor is implemented: loading simple
nested YAML dictionaries and merging two config trees.
"""

from __future__ import absolute_import

import os


class DictConfig(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value


def _to_config(value):
    if isinstance(value, dict):
        config = DictConfig()
        for key, item in value.items():
            config[key] = _to_config(item)
        return config
    if isinstance(value, list):
        return [_to_config(item) for item in value]
    return value


def _strip_comment(line):
    in_quotes = False
    quote_char = None
    result = []
    for char in line:
        if char in ('"', "'"):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif quote_char == char:
                in_quotes = False
                quote_char = None
        if char == '#' and not in_quotes:
            break
        result.append(char)
    return ''.join(result).rstrip()


def _parse_scalar(text):
    text = text.strip()
    if not text:
        return ''
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    lower = text.lower()
    if lower == 'true':
        return True
    if lower == 'false':
        return False
    if lower == 'none' or lower == 'null':
        return None
    try:
        if '.' in text:
            return float(text)
        return int(text)
    except Exception:
        return text


def _parse_yaml_text(text):
    root = {}
    stack = [(-1, root)]

    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(' '))
        stripped = line.strip()
        if ':' not in stripped:
            continue

        key, value = stripped.split(':', 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if not value:
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)

    return root


def _merge_dicts(left, right):
    merged = {}
    keys = set(left.keys()) | set(right.keys())
    for key in keys:
        if key in left and key in right and isinstance(left[key], dict) and isinstance(right[key], dict):
            merged[key] = _merge_dicts(left[key], right[key])
        elif key in right:
            merged[key] = right[key]
        else:
            merged[key] = left[key]
    return merged


class OmegaConf(object):
    @staticmethod
    def load(path):
        with open(path, 'r') as fp:
            parsed = _parse_yaml_text(fp.read())
        return _to_config(parsed)

    @staticmethod
    def merge(*configs):
        merged = {}
        for config in configs:
            merged = _merge_dicts(merged, config)
        return _to_config(merged)