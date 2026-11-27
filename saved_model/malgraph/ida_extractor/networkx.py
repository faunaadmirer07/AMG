"""Minimal networkx-compatible shim for the MalGraph IDA extractor.

The original extractor expects a small subset of the networkx API while it is
running inside IDA's bundled Python 2 runtime. This module provides just the
graph primitives the extractor uses so the extraction logic can stay intact.
"""

from __future__ import absolute_import


class _BaseGraph(object):
    def __init__(self):
        self._adj = {}
        self.node = {}

    def __iter__(self):
        return iter(self._adj)

    def __len__(self):
        return len(self._adj)

    def add_node(self, node):
        if node not in self._adj:
            self._adj[node] = set()
        if node not in self.node:
            self.node[node] = {}

    def add_edge(self, source, target):
        self.add_node(source)
        self.add_node(target)
        self._adj[source].add(target)

    def nodes(self):
        return list(self._adj.keys())

    def edges(self):
        edge_list = []
        for source, targets in self._adj.items():
            for target in targets:
                edge_list.append((source, target))
        return edge_list

    def successors(self, node):
        return list(self._adj.get(node, set()))

    def subgraph(self, nodes):
        node_set = set(nodes)
        sub_graph = self.__class__()
        for node in node_set:
            if node in self._adj:
                sub_graph.add_node(node)
                sub_graph.node[node] = dict(self.node.get(node, {}))
        for source in node_set:
            for target in self._adj.get(source, set()):
                if target in node_set:
                    sub_graph.add_edge(source, target)
        return sub_graph


class DiGraph(_BaseGraph):
    def to_undirected(self):
        graph = Graph()
        for node in self.nodes():
            graph.add_node(node)
            graph.node[node] = dict(self.node.get(node, {}))
        for source, target in self.edges():
            graph.add_edge(source, target)
        return graph


class Graph(_BaseGraph):
    def add_edge(self, source, target):
        self.add_node(source)
        self.add_node(target)
        self._adj[source].add(target)
        self._adj[target].add(source)


def connected_components(graph):
    visited = set()
    components = []

    for start_node in graph.nodes():
        if start_node in visited:
            continue

        stack = [start_node]
        component = set()

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in graph.successors(node):
                if neighbor not in visited:
                    stack.append(neighbor)

        components.append(component)

    return components


def betweenness_centrality(graph):
    return dict((node, 0.0) for node in graph.nodes())


def eigenvector_centrality(graph):
    return dict((node, 0.0) for node in graph.nodes())


def closeness_centrality(graph):
    return dict((node, 0.0) for node in graph.nodes())