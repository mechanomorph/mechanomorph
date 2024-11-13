"""Operations to find specific nodes in a tracking graph."""

from typing import List

import networkx as nx


def find_root_nodes(graph: nx.DiGraph) -> List[int]:
    """Find all roots in the graph.

    Roots are defined as nodes with 0 edges point towards then.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to find the roots in.

    root_nodes : List[int]
        The node ids of the roots.
    """
    return [node for node in graph.nodes() if graph.in_degree(node) == 0]


def find_leaf_nodes(graph: nx.DiGraph) -> List[int]:
    """Find all leaf nodes in the graph.

    Leaf nodes are defined as nodes without any out nodes
    (i.e., terminal nodes).

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to find the leaves in.

    leaf_nodes : List[int]
        The node ids of the leaves.
    """
    return [
        node
        for node in graph.nodes()
        if graph.in_degree(node) != 0 and graph.out_degree(node) == 0
    ]


def find_cell_divisions(graph: nx.DiGraph) -> List[int]:
    """Find all nodes where a cell division occurs.

    Cell division is defined as a node with 1 edge in and 2 edges out.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to find the leaves in.

    division_nodes : List[int]
        The node ids of the division nodes.
    """
    return [
        node
        for node in graph.nodes()
        if graph.in_degree(node) == 1 and graph.out_degree(node) == 2
    ]
