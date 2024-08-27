""""Utilities for processing tracking graphs."""
from typing import Tuple

import networkx as nx
import pandas as pd

def make_selected_node_data_table(
    tracking_graph: nx.DiGraph,
    nodes: Tuple[int, ...],
    data_keys: Tuple[str, ...]
) -> pd.DataFrame:
    """Make a table of the selected node data."""
    # iterate through the nodes and make the table
    table_data = []
    for node_key in nodes:
        node_data = tracking_graph.nodes[node_key]
        table_data.append(
            {data_key: node_data[data_key] for data_key in data_keys}
        )

    return pd.DataFrame(table_data)


def make_descendents_node_data_table(
    tracking_graph: nx.DiGraph,
    starting_node: int,
    data_keys: Tuple[str, ...],
    include_root: bool = True
) -> pd.DataFrame:
    """Make a table of the node data from all descendent nodes."""
    # get the descendant nodes
    descendant_nodes = nx.descendants(tracking_graph, starting_node)

    if include_root:
        # add the starting node
        descendant_nodes.add(starting_node)

    return make_selected_node_data_table(
        tracking_graph=tracking_graph,
        nodes=descendant_nodes,
        data_keys=data_keys
    )


def make_node_data_table(
    tracking_graph: nx.DiGraph,
    data_keys: Tuple[str, ...]
) -> pd.DataFrame:
    """Make a table of the specified node data."""
    table_data = []
    for node_key, node_data in tracking_graph.nodes(data=True):
        node_row = {data_key: node_data[data_key] for data_key in data_keys}
        node_row["node_key"] = node_key
        table_data.append(node_row)

    return pd.DataFrame(table_data)