"""
================================================================================
Repair Graph Generator for Repair2Skill
================================================================================
Creates a hierarchical dependency graph centered on the detected damaged part.
This mirrors Manual2Skill’s “Hierarchical Assembly Graph” but for repair tasks.

USAGE:
    from scripts.repair_graph import generate_repair_graph, visualize_repair_graph

    graph = generate_repair_graph("back_left_leg")
    visualize_repair_graph(graph, "outputs/repair_graph_back_left_leg.png")
================================================================================
"""

import os
import json
import matplotlib.pyplot as plt
import networkx as nx
from scripts.chair_graph import CHAIR_GRAPH, get_dependencies, find_parent


def generate_repair_graph(damaged_part: str) -> dict:
    """
    Generate a hierarchical graph of all dependent and parent parts
    around the damaged component.
    """
    graph = {}

    def build_subtree(part):
        children = get_dependencies(part)
        graph[part] = children
        for c in children:
            build_subtree(c)

    # Build downward dependencies
    build_subtree(damaged_part)

    # Include upward parent chain
    parent = find_parent(damaged_part)
    while parent:
        if parent not in graph:
            graph[parent] = [damaged_part]
        parent = find_parent(parent)

    return graph


def save_repair_graph_json(graph: dict, damaged_part: str, out_dir: str = "outputs"):
    """Save hierarchical graph as JSON."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"repair_graph_{damaged_part}.json")
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"[OK] Saved {path}")
    return path


def visualize_repair_graph(graph: dict, save_path: str = None):
    """
    Render the hierarchical repair graph as a directed diagram.
    """
    G = nx.DiGraph()
    for parent, children in graph.items():
        for child in children:
            G.add_edge(parent, child)

    plt.figure(figsize=(6, 5))
    pos = nx.spring_layout(G, seed=42, k=0.4)
    nx.draw_networkx_nodes(G, pos, node_size=1800, node_color="#9fd3c7")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

    plt.title("Repair Dependency Graph", fontsize=12, weight="bold")
    plt.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Repair graph visual saved to: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    damaged = "front_left_leg"
    graph = generate_repair_graph(damaged)
    save_repair_graph_json(graph, damaged)
    visualize_repair_graph(graph, f"outputs/repair_graph_{damaged}.png")
