import networkx as nx
import matplotlib.pyplot as plt
import json

# Define repair steps as edges (start to finish)
steps = [
    ("Start", "Remove broken left leg"),
    ("Remove broken left leg", "Insert replacement leg"),
    ("Insert replacement leg", "Fix seat crack"),
    ("Fix seat crack", "Reattach backrest"),
    ("Reattach backrest", "Inspect structure")
]

# Create directed graph
G = nx.DiGraph()
G.add_edges_from(steps)

# Draw the repair graph
plt.figure(figsize=(10, 6))
nx.draw(
    G,
    with_labels=True,
    node_size=2000,
    node_color='lightgreen',
    font_size=10,
    arrows=True
)
plt.title("Chair Repair Graph")
plt.tight_layout()
plt.show()

# Export the graph to JSON
graph_json = {
    "nodes": list(G.nodes),
    "edges": list(G.edges)
}

with open("repair_graph.json", "w") as f:
    json.dump(graph_json, f, indent=2)

print("✅ Repair graph exported to 'repair_graph.json'")
