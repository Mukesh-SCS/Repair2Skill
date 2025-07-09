import os
import json
import argparse
import collections


def parse_ikea_manual(json_path):
    """
    Parses IKEA manual JSON and converts into a simplified assembly graph format.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    graph = {
        "nodes": [],
        "edges": [],
    }

    node_id = 0
    for entry in data:
        item_id = entry.get("item_id", f"item_{node_id}")
        steps = entry.get("steps", [])

        for step in steps:
            parts = step.get("parts", [])
            instructions = step.get("instruction", "")
            dependencies = step.get("requires", [])

            node = {
                "id": node_id,
                "item_id": item_id,
                "instruction": instructions,
                "parts": parts,
                "requires": dependencies,
            }
            graph["nodes"].append(node)

            # Add edges for dependencies
            for req in dependencies:
                graph["edges"].append({
                    "from": req,
                    "to": node_id
                })

            node_id += 1

    os.makedirs("./outputs", exist_ok=True)
    output_path = "./outputs/ikea_assembly_graph.json"
    with open(output_path, 'w') as f:
        json.dump(graph, f, indent=2)

    print(f"Graph saved to: {output_path}")
    return output_path


def summarize_parts(graph_json, top_k=10):
    """
    Summarize the most frequently used part IDs in the graph.
    """
    with open(graph_json, 'r') as f:
        graph = json.load(f)

    part_count = collections.Counter()

    for node in graph["nodes"]:
        for part_group in node.get("parts", []):
            if isinstance(part_group, str):
                part_ids = [p.strip() for p in part_group.split(",") if p.strip().isdigit()]
                part_count.update(part_ids)

    print("\nTop Parts Used:")
    for part_id, count in part_count.most_common(top_k):
        print(f"Part {part_id}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Parse IKEA manual and generate graph")
    parser.add_argument('--input', type=str, required=True, help="Path to main_data.json")
    parser.add_argument('--summarize', action='store_true', help="Print top-used parts")

    args = parser.parse_args()

    graph_path = parse_ikea_manual(args.input)

    if args.summarize:
        summarize_parts(graph_path)


if __name__ == "__main__":
    main()
