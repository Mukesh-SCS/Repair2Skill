
import json

def execute_repair_sequence(graph_json_path):
    with open(graph_json_path, 'r') as f:
        graph = json.load(f)

    print("\n--- Repair Execution Plan ---")
    for step in graph.get("repair_sequence", []):
        print(f"Step {step['step_id']}: {step['action']} on {step['target_part']}")
        print(f" - Estimated Time: {step['estimated_time']}")
        print(f" - Difficulty: {step['difficulty']}")
    print("--- End of Plan ---")

if __name__ == "__main__":
    execute_repair_sequence("./outputs/stage2_assembly_graph.json")
