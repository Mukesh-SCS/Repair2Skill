from utils.openai_utils import generate_repair_plan

if __name__ == "__main__":
    plan = generate_repair_plan(
        furniture_type="Chair",
        damaged_part="Front Left Leg",
        assembly_step="Step 2: Attach leg using 4 screws provided."
    )
    print(plan)
