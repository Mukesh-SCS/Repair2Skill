import json
import openai

# Load part annotations
with open("../data/annotations.json", "r") as f:
    part_annotations = json.load(f)

# Load prompt template
with open("../openai_prompts/prompt_repair_graph.txt", "r") as f:
    prompt_template = f.read()

# Format the prompt with actual data
full_prompt = prompt_template.replace("Example Input:", f"Input:
{json.dumps(part_annotations, indent=2)}")

# Set your OpenAI key (or use environment variable)
openai.api_key = "your-api-key"

# Send prompt to GPT-4o
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful robot repair assistant."},
        {"role": "user", "content": full_prompt}
    ],
    max_tokens=1000
)

# Print the GPT-4o response
print("📋 Step-by-Step Repair Plan:")
print(response.choices[0].message.content)
