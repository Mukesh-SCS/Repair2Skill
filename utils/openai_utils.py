import openai
import base64
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image(image_path):
    encoded_image = encode_image(image_path)

    prompt = (
        "You are an expert in furniture part analysis. "
        "Given the following image of a piece of furniture, "
        "list all visible parts as a JSON array. "
        "For each part, include: name, label (as an integer), and role (as a string). "
        "Example format:\n"
        "[\n"
        "  {\"name\": \"side frame\", \"label\": 0, \"role\": \"support\"},\n"
        "  {\"name\": \"seat frame\", \"label\": 1, \"role\": \"seat\"}\n"
        "]\n"
        "Only output the JSON array."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Furniture repair analysis."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            },
        ],
        max_tokens=1000,
    )

    content = response.choices[0].message.content
    content = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(content)
    except Exception:
        return content  # fallback: return raw string if not valid JSON

def generate_repair_plan(furniture_type, damaged_part, assembly_step):
    prompt = (
        f"Furniture type: {furniture_type}\n"
        f"Damaged part: {damaged_part}\n"
        f"Assembly step: {assembly_step}\n"
        "Generate a step-by-step repair plan for the above scenario."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in furniture repair."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content