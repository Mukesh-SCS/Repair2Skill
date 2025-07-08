import openai
import base64
import os
import json
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def parse_manual(manual_path, stage1_json_path="./outputs/stage1_parts.json"):
    """
    Use a VLM (GPT-4o) to generate a hierarchical assembly graph from the manual image and Stage I JSON.
    """
    # Load Stage I JSON
    with open(stage1_json_path, "r") as f:
        stage1_json = json.load(f)

    # Encode manual image
    encoded_manual = encode_image(manual_path)

    prompt = (
        "You are an expert in furniture assembly. "
        "Given the following assembly manual page (image) and a JSON list of detected parts, "
        "generate a hierarchical assembly graph in JSON format. "
        "Each step should list the involved part labels as an array. "
        "Example output:\n"
        "{\n"
        "  \"steps\": [\n"
        "    {\"step\": 1, \"parts\": [0, 5]},\n"
        "    {\"step\": 2, \"parts\": [3, 4, 0, 5]}\n"
        "  ]\n"
        "}\n"
        "Only output the JSON."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Furniture assembly graph generation."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"Detected parts JSON:\n{json.dumps(stage1_json, indent=2)}"},
            {"role": "user", "content": {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_manual}",
                    "detail": "high"
                }
            }},
        ],
        max_tokens=1000,
    )

    content = response.choices[0].message.content
    content = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(content)
    except Exception:
        return content  # fallback: return raw string if not valid JSON