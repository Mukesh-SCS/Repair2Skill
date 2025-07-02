import openai
from PIL import Image
import base64
import json

openai.api_key = "your-openai-api-key"  # Replace this with your key

def encode_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_path = "chair.jpg"
base64_image = encode_image(image_path)

prompt = """
This is a broken chair image. Analyze the structure and return a JSON list describing each part with its label and condition.
Use the format:
[
  {"name": "left leg", "label": [0], "role": "broken"},
  {"name": "seat", "label": [1], "role": "intact"},
  ...
]
"""

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an expert in robotic repair and furniture diagnostics."},
        {"role": "user", "content": prompt},
        {
            "role": "user",
            "content": {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
            },
        },
    ],
    max_tokens=1000,
)

print("🔍 Detected Parts:")
print(response.choices[0].message.content)
