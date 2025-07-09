# ==================== utils/openai_utils.py ====================
import openai
import os
from dotenv import load_dotenv
import json

load_dotenv()

def generate_repair_plan(furniture_type, damaged_part, assembly_step, damage_type="missing"):
    """Generate repair plan using OpenAI GPT-4"""
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""
    You are an expert furniture repair technician. Given the following information about a damaged {furniture_type}:
    
    - Furniture Type: {furniture_type}
    - Damaged Part: {damaged_part}
    - Damage Type: {damage_type}
    - Original Assembly Step: {assembly_step}
    
    Please provide a detailed step-by-step repair plan. Include:
    1. Required tools and materials
    2. Safety precautions
    3. Step-by-step instructions
    4. Tips for best results
    
    Format your response as a structured JSON with the following format:
    {{
        "repair_plan": {{
            "tools_needed": ["tool1", "tool2", ...],
            "materials_needed": ["material1", "material2", ...],
            "safety_precautions": ["precaution1", "precaution2", ...],
            "steps": [
                {{
                    "step_number": 1,
                    "description": "Step description",
                    "estimated_time": "time estimate"
                }},
                ...
            ],
            "tips": ["tip1", "tip2", ...],
            "difficulty_level": "easy|medium|hard"
        }}
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful furniture repair expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        # Parse the response
        repair_plan = json.loads(response.choices[0].message.content)
        return repair_plan
        
    except Exception as e:
        print(f"Error generating repair plan: {e}")
        return {
            "error": "Failed to generate repair plan",
            "fallback_plan": {
                "tools_needed": ["screwdriver", "glue", "sandpaper"],
                "materials_needed": ["replacement part", "screws"],
                "steps": [
                    {"step_number": 1, "description": f"Remove damaged {damaged_part}", "estimated_time": "10 minutes"},
                    {"step_number": 2, "description": f"Clean the area", "estimated_time": "5 minutes"},
                    {"step_number": 3, "description": f"Install new {damaged_part}", "estimated_time": "15 minutes"}
                ]
            }
        }