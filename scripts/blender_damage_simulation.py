import bpy
import os
import random
import math
import time

def load_blend_file(filepath):
    bpy.ops.wm.open_mainfile(filepath=filepath)

# Then call this at the start of main:
def main():
    blend_file = "C:/Users/tripa/Downloads/GitHub_Projects/Repair2Skill/data/chair_scene.blend"  # put your blend file path here
    load_blend_file(blend_file)

def randomize_material(obj):
    mat = bpy.data.materials.new(name=f"RandMat_{obj.name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (random.uniform(0.3, 0.7), random.uniform(0.2, 0.6), random.uniform(0.1, 0.5), 1)
        bsdf.inputs['Roughness'].default_value = random.uniform(0.3, 0.7)
    obj.data.materials.clear()
    obj.data.materials.append(mat)

def randomize_light():
    # Remove all existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    # Add a sun light with fixed rotation to get consistent lighting
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = 5  # fixed medium brightness
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(45), 0, math.radians(45))

def randomize_camera():
    cam = bpy.data.objects.get('Camera')
    if cam:
        cam.location = (3, -4, 2)
        cam.rotation_euler = (math.radians(60), 0, math.radians(45))

def simulate_damage(parts, damage_prob=0.1):  # lower damage prob for visibility
    for obj in parts:
        if random.random() < damage_prob:
            if random.random() < 0.5:
                obj.hide_render = True
                obj.hide_viewport = True
            else:
                scale_factor = random.uniform(0.3, 0.7)
                obj.scale = (obj.scale[0], obj.scale[1]*scale_factor, obj.scale[2]*scale_factor)

def main():
    chair_parts_names = [
        "seat", "back", "front_left_leg", "front_right_leg",
        "back_left_leg", "back_right_leg", "armrest_left", "armrest_right"
    ]
    parts_objs = [bpy.data.objects.get(name) for name in chair_parts_names if bpy.data.objects.get(name)]

    for obj in parts_objs:
        randomize_material(obj)

    simulate_damage(parts_objs, damage_prob=0.1)

    randomize_light()
    randomize_camera()

    output_dir = "C:/Users/tripa/Downloads/GitHub_Projects/Repair2Skill/data/image"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"render_{int(time.time() * 1000)}.png"
    bpy.context.scene.render.filepath = os.path.join(output_dir, filename)

    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    main()
