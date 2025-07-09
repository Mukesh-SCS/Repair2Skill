import trimesh

def render_step_visual(mesh_path, highlighted_part_idx, save_path):
    mesh = trimesh.load(mesh_path)
    # Assume mesh is split into parts (e.g., via mesh.split())
    parts = mesh.split()
    colors = ['white'] * len(parts)
    if 0 <= highlighted_part_idx < len(parts):
        colors[highlighted_part_idx] = 'red'
    scene = trimesh.Scene()
    for part, color in zip(parts, colors):
        scene.add_geometry(part, node_name=color)
    # Render and save
    png = scene.save_image(resolution=(800, 600))
    with open(save_path, 'wb') as f:
        f.write(png)

if __name__ == "__main__":
    render_step_visual("../data/partnet_data/chair/model.obj",
                       highlighted_part_idx=2,
                       save_path="../data/visual_guides/repair_step1.png")
