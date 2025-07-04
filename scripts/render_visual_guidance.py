import pyvista as pv

def render_step_visual(mesh_path, highlighted_part_idx, save_path):
    mesh = pv.read(mesh_path)
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color="white", opacity=0.5)
    # Highlight part
    parts = mesh.split_bodies()
    plotter.add_mesh(parts[highlighted_part_idx], color="red")

    plotter.show(screenshot=save_path)

if __name__ == "__main__":
    render_step_visual("../data/partnet_data/chair/model.obj",
                       highlighted_part_idx=2,
                       save_path="../data/visual_guides/repair_step1.png")
