import open3d as o3d


def visualize_3d_model(model_path):
    mesh = o3d.io.read_triangle_mesh(model_path)
    if mesh.is_empty():
        print(f"Failed to load mesh from {model_path}")
        return
    o3d.visualization.draw_geometries([mesh], window_name="3D Model Viewer")