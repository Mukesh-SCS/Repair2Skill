import trimesh
import numpy as np
import os

def generate_missing_part(mesh_path, missing_part_idx, save_path):
    mesh = trimesh.load(mesh_path)
    mesh_parts = mesh.split()

    damaged_mesh = trimesh.util.concatenate(
        [part for idx, part in enumerate(mesh_parts) if idx != missing_part_idx])

    damaged_mesh.export(save_path)

if __name__ == "__main__":
    mesh_path = "../data/partnet_data/chair/model.obj"
    save_path = "../data/synthetic_damage/chair_missing_leg.obj"
    missing_part_idx = 2  # example index

    generate_missing_part(mesh_path, missing_part_idx, save_path)
