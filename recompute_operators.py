"""Recompute Poisson operators (cupy-compatible) for MF precomputes."""
import os, sys, glob, pickle
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
from utils.mesh_utils import get_mesh_operators

precompute_dir = "/data/sihun/multiface_align/precomputes"
obj_dir = "/data/sihun/multiface_align/obj"

for obj_file in sorted(glob.glob(os.path.join(obj_dir, "*.obj"))):
    name = os.path.basename(obj_file).replace(".obj", "").replace("_mesh", "")
    out_path = os.path.join(precompute_dir, f"{name}_operators.pkl")

    print(f"Recomputing: {name}")
    mesh = trimesh.load(obj_file, maintain_order=True, process=False)
    operators = get_mesh_operators(mesh)

    with open(out_path, 'wb') as f:
        pickle.dump(operators, f)
    print(f"  Saved: {out_path}")

print("Done!")
