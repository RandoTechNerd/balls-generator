from balls_lib import generate_flat_disc, generate_dome
import trimesh

print("Testing Flat Mode...")
try:
    mesh_flat = generate_flat_disc(diameter=50, shape_type="Spade")
    print(f"Flat Mesh: {len(mesh_flat.vertices)} verts")
    if not mesh_flat.is_watertight:
        print("Warning: Flat mesh not watertight")
    mesh_flat.export("test_flat.stl")
except Exception as e:
    print(f"FLAT FAIL: {e}")

print("Testing Dome Mode...")
try:
    mesh_dome = generate_dome(diameter=50, shape_type="Circle", part_type="Top")
    print(f"Dome Mesh: {len(mesh_dome.vertices)} verts")
    mesh_dome.export("test_dome.stl")
except Exception as e:
    print(f"DOME FAIL: {e}")
