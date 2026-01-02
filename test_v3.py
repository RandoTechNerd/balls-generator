from balls_lib import generate_ball_v3
import trimesh

print("Testing V3 Generation...")
try:
    mesh = generate_ball_v3(
        diameter=60, 
        mechanism_type="Screw", 
        part_type="Top",
        shape_type="Hexagon"
    )
    print(f"Success! Mesh has {len(mesh.vertices)} vertices.")
    mesh.export("test_v3.stl")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
