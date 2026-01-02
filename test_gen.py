from balls_lib import generate_capsule
import os

print("Generating capsule...")
mesh = generate_capsule(diameter=50, mode="Dome", mechanism="Screw", pattern="Circle")
print(f"Mesh generated with {len(mesh.vectors)} faces.")

mesh.save("test_ball.stl")
size = os.path.getsize("test_ball.stl")
print(f"STL saved. Size: {size} bytes")

if size > 100:
    print("SUCCESS")
else:
    print("FAILURE")
