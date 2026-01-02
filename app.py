import streamlit as st
import os

# --- CLOUD FIX: HEADLESS OPENGL ---
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import plotly.graph_objects as go
import tempfile
import trimesh
from balls_lib import generate_ball_v18

# --- CONFIG ---
# Fix for Docker/Linux case sensitivity and working dir issues
current_dir = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(current_dir, "ball_icon.png")

st.set_page_config(page_title="BALLS! (v18)", page_icon=icon_path, layout="wide")

col_head1, col_head2 = st.columns([1, 6])
with col_head1:
    if os.path.exists(icon_path):
        st.image(icon_path, width=80)
    else:
        # Fallback if image fails (Cloud path issue?)
        st.write("⚽") 
        
with col_head2:
    st.title("BALLS! (v18)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    
    mode = st.radio("Mode", ["Dome", "Flat"], horizontal=True)
    
    # 1. Main Dimensions
    diameter = st.slider("Diameter (mm)", 30.0, 150.0, 80.0, 1.0)
    wall_thick = st.slider("Wall Thickness", 1.2, 5.0, 2.0, 0.2)
    
    # 2. Pattern
    st.subheader("Pattern")
    shape = st.selectbox("Shape", ["Circle", "Hexagon", "Triangle", "Diamond", "Spade"])
    
    col1, col2 = st.columns(2)
    with col1:
        hole_size = st.slider("Hole Size", 2.0, 20.0, 8.0, 0.5)
    with col2:
        spacing = st.slider("Spacing", 0.5, 10.0, 1.5, 0.5)
        
    coverage = st.slider("Pattern Coverage", 0.5, 1.0, 0.95, 0.05)
    
    # 3. Mechanism
    st.subheader("Mechanism")
    
    supports = st.toggle("Add Smart Supports", value=False, help="Recommended for Dome-Down printing")
    
    if mode == "Dome":
        mech_type = st.selectbox("Type", ["Screw", "Magnets", "Hinge", "None"])
        
        if mech_type == "Hinge":
            part = "Dual"
            pitch, tol, th_len, back_h = 2.0, 0.4, 6.0, 8.0
            mag_count, mag_dia = 4, 6.0
            th_depth, back_th, j_off, force_lip, cut_th = 0.8, 2.0, 0.0, False, False
            
            st.markdown("---")
            latch_off = st.slider("Latch Finger Offset (Snugness)", 0.0, 5.0, 3.0, 0.1, help="Moves the finger outwards for looser fit or inwards for tighter fit.")
            
        elif mech_type == "Screw":
            part = st.radio("Part Type", ["Top", "Bottom"], horizontal=True)
            with st.expander("Thread Settings", expanded=True):
                pitch = st.slider("Thread Pitch", 1.0, 5.0, 2.5, 0.1)
                tol = st.slider("Tolerance (mm)", 0.1, 1.0, 0.4, 0.05)
                
                # Split Controls
                th_len = st.slider("Thread Coil Length", 2.0, 15.0, 6.0, 1.0)
                back_h = st.slider("Backing Cylinder Height (Lip)", 2.0, 20.0, 8.0, 1.0, help="Controls the solid inner wall height.")
                
                th_depth = st.slider("Thread Depth", 0.4, 2.0, 0.8, 0.1)
                back_th = st.slider("Backing Thickness", 1.0, 5.0, 2.0, 0.5)
                j_off = st.slider("Joint Offset", 0.0, 5.0, 0.0, 0.5)
                
            mag_count, mag_dia = 4, 6.0
            latch_off = 3.0
            
        elif mech_type == "Magnets":
            part = "Top"
            with st.expander("Magnet Settings", expanded=True):
                mag_count = st.slider("Magnet Count", 2, 8, 4, 2)
                mag_dia = st.slider("Magnet Diameter", 2.0, 10.0, 6.0, 0.1)
            pitch, tol, th_len, back_h = 2.0, 0.4, 6.0, 8.0
            th_depth, back_th, j_off, latch_off = 0.8, 2.0, 0.0, 3.0
        else:
            part = "Top"
            pitch, tol, th_len, back_h = 2.0, 0.4, 6.0, 8.0
            mag_count, mag_dia = 4, 6.0
            th_depth, back_th, j_off, latch_off = 0.8, 2.0, 0.0, 3.0
    else:
        mech_type = "None"
        part = "Top"
        pitch, tol, th_len, back_h = 2.0, 0.4, 6.0, 8.0
        mag_count, mag_dia = 4, 6.0
        th_depth, back_th, j_off, latch_off = 0.8, 2.0, 0.0, 3.0
        
    st.divider()
    if st.button("Generate Ball V18", type="primary"):
        st.session_state.needs_refresh = True

# --- GENERATION ---
if 'mesh_obj' not in st.session_state: 
    st.session_state.mesh_obj = None
    st.session_state.needs_refresh = False

if st.session_state.get('needs_refresh', False):
    with st.spinner("Processing Geometry..."):
        try:
            mesh = generate_ball_v18(
                mode=mode,
                diameter=diameter,
                wall_thickness=wall_thick,
                shape_type=shape,
                hole_size=hole_size,
                hole_spacing=spacing,
                coverage=coverage,
                part_type=part.split()[0],
                mechanism_type=mech_type,
                mag_count=mag_count,
                mag_dia=mag_dia,
                thread_pitch=pitch,
                thread_tolerance=tol,
                thread_len=th_len,   
                backing_height=back_h,   
                add_supports=supports,
                thread_depth=th_depth,
                backing_thickness=back_th,
                joint_offset=j_off,
                latch_offset=latch_off
            )
            st.session_state.mesh_obj = mesh
            st.session_state.error = None
        except Exception as e:
            st.session_state.error = str(e)
            st.error(f"Error: {e}")
            
        st.session_state.needs_refresh = False

# --- VISUALIZATION ---
if st.session_state.mesh_obj:
    mesh = st.session_state.mesh_obj
    
    view_mesh = mesh.copy()
    target_view_faces = 15000
    if len(view_mesh.faces) > target_view_faces:
        try:
             view_mesh = view_mesh.simplify_quadratic_decimation(target_view_faces)
        except Exception:
             pass

    v = view_mesh.vertices
    f = view_mesh.faces
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=v[:,0], y=v[:,1], z=v[:,2],
            i=f[:,0], j=f[:,1], k=f[:,2],
            color='#00cec9', 
            opacity=1.0, 
            flatshading=False
        )
    ])
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # Download
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        mesh.export(tmp.name)
        tmp_path = tmp.name
        
    fn_mode = "flat" if mode == "Flat" else f"dome_{mech_type}"
    with open(tmp_path, "rb") as f:
        st.download_button("Download STL", f, f"ball_v18_{fn_mode}.stl")
    os.remove(tmp_path)