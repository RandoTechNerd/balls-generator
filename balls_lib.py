import numpy as np
import math
import trimesh
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely import affinity
import shapely.ops
import mapbox_earcut

# --- CONSTANTS ---
DEFAULT_TOLERANCE = 0.4 
DEFAULT_PITCH = 2.0

# --- SHAPE GENERATORS (2D) ---
def create_2d_shape(shape_type, size):
    r = size / 2.0
    if shape_type == "Circle": return Point(0, 0).buffer(r, resolution=16)
    elif shape_type == "Triangle":
        return Polygon([(0, r), (r * math.sqrt(3)/2, -r/2), (-r * math.sqrt(3)/2, -r/2)])
    elif shape_type == "Diamond":
        return Polygon([(0, r), (r, 0), (0, -r), (-r, 0)])
    elif shape_type == "Hexagon":
        angle = math.pi / 3
        points = [(r * math.cos(i * angle + math.pi/6), r * math.sin(i * angle + math.pi/6)) for i in range(6)]
        return Polygon(points)
    elif shape_type == "Spade":
        s = size / 2.0
        tri = Polygon([(0, -s), (s*0.8, 0), (-s*0.8, 0)])
        c1 = Point(s*0.4, 0).buffer(s*0.43)
        c2 = Point(-s*0.4, 0).buffer(s*0.43)
        stem = Polygon([(-s*0.1, -s), (s*0.1, -s), (s*0.2, -s*1.2), (-s*0.2, -s*1.2)])
        return shapely.ops.unary_union([tri, c1, c2, stem])
    return Point(0, 0).buffer(r)

def generate_2d_pattern_sheet(radius_2d, shape_type="Circle", base_shape_size=8.0, min_gap=1.5, coverage=0.9):
    base_disc = Point(0, 0).buffer(radius_2d, resolution=64)
    holes = []
    max_r = (radius_2d * coverage) - (base_shape_size/2.0)
    step_size = base_shape_size + min_gap
    num_rings = int(math.floor(max_r / step_size))
    center_scale, edge_scale = 1.0, 0.5
    
    for i in range(num_rings + 1):
        r = i * step_size
        if r > max_r: break
        t = (i / num_rings) if num_rings > 0 else 0
        current_scale = center_scale + (edge_scale - center_scale) * t
        draw_size = max(0.1, base_shape_size * current_scale)
        circumference = 2 * math.pi * (0.1 if r == 0 else r)
        count = max(1, int(math.floor(circumference / (draw_size + min_gap))))
        angle_step = 360.0 / count
        for j in range(count):
            angle_deg = j * angle_step + (i * 15)
            cx = r * math.cos(math.radians(angle_deg))
            cy = r * math.sin(math.radians(angle_deg))
            shp = create_2d_shape(shape_type, draw_size)
            shp = affinity.rotate(shp, angle_deg + 90, origin=(0,0))
            shp = affinity.translate(shp, cx, cy)
            holes.append(shp)
                
    if holes:
        pattern_union = shapely.ops.unary_union(holes).buffer(0)
        final_2d = base_disc.difference(pattern_union)
    else:
        final_2d = base_disc
    return final_2d.buffer(0)

def fold_and_smooth(sheet_mesh, sphere_radius):
    v = sheet_mesh.vertices
    x, y = v[:, 0], v[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    phi = r / sphere_radius
    phi = np.clip(phi, 0, math.pi/2)
    x_3d = sphere_radius * np.sin(phi) * np.cos(theta)
    y_3d = sphere_radius * np.sin(phi) * np.sin(theta)
    z_3d = sphere_radius * np.cos(phi)
    sheet_mesh.vertices = np.stack([x_3d, y_3d, z_3d], axis=1)
    norms = np.linalg.norm(sheet_mesh.vertices, axis=1)
    mask = norms > 1e-6
    sheet_mesh.vertices[mask] = (sheet_mesh.vertices[mask] / norms[mask][:,None]) * sphere_radius
    return sheet_mesh

def generate_magnet_rim(radius, width, height, magnet_count, magnet_dia):
    outer_r = radius
    inner_r = radius - width
    ring = Point(0,0).buffer(outer_r, resolution=64).difference(Point(0,0).buffer(inner_r, resolution=64))
    holes = []
    angle_step = 360.0 / magnet_count
    magnet_r = (outer_r + inner_r) / 2.0
    for i in range(magnet_count):
        ang = math.radians(i * angle_step)
        cx = magnet_r * math.cos(ang)
        cy = magnet_r * math.sin(ang)
        holes.append(Point(cx, cy).buffer(magnet_dia/2.0, resolution=16))
    if holes:
        ring = ring.difference(shapely.ops.unary_union(holes))
    v, f = trimesh.creation.triangulate_polygon(ring)
    mesh_3d = trimesh.creation.extrude_polygon(ring, height)
    return mesh_3d

def generate_stabilizers_triangular(radius):
    fins = []
    gap = 0.6          
    fin_thick = 0.6    
    r_start = 6.0      
    r_end = radius * 0.7 
    
    points_top = []
    points_bottom = []
    r_vals = np.arange(r_start, r_end, 2.0)
    for r in r_vals:
        val = radius**2 - r**2
        if val < 0: val = 0
        z_surf = radius - math.sqrt(val)
        z_fin_top = max(0, z_surf - gap)
        points_top.append([r, z_fin_top])
        points_bottom.append([r, 0])
        
    poly_coords = points_top + points_bottom[::-1]
    if len(poly_coords) < 3: return trimesh.Trimesh()
    fin_poly = Polygon(poly_coords)
    fin_mesh = trimesh.creation.extrude_polygon(fin_poly, fin_thick)
    rot_align = trimesh.transformations.rotation_matrix(math.pi/2, [1, 0, 0])
    fin_mesh.apply_transform(rot_align)
    fin_mesh.apply_translation([0, -fin_thick/2, 0])
    
    contacts = []
    contact_interval = 2.0 
    for r in np.arange(r_start + 1.0, r_end, contact_interval):
        val = radius**2 - r**2
        if val < 0: val = 0
        z_surf = radius - math.sqrt(val)
        z_bot = max(0, z_surf - gap)
        height = (z_surf - z_bot) + 0.2 
        if height > 0:
            s = fin_thick
            v_pyr = [[0, 0, height], [s/2, s/2, 0], [s/2, -s/2, 0], [-s/2, -s/2, 0], [-s/2, s/2, 0]]
            f_pyr = [[0,1,2], [0,2,3], [0,3,4], [0,4,1], [1,4,3], [1,3,2]]
            pyr = trimesh.Trimesh(vertices=v_pyr, faces=f_pyr)
            pyr.apply_translation([r, 0, z_bot])
            contacts.append(pyr)
    branch_parts = [fin_mesh] + contacts
    branch = trimesh.util.concatenate(branch_parts)
    for i in range(4):
        f = branch.copy()
        angle = math.radians(i * 90)
        rot = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        f.apply_transform(rot)
        fins.append(f)
    return trimesh.util.concatenate(fins)

def generate_thread_profile_mesh(
    base_radius, 
    thread_height,      
    backing_height,     
    pitch, 
    tooth_depth, 
    tolerance, 
    internal=False, 
    backing_thickness=2.0, 
    extra_overlap=1.0
):
    segs = 128
    final_cyl_h = backing_height
            
    if not internal: # MALE
        r_inner = base_radius - backing_thickness
        r_root = base_radius 
        r_tip = base_radius + tooth_depth
        cyl = trimesh.creation.annulus(r_min=r_inner, r_max=r_root, height=final_cyl_h)
        r_root_thread = r_root - 0.4 # DEEP EMBED
        r_tip_thread = r_tip 
    else: # FEMALE
        r_root = base_radius
        r_outer = base_radius + backing_thickness
        r_tip = base_radius - tooth_depth
        cyl = trimesh.creation.annulus(r_min=r_root, r_max=r_outer, height=final_cyl_h)
        r_root_thread = r_root + 0.4 # DEEP EMBED
        r_tip_thread = r_tip 

    # 2. Thread Coil Generation
    effective_h = thread_height 
    turns = (effective_h + 1.0) / pitch 
    total_steps = int(turns * segs)
    coil_verts = []
    coil_faces = []
    start_z = 0.5 
    
    for i in range(total_steps): 
        angle = (i / segs) * 2 * math.pi
        z_rel = (i / segs) * pitch
        z_final = start_z + z_rel
        c, s = math.cos(angle), math.sin(angle)
        z_rb = z_final - pitch * 0.45
        z_tip = z_final
        z_rt = z_final + pitch * 0.45
        p_rb  = [r_root_thread * c, r_root_thread * s, z_rb]
        p_tip = [r_tip_thread * c,  r_tip_thread * s,  z_tip]
        p_rt  = [r_root_thread * c, r_root_thread * s, z_rt]
        idx = len(coil_verts)
        coil_verts.extend([p_rb, p_tip, p_rt])
        if i > 0:
            p_idx = idx - 3
            coil_faces.append([p_idx, p_idx+1, idx+1])
            coil_faces.append([p_idx, idx+1, idx])
            coil_faces.append([p_idx+1, p_idx+2, idx+2])
            coil_faces.append([p_idx+1, idx+2, idx+1])
            
    if len(coil_faces) > 0:
        coil = trimesh.Trimesh(vertices=coil_verts, faces=coil_faces)
        return trimesh.util.concatenate([cyl, coil])
    else:
        return cyl 

# --- MAIN GENERATOR V18 ---
def generate_ball_v18(
    mode="Dome",
    diameter=50.0,
    wall_thickness=2.0,
    shape_type="Circle",
    hole_size=8.0,
    hole_spacing=1.5,
    coverage=0.9,
    part_type="Top",
    mechanism_type="Screw",
    mag_count=4,
    mag_dia=6.0,
    thread_pitch=2.0,
    thread_tolerance=0.4,
    thread_len=6.0,
    backing_height=8.0, 
    add_supports=False,
    thread_depth=0.8,
    backing_thickness=2.0, 
    joint_offset=0.0,
    latch_offset=3.0    
):
    radius = diameter / 2.0
    if mode == "Dome": r_flat = (math.pi / 2.0) * radius
    else: r_flat = radius
    flat_poly = generate_2d_pattern_sheet(r_flat, shape_type, hole_size, hole_spacing, coverage)
    v, f = trimesh.creation.triangulate_polygon(flat_poly)
    mesh_2d = trimesh.Trimesh(vertices=np.column_stack([v, np.zeros(len(v))]), faces=f)
    
    target_len = 0.5 
    max_faces = 200000 
    for _ in range(5):
        if len(mesh_2d.faces) > max_faces: break
        edges = mesh_2d.vertices[mesh_2d.edges_unique]
        lengths = np.linalg.norm(edges[:,0]-edges[:,1], axis=1)
        if lengths.max() > target_len:
            mesh_2d = mesh_2d.subdivide()
        else: break
            
    if mode == "Flat": return trimesh.creation.extrude_polygon(flat_poly, wall_thickness)

    outer = fold_and_smooth(mesh_2d.copy(), radius)
    outer.fix_normals()
    inner = outer.copy()
    scale_factor = (radius - wall_thickness) / radius
    inner.vertices *= scale_factor
    inner.invert()
    inner.fix_normals()
    edges = outer.edges_sorted
    groups = trimesh.grouping.group_rows(edges, require_count=1)
    boundary_edges = edges[groups]
    wall_faces = []
    n_v = len(outer.vertices)
    for v1, v2 in boundary_edges:
        wall_faces.append([v1, v2, v2 + n_v])
        wall_faces.append([v1, v2 + n_v, v1 + n_v])
    all_v = np.vstack([outer.vertices, inner.vertices])
    all_f = np.vstack([outer.faces, inner.faces + n_v, wall_faces])
    dome_solid = trimesh.Trimesh(vertices=all_v, faces=all_f)
    dome_solid.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [0,1,0], point=[0,0,0]))
    dome_solid.apply_translation([0, 0, radius])
    
    parts = []
    
    if mechanism_type == "Hinge":
        d1 = dome_solid.copy()
        d2 = dome_solid.copy()
        gap = 2.0
        d1.apply_translation([-radius - gap/2, 0, 0])
        d2.apply_translation([radius + gap/2, 0, 0])
        parts.extend([d1, d2])
        
        bw = 6.0 
        h_thick = 4.0
        h_thin = 0.4
        def make_bridge_segment(x1, h1, x2, h2):
            z_top = radius
            verts = [[x1, -bw, z_top], [x1, bw, z_top], [x1, bw, z_top-h1], [x1, -bw, z_top-h1], 
                     [x2, -bw, z_top], [x2, bw, z_top], [x2, bw, z_top-h2], [x2, -bw, z_top-h2]]
            return trimesh.Trimesh(vertices=verts, faces=[[0,1,5],[0,5,4], [2,3,7],[2,7,6], [0,4,7],[0,7,3], [1,2,6],[1,6,5], [0,3,2],[0,2,1], [4,5,6],[4,6,7]])
        b_left = make_bridge_segment(-gap/2 - 1.0, h_thick, 0, h_thin)
        b_right = make_bridge_segment(0, h_thin, gap/2 + 1.0, h_thick)
        parts.append(b_left)
        parts.append(b_right)

        # --- LATCH V7 (Perfect Alignment) ---
        hook_w = 8.0
        hook_h = 8.0
        hook_th = 2.0
        base_x = (radius * 2) + (gap / 2)
        offset = latch_offset
        
        # 1. Finger Vertical Arm
        # SHIFTED OVER towards model by 'hook_th' to sit perfectly on buttress end.
        # Original logic: finger_center_x = base_x + offset + hook_th/2
        # New logic: finger_center_x = base_x + offset - hook_th/2
        finger_center_x = base_x + offset - hook_th/2
        
        finger = trimesh.creation.box([hook_th, hook_w, hook_h])
        finger.apply_translation([finger_center_x, 0, radius + hook_h/2])
        parts.append(finger)
        
        # 2. Buttress (Ramp)
        z_low = radius - 4.0
        z_high = radius
        x_wall = base_x
        x_finger_outer = base_x + offset
        
        v_buttress = [
            [x_wall, -hook_w/2, z_low], [x_wall, hook_w/2, z_low],
            [x_finger_outer, hook_w/2, z_high], [x_finger_outer, -hook_w/2, z_high],
            [x_wall, hook_w/2, z_high], [x_wall, -hook_w/2, z_high]
        ]
        buttress = trimesh.Trimesh(vertices=v_buttress, faces=[
            [0,1,2],[0,2,3], [0,3,5],[0,5,1], [1,5,4],[1,4,2], [3,2,4],[3,4,5]
        ])
        parts.append(buttress)
        
        # 3. Nub
        nub = trimesh.creation.icosphere(subdivisions=1, radius=1.2)
        # Position: Inner face of finger tip.
        # Finger Inner face is at (finger_center_x - hook_th/2)
        nub.apply_translation([finger_center_x - hook_th/2 - 0.2, 0, radius + hook_h - 2.0])
        parts.append(nub)
        
        # 4. Clasp (Wedge)
        clasp_base_x = -(radius * 2) - (gap / 2)
        clasp_len = 6.0
        clasp_h = 1.5
        clasp_w = 8.0
        z_start = radius - 4.0
        v_wedge = [
            [clasp_base_x, -clasp_w/2, z_start], 
            [clasp_base_x - clasp_h, -clasp_w/2, z_start + 2.0],
            [clasp_base_x - clasp_h, -clasp_w/2, z_start + 4.0],
            [clasp_base_x, -clasp_w/2, z_start + 6.0], 
            [clasp_base_x, clasp_w/2, z_start], 
            [clasp_base_x - clasp_h, clasp_w/2, z_start + 2.0],
            [clasp_base_x - clasp_h, clasp_w/2, z_start + 4.0],
            [clasp_base_x, clasp_w/2, z_start + 6.0]
        ]
        f_wedge = [[0,1,5],[0,5,4], [1,2,6],[1,6,5], [2,3,7],[2,7,6], [0,4,7],[0,7,3], [0,3,2],[0,2,1], [4,5,6],[4,6,7]]
        wedge_mesh = trimesh.Trimesh(vertices=v_wedge, faces=f_wedge)
        parts.append(wedge_mesh)

        if add_supports:
            parts.append(generate_stabilizers_triangular(radius).apply_translation([-radius - gap/2, 0, 0]))
            parts.append(generate_stabilizers_triangular(radius).apply_translation([radius + gap/2, 0, 0]))

    elif mechanism_type == "Magnets":
        rim_h = 4.0 
        mag_mesh = generate_magnet_rim(radius - wall_thickness + 0.5, 5.0, rim_h, mag_count, mag_dia)
        mag_mesh.apply_translation([0, 0, radius - rim_h])
        parts.append(dome_solid)
        parts.append(mag_mesh)
        if add_supports: parts.append(generate_stabilizers_triangular(radius))
            
    elif mechanism_type == "Screw":
        r_dome_inner = radius - wall_thickness
        r_interface = r_dome_inner - joint_offset
        lip_od = r_interface - thread_tolerance
        
        if part_type == "Top":
            if r_dome_inner > lip_od:
                conn = trimesh.creation.annulus(r_min=lip_od - 0.1, r_max=r_dome_inner + 0.1, height=wall_thickness)
                conn.apply_translation([0, 0, radius + wall_thickness/2])
                parts.append(conn)
            
            th_mesh = generate_thread_profile_mesh(
                base_radius=lip_od - thread_depth, 
                thread_height=thread_len, 
                backing_height=backing_height, 
                pitch=thread_pitch, 
                tooth_depth=thread_depth, 
                tolerance=0,
                internal=False,
                backing_thickness=backing_thickness
            )
            th_mesh.apply_translation([0, 0, radius + wall_thickness])
            parts.append(th_mesh)
        else:
            backing_needed = (r_dome_inner - r_interface) + backing_thickness
            th_mesh = generate_thread_profile_mesh(
                base_radius=r_interface,
                thread_height=thread_len,
                backing_height=backing_height,
                pitch=thread_pitch,
                tooth_depth=thread_depth,
                tolerance=0,
                internal=True,
                backing_thickness=backing_needed
            )
            th_mesh.apply_translation([0, 0, radius - backing_height]) 
            parts.append(th_mesh)
            
        parts.append(dome_solid)
        if add_supports: parts.append(generate_stabilizers_triangular(radius))

    return trimesh.util.concatenate(parts)
