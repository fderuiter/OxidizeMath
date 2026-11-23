use super::*;
use std::collections::{HashMap, HashSet};

fn count_open_boundary_edges(mesh: &Mesh) -> usize {
    let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
    let (triangles, _) = mesh.indices.as_chunks::<3>();
    for &[a, b, c] in triangles {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            *directed.entry((u, v)).or_insert(0) += 1;
        }
    }
    directed
        .iter()
        .filter(|&(&(u, v), &cnt)| directed.get(&(v, u)).copied().unwrap_or(0) != cnt)
        .count()
}

fn euler_characteristic(mesh: &Mesh) -> isize {
    let mut undirected: HashSet<(usize, usize)> = HashSet::new();
    let (triangles, _) = mesh.indices.as_chunks::<3>();
    for &[a, b, c] in triangles {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            undirected.insert(if u < v { (u, v) } else { (v, u) });
        }
    }
    (mesh.vertices.len() as isize) - (undirected.len() as isize) + (triangles.len() as isize)
}

fn signed_volume(mesh: &Mesh) -> f32 {
    let (triangles, _) = mesh.indices.as_chunks::<3>();
    let mut vol6 = 0.0_f32;
    for &[i0, i1, i2] in triangles {
        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];
        vol6 += v0.dot(v1.cross(v2));
    }
    vol6 / 6.0
}

fn assert_closed_orientable_manifold(mesh: &Mesh, expected_euler: isize) {
    assert!(!mesh.vertices.is_empty());
    assert_eq!(mesh.vertices.len(), mesh.normals.len());
    assert!(mesh.vertices.iter().all(Point3D::is_finite));
    assert!(
        mesh.normals
            .iter()
            .all(|n| n.is_finite() && (n.magnitude() - 1.0).abs() < 1.0e-3)
    );
    assert_eq!(count_open_boundary_edges(mesh), 0, "Expected 0 open edges");
    assert_eq!(euler_characteristic(mesh), expected_euler);
    assert!(
        signed_volume(mesh) > 0.0,
        "Expected positive outward-wound signed volume, got {}",
        signed_volume(mesh)
    );
}

#[test]
fn test_stl_binary_size_and_ascii_round_trip() {
    let sphere = build_uv_sphere(Point3D::new(0.0, 0.0, 0.0), 1.5, 16, 12).expect("sphere");
    let face_count = sphere.indices.len() / 3;

    let stl_bin = export_mesh_to_stl_binary(&sphere).expect("binary STL");
    assert_eq!(stl_bin.len(), 84 + 50 * face_count);
    assert!(!stl_bin[..5].eq_ignore_ascii_case(b"solid"));
    let stored_faces = u32::from_le_bytes(stl_bin[80..84].try_into().unwrap()) as usize;
    assert_eq!(stored_faces, face_count);

    for f in 0..face_count {
        let base = 84 + f * 50;
        for k in 0..12 {
            let off = base + k * 4;
            let val = f32::from_le_bytes(stl_bin[off..off + 4].try_into().unwrap());
            assert!(val.is_finite(), "Non-finite STL float at facet {f}");
        }
    }

    let stl_ascii = export_mesh_to_stl_ascii(&sphere).expect("ASCII STL");
    let parsed = parse_stl_ascii(&stl_ascii).expect("parse ASCII STL");
    assert_eq!(parsed.indices.len() / 3, face_count);
    assert!((signed_volume(&parsed) - signed_volume(&sphere)).abs() < 1.0e-2);
}

#[test]
fn test_colored_ply_binary_and_ascii_serializers() {
    let cone = build_cone(
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(0.0, 0.0, 2.0),
        1.0,
        12,
    )
    .expect("cone");
    let colors: Vec<VertexColor> = (0..cone.vertices.len())
        .map(|i| VertexColor::new((i * 17) as u8, 128, 255 - (i * 11) as u8))
        .collect();
    let colored = ColoredMesh::new(cone.clone(), colors);

    let ply_bin = export_colored_mesh_to_ply_binary(&colored).expect("colored binary PLY");
    let header_end = ply_bin
        .windows(11)
        .position(|w| w == b"end_header\n")
        .expect("PLY end_header")
        + 11;
    let header_str = std::str::from_utf8(&ply_bin[..header_end]).unwrap();
    assert!(header_str.contains("format binary_little_endian 1.0"));
    assert!(header_str.contains("property uchar red"));
    let expected_payload = cone.vertices.len() * 27 + (cone.indices.len() / 3) * 13;
    assert_eq!(ply_bin.len() - header_end, expected_payload);

    let ply_ascii = export_colored_mesh_to_ply_ascii(&colored).expect("colored ASCII PLY");
    assert!(ply_ascii.contains("format ascii 1.0"));
    assert!(ply_ascii.contains("128"));

    let uncolored_bin = export_mesh_to_ply_binary(&cone).expect("uncolored binary PLY");
    assert!(!uncolored_bin.is_empty());
    let uncolored_ascii = export_mesh_to_ply_ascii(&cone).expect("uncolored ASCII PLY");
    assert!(uncolored_ascii.contains("element face 24"));
}

#[test]
fn test_obj_hardening_rejects_nan_vertices_and_replaces_degenerate_normals() {
    let mut bad_mesh = Mesh {
        vertices: vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(f32::NAN, 1.0, 0.0),
        ],
        normals: vec![Point3D::new(0.0, 0.0, 1.0); 3],
        indices: vec![0, 1, 2],
    };
    assert!(export_mesh_to_obj_string(&bad_mesh).is_err());
    assert!(export_mesh_to_stl_binary(&bad_mesh).is_err());
    assert!(export_mesh_to_ply_binary(&bad_mesh).is_err());

    bad_mesh.vertices[2] = Point3D::new(0.0, 1.0, 0.0);
    bad_mesh.normals = vec![
        Point3D::new(f32::NAN, 0.0, 0.0),
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(f32::INFINITY, 0.0, 0.0),
    ];
    let obj = export_mesh_to_obj_string(&bad_mesh).expect("sanitized OBJ");
    assert!(!obj.contains("NaN") && !obj.contains("inf"));
    assert!(obj.contains("vn 0 0 1"));
}

#[test]
fn test_3dgs_splat_ply_binary_encoding_and_validation() {
    let splat = GaussianSplatPly {
        mean: Point3D::new(1.25, -0.5, 2.0),
        scale: Point3D::new(0.25, 0.5, 1.0),
        rotation: [2.0, 0.0, 0.0, 0.0],
        opacity: 0.8,
        color: [0.5, 1.0, 0.0],
    };
    let bytes = export_splats_to_ply_binary(&[splat]).expect("3DGS PLY");
    let header_end = bytes
        .windows(11)
        .position(|w| w == b"end_header\n")
        .expect("end_header")
        + 11;
    let header = std::str::from_utf8(&bytes[..header_end]).unwrap();
    for prop in ["f_dc_0", "opacity", "scale_0", "rot_0", "rot_3"] {
        assert!(header.contains(prop), "Missing 3DGS property {prop}");
    }
    assert_eq!(bytes.len() - header_end, 68);

    let payload = &bytes[header_end..];
    let vals: Vec<f32> = (0..17)
        .map(|i| f32::from_le_bytes(payload[i * 4..(i + 1) * 4].try_into().unwrap()))
        .collect();
    assert!((vals[0] - 1.25).abs() < 1.0e-6);
    assert!(vals[6].abs() < 1.0e-6);
    assert!((vals[7] - 0.5 / SH_C0).abs() < 1.0e-4);
    assert!((vals[9] - (0.8_f32 / 0.2_f32).ln()).abs() < 1.0e-5);
    assert!((vals[10] - 0.25_f32.ln()).abs() < 1.0e-5);
    assert!(vals[12].abs() < 1.0e-6);
    assert!((vals[13] - 1.0).abs() < 1.0e-6);

    assert!(export_splats_to_ply_binary(&[]).is_err());
}

#[test]
fn test_primitives_and_tubes_are_watertight_2_manifolds() {
    let sphere = build_uv_sphere(Point3D::new(0.0, 0.0, 0.0), 1.0, 20, 14).unwrap();
    assert_eq!(sphere.normals[0], Point3D::new(0.0, 0.0, 1.0));
    assert_eq!(
        *sphere.normals.last().unwrap(),
        Point3D::new(0.0, 0.0, -1.0)
    );
    assert_closed_orientable_manifold(&sphere, 2);

    let cyl = build_cylinder(
        Point3D::new(-1.0, 0.0, 0.0),
        Point3D::new(1.0, 0.5, -0.5),
        0.3,
        16,
    )
    .unwrap();
    assert_closed_orientable_manifold(&cyl, 2);

    let cone = build_cone(
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(0.0, 2.0, 0.0),
        0.6,
        16,
    )
    .unwrap();
    assert_closed_orientable_manifold(&cone, 2);

    let arrow = build_arrow(
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(0.5, 0.5, 1.2),
        0.05,
        0.12,
        0.3,
        16,
    )
    .unwrap();
    assert_closed_orientable_manifold(&arrow, 2);

    let helix: Vec<Point3D> = (0..40)
        .map(|i| {
            let t = i as f32 * 0.2;
            Point3D::new(t.cos(), t.sin(), t * 0.3)
        })
        .collect();
    let open_tube = extrude_polyline_tube(&helix, 0.12, 64, false).unwrap();
    assert_eq!(open_tube.vertices.len(), 40 * MAX_TUBE_RADIAL_SEGMENTS + 2);
    assert_closed_orientable_manifold(&open_tube, 2);

    let ring_pts: Vec<Point3D> = (0..24)
        .map(|i| {
            let t = (i as f32) * std::f32::consts::TAU / 24.0;
            Point3D::new(2.0 * t.cos(), 2.0 * t.sin(), 0.3 * (2.0 * t).sin())
        })
        .collect();
    let closed_tube = extrude_polyline_tube(&ring_pts, 0.2, 12, true).unwrap();
    assert_closed_orientable_manifold(&closed_tube, 0);
}
