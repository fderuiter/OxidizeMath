use super::types::{Mesh, Point3D};
use std::fmt::Write;
use std::io::{Error, ErrorKind};

const STL_HEADER_LEN: usize = 80;
const STL_FACET_BYTES: usize = 50;
const STL_HEADER_PREFIX: &[u8] = b"OxidizeMath Binary STL (units: mm)";

/// Serializes an indexed [`Mesh`] to a Wavefront `.obj` string with vertex normals.
pub fn export_mesh_to_obj_string(mesh: &Mesh) -> Result<String, Error> {
    let estimated_size =
        mesh.vertices.len() * 25 + mesh.normals.len() * 25 + (mesh.indices.len() / 3) * 30 + 100;
    let mut out = String::with_capacity(estimated_size);
    let has_vertex_normals = mesh.normals.len() >= mesh.vertices.len();

    out.push_str("# OxidizeMath Wavefront OBJ (units: mm)\n");
    out.push_str("o oxidize_mesh\n");
    for v in &mesh.vertices {
        let _ = writeln!(out, "v {} {} {}", v.x, v.y, v.z);
    }
    for n in &mesh.normals {
        let sanitized = sanitize_unit_vector(*n, Point3D::new(0.0, 0.0, 1.0));
        let _ = writeln!(out, "vn {} {} {}", sanitized.x, sanitized.y, sanitized.z);
    }
    for &[c0, c1, c2] in mesh.indices.as_chunks::<3>().0 {
        let i1 = c0 + 1;
        let i2 = c1 + 1;
        let i3 = c2 + 1;
        if has_vertex_normals {
            let _ = writeln!(out, "f {i1}//{i1} {i2}//{i2} {i3}//{i3}");
        } else {
            let _ = writeln!(out, "f {i1} {i2} {i3}");
        }
    }

    Ok(out)
}

/// Serializes an indexed [`Mesh`] to a standard little-endian binary `.stl` byte vector
/// (`80`-byte header, `u32` facet count, and `50`-byte triangle records in millimeters).
pub fn export_mesh_to_stl_bytes(mesh: &Mesh) -> Result<Vec<u8>, Error> {
    let tri_count = mesh.indices.len() / 3;
    let tri_count_u32 = u32::try_from(tri_count)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "mesh exceeds u32 STL facet limit"))?;
    let v_len = mesh.vertices.len();
    for &idx in mesh.indices.iter().take(tri_count * 3) {
        if idx >= v_len {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "mesh triangle index out of bounds",
            ));
        }
    }

    let mut bytes = Vec::with_capacity(STL_HEADER_LEN + 4 + tri_count * STL_FACET_BYTES);
    let mut header = [0_u8; STL_HEADER_LEN];
    header[..STL_HEADER_PREFIX.len()].copy_from_slice(STL_HEADER_PREFIX);
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&tri_count_u32.to_le_bytes());

    for tri in mesh.indices.as_chunks::<3>().0 {
        let p0 = sanitize_point(mesh.vertices[tri[0]]);
        let p1 = sanitize_point(mesh.vertices[tri[1]]);
        let p2 = sanitize_point(mesh.vertices[tri[2]]);
        let normal = facet_normal(mesh, tri, p0, p1, p2);
        write_point_le(&mut bytes, normal);
        write_point_le(&mut bytes, p0);
        write_point_le(&mut bytes, p1);
        write_point_le(&mut bytes, p2);
        bytes.extend_from_slice(&0_u16.to_le_bytes());
    }

    Ok(bytes)
}

fn facet_normal(mesh: &Mesh, tri: &[usize], p0: Point3D, p1: Point3D, p2: Point3D) -> Point3D {
    let geom = (p1 - p0).cross(p2 - p0);
    let geom_mag = geom.magnitude();
    if geom_mag > Point3D::EPSILON && geom_mag.is_finite() {
        return geom / geom_mag;
    }
    if mesh.normals.len() > tri[0] && mesh.normals.len() > tri[1] && mesh.normals.len() > tri[2] {
        let avg = mesh.normals[tri[0]] + mesh.normals[tri[1]] + mesh.normals[tri[2]];
        return sanitize_unit_vector(avg, Point3D::new(0.0, 0.0, 1.0));
    }
    Point3D::new(0.0, 0.0, 1.0)
}

fn sanitize_point(p: Point3D) -> Point3D {
    Point3D::new(
        if p.x.is_finite() { p.x } else { 0.0 },
        if p.y.is_finite() { p.y } else { 0.0 },
        if p.z.is_finite() { p.z } else { 0.0 },
    )
}

fn sanitize_unit_vector(v: Point3D, fallback: Point3D) -> Point3D {
    if !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite() {
        return fallback;
    }
    let mag = v.magnitude();
    if mag > Point3D::EPSILON && mag.is_finite() {
        v / mag
    } else {
        fallback
    }
}

fn write_point_le(out: &mut Vec<u8>, p: Point3D) {
    out.extend_from_slice(&p.x.to_le_bytes());
    out.extend_from_slice(&p.y.to_le_bytes());
    out.extend_from_slice(&p.z.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::analyze_mesh;

    fn unit_tetrahedron_mm(scale: f32) -> Mesh {
        let vertices = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(scale, 0.0, 0.0),
            Point3D::new(0.0, scale, 0.0),
            Point3D::new(0.0, 0.0, scale),
        ];
        // Outward CCW winding for all 4 faces
        let indices = vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
        let normals = vec![Point3D::new(0.0, 0.0, 1.0); 4];
        Mesh {
            vertices,
            normals,
            indices,
        }
    }

    #[test]
    fn test_binary_stl_header_facet_layout_and_watertight_diagnostics() {
        let mesh = unit_tetrahedron_mm(10.0);
        let diag = analyze_mesh(&mesh);
        assert!(diag.is_watertight());
        assert_eq!(diag.open_edges, 0);
        assert_eq!(diag.nan_normals, 0);
        assert_eq!(diag.dimensions_mm, Point3D::new(10.0, 10.0, 10.0));
        assert!((diag.signed_volume_mm3 - (1000.0 / 6.0)).abs() < 1e-4);

        let stl = export_mesh_to_stl_bytes(&mesh).expect("stl export");
        assert_eq!(stl.len(), 84 + 4 * 50);
        assert!(!stl[..80].starts_with(b"solid "));
        let count = u32::from_le_bytes(stl[80..84].try_into().unwrap());
        assert_eq!(count, 4);

        for f in 0..4 {
            let base = 84 + f * 50;
            let nx = f32::from_le_bytes(stl[base..base + 4].try_into().unwrap());
            let ny = f32::from_le_bytes(stl[base + 4..base + 8].try_into().unwrap());
            let nz = f32::from_le_bytes(stl[base + 8..base + 12].try_into().unwrap());
            let mag = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!((mag - 1.0).abs() < 1e-5);
            let attr = u16::from_le_bytes(stl[base + 48..base + 50].try_into().unwrap());
            assert_eq!(attr, 0);
        }
    }

    #[test]
    fn test_stl_and_obj_sanitize_degenerate_normals() {
        let mesh = Mesh {
            vertices: vec![
                Point3D::new(0.0, 0.0, 0.0),
                Point3D::new(0.0, 0.0, 0.0),
                Point3D::new(0.0, 0.0, 0.0),
            ],
            normals: vec![Point3D::new(f32::NAN, 0.0, 0.0); 3],
            indices: vec![0, 1, 2],
        };
        let stl = export_mesh_to_stl_bytes(&mesh).expect("stl degenerate");
        let nz = f32::from_le_bytes(stl[92..96].try_into().unwrap());
        assert_eq!(nz, 1.0);
        let obj = export_mesh_to_obj_string(&mesh).expect("obj degenerate");
        assert!(!obj.contains("NaN"));
    }
}
