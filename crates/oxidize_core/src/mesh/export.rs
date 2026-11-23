use super::types::{Mesh, Point3D};
use std::fmt::Write;
use std::io::{Error, ErrorKind};
const STL_HEADER_LEN: usize = 80;
const STL_FACET_BYTES: usize = 50;
const STL_HEADER_PREFIX: &[u8] = b"OxidizeMath Binary STL (units: mm)";

const FALLBACK_NORMAL: Point3D = Point3D {
    x: 0.0,
    y: 0.0,
    z: 1.0,
};

pub(super) fn invalid_mesh_error(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

pub(super) fn validate_mesh_geometry(mesh: &Mesh, require_faces: bool) -> Result<(), Error> {
    if mesh.vertices.is_empty() && (require_faces || !mesh.indices.is_empty()) {
        return Err(invalid_mesh_error("Mesh contains no vertices"));
    }
    for (idx, v) in mesh.vertices.iter().enumerate() {
        if !v.is_finite() {
            return Err(invalid_mesh_error(&format!(
                "Mesh vertex {idx} contains non-finite coordinates"
            )));
        }
    }
    if !mesh.indices.len().is_multiple_of(3) {
        return Err(invalid_mesh_error(
            "Mesh index buffer length must be a multiple of 3",
        ));
    }
    if require_faces && mesh.indices.is_empty() {
        return Err(invalid_mesh_error("Mesh contains zero triangles"));
    }
    for &v_idx in &mesh.indices {
        if v_idx >= mesh.vertices.len() {
            return Err(invalid_mesh_error(&format!(
                "Mesh face index {v_idx} out of bounds for {} vertices",
                mesh.vertices.len()
            )));
        }
    }
    Ok(())
}

pub(super) fn resolve_triangle_normal(mesh: &Mesh, i0: usize, i1: usize, i2: usize) -> Point3D {
    let v0 = mesh.vertices[i0];
    let v1 = mesh.vertices[i1];
    let v2 = mesh.vertices[i2];
    let cross = (v1 - v0).cross(v2 - v0);
    if cross.is_finite() && cross.magnitude() > Point3D::EPSILON {
        return cross.normalize();
    }
    let mut sum = Point3D::new(0.0, 0.0, 0.0);
    for idx in [i0, i1, i2] {
        if let Some(vn) = mesh.normals.get(idx)
            && vn.is_finite()
            && vn.magnitude() > Point3D::EPSILON
        {
            sum = sum + vn.normalize();
        }
    }
    if sum.is_finite() && sum.magnitude() > Point3D::EPSILON {
        sum.normalize()
    } else {
        FALLBACK_NORMAL
    }
}

pub(super) fn sanitized_vertex_normals(mesh: &Mesh) -> Vec<Point3D> {
    let mut facet_accum = vec![Point3D::new(0.0, 0.0, 0.0); mesh.vertices.len()];
    let (triangles, _) = mesh.indices.as_chunks::<3>();
    for &[i0, i1, i2] in triangles {
        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];
        let fn_raw = (v1 - v0).cross(v2 - v0);
        let fn_safe = if fn_raw.is_finite() && fn_raw.magnitude() > Point3D::EPSILON {
            fn_raw.normalize()
        } else {
            FALLBACK_NORMAL
        };
        facet_accum[i0] = facet_accum[i0] + fn_safe;
        facet_accum[i1] = facet_accum[i1] + fn_safe;
        facet_accum[i2] = facet_accum[i2] + fn_safe;
    }

    (0..mesh.vertices.len())
        .map(|idx| {
            if let Some(n) = mesh.normals.get(idx)
                && n.is_finite()
                && n.magnitude() > Point3D::EPSILON
            {
                n.normalize()
            } else if facet_accum[idx].is_finite()
                && facet_accum[idx].magnitude() > Point3D::EPSILON
            {
                facet_accum[idx].normalize()
            } else {
                FALLBACK_NORMAL
            }
        })
        .collect()
}

/// Serializes an indexed [`Mesh`] to a Wavefront `.obj` string with sanitized vertex normals.
pub fn export_mesh_to_obj_string(mesh: &Mesh) -> Result<String, Error> {
    validate_mesh_geometry(mesh, false)?;
    let normals = sanitized_vertex_normals(mesh);
    let (triangles, _) = mesh.indices.as_chunks::<3>();
    let estimated_size = mesh.vertices.len() * 25 + normals.len() * 25 + triangles.len() * 30 + 100;
    let mut out = String::with_capacity(estimated_size);

    out.push_str("# OxidizeMath Wavefront OBJ (units: mm)\n");
    out.push_str("o oxidize_mesh\n");
    for v in &mesh.vertices {
        let _ = writeln!(out, "v {} {} {}", v.x, v.y, v.z);
    }
    for n in &normals {
        let _ = writeln!(out, "vn {} {} {}", n.x, n.y, n.z);
    }
    for &[c0, c1, c2] in triangles {
        let i1 = c0 + 1;
        let i2 = c1 + 1;
        let i3 = c2 + 1;
        let _ = writeln!(out, "f {i1}//{i1} {i2}//{i2} {i3}//{i3}");
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
