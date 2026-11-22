use super::types::{Mesh, Point3D};
use std::fmt::Write;
use std::io::{Error, ErrorKind};

const STL_HEADER_BYTES: usize = 80;
const STL_TRIANGLE_BYTES: usize = 50;
const STL_HEADER_LABEL: &[u8] = b"OxidizeMath Binary STL Mesh";

#[allow(missing_docs)]
pub fn export_mesh_to_obj_string(mesh: &Mesh) -> Result<String, std::io::Error> {
    let estimated_size =
        mesh.vertices.len() * 25 + mesh.normals.len() * 25 + (mesh.indices.len() / 3) * 30 + 100;
    let mut out = String::with_capacity(estimated_size);

    out.push_str("o oxidize_mesh\n");
    for v in &mesh.vertices {
        let _ = writeln!(out, "v {} {} {}", v.x, v.y, v.z);
    }
    for n in &mesh.normals {
        let _ = writeln!(out, "vn {} {} {}", n.x, n.y, n.z);
    }
    for chunk in mesh.indices.chunks(3) {
        if chunk.len() == 3 {
            let i1 = chunk[0] + 1;
            let i2 = chunk[1] + 1;
            let i3 = chunk[2] + 1;
            let _ = writeln!(out, "f {}//{} {}//{} {}//{}", i1, i1, i2, i2, i3, i3);
        }
    }

    Ok(out)
}

/// Serializes a triangle [`Mesh`] into the standard binary STL byte layout:
/// an 80-byte header, a 4-byte little-endian `u32` triangle count, and
/// `50 * triangle_count` bytes of facet records (`normal`, `v0`, `v1`, `v2`, `attr`).
pub fn export_mesh_to_binary_stl(mesh: &Mesh) -> Result<Vec<u8>, std::io::Error> {
    if !mesh.indices.len().is_multiple_of(3) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "mesh index buffer length must be a multiple of 3",
        ));
    }
    let tri_count = mesh.indices.len() / 3;
    let tri_count_u32 = u32::try_from(tri_count).map_err(|_| {
        Error::new(
            ErrorKind::InvalidInput,
            "triangle count exceeds binary STL u32 capacity",
        )
    })?;

    let mut out = Vec::with_capacity(STL_HEADER_BYTES + 4 + tri_count * STL_TRIANGLE_BYTES);
    let mut header = [0_u8; STL_HEADER_BYTES];
    header[..STL_HEADER_LABEL.len()].copy_from_slice(STL_HEADER_LABEL);
    out.extend_from_slice(&header);
    out.extend_from_slice(&tri_count_u32.to_le_bytes());

    let vertex_count = mesh.vertices.len();
    for &[i0, i1, i2] in mesh.indices.as_chunks::<3>().0 {
        if i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "triangle vertex index out of bounds",
            ));
        }
        let v0 = sanitize_point(mesh.vertices[i0]);
        let v1 = sanitize_point(mesh.vertices[i1]);
        let v2 = sanitize_point(mesh.vertices[i2]);
        let normal = compute_facet_normal(mesh, [i0, i1, i2], [v0, v1, v2]);

        write_point_le(&mut out, normal);
        write_point_le(&mut out, v0);
        write_point_le(&mut out, v1);
        write_point_le(&mut out, v2);
        out.extend_from_slice(&0_u16.to_le_bytes());
    }

    Ok(out)
}

#[inline]
fn sanitize_f32(val: f32) -> f32 {
    if !val.is_finite() || val == 0.0 {
        0.0
    } else {
        val
    }
}

#[inline]
fn sanitize_point(p: Point3D) -> Point3D {
    Point3D::new(sanitize_f32(p.x), sanitize_f32(p.y), sanitize_f32(p.z))
}

#[inline]
fn write_point_le(out: &mut Vec<u8>, p: Point3D) {
    out.extend_from_slice(&p.x.to_le_bytes());
    out.extend_from_slice(&p.y.to_le_bytes());
    out.extend_from_slice(&p.z.to_le_bytes());
}

fn compute_facet_normal(mesh: &Mesh, idx: [usize; 3], verts: [Point3D; 3]) -> Point3D {
    let geom = (verts[1] - verts[0]).cross(verts[2] - verts[0]);
    if geom.magnitude() > Point3D::EPSILON {
        return sanitize_point(geom.normalize());
    }
    if mesh.normals.len() == mesh.vertices.len() {
        let avg = sanitize_point(mesh.normals[idx[0]])
            + sanitize_point(mesh.normals[idx[1]])
            + sanitize_point(mesh.normals[idx[2]]);
        return sanitize_point(avg.normalize());
    }
    Point3D::new(0.0, 0.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tetrahedron() -> Mesh {
        let vertices = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(0.0, 0.0, 1.0),
        ];
        let normals = vertices.clone();
        let indices = vec![0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3];
        Mesh {
            vertices,
            normals,
            indices,
        }
    }

    fn read_f32_le(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn test_binary_stl_exact_layout_and_finite_records() {
        let mesh = sample_tetrahedron();
        let bytes = export_mesh_to_binary_stl(&mesh).expect("export binary stl");
        assert_eq!(bytes.len(), 84 + 50 * 4);
        assert!(!bytes[..5].eq_ignore_ascii_case(b"solid"));
        let tri_count = u32::from_le_bytes(bytes[80..84].try_into().unwrap());
        assert_eq!(tri_count, 4);

        for tri_idx in 0..4 {
            let base = 84 + tri_idx * 50;
            for float_idx in 0..12 {
                let val = read_f32_le(&bytes, base + float_idx * 4);
                assert!(val.is_finite(), "record float must be finite");
            }
            let attr = u16::from_le_bytes(bytes[base + 48..base + 50].try_into().unwrap());
            assert_eq!(attr, 0);
        }
    }

    #[test]
    fn test_binary_stl_sanitizes_non_finite_and_rejects_bad_indices() {
        let nan_mesh = Mesh {
            vertices: vec![
                Point3D::new(f32::NAN, -0.0, 1.0),
                Point3D::new(1.0, 0.0, 0.0),
                Point3D::new(0.0, 1.0, f32::INFINITY),
            ],
            normals: vec![Point3D::new(0.0, 0.0, 1.0); 3],
            indices: vec![0, 1, 2],
        };
        let bytes = export_mesh_to_binary_stl(&nan_mesh).expect("sanitize non-finite");
        assert_eq!(bytes.len(), 134);
        for float_idx in 0..12 {
            assert!(read_f32_le(&bytes, 84 + float_idx * 4).is_finite());
        }

        let bad_count = Mesh {
            vertices: nan_mesh.vertices.clone(),
            normals: vec![],
            indices: vec![0, 1],
        };
        assert!(export_mesh_to_binary_stl(&bad_count).is_err());

        let oob = Mesh {
            vertices: nan_mesh.vertices,
            normals: vec![],
            indices: vec![0, 1, 9],
        };
        assert!(export_mesh_to_binary_stl(&oob).is_err());
    }
}
