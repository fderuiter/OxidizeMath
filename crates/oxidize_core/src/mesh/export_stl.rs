//! Binary and ASCII stereolithography (`.stl`) serializers and round-trip parser.

use super::export::{invalid_mesh_error, resolve_triangle_normal, validate_mesh_geometry};
use super::types::{Mesh, Point3D};
use std::fmt::Write as _;
use std::io::{Error, ErrorKind};

const STL_HEADER_TEXT: &[u8] = b"OxidizeMath binary STL export";

fn write_point_le(buf: &mut Vec<u8>, pt: Point3D) {
    buf.extend_from_slice(&pt.x.to_le_bytes());
    buf.extend_from_slice(&pt.y.to_le_bytes());
    buf.extend_from_slice(&pt.z.to_le_bytes());
}

/// Serializes an indexed triangle [`Mesh`] into standard binary little-endian `.stl` bytes (`84 + 50 * F`).
pub fn export_mesh_to_stl_binary(mesh: &Mesh) -> Result<Vec<u8>, Error> {
    validate_mesh_geometry(mesh, true)?;
    let (triangles, _) = mesh.indices.as_chunks::<3>();
    let face_count = u32::try_from(triangles.len())
        .map_err(|_| invalid_mesh_error("STL triangle count exceeds u32::MAX"))?;

    let mut out = Vec::with_capacity(84 + 50 * triangles.len());
    let mut header = [0_u8; 80];
    let copy_len = STL_HEADER_TEXT.len().min(header.len());
    header[..copy_len].copy_from_slice(&STL_HEADER_TEXT[..copy_len]);
    out.extend_from_slice(&header);
    out.extend_from_slice(&face_count.to_le_bytes());

    for &[i0, i1, i2] in triangles {
        let normal = resolve_triangle_normal(mesh, i0, i1, i2);
        write_point_le(&mut out, normal);
        write_point_le(&mut out, mesh.vertices[i0]);
        write_point_le(&mut out, mesh.vertices[i1]);
        write_point_le(&mut out, mesh.vertices[i2]);
        out.extend_from_slice(&0_u16.to_le_bytes());
    }

    Ok(out)
}

/// Serializes an indexed triangle [`Mesh`] into human-readable ASCII `.stl` text.
pub fn export_mesh_to_stl_ascii(mesh: &Mesh) -> Result<String, Error> {
    validate_mesh_geometry(mesh, true)?;
    let (triangles, _) = mesh.indices.as_chunks::<3>();
    let mut out = String::with_capacity(64 + triangles.len() * 180);
    out.push_str("solid oxidize_mesh\n");

    for &[i0, i1, i2] in triangles {
        let n = resolve_triangle_normal(mesh, i0, i1, i2);
        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];
        let _ = writeln!(out, "  facet normal {:.6} {:.6} {:.6}", n.x, n.y, n.z);
        out.push_str("    outer loop\n");
        let _ = writeln!(out, "      vertex {:.6} {:.6} {:.6}", v0.x, v0.y, v0.z);
        let _ = writeln!(out, "      vertex {:.6} {:.6} {:.6}", v1.x, v1.y, v1.z);
        let _ = writeln!(out, "      vertex {:.6} {:.6} {:.6}", v2.x, v2.y, v2.z);
        out.push_str("    endloop\n  endfacet\n");
    }

    out.push_str("endsolid oxidize_mesh\n");
    Ok(out)
}

fn parse_triplet(tokens: &[&str], label: &str) -> Result<Point3D, Error> {
    if tokens.len() != 3 {
        return Err(invalid_mesh_error(&format!(
            "Expected 3 coordinates for {label}"
        )));
    }
    let x: f32 = tokens[0]
        .parse()
        .map_err(|_| invalid_mesh_error("Invalid STL float X"))?;
    let y: f32 = tokens[1]
        .parse()
        .map_err(|_| invalid_mesh_error("Invalid STL float Y"))?;
    let z: f32 = tokens[2]
        .parse()
        .map_err(|_| invalid_mesh_error("Invalid STL float Z"))?;
    let pt = Point3D::new(x, y, z);
    if !pt.is_finite() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Non-finite coordinate in ASCII STL",
        ));
    }
    Ok(pt)
}

/// Parses an ASCII `.stl` document back into a triangle soup [`Mesh`].
pub fn parse_stl_ascii(stl: &str) -> Result<Mesh, Error> {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut current_normal = Point3D::new(0.0, 0.0, 1.0);
    let mut facet_verts: Vec<Point3D> = Vec::with_capacity(3);
    let mut saw_solid = false;

    for raw_line in stl.lines() {
        let parts: Vec<&str> = raw_line.split_whitespace().collect();
        match parts.as_slice() {
            ["solid", ..] => saw_solid = true,
            ["facet", "normal", rest @ ..] => {
                current_normal = parse_triplet(rest, "facet normal")?;
                facet_verts.clear();
            }
            ["vertex", rest @ ..] => {
                facet_verts.push(parse_triplet(rest, "vertex")?);
            }
            ["endfacet"] => {
                if facet_verts.len() != 3 {
                    return Err(invalid_mesh_error("ASCII STL facet must have 3 vertices"));
                }
                let base = vertices.len();
                vertices.extend_from_slice(&facet_verts);
                normals.extend_from_slice(&[current_normal; 3]);
                indices.extend_from_slice(&[base, base + 1, base + 2]);
                facet_verts.clear();
            }
            _ => {}
        }
    }

    if !saw_solid || indices.is_empty() {
        return Err(invalid_mesh_error("ASCII STL contained no valid facets"));
    }
    Ok(Mesh {
        vertices,
        normals,
        indices,
    })
}
