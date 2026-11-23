use super::types::{Mesh, Point3D};
use std::fmt::Write;
use std::io::{Error, ErrorKind};

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

#[allow(missing_docs)]
pub fn export_mesh_to_obj_string(mesh: &Mesh) -> Result<String, Error> {
    validate_mesh_geometry(mesh, false)?;
    let normals = sanitized_vertex_normals(mesh);
    let (triangles, _) = mesh.indices.as_chunks::<3>();
    let estimated_size = mesh.vertices.len() * 25 + normals.len() * 25 + triangles.len() * 30 + 100;
    let mut out = String::with_capacity(estimated_size);

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
        let _ = writeln!(out, "f {}//{} {}//{} {}//{}", i1, i1, i2, i2, i3, i3);
    }

    Ok(out)
}
