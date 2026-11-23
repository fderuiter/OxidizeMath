//! Watertight 2-manifold swept-tube and solid primitive mesh builders.

use super::export::invalid_mesh_error;
use super::types::{ColoredMesh, Mesh, Point3D, VertexColor};
use std::f32::consts::{PI, TAU};
use std::io::Error;

/// Maximum radial segment count permitted on swept tubes (`3..=32`) to bound browser memory.
pub const MAX_TUBE_RADIAL_SEGMENTS: usize = 32;

/// Maximum angular or longitudinal tessellation resolution permitted on solid primitives.
pub const MAX_PRIMITIVE_SEGMENTS: usize = 1000;

fn orthonormal_frame(tangent: Point3D) -> (Point3D, Point3D) {
    let ref_axis = if tangent.x.abs() < 0.9 {
        Point3D::new(1.0, 0.0, 0.0)
    } else {
        Point3D::new(0.0, 1.0, 0.0)
    };
    let r = ref_axis.cross(tangent).normalize();
    let s = tangent.cross(r).normalize();
    (r, s)
}

fn sanitize_polyline(points: &[Point3D], closed: bool) -> Result<Vec<Point3D>, Error> {
    let mut clean: Vec<Point3D> = Vec::with_capacity(points.len());
    for &pt in points {
        if !pt.is_finite() {
            return Err(invalid_mesh_error(
                "Tube polyline contains non-finite point",
            ));
        }
        if clean
            .last()
            .is_none_or(|prev| (pt - *prev).magnitude() > 1.0e-5)
        {
            clean.push(pt);
        }
    }
    if closed && clean.len() >= 2 && (clean[0] - *clean.last().unwrap()).magnitude() <= 1.0e-4 {
        clean.pop();
    }
    let min_pts = if closed { 3 } else { 2 };
    if clean.len() < min_pts {
        return Err(invalid_mesh_error(
            "Insufficient distinct points for tube extrusion",
        ));
    }
    Ok(clean)
}

fn compute_tangents(pts: &[Point3D], closed: bool) -> Vec<Point3D> {
    let n = pts.len();
    let mut tangents = Vec::with_capacity(n);
    for i in 0..n {
        let t = if !closed && i == 0 {
            (pts[1] - pts[0]).normalize()
        } else if !closed && i + 1 == n {
            (pts[n - 1] - pts[n - 2]).normalize()
        } else {
            let prev = pts[(i + n - 1) % n];
            let next = pts[(i + 1) % n];
            let d0 = (pts[i] - prev).normalize();
            let d1 = (next - pts[i]).normalize();
            let sum = d0 + d1;
            if sum.magnitude() > Point3D::EPSILON {
                sum.normalize()
            } else {
                d1
            }
        };
        tangents.push(t);
    }
    tangents
}

fn advance_rmf(p0: Point3D, t0: Point3D, r0: Point3D, p1: Point3D, t1: Point3D) -> Point3D {
    let v1 = p1 - p0;
    let c1 = v1.dot(v1);
    if c1 <= 1.0e-12 {
        return r0;
    }
    let r_l = r0 - (2.0 / c1) * v1.dot(r0) * v1;
    let t_l = t0 - (2.0 / c1) * v1.dot(t0) * v1;
    let v2 = t1 - t_l;
    let c2 = v2.dot(v2);
    let r_next = if c2 > 1.0e-12 {
        r_l - (2.0 / c2) * v2.dot(r_l) * v2
    } else {
        r_l
    };
    let ortho = r_next - t1 * r_next.dot(t1);
    if ortho.magnitude() > Point3D::EPSILON {
        ortho.normalize()
    } else {
        orthonormal_frame(t1).0
    }
}

fn compute_rmf_frames(
    pts: &[Point3D],
    tangents: &[Point3D],
    closed: bool,
) -> Vec<(Point3D, Point3D)> {
    let n = pts.len();
    let mut r_vecs = Vec::with_capacity(n);
    let (r0, _) = orthonormal_frame(tangents[0]);
    r_vecs.push(r0);
    for i in 0..n - 1 {
        let next_r = advance_rmf(pts[i], tangents[i], r_vecs[i], pts[i + 1], tangents[i + 1]);
        r_vecs.push(next_r);
    }

    if closed {
        let r_end = advance_rmf(
            pts[n - 1],
            tangents[n - 1],
            r_vecs[n - 1],
            pts[0],
            tangents[0],
        );
        let s0 = tangents[0].cross(r0).normalize();
        let twist = r_end.dot(s0).atan2(r_end.dot(r0));
        for (i, r_ref) in r_vecs.iter_mut().enumerate() {
            let angle = -twist * (i as f32) / (n as f32);
            let s_i = tangents[i].cross(*r_ref).normalize();
            let (sin_a, cos_a) = angle.sin_cos();
            *r_ref = (*r_ref * cos_a + s_i * sin_a).normalize();
        }
    }

    r_vecs
        .into_iter()
        .enumerate()
        .map(|(i, r)| (r, tangents[i].cross(r).normalize()))
        .collect()
}

fn push_tube_quads(indices: &mut Vec<usize>, ring_a: usize, ring_b: usize, radial: usize) {
    for k in 0..radial {
        let k_next = (k + 1) % radial;
        let v00 = ring_a + k;
        let v01 = ring_a + k_next;
        let v10 = ring_b + k;
        let v11 = ring_b + k_next;
        indices.extend_from_slice(&[v00, v01, v11, v00, v11, v10]);
    }
}

/// Sweeps a circular cross-section along `points` using double-reflection rotation-minimizing frames.
///
/// Open curves are sealed with welded start and end triangle-fan caps ($\chi = 2$); closed curves
/// distribute holonomy twist smoothly and weld the closing seam ($\chi = 0$).
pub fn extrude_polyline_tube(
    points: &[Point3D],
    radius: f32,
    radial_segments: usize,
    closed: bool,
) -> Result<Mesh, Error> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(invalid_mesh_error(
            "Tube radius must be finite and positive",
        ));
    }
    let pts = sanitize_polyline(points, closed)?;
    let radial = radial_segments.clamp(3, MAX_TUBE_RADIAL_SEGMENTS);
    let tangents = compute_tangents(&pts, closed);
    let frames = compute_rmf_frames(&pts, &tangents, closed);
    let m = pts.len();

    let cap_verts = if closed { 0 } else { 2 };
    let mut vertices = Vec::with_capacity(m * radial + cap_verts);
    let mut normals = Vec::with_capacity(m * radial + cap_verts);

    for (i, &(r_vec, s_vec)) in frames.iter().enumerate() {
        for k in 0..radial {
            let theta = TAU * (k as f32) / (radial as f32);
            let (sin_t, cos_t) = theta.sin_cos();
            let normal = (r_vec * cos_t + s_vec * sin_t).normalize();
            vertices.push(pts[i] + normal * radius);
            normals.push(normal);
        }
    }

    let seg_count = if closed { m } else { m - 1 };
    let mut indices = Vec::with_capacity((seg_count * 2 + cap_verts) * radial * 3);
    for i in 0..m - 1 {
        push_tube_quads(&mut indices, i * radial, (i + 1) * radial, radial);
    }
    if closed {
        push_tube_quads(&mut indices, (m - 1) * radial, 0, radial);
    } else {
        let start_center = vertices.len();
        vertices.push(pts[0]);
        normals.push(-tangents[0]);
        let end_center = vertices.len();
        vertices.push(pts[m - 1]);
        normals.push(tangents[m - 1]);
        let last_ring = (m - 1) * radial;
        for k in 0..radial {
            let k_next = (k + 1) % radial;
            indices.extend_from_slice(&[start_center, k_next, k]);
            indices.extend_from_slice(&[end_center, last_ring + k, last_ring + k_next]);
        }
    }

    Ok(Mesh {
        vertices,
        normals,
        indices,
    })
}

/// Builds a closed 2-manifold UV sphere ($\chi = 2$) with welded poles and exact `(0, 0, ±1)` pole normals.
pub fn build_uv_sphere(
    center: Point3D,
    radius: f32,
    u_segments: usize,
    v_segments: usize,
) -> Result<Mesh, Error> {
    if !center.is_finite() || !radius.is_finite() || radius <= 0.0 {
        return Err(invalid_mesh_error(
            "Sphere center and radius must be finite and positive",
        ));
    }
    let nu = u_segments.clamp(3, MAX_PRIMITIVE_SEGMENTS);
    let nv = v_segments.clamp(2, MAX_PRIMITIVE_SEGMENTS);

    let mut vertices = Vec::with_capacity(2 + (nv - 1) * nu);
    let mut normals = Vec::with_capacity(2 + (nv - 1) * nu);
    vertices.push(center + Point3D::new(0.0, 0.0, radius));
    normals.push(Point3D::new(0.0, 0.0, 1.0));

    for j in 1..nv {
        let v = PI * (j as f32) / (nv as f32);
        let (sin_v, cos_v) = v.sin_cos();
        for i in 0..nu {
            let u = TAU * (i as f32) / (nu as f32);
            let (sin_u, cos_u) = u.sin_cos();
            let dir = Point3D::new(sin_v * cos_u, sin_v * sin_u, cos_v);
            vertices.push(center + dir * radius);
            normals.push(dir);
        }
    }

    let south_idx = vertices.len();
    vertices.push(center + Point3D::new(0.0, 0.0, -radius));
    normals.push(Point3D::new(0.0, 0.0, -1.0));

    let mut indices = Vec::with_capacity(2 * (nv - 1) * nu * 3);
    for i in 0..nu {
        let i_next = (i + 1) % nu;
        indices.extend_from_slice(&[0, 1 + i, 1 + i_next]);
    }
    for j in 0..nv.saturating_sub(2) {
        let row_a = 1 + j * nu;
        let row_b = row_a + nu;
        for i in 0..nu {
            let i_next = (i + 1) % nu;
            indices.extend_from_slice(&[
                row_a + i,
                row_b + i,
                row_a + i_next,
                row_a + i_next,
                row_b + i,
                row_b + i_next,
            ]);
        }
    }
    let last_row = 1 + (nv - 2) * nu;
    for i in 0..nu {
        let i_next = (i + 1) % nu;
        indices.extend_from_slice(&[south_idx, last_row + i_next, last_row + i]);
    }

    Ok(Mesh {
        vertices,
        normals,
        indices,
    })
}

/// Builds a closed 2-manifold cylinder ($\chi = 2$) between `start` and `end` with triangle-fan caps.
pub fn build_cylinder(
    start: Point3D,
    end: Point3D,
    radius: f32,
    radial_segments: usize,
) -> Result<Mesh, Error> {
    extrude_polyline_tube(&[start, end], radius, radial_segments, false)
}

/// Builds a closed 2-manifold cone ($\chi = 2$) from `base_center` to `apex` with a welded base fan.
pub fn build_cone(
    base_center: Point3D,
    apex: Point3D,
    base_radius: f32,
    radial_segments: usize,
) -> Result<Mesh, Error> {
    if !base_center.is_finite()
        || !apex.is_finite()
        || !base_radius.is_finite()
        || base_radius <= 0.0
    {
        return Err(invalid_mesh_error(
            "Cone parameters must be finite with positive radius",
        ));
    }
    let axis = apex - base_center;
    let height = axis.magnitude();
    if height <= Point3D::EPSILON {
        return Err(invalid_mesh_error("Cone height must be non-zero"));
    }
    let dir = axis / height;
    let (r_vec, s_vec) = orthonormal_frame(dir);
    let radial = radial_segments.clamp(3, MAX_TUBE_RADIAL_SEGMENTS);

    let mut vertices = Vec::with_capacity(radial + 2);
    let mut normals = Vec::with_capacity(radial + 2);
    for k in 0..radial {
        let theta = TAU * (k as f32) / (radial as f32);
        let (sin_t, cos_t) = theta.sin_cos();
        let rad_dir = r_vec * cos_t + s_vec * sin_t;
        vertices.push(base_center + rad_dir * base_radius);
        normals.push((rad_dir * height + dir * base_radius).normalize());
    }
    let base_idx = vertices.len();
    vertices.push(base_center);
    normals.push(-dir);
    let apex_idx = vertices.len();
    vertices.push(apex);
    normals.push(dir);

    let mut indices = Vec::with_capacity(radial * 6);
    for k in 0..radial {
        let k_next = (k + 1) % radial;
        indices.extend_from_slice(&[k, k_next, apex_idx, base_idx, k_next, k]);
    }
    Ok(Mesh {
        vertices,
        normals,
        indices,
    })
}

/// Builds a single-piece welded 3D arrow ($\chi = 2$) combining a cylindrical shaft, collar, and conical tip.
pub fn build_arrow(
    start: Point3D,
    end: Point3D,
    shaft_radius: f32,
    head_radius: f32,
    head_length: f32,
    radial_segments: usize,
) -> Result<Mesh, Error> {
    if !start.is_finite()
        || !end.is_finite()
        || !shaft_radius.is_finite()
        || !head_radius.is_finite()
        || !head_length.is_finite()
        || shaft_radius <= 0.0
    {
        return Err(invalid_mesh_error(
            "Arrow parameters must be finite and positive",
        ));
    }
    let axis = end - start;
    let total_len = axis.magnitude();
    if total_len <= Point3D::EPSILON {
        return Err(invalid_mesh_error("Arrow length must be non-zero"));
    }
    let dir = axis / total_len;
    let h_len = head_length.clamp(0.1 * total_len, 0.8 * total_len);
    let h_rad = head_radius.max(shaft_radius * 1.25);
    let neck = end - dir * h_len;
    let (r_vec, s_vec) = orthonormal_frame(dir);
    let radial = radial_segments.clamp(3, MAX_TUBE_RADIAL_SEGMENTS);

    let mut vertices = Vec::with_capacity(3 * radial + 2);
    let mut normals = Vec::with_capacity(3 * radial + 2);
    for &(origin, rad, axial_w) in &[
        (start, shaft_radius, 0.0_f32),
        (neck, shaft_radius, 0.0_f32),
        (neck, h_rad, h_rad / h_len),
    ] {
        for k in 0..radial {
            let theta = TAU * (k as f32) / (radial as f32);
            let (sin_t, cos_t) = theta.sin_cos();
            let rad_dir = r_vec * cos_t + s_vec * sin_t;
            vertices.push(origin + rad_dir * rad);
            normals.push((rad_dir + dir * axial_w).normalize());
        }
    }
    let start_cap = vertices.len();
    vertices.push(start);
    normals.push(-dir);
    let apex_idx = vertices.len();
    vertices.push(end);
    normals.push(dir);

    let mut indices = Vec::with_capacity(radial * 18);
    push_tube_quads(&mut indices, 0, radial, radial);
    push_tube_quads(&mut indices, radial, 2 * radial, radial);
    for k in 0..radial {
        let k_next = (k + 1) % radial;
        indices.extend_from_slice(&[start_cap, k_next, k]);
        indices.extend_from_slice(&[2 * radial + k, 2 * radial + k_next, apex_idx]);
    }

    Ok(Mesh {
        vertices,
        normals,
        indices,
    })
}

/// Appends `src` geometry into `dst`, offsetting triangle vertex indices.
pub fn append_mesh(dst: &mut Mesh, src: &Mesh) {
    let offset = dst.vertices.len();
    dst.vertices.extend_from_slice(&src.vertices);
    dst.normals.extend_from_slice(&src.normals);
    dst.indices
        .extend(src.indices.iter().map(|&idx| idx + offset));
}

/// Appends `src` geometry into `dst` with a uniform `color` assigned to every appended vertex.
pub fn append_colored_part(dst: &mut ColoredMesh, src: &Mesh, color: VertexColor) {
    append_mesh(&mut dst.mesh, src);
    dst.colors
        .extend(std::iter::repeat_n(color, src.vertices.len()));
}
