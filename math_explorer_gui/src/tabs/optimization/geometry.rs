#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Shared 2D polygon half-space clipping, constraint line clipping, and marching-squares contours.

use eframe::egui::Color32;
use math_explorer::pure_math::analysis::optimization::ConstraintSense;

/// Clips a convex polygon `poly` against $a_1 x_1 + a_2 x_2 \le b$ (or $\ge b$).
#[must_use]
pub fn clip_halfspace_polygon(
    poly: &[[f64; 2]],
    a1: f64,
    a2: f64,
    sense: ConstraintSense,
    rhs: f64,
) -> Vec<[f64; 2]> {
    match sense {
        ConstraintSense::LessEqual => clip_le_halfspace(poly, a1, a2, rhs),
        ConstraintSense::GreaterEqual => clip_le_halfspace(poly, -a1, -a2, -rhs),
        ConstraintSense::Equal => {
            let p1 = clip_le_halfspace(poly, a1, a2, rhs + 0.04);
            clip_le_halfspace(&p1, -a1, -a2, -rhs + 0.04)
        }
    }
}

fn clip_le_halfspace(poly: &[[f64; 2]], a1: f64, a2: f64, rhs: f64) -> Vec<[f64; 2]> {
    if poly.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(poly.len() + 2);
    let n = poly.len();
    for i in 0..n {
        let p = poly[i];
        let q = poly[(i + 1) % n];
        let vp = a1 * p[0] + a2 * p[1] - rhs;
        let vq = a1 * q[0] + a2 * q[1] - rhs;
        let p_in = vp <= 1e-9;
        let q_in = vq <= 1e-9;
        if p_in {
            out.push(p);
        }
        if p_in != q_in {
            let denom = vp - vq;
            if denom.abs() > 1e-12 {
                let t = (vp / denom).clamp(0.0, 1.0);
                out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
            }
        }
    }
    out
}

/// Computes the clipped line segment of $a_1 x_1 + a_2 x_2 = b$ across `[x_min, x_max] x [y_min, y_max]`.
#[must_use]
pub fn constraint_boundary_segment(
    a1: f64,
    a2: f64,
    rhs: f64,
    x_range: [f64; 2],
    y_range: [f64; 2],
) -> Option<[[f64; 2]; 2]> {
    let mut hits: Vec<[f64; 2]> = Vec::with_capacity(4);
    if a2.abs() > 1e-9 {
        for &x in &x_range {
            let y = (rhs - a1 * x) / a2;
            if y >= y_range[0] - 1e-7 && y <= y_range[1] + 1e-7 {
                push_unique_hit(&mut hits, [x, y.clamp(y_range[0], y_range[1])]);
            }
        }
    }
    if a1.abs() > 1e-9 {
        for &y in &y_range {
            let x = (rhs - a2 * y) / a1;
            if x >= x_range[0] - 1e-7 && x <= x_range[1] + 1e-7 {
                push_unique_hit(&mut hits, [x.clamp(x_range[0], x_range[1]), y]);
            }
        }
    }
    if hits.len() >= 2 {
        Some([hits[0], hits[1]])
    } else {
        None
    }
}

fn push_unique_hit(hits: &mut Vec<[f64; 2]>, pt: [f64; 2]) {
    let dup = hits
        .iter()
        .any(|q| (q[0] - pt[0]).hypot(q[1] - pt[1]) < 1e-5);
    if !dup {
        hits.push(pt);
    }
}

/// Samples marching-squares iso-contour segments across `[x0, x1] x [y0, y1]`.
pub fn sample_iso_contours(
    eval: impl Fn(f64, f64) -> f64,
    bounds: [f64; 4],
    res: usize,
    num_levels: usize,
) -> Vec<(f32, Vec<[[f64; 2]; 2]>)> {
    let n = res.max(8);
    let (x0, x1, y0, y1) = (bounds[0], bounds[1], bounds[2], bounds[3]);
    let mut vals = vec![0.0; n * n];
    let (mut z_min, mut z_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for iy in 0..n {
        let y = y0 + (iy as f64 / (n - 1) as f64) * (y1 - y0);
        for ix in 0..n {
            let x = x0 + (ix as f64 / (n - 1) as f64) * (x1 - x0);
            let z = eval(x, y);
            vals[math_commons::math_kernel::types::flatten_2d_index(ix, iy, n)] = z;
            if z.is_finite() {
                z_min = z_min.min(z);
                z_max = z_max.max(z);
            }
        }
    }
    if !z_min.is_finite() || (z_max - z_min) < 1e-12 {
        return Vec::new();
    }
    build_contour_levels(&vals, n, bounds, z_min, z_max, num_levels)
}

fn build_contour_levels(
    vals: &[f64],
    n: usize,
    bounds: [f64; 4],
    z_min: f64,
    z_max: f64,
    num_levels: usize,
) -> Vec<(f32, Vec<[[f64; 2]; 2]>)> {
    let (x0, x1, y0, y1) = (bounds[0], bounds[1], bounds[2], bounds[3]);
    let dx = (x1 - x0) / (n - 1) as f64;
    let dy = (y1 - y0) / (n - 1) as f64;
    let mut out = Vec::with_capacity(num_levels);
    for lv in 0..num_levels {
        let frac = ((lv + 1) as f64 / (num_levels + 1) as f64).powf(1.55);
        let iso = z_min + frac * (z_max - z_min);
        let mut segs = Vec::new();
        for iy in 0..(n - 1) {
            let cy = y0 + iy as f64 * dy;
            for ix in 0..(n - 1) {
                let cx = x0 + ix as f64 * dx;
                let z00 = vals[math_commons::math_kernel::types::flatten_2d_index(ix, iy, n)];
                let z10 = vals[math_commons::math_kernel::types::flatten_2d_index(ix + 1, iy, n)];
                let z11 =
                    vals[math_commons::math_kernel::types::flatten_2d_index(ix + 1, iy + 1, n)];
                let z01 = vals[math_commons::math_kernel::types::flatten_2d_index(ix, iy + 1, n)];
                cell_contour_segments([cx, cy, dx, dy], [z00, z10, z11, z01], iso, &mut segs);
            }
        }
        out.push((frac as f32, segs));
    }
    out
}

fn cell_contour_segments(rect: [f64; 4], z: [f64; 4], iso: f64, segs: &mut Vec<[[f64; 2]; 2]>) {
    let [cx, cy, dx, dy] = rect;
    let corners = [[cx, cy], [cx + dx, cy], [cx + dx, cy + dy], [cx, cy + dy]];
    let mut pts = [[0.0; 2]; 4];
    let mut cnt = 0;
    for e in 0..4 {
        let j = (e + 1) % 4;
        if (z[e] <= iso) != (z[j] <= iso) {
            let denom = z[j] - z[e];
            let t = if denom.abs() > 1e-12 {
                ((iso - z[e]) / denom).clamp(0.0, 1.0)
            } else {
                0.5
            };
            pts[cnt] = [
                corners[e][0] + t * (corners[j][0] - corners[e][0]),
                corners[e][1] + t * (corners[j][1] - corners[e][1]),
            ];
            cnt += 1;
        }
    }
    if cnt >= 2 {
        segs.push([pts[0], pts[1]]);
    }
    if cnt == 4 {
        segs.push([pts[2], pts[3]]);
    }
}

/// Linearly interpolates between two `SemanticPalette` colors in RGBA space.
#[inline]
#[must_use]
pub fn blend_palette_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let s = t.clamp(0.0, 1.0);
    let lerp = |u: u8, v: u8| ((u as f32) + s * ((v as f32) - (u as f32))).round() as u8;
    Color32::from_rgba_unmultiplied(
        lerp(a.r(), b.r()),
        lerp(a.g(), b.g()),
        lerp(a.b(), b.b()),
        lerp(a.a(), b.a()),
    )
}

// [cite:lorahub]
