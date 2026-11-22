//! Analytical tools and derived quantities for Fluid Dynamics.

use super::types::{FlowState, FluidProperties};

/// Calculates the Reynolds Number ($Re$).
///
/// $$Re = \frac{\rho u L}{\mu} = \frac{u L}{\nu}$$
///
/// * `properties`: Fluid properties.
/// * `velocity_magnitude`: Characteristic velocity scale $u$.
/// * `characteristic_length`: Characteristic length scale $L$.
#[verified_engine::verified]
pub fn reynolds_number(
    properties: &FluidProperties,
    velocity_magnitude: f64,
    characteristic_length: f64,
) -> f64 {
    (properties.density() * velocity_magnitude * characteristic_length)
        / properties.dynamic_viscosity()
}

/// Calculates the Bernoulli constant along a streamline for steady, incompressible, inviscid flow.
///
/// $$C = p + \frac{1}{2}\rho v^2 + \rho g h$$
///
/// * `state`: Fluid state ($p, \mathbf{u}$).
/// * `properties`: Fluid properties (specifically density $\rho$).
/// * `height`: Elevation $h$ relative to a datum.
/// * `gravity`: Gravitational acceleration $g$ (magnitude, usually 9.81).
#[verified_engine::verified]
pub fn bernoulli_constant(
    state: &FlowState,
    properties: &FluidProperties,
    height: f64,
    gravity: f64,
) -> f64 {
    let v_sq = state.velocity.norm_squared();
    state.pressure + 0.5 * properties.density() * v_sq + properties.density() * gravity * height
}

/// Calculates Shear Stress ($\tau$) in a Boundary Layer (Newtonian Fluid).
///
/// $$\tau = \mu \frac{\partial u}{\partial y}$$
///
/// * `properties`: Fluid properties (specifically dynamic viscosity $\mu$).
/// * `velocity_gradient_normal`: Gradient of velocity perpendicular to the wall ($\frac{\partial u}{\partial y}$).
#[verified_engine::verified]
pub fn shear_stress(properties: &FluidProperties, velocity_gradient_normal: f64) -> f64 {
    properties.dynamic_viscosity() * velocity_gradient_normal
}

#[verified_engine::verified]
pub(super) fn march_streamlines(
    field: &super::potential_flow::PotentialFlowField,
    bounds: (f64, f64, f64, f64),
    resolution: usize,
    iso_levels: &[f64],
) -> Vec<([f64; 2], [f64; 2])> {
    use super::potential_flow::FlowElement;
    let (x_min, x_max, y_min, y_max) = bounds;
    if !x_min.is_finite()
        || !x_max.is_finite()
        || !y_min.is_finite()
        || !y_max.is_finite()
        || x_max <= x_min
        || y_max <= y_min
        || iso_levels.is_empty()
    {
        return Vec::new();
    }
    let n = resolution.clamp(2, math_commons::registry::MAX_GRID_SIZE);
    let (hx, hy) = ((x_max - x_min) / (n as f64), (y_max - y_min) / (n as f64));
    let core_r2 = (0.65 * (hx * hx + hy * hy)).max(math_commons::registry::TOLERANCE_HIGH);
    debug_assert!(n >= 2 && core_r2 >= math_commons::registry::TOLERANCE_HIGH);
    let mut segments = Vec::with_capacity(n * iso_levels.len() * 2);
    for iy in 0..n {
        let (y0, y1) = (y_min + (iy as f64) * hy, y_min + ((iy + 1) as f64) * hy);
        for ix in 0..n {
            let (x0, x1) = (x_min + (ix as f64) * hx, x_min + ((ix + 1) as f64) * hx);
            let c = [
                field.stream_function(x0, y0),
                field.stream_function(x1, y0),
                field.stream_function(x1, y1),
                field.stream_function(x0, y1),
            ];
            if is_smooth_cell(field, x0, x1, y0, y1, c, core_r2) {
                append_cell_contours(x0, x1, y0, y1, c, iso_levels, &mut segments);
            }
        }
    }
    segments
}

#[verified_engine::verified]
pub(super) fn find_stagnation_points(
    field: &super::potential_flow::PotentialFlowField,
    bounds: (f64, f64, f64, f64),
    resolution: usize,
) -> Vec<nalgebra::Vector2<f64>> {
    let (x_min, x_max, y_min, y_max) = bounds;
    if !x_min.is_finite()
        || !x_max.is_finite()
        || !y_min.is_finite()
        || !y_max.is_finite()
        || x_max <= x_min
        || y_max <= y_min
        || field.elements.is_empty()
    {
        return Vec::new();
    }
    let n = resolution.clamp(2, math_commons::registry::MAX_GRID_SIZE);
    let (hx, hy) = ((x_max - x_min) / (n as f64), (y_max - y_min) / (n as f64));
    let diag = hx.hypot(hy);
    let core_r2 = (0.65 * diag * diag).max(math_commons::registry::TOLERANCE_HIGH);
    debug_assert!(diag > 0.0);
    let mut points: Vec<nalgebra::Vector2<f64>> = Vec::new();
    for iy in 0..n {
        let (y0, y1) = (y_min + (iy as f64) * hy, y_min + ((iy + 1) as f64) * hy);
        for ix in 0..n {
            let (x0, x1) = (x_min + (ix as f64) * hx, x_min + ((ix + 1) as f64) * hx);
            let seed = nalgebra::Vector2::new(0.5 * (x0 + x1), 0.5 * (y0 + y1));
            if cell_may_contain_stagnation(field, x0, x1, y0, y1, core_r2)
                && let Some(pt) = refine_stagnation(field, seed, diag, core_r2)
                && pt.x >= x_min
                && pt.x <= x_max
                && pt.y >= y_min
                && pt.y <= y_max
                && points.iter().all(|q| (q - pt).norm() > 0.35 * diag)
            {
                points.push(pt);
            }
        }
    }
    points
}

#[verified_engine::verified]
fn is_smooth_cell(
    field: &super::potential_flow::PotentialFlowField,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    c: [f64; 4],
    core_r2: f64,
) -> bool {
    use super::potential_flow::FlowElement;
    if !c[0].is_finite() || !c[1].is_finite() || !c[2].is_finite() || !c[3].is_finite() {
        return false;
    }
    let (xc, yc) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
    if field.is_near_singularity(xc, yc, core_r2) {
        return false;
    }
    let (hx, hy) = (x1 - x0, y1 - y0);
    debug_assert!(hx > 0.0 && hy > 0.0);
    let vc = field.velocity(xc, yc);
    if !vc.x.is_finite() || !vc.y.is_finite() {
        return false;
    }
    let fast_tol = (vc.norm() * hx.hypot(hy) * 3.5).max(1e-3);
    if (c[1] - c[0] + vc.y * hx).abs() <= fast_tol
        && (c[2] - c[3] + vc.y * hx).abs() <= fast_tol
        && (c[3] - c[0] - vc.x * hy).abs() <= fast_tol
        && (c[2] - c[1] - vc.x * hy).abs() <= fast_tol
    {
        return true;
    }
    let (vb, vt) = (field.velocity(xc, y0), field.velocity(xc, y1));
    let (vl, vr) = (field.velocity(x0, yc), field.velocity(x1, yc));
    let vmax = vb.norm().max(vt.norm()).max(vl.norm()).max(vr.norm());
    let edge_tol = (vmax * hx.hypot(hy) * 2.5).max(1e-3);
    (c[1] - c[0] + vb.y * hx).abs() <= edge_tol
        && (c[2] - c[3] + vt.y * hx).abs() <= edge_tol
        && (c[3] - c[0] - vl.x * hy).abs() <= edge_tol
        && (c[2] - c[1] - vr.x * hy).abs() <= edge_tol
}

#[verified_engine::verified]
fn edge_point(x0: f64, x1: f64, y0: f64, y1: f64, c: [f64; 4], edge: u8, level: f64) -> [f64; 2] {
    let lerp_t = |va: f64, vb: f64| -> f64 {
        let dv = vb - va;
        if dv.abs() <= math_commons::registry::TOLERANCE_HIGH {
            0.5
        } else {
            ((level - va) / dv).clamp(0.0, 1.0)
        }
    };
    debug_assert!(edge < 4);
    match edge {
        0 => [x0 + lerp_t(c[0], c[1]) * (x1 - x0), y0],
        1 => [x1, y0 + lerp_t(c[1], c[2]) * (y1 - y0)],
        2 => [x0 + lerp_t(c[3], c[2]) * (x1 - x0), y1],
        _ => [x0, y0 + lerp_t(c[0], c[3]) * (y1 - y0)],
    }
}

#[verified_engine::verified]
fn append_cell_contours(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    c: [f64; 4],
    iso_levels: &[f64],
    out: &mut Vec<([f64; 2], [f64; 2])>,
) {
    let lo = c[0].min(c[1]).min(c[2]).min(c[3]);
    let hi = c[0].max(c[1]).max(c[2]).max(c[3]);
    let center = 0.25 * (c[0] + c[1] + c[2] + c[3]);
    debug_assert!(lo <= hi);
    for &level in iso_levels {
        if level < lo || level > hi {
            continue;
        }
        let mask = u8::from(c[0] >= level)
            | (u8::from(c[1] >= level) << 1)
            | (u8::from(c[2] >= level) << 2)
            | (u8::from(c[3] >= level) << 3);
        let pairs: &[(u8, u8)] = match mask {
            1 | 14 => &[(3, 0)],
            2 | 13 => &[(0, 1)],
            3 | 12 => &[(3, 1)],
            4 | 11 => &[(1, 2)],
            5 => {
                if center >= level {
                    &[(3, 2), (0, 1)]
                } else {
                    &[(3, 0), (1, 2)]
                }
            }
            6 | 9 => &[(0, 2)],
            7 | 8 => &[(3, 2)],
            10 => {
                if center >= level {
                    &[(3, 0), (1, 2)]
                } else {
                    &[(3, 2), (0, 1)]
                }
            }
            _ => &[],
        };
        for &(ea, eb) in pairs {
            out.push((
                edge_point(x0, x1, y0, y1, c, ea, level),
                edge_point(x0, x1, y0, y1, c, eb, level),
            ));
        }
    }
}

#[verified_engine::verified]
fn cell_may_contain_stagnation(
    field: &super::potential_flow::PotentialFlowField,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    core_r2: f64,
) -> bool {
    use super::potential_flow::FlowElement;
    let (xc, yc) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
    if field.is_near_singularity(xc, yc, core_r2) {
        return false;
    }
    let vs = [
        field.velocity(x0, y0),
        field.velocity(x1, y0),
        field.velocity(x1, y1),
        field.velocity(x0, y1),
    ];
    let vc = field.velocity(xc, yc);
    let u_min = vs[0].x.min(vs[1].x).min(vs[2].x).min(vs[3].x).min(vc.x);
    let u_max = vs[0].x.max(vs[1].x).max(vs[2].x).max(vs[3].x).max(vc.x);
    let v_min = vs[0].y.min(vs[1].y).min(vs[2].y).min(vs[3].y).min(vc.y);
    let v_max = vs[0].y.max(vs[1].y).max(vs[2].y).max(vs[3].y).max(vc.y);
    let max_mag = vs[0]
        .norm()
        .max(vs[1].norm())
        .max(vs[2].norm())
        .max(vs[3].norm());
    debug_assert!(u_min <= u_max && v_min <= v_max);
    max_mag > 1e-6
        && ((u_min <= 0.0 && u_max >= 0.0 && v_min <= 0.0 && v_max >= 0.0)
            || vc.norm() <= 0.35 * max_mag)
}

#[verified_engine::verified]
fn refine_stagnation(
    field: &super::potential_flow::PotentialFlowField,
    seed: nalgebra::Vector2<f64>,
    diag: f64,
    core_r2: f64,
) -> Option<nalgebra::Vector2<f64>> {
    use super::potential_flow::FlowElement;
    let mut p = seed;
    let eps = (diag * 1e-4).max(1e-6);
    for _ in 0..8 {
        if field.is_near_singularity(p.x, p.y, core_r2) || (p - seed).norm() > 1.75 * diag {
            return None;
        }
        let f = field.velocity(p.x, p.y);
        if f.norm() <= 1e-7 {
            return Some(p);
        }
        let df_dx = (field.velocity(p.x + eps, p.y) - field.velocity(p.x - eps, p.y)) / (2.0 * eps);
        let df_dy = (field.velocity(p.x, p.y + eps) - field.velocity(p.x, p.y - eps)) / (2.0 * eps);
        let det = df_dx.x * df_dy.y - df_dy.x * df_dx.y;
        if !det.is_finite() || det.abs() <= math_commons::registry::TOLERANCE_HIGH {
            break;
        }
        let step = nalgebra::Vector2::new(
            df_dy.y * f.x - df_dy.x * f.y,
            -df_dx.y * f.x + df_dx.x * f.y,
        ) / det;
        let scale = (0.75 * diag / step.norm().max(diag)).min(1.0);
        p -= step * scale;
    }
    let final_v = field.velocity(p.x, p.y);
    debug_assert!(eps > 0.0);
    (!field.is_near_singularity(p.x, p.y, core_r2)
        && (p - seed).norm() <= 1.75 * diag
        && final_v.norm() <= 1e-3)
        .then_some(p)
}
