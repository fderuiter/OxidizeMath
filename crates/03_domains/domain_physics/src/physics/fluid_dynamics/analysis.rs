//! Analytical tools and derived quantities for Fluid Dynamics.

use super::potential_flow::{FlowElement, PotentialFlowField};
use super::types::{FlowState, FluidProperties};
use nalgebra::Vector2;

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

const MARCHING_EDGES: &[&[(usize, usize)]] = &[
    &[],
    &[(3, 0)],
    &[(0, 1)],
    &[(3, 1)],
    &[(1, 2)],
    &[(3, 0), (1, 2)],
    &[(0, 2)],
    &[(3, 2)],
    &[(2, 3)],
    &[(0, 2)],
    &[(0, 3), (2, 1)],
    &[(2, 1)],
    &[(1, 3)],
    &[(1, 0)],
    &[(3, 0)],
    &[],
];

#[verified_engine::verified]
fn has_valid_bounds(bounds: (f64, f64, f64, f64), resolution: usize) -> bool {
    let (x_min, x_max, y_min, y_max) = bounds;
    resolution >= 2
        && x_min.is_finite()
        && x_max.is_finite()
        && y_min.is_finite()
        && y_max.is_finite()
        && x_max > x_min
        && y_max > y_min
}

/// Rejects cells straddling singularity cores or `atan2` branch-cut discontinuities.
#[verified_engine::verified]
fn is_smooth_cell(
    field: &PotentialFlowField,
    corners: [[f64; 2]; 4],
    psi: [f64; 4],
    dx: f64,
    dy: f64,
) -> bool {
    if !psi.iter().all(|v| v.is_finite()) {
        return false;
    }
    let core_r2 = (dx * dx + dy * dy) * 0.64;
    let cx = 0.5 * (corners[0][0] + corners[1][0]);
    let cy = 0.5 * (corners[0][1] + corners[3][1]);
    if field.is_near_singularity(cx, cy, core_r2) {
        return false;
    }
    for &[x, y] in &corners {
        if field.is_near_singularity(x, y, core_r2 * 0.25) {
            return false;
        }
    }
    let vel = field.velocity(cx, cy);
    let expected = vel.x.abs() * dy + vel.y.abs() * dx + 1e-6;
    for (a, b) in [(0, 1), (1, 2), (3, 2), (0, 3)] {
        if (psi[b] - psi[a]).abs() > expected * 3.5 + 1e-4 {
            return false;
        }
    }
    true
}

#[verified_engine::verified]
fn edge_point(edge: usize, corners: [[f64; 2]; 4], psi: [f64; 4], level: f64) -> [f64; 2] {
    let (i0, i1) = match edge {
        0 => (0, 1),
        1 => (1, 2),
        2 => (3, 2),
        _ => (0, 3),
    };
    let denom = psi[i1] - psi[i0];
    let t = if denom.abs() < 1e-12 {
        0.5
    } else {
        ((level - psi[i0]) / denom).clamp(0.0, 1.0)
    };
    [
        corners[i0][0] + t * (corners[i1][0] - corners[i0][0]),
        corners[i0][1] + t * (corners[i1][1] - corners[i0][1]),
    ]
}

#[verified_engine::verified]
fn append_cell_contours(
    corners: [[f64; 2]; 4],
    psi: [f64; 4],
    iso_levels: &[f64],
    segments: &mut Vec<([f64; 2], [f64; 2])>,
) {
    let min_psi = psi[0].min(psi[1]).min(psi[2]).min(psi[3]);
    let max_psi = psi[0].max(psi[1]).max(psi[2]).max(psi[3]);
    for &level in iso_levels {
        if !level.is_finite() || level < min_psi || level > max_psi {
            continue;
        }
        let mut mask = 0usize;
        for (bit, &val) in psi.iter().enumerate() {
            if val >= level {
                mask |= 1 << bit;
            }
        }
        for &(e0, e1) in MARCHING_EDGES[mask] {
            let p0 = edge_point(e0, corners, psi, level);
            let p1 = edge_point(e1, corners, psi, level);
            segments.push((p0, p1));
        }
    }
}

/// Extracts iso-$\psi$ contour line segments over `(x_min, x_max, y_min, y_max)` via marching squares.
#[verified_engine::verified]
pub fn march_streamlines(
    field: &PotentialFlowField,
    bounds: (f64, f64, f64, f64),
    resolution: usize,
    iso_levels: &[f64],
) -> Vec<([f64; 2], [f64; 2])> {
    if iso_levels.is_empty() || !has_valid_bounds(bounds, resolution) {
        return Vec::new();
    }
    let (x_min, x_max, y_min, y_max) = bounds;
    let nx = resolution;
    let ny = resolution;
    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;
    let stride = nx + 1;
    let mut grid = vec![0.0; stride * (ny + 1)];
    for j in 0..=ny {
        let y = y_min + j as f64 * dy;
        for i in 0..=nx {
            let x = x_min + i as f64 * dx;
            let idx = math_commons::math_kernel::types::flatten_2d_index(i, j, stride);
            grid[idx] = field.stream_function(x, y);
        }
    }
    let mut segments = Vec::new();
    for j in 0..ny {
        let y0 = y_min + j as f64 * dy;
        let y1 = y0 + dy;
        for i in 0..nx {
            let x0 = x_min + i as f64 * dx;
            let x1 = x0 + dx;
            let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
            let psi = [
                grid[math_commons::math_kernel::types::flatten_2d_index(i, j, stride)],
                grid[math_commons::math_kernel::types::flatten_2d_index(i + 1, j, stride)],
                grid[math_commons::math_kernel::types::flatten_2d_index(i + 1, j + 1, stride)],
                grid[math_commons::math_kernel::types::flatten_2d_index(i, j + 1, stride)],
            ];
            if is_smooth_cell(field, corners, psi, dx, dy) {
                append_cell_contours(corners, psi, iso_levels, &mut segments);
            }
        }
    }
    segments
}

#[verified_engine::verified]
fn cell_may_contain_stagnation(vels: [Vector2<f64>; 4], tol: f64) -> bool {
    let min_u = vels[0].x.min(vels[1].x).min(vels[2].x).min(vels[3].x);
    let max_u = vels[0].x.max(vels[1].x).max(vels[2].x).max(vels[3].x);
    let min_v = vels[0].y.min(vels[1].y).min(vels[2].y).min(vels[3].y);
    let max_v = vels[0].y.max(vels[1].y).max(vels[2].y).max(vels[3].y);
    min_u <= tol && max_u >= -tol && min_v <= tol && max_v >= -tol
}

#[verified_engine::verified]
fn refine_stagnation(
    field: &PotentialFlowField,
    seed: Vector2<f64>,
    core_r2: f64,
    max_step: f64,
) -> Option<Vector2<f64>> {
    let mut p = seed;
    let h = 1e-5;
    for _ in 0..14 {
        if field.is_near_singularity(p.x, p.y, core_r2) {
            return None;
        }
        let f = field.velocity(p.x, p.y);
        if f.norm_squared() < 1e-14 {
            return Some(p);
        }
        let fx = (field.velocity(p.x + h, p.y) - field.velocity(p.x - h, p.y)) / (2.0 * h);
        let fy = (field.velocity(p.x, p.y + h) - field.velocity(p.x, p.y - h)) / (2.0 * h);
        let det = fx.x * fy.y - fy.x * fx.y;
        if det.abs() < 1e-14 {
            break;
        }
        let mut step = Vector2::new(
            -(fy.y * f.x - fy.x * f.y) / det,
            -(-fx.y * f.x + fx.x * f.y) / det,
        );
        let norm = step.norm();
        if norm > max_step {
            step *= max_step / norm;
        }
        p += step;
    }
    let residual = field.velocity(p.x, p.y).norm_squared();
    if !field.is_near_singularity(p.x, p.y, core_r2) && residual < 1e-8 {
        Some(p)
    } else {
        None
    }
}

/// Detects stagnation points ($\mathbf{V} = \mathbf{0}$) outside singularity cores via damped Newton refinement.
#[verified_engine::verified]
pub fn find_stagnation_points(
    field: &PotentialFlowField,
    bounds: (f64, f64, f64, f64),
    resolution: usize,
) -> Vec<Vector2<f64>> {
    if field.elements.is_empty() || !has_valid_bounds(bounds, resolution) {
        return Vec::new();
    }
    let (x_min, x_max, y_min, y_max) = bounds;
    let dx = (x_max - x_min) / resolution as f64;
    let dy = (y_max - y_min) / resolution as f64;
    let core_r2 = (dx * dx + dy * dy) * 0.36;
    let merge_r2 = (dx * dx + dy * dy) * 0.25;
    let max_step = dx.hypot(dy);
    let mut roots: Vec<Vector2<f64>> = Vec::new();

    for j in 0..resolution {
        let y0 = y_min + j as f64 * dy;
        for i in 0..resolution {
            let x0 = x_min + i as f64 * dx;
            let vels = [
                field.velocity(x0, y0),
                field.velocity(x0 + dx, y0),
                field.velocity(x0 + dx, y0 + dy),
                field.velocity(x0, y0 + dy),
            ];
            let scale = (vels[0].norm() + vels[2].norm()) * 0.25 + 1e-4;
            if !cell_may_contain_stagnation(vels, scale) {
                continue;
            }
            let seed = Vector2::new(x0 + 0.5 * dx, y0 + 0.5 * dy);
            if let Some(root) = refine_stagnation(field, seed, core_r2, max_step)
                && root.x >= x_min
                && root.x <= x_max
                && root.y >= y_min
                && root.y <= y_max
                && !roots.iter().any(|r| (r - root).norm_squared() <= merge_r2)
            {
                roots.push(root);
            }
        }
    }
    roots
}
