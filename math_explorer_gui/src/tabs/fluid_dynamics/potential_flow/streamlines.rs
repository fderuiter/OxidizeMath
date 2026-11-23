//! Stream-function marching-squares contour extraction, stagnation-point detection,
//! and Kutta-Joukowski aerodynamic observables for [`super::PotentialFlowTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ElementType, PotentialFlowTool};
use math_explorer::physics::fluid_dynamics::potential_flow::{FlowElement, Vortex};
use std::f64::consts::PI;

const CONTOUR_GRID_N: usize = 48;
const NUM_CONTOUR_LEVELS: usize = 15;

impl PotentialFlowTool {
    pub(crate) fn refresh_streamline_cache(&mut self) {
        self.cached_view_range = self.view_range;
        self.cached_stagnation = self.detect_stagnation_points();
        let segments = self.compute_streamline_segments();
        self.cached_polylines = chain_segments_into_polylines(&segments);
    }

    pub(crate) fn compute_arrow_field(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        if !self.view_range.is_finite()
            || self.view_range <= 0.0
            || !(2..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_size)
        {
            return (Vec::new(), Vec::new());
        }
        let step = (2.0 * self.view_range) / (self.grid_size as f64);
        if !step.is_finite() || step <= 0.0 {
            return (Vec::new(), Vec::new());
        }
        let min = -self.view_range;
        let cyl = self.inferred_cylinder();
        let mut origins = Vec::new();
        let mut tips = Vec::new();
        for ix in 0..=self.grid_size {
            let x = min + (ix as f64) * step;
            for iy in 0..=self.grid_size {
                let y = min + (iy as f64) * step;
                if is_inside_cylinder(x, y, cyl, 0.92) {
                    continue;
                }
                let v = self.field.velocity(x, y);
                let mag = v.norm();
                if !v.x.is_finite() || !v.y.is_finite() || !mag.is_finite() || mag <= 1e-6 {
                    continue;
                }
                let arrow_len = (mag * 0.1).min(step * 0.8);
                let end = [x + (v.x / mag) * arrow_len, y + (v.y / mag) * arrow_len];
                if arrow_len > 0.0 && end[0].is_finite() && end[1].is_finite() {
                    origins.push([x, y]);
                    tips.push(end);
                }
            }
        }
        (origins, tips)
    }

    pub(crate) fn aerodynamic_summary(&self) -> (f64, f64, f64, f64) {
        let (mut ux, mut uy, mut gamma, mut net_q) = (0.0, 0.0, 0.0, 0.0);
        for el in &self.elements {
            match el.element_type {
                ElementType::Uniform => {
                    let rad = el.angle.to_radians();
                    ux += el.strength * rad.cos();
                    uy += el.strength * rad.sin();
                }
                ElementType::Vortex => gamma += el.strength,
                ElementType::Source => net_q += el.strength,
                ElementType::Sink => net_q -= el.strength,
                ElementType::Doublet => {}
            }
        }
        let u_inf = ux.hypot(uy);
        let lift = -u_inf * gamma;
        let drag = if net_q.abs() <= 1e-6 {
            0.0
        } else {
            (u_inf * net_q).abs()
        };
        (u_inf, gamma, lift, drag)
    }

    pub(crate) fn flow_observables(&self) -> Vec<(&'static str, f64)> {
        let (u_inf, gamma, lift, drag) = self.aerodynamic_summary();
        let r_cyl = self.inferred_cylinder().map_or(0.0, |(_, _, r)| r);
        vec![
            ("freestream_speed", u_inf),
            ("circulation", gamma),
            ("circulation_abs", gamma.abs()),
            ("kutta_joukowski_lift", lift),
            ("lift_magnitude", lift.abs()),
            ("pressure_drag", drag),
            ("cylinder_radius", r_cyl),
            ("stagnation_count", self.cached_stagnation.len() as f64),
        ]
    }

    fn inferred_cylinder(&self) -> Option<(f64, f64, f64)> {
        let (u_inf, _, _, _) = self.aerodynamic_summary();
        let doublet = self
            .elements
            .iter()
            .find(|e| e.element_type == ElementType::Doublet && e.strength > 1e-6)?;
        if u_inf <= 1e-6 {
            return None;
        }
        let r = (doublet.strength / (2.0 * PI * u_inf)).sqrt();
        (r.is_finite() && r > 0.05).then_some((doublet.x, doublet.y, r))
    }

    fn sample_psi(&self, x: f64, y: f64, cyl: Option<(f64, f64, f64)>) -> Option<f64> {
        if is_inside_cylinder(x, y, cyl, 0.90) || self.is_near_singularity(x, y) {
            return None;
        }
        let mut psi = self.field.stream_function(x, y);
        for el in &self.elements {
            if el.element_type == ElementType::Vortex {
                psi -= 2.0 * Vortex::new(el.strength, el.x, el.y).stream_function(x, y);
            }
        }
        psi.is_finite().then_some(psi)
    }

    fn is_near_singularity(&self, x: f64, y: f64) -> bool {
        let min_r = (self.view_range * 0.025).max(0.12);
        self.elements
            .iter()
            .any(|el| el.element_type != ElementType::Uniform && (x - el.x).hypot(y - el.y) < min_r)
    }

    fn max_source_jump(&self) -> f64 {
        self.elements
            .iter()
            .filter_map(|el| match el.element_type {
                ElementType::Source | ElementType::Sink => Some(el.strength.abs()),
                _ => None,
            })
            .fold(0.0_f64, f64::max)
    }

    pub(crate) fn compute_streamline_segments(&self) -> Vec<[[f64; 2]; 2]> {
        if self.elements.is_empty() || !self.view_range.is_finite() || self.view_range <= 0.0 {
            return Vec::new();
        }
        let cyl = self.inferred_cylinder();
        let grid = self.sample_psi_grid(cyl);
        let levels = self.select_contour_levels(&grid, cyl);
        if levels.is_empty() {
            return Vec::new();
        }
        let jump_cap = {
            let q = self.max_source_jump();
            if q > 1e-6 {
                0.42 * q
            } else {
                f64::INFINITY
            }
        };
        let step = (2.0 * self.view_range) / (CONTOUR_GRID_N as f64);
        let min = -self.view_range;
        let mut segments = Vec::new();
        for ix in 0..CONTOUR_GRID_N {
            let x0 = min + (ix as f64) * step;
            for iy in 0..CONTOUR_GRID_N {
                let y0 = min + (iy as f64) * step;
                let cell = [
                    grid[ix][iy],
                    grid[ix + 1][iy],
                    grid[ix + 1][iy + 1],
                    grid[ix][iy + 1],
                ];
                let Some(vals) = unpack_cell(cell, jump_cap) else {
                    continue;
                };
                let corners = [
                    [x0, y0],
                    [x0 + step, y0],
                    [x0 + step, y0 + step],
                    [x0, y0 + step],
                ];
                for &lvl in &levels {
                    emit_cell_iso_segments(corners, vals, lvl, &mut segments);
                }
            }
        }
        segments
    }

    fn sample_psi_grid(&self, cyl: Option<(f64, f64, f64)>) -> Vec<Vec<Option<f64>>> {
        let step = (2.0 * self.view_range) / (CONTOUR_GRID_N as f64);
        let min = -self.view_range;
        (0..=CONTOUR_GRID_N)
            .map(|ix| {
                let x = min + (ix as f64) * step;
                (0..=CONTOUR_GRID_N)
                    .map(|iy| self.sample_psi(x, min + (iy as f64) * step, cyl))
                    .collect()
            })
            .collect()
    }

    fn select_contour_levels(
        &self,
        grid: &[Vec<Option<f64>>],
        cyl: Option<(f64, f64, f64)>,
    ) -> Vec<f64> {
        let mut min_psi = f64::INFINITY;
        let mut max_psi = f64::NEG_INFINITY;
        for col in grid {
            for &v in col.iter().flatten() {
                min_psi = min_psi.min(v);
                max_psi = max_psi.max(v);
            }
        }
        if !min_psi.is_finite() || !max_psi.is_finite() || (max_psi - min_psi).abs() < 1e-9 {
            return Vec::new();
        }
        let span = max_psi - min_psi;
        let mut levels: Vec<f64> = (1..=NUM_CONTOUR_LEVELS)
            .map(|k| min_psi + span * (k as f64) / ((NUM_CONTOUR_LEVELS + 1) as f64))
            .collect();
        for &[sx, sy] in &self.cached_stagnation {
            if let Some(psi_s) = self.sample_psi(sx, sy, None) {
                levels.push(psi_s);
            }
        }
        if let Some((cx, cy, r)) = cyl {
            if let Some(psi_cyl) = self.sample_psi(cx, cy + r * 1.001, None) {
                levels.push(psi_cyl);
            }
        }
        levels.sort_by(|a, b| a.total_cmp(b));
        levels.dedup_by(|a, b| (*a - *b).abs() < span * 1e-3);
        levels
    }

    pub(crate) fn detect_stagnation_points(&self) -> Vec<[f64; 2]> {
        if self.elements.len() < 2 || !self.view_range.is_finite() || self.view_range <= 0.0 {
            return Vec::new();
        }
        let (u_inf, _, _, _) = self.aerodynamic_summary();
        let ref_speed = u_inf.max(1.0);
        let cyl = self.inferred_cylinder();
        let n = 36_usize;
        let step = (2.0 * self.view_range) / (n as f64);
        let min = -self.view_range;
        let mut found: Vec<[f64; 2]> = Vec::new();
        for ix in 0..=n {
            for iy in 0..=n {
                let seed = [min + (ix as f64) * step, min + (iy as f64) * step];
                if self.field.velocity(seed[0], seed[1]).norm() > ref_speed * 0.65 {
                    continue;
                }
                if let Some(pt) = self.refine_stagnation(seed, ref_speed, cyl) {
                    if found
                        .iter()
                        .all(|p| (p[0] - pt[0]).hypot(p[1] - pt[1]) > 0.25)
                    {
                        found.push(pt);
                    }
                }
            }
        }
        found
    }

    fn refine_stagnation(
        &self,
        mut p: [f64; 2],
        ref_speed: f64,
        cyl: Option<(f64, f64, f64)>,
    ) -> Option<[f64; 2]> {
        let h = 1e-4;
        for _ in 0..8 {
            let v = self.field.velocity(p[0], p[1]);
            if !v.x.is_finite() || !v.y.is_finite() {
                return None;
            }
            if v.norm() < ref_speed * 1e-4 {
                break;
            }
            let vx = (self.field.velocity(p[0] + h, p[1]) - self.field.velocity(p[0] - h, p[1]))
                / (2.0 * h);
            let vy = (self.field.velocity(p[0], p[1] + h) - self.field.velocity(p[0], p[1] - h))
                / (2.0 * h);
            let det = vx.x * vy.y - vx.y * vy.x;
            if !det.is_finite() || det.abs() < 1e-10 {
                return None;
            }
            let dx = (vy.y * v.x - vy.x * v.y) / det;
            let dy = (-vx.y * v.x + vx.x * v.y) / det;
            let scale = 1.0_f64.min(0.6 / dx.hypot(dy).max(1e-9));
            p[0] -= dx * scale;
            p[1] -= dy * scale;
        }
        let speed = self.field.velocity(p[0], p[1]).norm();
        let in_bounds = p[0].abs() <= self.view_range && p[1].abs() <= self.view_range;
        (speed < ref_speed * 2e-3
            && in_bounds
            && !is_inside_cylinder(p[0], p[1], cyl, 0.96)
            && !self.is_near_singularity(p[0], p[1]))
        .then_some(p)
    }
}

fn is_inside_cylinder(x: f64, y: f64, cyl: Option<(f64, f64, f64)>, ratio: f64) -> bool {
    cyl.is_some_and(|(cx, cy, r)| (x - cx).hypot(y - cy) < r * ratio)
}

fn unpack_cell(cell: [Option<f64>; 4], jump_cap: f64) -> Option<[f64; 4]> {
    let vals = [cell[0]?, cell[1]?, cell[2]?, cell[3]?];
    let min_v = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ((max_v - min_v) <= jump_cap).then_some(vals)
}

fn emit_cell_iso_segments(
    corners: [[f64; 2]; 4],
    vals: [f64; 4],
    level: f64,
    out: &mut Vec<[[f64; 2]; 2]>,
) {
    let mut hits = [[0.0_f64; 2]; 4];
    let mut count = 0_usize;
    for edge in 0..4 {
        let next = (edge + 1) % 4;
        let (v0, v1) = (vals[edge] - level, vals[next] - level);
        if (v0 <= 0.0 && v1 > 0.0) || (v0 > 0.0 && v1 <= 0.0) {
            let denom = v0 - v1;
            if denom.abs() > 1e-12 {
                let t = (v0 / denom).clamp(0.0, 1.0);
                hits[count] = [
                    corners[edge][0] + t * (corners[next][0] - corners[edge][0]),
                    corners[edge][1] + t * (corners[next][1] - corners[edge][1]),
                ];
                count += 1;
            }
        }
    }
    if count == 2 {
        out.push([hits[0], hits[1]]);
    } else if count == 4 {
        out.push([hits[0], hits[1]]);
        out.push([hits[2], hits[3]]);
    }
}

fn chain_segments_into_polylines(segments: &[[[f64; 2]; 2]]) -> Vec<Vec<[f64; 2]>> {
    let mut used = vec![false; segments.len()];
    let mut polylines = Vec::new();
    let eps = 1e-4;
    for start in 0..segments.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut poly = vec![segments[start][0], segments[start][1]];
        loop {
            let tail = *poly.last().unwrap();
            let mut matched = None;
            for (i, seg) in segments.iter().enumerate() {
                if used[i] {
                    continue;
                }
                if (seg[0][0] - tail[0]).hypot(seg[0][1] - tail[1]) < eps {
                    matched = Some((i, seg[1]));
                    break;
                }
                if (seg[1][0] - tail[0]).hypot(seg[1][1] - tail[1]) < eps {
                    matched = Some((i, seg[0]));
                    break;
                }
            }
            let Some((idx, next_pt)) = matched else { break };
            used[idx] = true;
            poly.push(next_pt);
        }
        polylines.push(poly);
    }
    polylines
}

#[cfg(test)]
mod tests {
    use super::super::{ElementType, PotentialFlowTool, SerializedFlowElement};
    use std::f64::consts::PI;

    #[test]
    fn test_cylinder_streamlines_stagnation_and_kutta_lift() {
        let tool = PotentialFlowTool::default();
        assert!(
            !tool.cached_polylines.is_empty(),
            "streamline contours must be generated"
        );
        assert_eq!(
            tool.cached_stagnation.len(),
            2,
            "symmetric cylinder has 2 stagnation points"
        );

        let mut spinning = PotentialFlowTool::default();
        spinning.elements.push(SerializedFlowElement {
            element_type: ElementType::Vortex,
            strength: -40.0 * PI,
            x: 0.0,
            y: 0.0,
            angle: 0.0,
        });
        spinning.rebuild_field();
        let obs = spinning.flow_observables();
        let lift = obs
            .iter()
            .find(|(k, _)| *k == "kutta_joukowski_lift")
            .unwrap()
            .1;
        let drag = obs.iter().find(|(k, _)| *k == "pressure_drag").unwrap().1;
        assert!(lift > 1250.0 && drag.abs() < 1e-9);
        assert_eq!(spinning.cached_stagnation.len(), 2);
    }
}
