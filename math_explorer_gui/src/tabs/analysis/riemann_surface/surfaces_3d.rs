//! Multi-sheet 3D Riemann surface tessellation (`NthRootSurface`, `LogSurface`,
//! `RiemannSurface`) and branch-cut monodromy path lifting for [`super::RiemannSurfaceStudio`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{RiemannBranchFamily, RiemannSurfaceStudio, SheetQuad3D, SurfaceHeightChannel};
use math_explorer::pure_math::analysis::complex::surfaces::{
    LogSurface, NthRootSurface, RiemannSurface,
};
use num_complex::Complex64;
use std::f64::consts::{PI, TAU};

fn map_complex_to_3d_and_color(
    z: Complex64,
    w: Complex64,
    channel: SurfaceHeightChannel,
) -> ([f64; 3], f64) {
    let (height, color_val) = match channel {
        SurfaceHeightChannel::ImaginaryPart => (w.im * 0.38, w.re),
        SurfaceHeightChannel::RealPart => (w.re * 0.65, w.im),
    };
    let clamped_h = if height.is_finite() {
        height.clamp(-3.5, 3.5)
    } else {
        0.0
    };
    let safe_color = if color_val.is_finite() {
        color_val
    } else {
        0.0
    };
    ([z.re, z.im, clamped_h], safe_color)
}

fn append_sheet_quads(
    surface: &dyn RiemannSurface,
    sheet: isize,
    has_bridge: bool,
    target_sheet: isize,
    res: (f64, usize, usize),
    channel: SurfaceHeightChannel,
    quads: &mut Vec<SheetQuad3D>,
) -> (f64, f64) {
    let (r_max, r_steps, theta_steps) = res;
    let r_min = 0.06_f64;
    let (theta_min, theta_max) = (-PI + 1e-4, PI - 1e-4);
    let mut grid = vec![([0.0_f64; 3], 0.0_f64); (r_steps + 1) * (theta_steps + 1)];
    let (mut c_min, mut c_max) = (f64::INFINITY, f64::NEG_INFINITY);

    for ir in 0..=r_steps {
        let r = r_min + (r_max - r_min) * (ir as f64 / r_steps as f64);
        for it in 0..=theta_steps {
            let theta = theta_min + (theta_max - theta_min) * (it as f64 / theta_steps as f64);
            let z = Complex64::from_polar(r, theta);
            let (pt, c_val) = map_complex_to_3d_and_color(z, surface.evaluate(z, sheet), channel);
            c_min = c_min.min(c_val);
            c_max = c_max.max(c_val);
            grid[ir * (theta_steps + 1) + it] = (pt, c_val);
        }
    }
    for ir in 0..r_steps {
        for it in 0..theta_steps {
            let (p00, c00) = grid[ir * (theta_steps + 1) + it];
            let (p10, c10) = grid[(ir + 1) * (theta_steps + 1) + it];
            let (p11, c11) = grid[(ir + 1) * (theta_steps + 1) + it + 1];
            let (p01, c01) = grid[ir * (theta_steps + 1) + it + 1];
            quads.push(SheetQuad3D {
                corners: [p00, p10, p11, p01],
                color_scalar: 0.25 * (c00 + c10 + c11 + c01),
            });
        }
    }
    if has_bridge {
        for ir in 0..r_steps {
            let r0 = r_min + (r_max - r_min) * (ir as f64 / r_steps as f64);
            let r1 = r_min + (r_max - r_min) * ((ir + 1) as f64 / r_steps as f64);
            let (p00, c00) = grid[ir * (theta_steps + 1) + theta_steps];
            let (p10, c10) = grid[(ir + 1) * (theta_steps + 1) + theta_steps];
            let z0_n = Complex64::from_polar(r0, theta_min);
            let z1_n = Complex64::from_polar(r1, theta_min);
            let (p01, c01) =
                map_complex_to_3d_and_color(z0_n, surface.evaluate(z0_n, target_sheet), channel);
            let (p11, c11) =
                map_complex_to_3d_and_color(z1_n, surface.evaluate(z1_n, target_sheet), channel);
            quads.push(SheetQuad3D {
                corners: [p00, p10, p11, p01],
                color_scalar: 0.25 * (c00 + c10 + c11 + c01),
            });
        }
    }
    (c_min, c_max)
}

fn build_sheets_for_surface(
    surface: &dyn RiemannSurface,
    sheets: &[isize],
    r_max: f64,
    r_steps: usize,
    theta_steps: usize,
    channel: SurfaceHeightChannel,
) -> (Vec<SheetQuad3D>, (f64, f64)) {
    let mut quads = Vec::with_capacity(sheets.len() * r_steps * (theta_steps + 1));
    let (mut c_min, mut c_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &sheet in sheets {
        let target = surface.next_sheet(sheet, 1);
        let (s_min, s_max) = append_sheet_quads(
            surface,
            sheet,
            sheets.contains(&target),
            target,
            (r_max, r_steps, theta_steps),
            channel,
            &mut quads,
        );
        c_min = c_min.min(s_min);
        c_max = c_max.max(s_max);
    }
    if !c_min.is_finite() || !c_max.is_finite() || (c_max - c_min).abs() < 1e-9 {
        c_min = -1.0;
        c_max = 1.0;
    }
    (quads, (c_min, c_max))
}

fn trace_monodromy_loop(
    surface: &dyn RiemannSurface,
    center: [f64; 2],
    radius: f64,
    turns: isize,
    start_sheet: isize,
    channel: SurfaceHeightChannel,
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>, isize, Complex64, Complex64) {
    let steps = (96 * turns.unsigned_abs().max(1)).clamp(96, 384);
    let c = Complex64::new(center[0], center[1]);
    let normalized_start = surface.next_sheet(start_sheet, 0);
    let mut current_sheet = normalized_start;

    let z0 = c + Complex64::from_polar(radius, 0.0);
    let safe_z0 = if z0.norm() < 1e-4 {
        Complex64::new(1e-4, 0.0)
    } else {
        z0
    };
    let mut prev_arg = safe_z0.arg();
    let start_val = surface.evaluate(safe_z0, current_sheet);

    let mut path_3d = Vec::with_capacity(steps + 1);
    let mut loop_2d = Vec::with_capacity(steps + 1);

    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let phi = TAU * (turns as f64) * t;
        let raw_z = c + Complex64::from_polar(radius, phi);
        let z = if raw_z.norm() < 1e-4 {
            Complex64::new(1e-4, 1e-4)
        } else {
            raw_z
        };
        let arg = z.arg();
        if i > 0 {
            let diff = arg - prev_arg;
            if diff < -PI {
                // Crossed branch cut counter-clockwise (+pi -> -pi)
                current_sheet = surface.next_sheet(current_sheet, 1);
            } else if diff > PI {
                // Crossed branch cut clockwise (-pi -> +pi)
                current_sheet = surface.next_sheet(current_sheet, -1);
            }
        }
        prev_arg = arg;
        let w = surface.evaluate(z, current_sheet);
        let (mut pt3, _) = map_complex_to_3d_and_color(z, w, channel);
        pt3[2] += 0.025; // Slight normal lift so path remains crisp over surface quads
        path_3d.push(pt3);
        loop_2d.push([z.re, z.im]);
    }

    let end_z = c + Complex64::from_polar(radius, TAU * (turns as f64));
    let safe_end = if end_z.norm() < 1e-4 {
        Complex64::new(1e-4, 0.0)
    } else {
        end_z
    };
    let end_val = surface.evaluate(safe_end, current_sheet);
    (path_3d, loop_2d, current_sheet, start_val, end_val)
}

impl RiemannSurfaceStudio {
    pub(crate) fn rebuild_surface_and_monodromy(&mut self) {
        match self.branch_family {
            RiemannBranchFamily::NthRoot => {
                let n = self.root_n.clamp(2, 6);
                let surf = NthRootSurface::new(n);
                let sheets: Vec<isize> = (0..n as isize).collect();
                self.apply_surface_and_monodromy(&surf, &sheets);
            }
            RiemannBranchFamily::Logarithm => {
                let span = self.log_sheet_span.clamp(1, 3) as isize;
                let surf = LogSurface;
                let sheets: Vec<isize> = (-span..=span).collect();
                self.apply_surface_and_monodromy(&surf, &sheets);
            }
        }
    }

    fn apply_surface_and_monodromy(&mut self, surf: &dyn RiemannSurface, sheets: &[isize]) {
        let max_res = crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION;
        let r_max = self.radial_max.clamp(0.6, 2.5);
        let r_steps = self.radial_steps.clamp(8, max_res);
        let t_steps = self.angular_steps.clamp(16, max_res);
        self.diagnostics.sheet_count = surf.num_sheets();
        let (quads, range) =
            build_sheets_for_surface(surf, sheets, r_max, r_steps, t_steps, self.height_channel);
        let (p3, p2, end_s, w0, w1) = trace_monodromy_loop(
            surf,
            self.loop_center,
            self.loop_radius,
            self.winding_turns,
            self.start_sheet,
            self.height_channel,
        );
        self.sheet_quads = quads;
        self.color_range = range;
        self.lifted_path_3d = p3;
        self.base_loop_2d = p2;
        self.diagnostics.final_sheet = end_s;
        self.diagnostics.start_value = w0;
        self.diagnostics.end_value = w1;
        self.diagnostics.monodromy_jump = (w1 - w0).norm();
    }
}
