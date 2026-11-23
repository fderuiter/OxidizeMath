#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{
    default_gs_f, default_gs_k, MorphogenesisModelKind, MorphogenesisToolState, SimpleRng,
};
use super::MorphogenesisTool;
use crate::async_sim::SimWorkerState;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::InteractiveTool;
use eframe::egui;
use eframe::egui::ColorImage;
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{
    GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use math_explorer::biology::reaction_diffusion::ChemicalState;

impl MorphogenesisTool {
    pub(crate) fn apply_named_preset(&mut self, a: f64, b: f64, d_u: f64, d_v: f64, dt: f64) {
        self.apply_model_preset(
            MorphogenesisModelKind::Schnakenberg,
            (a, b),
            (default_gs_f(), default_gs_k()),
            (d_u, d_v, dt),
        );
    }

    pub(crate) fn apply_model_preset(
        &mut self,
        model: MorphogenesisModelKind,
        ab: (f64, f64),
        fk: (f64, f64),
        diff_dt: (f64, f64, f64),
    ) {
        let (a, b) = ab;
        let (f, k) = fk;
        let (d_u, d_v, dt) = diff_dt;
        let target_state = if self.worker.is_running() {
            SimWorkerState::Running
        } else {
            SimWorkerState::Ready
        };
        let (xk, yk) = self.atlas.axis_keys();
        let preset_state = MorphogenesisToolState {
            a,
            b,
            d_u,
            d_v,
            dt,
            width: self.width,
            height: self.height,
            model,
            f,
            k,
            state: Some(target_state),
            paused: None,
            pending_initial_start: false,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        };
        if let Ok(json) = serde_json::to_string(&preset_state) {
            let _ = self.try_load_state(&json);
        }
    }

    pub(crate) fn initialize_system(
        system: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
        width: usize,
        height: usize,
    ) {
        let n = width * height;
        let mut rng = SimpleRng::new(12345);
        let a = system.kinetics.a;
        let b = system.kinetics.b;
        let u_eq = a + b;
        let v_eq = if u_eq.abs() > 1e-9 {
            b / (u_eq * u_eq)
        } else {
            0.0
        };
        for i in 0..n {
            system.u_mut()[i] = u_eq + rng.range(-0.1, 0.1);
            system.v_mut()[i] = v_eq + rng.range(-0.1, 0.1);
        }
    }

    pub(crate) fn initialize_gray_scott_system(
        system: &mut TuringSystem<2, GrayScottKinetics, FiniteDifference2D>,
        width: usize,
        height: usize,
    ) {
        system.u_mut().fill(1.0);
        system.v_mut().fill(0.0);
        let mut rng = SimpleRng::new(54321);
        let cx = width / 2;
        let cy = height / 2;
        let r = (width.min(height) / 10).max(2);
        for y in cy.saturating_sub(r)..=(cy + r).min(height.saturating_sub(1)) {
            for x in cx.saturating_sub(r)..=(cx + r).min(width.saturating_sub(1)) {
                let idx = flatten_2d_index(x, y, width);
                system.u_mut()[idx] = (0.50 + rng.range(-0.04, 0.04)).clamp(0.0, 1.0);
                system.v_mut()[idx] = (0.25 + rng.range(-0.04, 0.04)).clamp(0.0, 1.0);
            }
        }
    }

    pub(crate) fn initialize_gray_scott_slices(
        u: &mut [f64],
        v: &mut [f64],
        width: usize,
        height: usize,
    ) {
        u.fill(1.0);
        v.fill(0.0);
        let mut rng = SimpleRng::new(54321);
        let cx = width / 2;
        let cy = height / 2;
        let r = (width.min(height) / 10).max(2);
        for y in cy.saturating_sub(r)..=(cy + r).min(height.saturating_sub(1)) {
            for x in cx.saturating_sub(r)..=(cx + r).min(width.saturating_sub(1)) {
                let idx = flatten_2d_index(x, y, width);
                u[idx] = (0.50 + rng.range(-0.04, 0.04)).clamp(0.0, 1.0);
                v[idx] = (0.25 + rng.range(-0.04, 0.04)).clamp(0.0, 1.0);
            }
        }
    }

    pub(crate) fn initialize_chemical_state(
        state: &mut ChemicalState,
        width: usize,
        height: usize,
    ) {
        let n = width * height;
        if state.num_species() < 2 || state.grid_size() != n {
            return;
        }
        let (u, v) = state.grid.as_mut_slice().split_at_mut(n);
        Self::initialize_gray_scott_slices(u, v, width, height);
    }

    #[cfg(test)]
    pub(crate) fn plot_concentration(data: &[f64], width: usize, height: usize) -> ColorImage {
        Self::plot_concentration_with_contrast(data, width, height, false)
    }

    pub(crate) fn plot_concentration_with_contrast(
        data: &[f64],
        width: usize,
        height: usize,
        high_contrast: bool,
    ) -> ColorImage {
        let num_pixels = width * height;
        let mut pixels = Vec::with_capacity(num_pixels);
        let (min, max) =
            data.iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                    if val.is_finite() {
                        (min.min(val), max.max(val))
                    } else {
                        (min, max)
                    }
                });
        let (min, max) = if min.is_finite() && max.is_finite() && max >= min {
            (min, max)
        } else {
            (0.0, 1.0)
        };
        let range = max - min;
        let inv_range = if range > 1e-6 { 1.0 / range } else { 1.0 };

        for &val in data {
            let safe_val = if val.is_finite() { val } else { min };
            let norm = (safe_val - min) * inv_range;
            let safe_norm = if norm.is_finite() {
                norm.clamp(0.0, 1.0)
            } else {
                0.0
            };
            let rgb = if high_contrast {
                heatmap_color_high_contrast(safe_norm)
            } else {
                heatmap_color(safe_norm)
            };
            pixels.push(egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
        }

        ColorImage {
            size: [width, height],
            pixels,
            ..Default::default()
        }
    }
}

impl FiniteStateGuard for MorphogenesisTool {
    fn is_finite_state(&self) -> bool {
        self.worker.state() != SimWorkerState::Faulted && self.worker.diagnostic_report().is_none()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.worker.diagnostic_report()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.dt = (p.dt * 0.5).max(1e-4);
        drop(p);
        self.reset_system();
    }

    fn clamp_parameters(&mut self) {
        let model = {
            let mut mp = self.model_params.write().unwrap_or_else(|e| e.into_inner());
            mp.f = mp.f.clamp(0.005, 0.12);
            mp.k = mp.k.clamp(0.03, 0.08);
            mp.model
        };
        let (du_min, dv_min, dt_max) = match model {
            MorphogenesisModelKind::Schnakenberg => (0.1, 10.0, 0.1),
            MorphogenesisModelKind::GrayScott
            | MorphogenesisModelKind::GenericReactionDiffusion => (0.01, 0.01, 1.0),
        };
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = p.a.clamp(0.0, 1.0);
        p.b = p.b.clamp(0.0, 2.0);
        p.d_u = p.d_u.clamp(du_min, 5.0);
        p.d_v = p.d_v.clamp(dv_min, 200.0);
        p.dt = p.dt.clamp(0.001, dt_max);
        drop(p);
        self.reset_system();
    }

    fn reset_parameters(&mut self) {
        {
            let mut mp = self.model_params.write().unwrap_or_else(|e| e.into_inner());
            mp.f = default_gs_f();
            mp.k = default_gs_k();
            mp.model = MorphogenesisModelKind::Schnakenberg;
        }
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = 0.1;
        p.b = 0.9;
        p.d_u = 1.0;
        p.d_v = 100.0;
        p.dt = 0.05;
        drop(p);
        self.reset_system();
    }
}
