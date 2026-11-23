//! Grid seeding, preset transitions, and concentration colormap helpers for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::MorphogenesisToolState;
use super::runner::MorphogenesisKineticsAdapter;
use super::{MorphogenesisTool, SimpleRng};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::InteractiveTool;
use crate::presets::MorphogenesisKineticsMode;
#[cfg(test)]
use eframe::egui;
#[cfg(test)]
use eframe::egui::ColorImage;
#[cfg(test)]
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::biology::diffusion::FiniteDifference2D;
#[cfg(test)]
use math_explorer::biology::morphogenesis::SchnakenbergKinetics;
use math_explorer::biology::morphogenesis::TuringSystem;
use math_explorer::rand::Rng;
use math_explorer::rng::OxidizeRng;

impl MorphogenesisTool {
    pub(crate) fn apply_named_preset(&mut self, a: f64, b: f64, d_u: f64, d_v: f64, dt: f64) {
        let (f, k) = {
            let ks = self
                .kinetics_state
                .read()
                .unwrap_or_else(|e| e.into_inner());
            (ks.f, ks.k)
        };
        self.apply_full_preset(
            MorphogenesisKineticsMode::Schnakenberg,
            a,
            b,
            f,
            k,
            d_u,
            d_v,
            dt,
        );
    }

    pub(crate) fn apply_gray_scott_preset(&mut self, f: f64, k: f64) {
        let (a, b) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b)
        };
        self.apply_full_preset(
            MorphogenesisKineticsMode::GrayScott,
            a,
            b,
            f,
            k,
            0.16,
            0.08,
            1.0,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_full_preset(
        &mut self,
        kinetics_mode: MorphogenesisKineticsMode,
        a: f64,
        b: f64,
        f: f64,
        k: f64,
        d_u: f64,
        d_v: f64,
        dt: f64,
    ) {
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
            kinetics_mode,
            f,
            k,
            state: Some(target_state),
            paused: None,
            pending_initial_start: false,
            fork: None,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        };
        if let Ok(json) = serde_json::to_string(&preset_state) {
            let _ = self.try_load_state(&json);
        }
    }

    pub(super) fn apply_dimensions_and_target_state(
        &mut self,
        width: usize,
        height: usize,
        target_state: SimWorkerState,
        pending_initial_start: bool,
    ) {
        if width != self.width || height != self.height {
            self.width = width;
            self.height = height;
            self.local_runner = Self::create_kinetics_runner(
                &self.params,
                &self.kinetics_state,
                self.width,
                self.height,
            );
            let runner = Self::create_kinetics_runner(
                &self.params,
                &self.kinetics_state,
                self.width,
                self.height,
            );
            self.worker = BackgroundWorker::new(runner);
            self.branch_engine.clear();
            self.stepped_pixels = None;
            self.step_runner = None;
            self.dirty = true;
        } else {
            self.reset();
        }
        if target_state == SimWorkerState::Running {
            self.pending_initial_start = false;
            if !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            }
        } else {
            self.pending_initial_start = pending_initial_start;
            if self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Pause);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn initialize_schnakenberg(
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

    #[cfg(test)]
    pub(crate) fn initialize_system(
        system: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
        width: usize,
        height: usize,
    ) {
        Self::initialize_schnakenberg(system, width, height);
    }

    pub(crate) fn initialize_adapter_system(
        system: &mut TuringSystem<2, MorphogenesisKineticsAdapter, FiniteDifference2D>,
        width: usize,
        height: usize,
    ) {
        let mut rng = SimpleRng::new(12345);
        Self::initialize_adapter_system_with_rng(system, width, height, &mut rng);
    }

    pub(crate) fn initialize_adapter_system_with_rng(
        system: &mut TuringSystem<2, MorphogenesisKineticsAdapter, FiniteDifference2D>,
        width: usize,
        height: usize,
        rng: &mut SimpleRng,
    ) {
        match system.kinetics.mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                let n = width * height;
                let u_eq = system.kinetics.a + system.kinetics.b;
                let v_eq = if u_eq.abs() > 1e-9 {
                    system.kinetics.b / (u_eq * u_eq)
                } else {
                    0.0
                };
                for i in 0..n {
                    system.u_mut()[i] = u_eq + rng.range(-0.1, 0.1);
                    system.v_mut()[i] = v_eq + rng.range(-0.1, 0.1);
                }
            }
            MorphogenesisKineticsMode::GrayScott => {
                Self::seed_gray_scott_patches(system, width, height);
            }
        }
    }

    fn seed_gray_scott_patches(
        system: &mut TuringSystem<2, MorphogenesisKineticsAdapter, FiniteDifference2D>,
        width: usize,
        height: usize,
    ) {
        system.u_mut().fill(1.0);
        system.v_mut().fill(0.0);
        if width == 0 || height == 0 {
            return;
        }
        let mut rng = OxidizeRng::new(12345);
        let cx = width / 2;
        let cy = height / 2;
        let seeds = [
            (cx, cy, 5usize),
            (
                cx.saturating_sub(width / 5),
                cy.saturating_sub(height / 6),
                3,
            ),
            (
                (cx + width / 5).min(width - 1),
                (cy + height / 6).min(height - 1),
                3,
            ),
        ];
        for (sx, sy, half) in seeds {
            let x0 = sx.saturating_sub(half);
            let x1 = (sx + half).min(width - 1);
            let y0 = sy.saturating_sub(half);
            let y1 = (sy + half).min(height - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let idx = flatten_2d_index(x, y, width);
                    let u_val = 0.50_f64 + rng.gen_range(-0.02_f64..=0.02_f64);
                    let v_val = 0.25_f64 + rng.gen_range(-0.02_f64..=0.02_f64);
                    system.u_mut()[idx] = u_val.clamp(0.0, 1.0);
                    system.v_mut()[idx] = v_val.clamp(0.0, 1.0);
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn plot_concentration(data: &[f64], width: usize, height: usize) -> ColorImage {
        Self::plot_concentration_with_contrast(data, width, height, false)
    }

    #[cfg(test)]
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
