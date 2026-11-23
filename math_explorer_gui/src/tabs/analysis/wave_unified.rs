//! Unified 2D acoustic wave tank model (`WaveUnified`) supporting zero-allocation
//! `TimelineRingBuffer` scrubbing and perceptual raster shadow-twin difference maps.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::unified::{UnifiedCheckpointSlot, UnifiedModel};
use crate::async_sim::{SimCommand, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use eframe::egui::Color32;
use math_commons::math_kernel::colormap::heatmap_color;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::analysis::pde::wave_2d::{
    apply_multi_slit_barrier, apply_sponge_layer, BoundaryMode2D, Wave2DMedium, Wave2DParams,
    Wave2DSolver,
};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Default grid resolution (`64 x 64`) for [`WaveUnified`], fitting ~20 Verlet checkpoints in `2 MiB`.
pub const WAVE_UNIFIED_GRID: usize = 64;
/// Fixed integration timestep per worker tick.
pub const WAVE_UNIFIED_DT: f64 = 0.008;

/// 2D Störmer-Verlet acoustic wave tank implementing [`UnifiedModel`] with raster shadow-twin branching.
#[derive(Clone)]
pub struct WaveUnified {
    solver: Wave2DSolver,
    wave_params: Wave2DParams,
    slit_width: usize,
    step_count: usize,
    diverged: bool,
    diagnostic_report: Option<DiagnosticReport>,
}

impl WaveUnified {
    /// Builds a standalone [`crate::async_sim::unified::UnifiedSimTool`] wrapping [`WaveUnified`].
    #[must_use]
    pub fn build_tool() -> Box<dyn crate::framework::InteractiveTool> {
        Box::new(crate::async_sim::unified::UnifiedSimTool::<Self>::new())
    }

    fn sync_params(&mut self, params: &HashMap<String, f64>) {
        let c = params.get("wave_speed").copied().unwrap_or(1.0);
        let damping = params.get("damping").copied().unwrap_or(0.02);
        let next_slit = params
            .get("slit_width")
            .copied()
            .unwrap_or(6.0)
            .round()
            .clamp(2.0, 18.0) as usize;
        self.wave_params.wave_speed = c;
        self.wave_params.damping = damping;
        if next_slit != self.slit_width {
            self.slit_width = next_slit;
            self.rebuild_geometry();
        }
    }

    fn rebuild_geometry(&mut self) {
        let w = self.solver.width();
        let h = self.solver.height();
        self.solver.wall_mask.fill(1.0);
        apply_sponge_layer(&mut self.solver.sponge_mask, w, h, 4, 1.5);
        apply_multi_slit_barrier(&mut self.solver.wall_mask, w, h, w / 2, 2, 2, self.slit_width, 14.0);
    }

    fn seed_initial_pulse(&mut self, params: &HashMap<String, f64>) {
        let amp = params.get("pulse_amp").copied().unwrap_or(1.0);
        self.solver.clear_wave_state();
        self.rebuild_geometry();
        self.solver.inject_gaussian_pulse(0.25, 0.5, 0.07, amp);
    }
}

impl FiniteStateGuard for WaveUnified {
    fn is_finite_state(&self) -> bool {
        !self.diverged && self.solver.displacement().iter().all(|v| v.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic_report.as_ref()
    }

    fn diverged(&self) -> bool {
        self.diverged || !FiniteStateGuard::is_finite_state(self)
    }

    fn reset_parameters(&mut self) {
        self.diverged = false;
        self.diagnostic_report = None;
        self.step_count = 0;
    }
}

impl UnifiedModel for WaveUnified {
    fn domain() -> &'static str {
        "analysis"
    }

    fn name() -> &'static str {
        "2D Acoustic Wave Tank (Unified)"
    }

    fn is_raster_model() -> bool {
        true
    }

    fn supports_timeline_branch() -> bool {
        true
    }

    fn time_unit() -> &'static str {
        "s"
    }

    fn new(params: &HashMap<String, f64>) -> Self {
        let c = params.get("wave_speed").copied().unwrap_or(1.0);
        let damping = params.get("damping").copied().unwrap_or(0.02);
        let slit_width = params
            .get("slit_width")
            .copied()
            .unwrap_or(6.0)
            .round()
            .clamp(2.0, 18.0) as usize;
        let mut model = Self {
            solver: Wave2DSolver::new_unit_square(WAVE_UNIFIED_GRID),
            wave_params: Wave2DParams {
                medium: Wave2DMedium::AcousticWave,
                boundary: BoundaryMode2D::Sponge,
                wave_speed: c,
                plate_stiffness: 0.0,
                damping,
            },
            slit_width,
            step_count: 0,
            diverged: false,
            diagnostic_report: None,
        };
        model.seed_initial_pulse(params);
        model
    }

    fn step(&mut self, params: &HashMap<String, f64>) {
        if self.diverged {
            return;
        }
        self.sync_params(params);
        if !self.wave_params.wave_speed.is_finite() || self.wave_params.wave_speed <= 0.0 {
            self.diverged = true;
            self.diagnostic_report = Some(DiagnosticReport::new(
                self.solver.time(),
                self.step_count,
                "wave_speed",
                "Wave speed must be finite and positive.",
            ));
            return;
        }
        self.solver.step(&self.wave_params, WAVE_UNIFIED_DT);
        self.step_count += 1;
        if !FiniteStateGuard::is_finite_state(self) {
            self.diverged = true;
            self.diagnostic_report = Some(DiagnosticReport::new(
                self.solver.time(),
                self.step_count,
                "displacement",
                "Non-finite displacement encountered in 2D wave solver.",
            ));
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let w = self.solver.width();
        let h = self.solver.height();
        let disp = self.solver.displacement();
        let mut pixels = vec![Color32::BLACK; w * h];
        for y in 0..h {
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                if self.solver.wall_mask[idx] <= 0.5 {
                    pixels[idx] = Color32::from_gray(55);
                } else {
                    let norm = (0.5 + disp[idx] * 0.85).clamp(0.0, 1.0);
                    let rgb = heatmap_color(norm);
                    pixels[idx] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                }
            }
        }
        StateSnapshot {
            width: w,
            height: h,
            pixels: Arc::new(RwLock::new(pixels)),
            custom_data: vec![self.solver.time(), self.step_count as f64],
            structured_data: None,
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn process_command(&mut self, cmd: SimCommand, params: &HashMap<String, f64>) {
        if let SimCommand::Custom(ref action) = cmd {
            if action == "InjectPulse" {
                let amp = params.get("pulse_amp").copied().unwrap_or(1.0);
                self.solver.inject_gaussian_pulse(0.25, 0.5, 0.07, amp);
            }
        }
    }

    fn parameters() -> HashMap<String, ParameterConstraint> {
        let c = |min, max, step| ParameterConstraint { min, max, step };
        HashMap::from([
            ("wave_speed".to_string(), c(0.2, 2.5, 0.05)),
            ("damping".to_string(), c(0.0, 0.5, 0.01)),
            ("slit_width".to_string(), c(2.0, 18.0, 1.0)),
            ("pulse_amp".to_string(), c(0.2, 2.0, 0.1)),
        ])
    }

    fn time_step_dt(&self, _params: &HashMap<String, f64>) -> f64 {
        WAVE_UNIFIED_DT
    }

    fn checkpoint_template(&self) -> UnifiedCheckpointSlot {
        let cells = self.solver.width() * self.solver.height();
        UnifiedCheckpointSlot {
            scalars: [0.0; 8],
            field: vec![0.0; self.solver.verlet_checkpoint_field_len()],
            mask: vec![1; cells],
        }
    }

    fn write_checkpoint(&self, slot: &mut UnifiedCheckpointSlot) {
        let mut s4 = [0.0; 4];
        self.solver
            .export_verlet_checkpoint(&mut s4, &mut slot.field, &mut slot.mask);
        slot.scalars[0..4].copy_from_slice(&s4);
        slot.scalars[4] = self.step_count as f64;
        slot.scalars[5] = self.slit_width as f64;
    }

    fn restore_checkpoint(&mut self, slot: &UnifiedCheckpointSlot, params: &HashMap<String, f64>) {
        let mut s4 = [0.0; 4];
        s4.copy_from_slice(&slot.scalars[0..4]);
        self.solver
            .restore_verlet_checkpoint(&s4, &slot.field, &slot.mask);
        self.step_count = slot.scalars[4].max(0.0) as usize;
        self.slit_width = slot.scalars[5].round().clamp(2.0, 18.0) as usize;
        self.wave_params.wave_speed = params.get("wave_speed").copied().unwrap_or(1.0);
        self.wave_params.damping = params.get("damping").copied().unwrap_or(0.02);
        self.diverged = false;
        self.diagnostic_report = None;
    }

    fn fork_shadow_twin(&self) -> Option<Self> {
        Some(self.clone())
    }

    fn scalar_field_len(&self) -> usize {
        self.solver.displacement().len()
    }

    fn read_scalar_field(&self, out: &mut [f64]) -> usize {
        let disp = self.solver.displacement();
        let n = disp.len().min(out.len());
        out[..n].copy_from_slice(&disp[..n]);
        n
    }
}

impl TheoryDescribable for WaveUnified {
    fn theory_description(&self) -> String {
        "2D Störmer-Verlet acoustic wave equation ∂²u/∂t² = c²∇²u - γ∂u/∂t with double-slit diffraction barrier.".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Two dimensional Stoermer Verlet acoustic wave equation with double slit diffraction barrier.".to_string()
    }

    fn theory_citation(&self) -> String {
        "Courant, R., Friedrichs, K., & Lewy, H. (1928). On the partial difference equations of mathematical physics.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::from([("summary".to_string(), self.theory_description())])
    }
}
