#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use eframe::egui;
use math_explorer::pure_math::analysis::pde::wave_2d::SpectralMode2D;

use super::{ChladniWaveTankTool, GRID_SIZE};
use crate::audio::mappings::wave_plate::{chladni_wave_tank_frame, MAX_CHLADNI_MODES};
use crate::audio::{render_sonification_panel, SonificationFrame};

pub(crate) const SONIFICATION_TOOL_ID: &str = "chladni_wave_tank";
pub(crate) const MAX_CUSTOM_CLAMPS: usize = 16;
pub(crate) const CLAMP_COORD_MIN: f64 = 0.02;
pub(crate) const CLAMP_COORD_MAX: f64 = 0.98;

#[derive(Debug, Clone)]
pub(crate) struct ChladniSonificationState {
    pub(crate) top_modes: [SpectralMode2D; MAX_CHLADNI_MODES],
    pub(crate) mode_count: usize,
    pub(crate) pending_impulse: f32,
    pub(crate) latest_frame: SonificationFrame,
}

impl Default for ChladniSonificationState {
    fn default() -> Self {
        Self {
            top_modes: [SpectralMode2D::default(); MAX_CHLADNI_MODES],
            mode_count: 0,
            pending_impulse: 0.0,
            latest_frame: SonificationFrame::silent(),
        }
    }
}

pub(super) fn are_custom_clamps_valid(clamps: &[[f64; 2]]) -> bool {
    clamps.len() <= MAX_CUSTOM_CLAMPS
        && clamps.iter().all(|&[nx, ny]| {
            nx.is_finite()
                && (CLAMP_COORD_MIN..=CLAMP_COORD_MAX).contains(&nx)
                && ny.is_finite()
                && (CLAMP_COORD_MIN..=CLAMP_COORD_MAX).contains(&ny)
        })
}

pub(super) fn sanitize_custom_clamps(clamps: &mut Vec<[f64; 2]>) {
    clamps.truncate(MAX_CUSTOM_CLAMPS);
    for [nx, ny] in clamps.iter_mut() {
        *nx = if nx.is_finite() {
            nx.clamp(CLAMP_COORD_MIN, CLAMP_COORD_MAX)
        } else {
            0.5
        };
        *ny = if ny.is_finite() {
            ny.clamp(CLAMP_COORD_MIN, CLAMP_COORD_MAX)
        } else {
            0.5
        };
    }
}

impl ChladniWaveTankTool {
    pub(super) fn trigger_mallet_pulse(
        &mut self,
        nx: f64,
        ny: f64,
        sigma: f64,
        amplitude: f64,
        impulse: f32,
    ) {
        self.solver
            .inject_gaussian_pulse(nx, ny, sigma, amplitude);
        if impulse.is_finite() {
            self.audio.pending_impulse = self.audio.pending_impulse.max(impulse.clamp(0.0, 1.0));
        }
        self.refresh_analysis_outputs();
    }

    pub(super) fn update_sonification_frame(&mut self) {
        let field_finite = self.solver.displacement().iter().all(|v| v.is_finite());
        if !field_finite {
            self.audio.mode_count = 0;
            self.audio.pending_impulse = 0.0;
            self.audio.latest_frame = SonificationFrame::silent();
            crate::audio::notify_solver_diverged();
            return;
        }
        self.audio.mode_count = self.fft.extract_top_modes(&mut self.audio.top_modes);
        let base_len = if self.solver.dx() >= 0.5 {
            1.0
        } else {
            (((GRID_SIZE - 1) as f64) * self.solver.dx()).max(1.0e-3)
        };
        let preset_pins = match self.state.clamp_layout {
            super::ClampLayout::None => 0_usize,
            super::ClampLayout::CenterPin => 1,
            super::ClampLayout::QuadSymmetry | super::ClampLayout::EdgeMidpoints => 4,
        };
        let total_pins = preset_pins + self.state.custom_clamps.len();
        let domain_len =
            base_len / (1.0 + 0.035 * (total_pins as f64)).sqrt();
        let params = self.pde_params();
        self.audio.latest_frame = chladni_wave_tank_frame(
            &self.audio.top_modes[..self.audio.mode_count],
            &params,
            domain_len,
            &self.detector_buf,
            self.audio.pending_impulse,
        );
    }

    pub(super) fn draw_sonification_section(&mut self, ui: &mut egui::Ui) {
        render_sonification_panel(
            ui,
            SONIFICATION_TOOL_ID,
            &self.audio.latest_frame,
            self.paused,
        );
        if self.audio.pending_impulse > 0.0 {
            self.audio.pending_impulse = 0.0;
            for voice in &mut self.audio.latest_frame.voices {
                voice.transient_impulse = 0.0;
            }
        }
    }
}
