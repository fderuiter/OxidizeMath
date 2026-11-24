#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use math_commons::ast::CompiledExpr;
use math_explorer::pure_math::analysis::pde::wave_2d::{Wave2DMedium, Wave2DParams, Wave2DSolver};

use super::{WavetableSourceMode, WavetableStudioState};
use crate::audio::wav::encode_wavetable_wav;
use crate::audio::wavetable::{
    max_nyquist_harmonic, WavetableSpectralProcessor, PREVIEW_PERIODIC_WAVE_HARMONICS,
    WAVETABLE_FRAME_SAMPLES,
};
use crate::framework::MathExpressionInputBar;

pub(crate) const PDE_SLICE_GRID: usize = 32;

/// Pre-allocated single-cycle and multi-frame wavetable synthesis engine.
pub(crate) struct WavetableBankEngine {
    pub(crate) spectral: WavetableSpectralProcessor,
    pub(crate) pde_solver: Wave2DSolver,
    pub(crate) active_frame: [f32; WAVETABLE_FRAME_SAMPLES],
    pub(crate) keyframe_start: [f32; WAVETABLE_FRAME_SAMPLES],
    pub(crate) keyframe_end: [f32; WAVETABLE_FRAME_SAMPLES],
    pub(crate) periodic_real: [f32; PREVIEW_PERIODIC_WAVE_HARMONICS + 1],
    pub(crate) periodic_imag: [f32; PREVIEW_PERIODIC_WAVE_HARMONICS + 1],
    pub(crate) harmonic_mags: [f32; PREVIEW_PERIODIC_WAVE_HARMONICS],
    pub(crate) bank_samples: Vec<f32>,
    pub(crate) last_dc_mean: f32,
    pub(crate) last_nyquist_bin: usize,
}

impl WavetableBankEngine {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            spectral: WavetableSpectralProcessor::new(),
            pde_solver: Wave2DSolver::new_unit_square(PDE_SLICE_GRID),
            active_frame: [0.0; WAVETABLE_FRAME_SAMPLES],
            keyframe_start: [0.0; WAVETABLE_FRAME_SAMPLES],
            keyframe_end: [0.0; WAVETABLE_FRAME_SAMPLES],
            periodic_real: [0.0; PREVIEW_PERIODIC_WAVE_HARMONICS + 1],
            periodic_imag: [0.0; PREVIEW_PERIODIC_WAVE_HARMONICS + 1],
            harmonic_mags: [0.0; PREVIEW_PERIODIC_WAVE_HARMONICS],
            bank_samples: Vec::new(),
            last_dc_mean: 0.0,
            last_nyquist_bin: 128,
        }
    }

    /// Re-seeds and steps the `32x32` 2D PDE field so cross-section slices contain rich modal structure.
    pub(crate) fn rebuild_pde_field(&mut self, state: &WavetableStudioState) {
        self.pde_solver.reset_all();
        self.pde_solver
            .inject_gaussian_pulse(0.28, 0.36, 0.065, 1.0);
        self.pde_solver
            .inject_gaussian_pulse(0.71, 0.68, 0.050, -0.75);
        let params = Wave2DParams {
            medium: state.pde_medium,
            boundary: state.pde_boundary,
            wave_speed: if state.pde_medium == Wave2DMedium::AcousticWave {
                state.pde_wave_speed
            } else {
                0.0
            },
            plate_stiffness: if state.pde_medium == Wave2DMedium::BiharmonicPlate {
                state.pde_stiffness
            } else {
                0.0
            },
            damping: 0.02,
        };
        let dt = if state.pde_medium == Wave2DMedium::BiharmonicPlate {
            0.004
        } else {
            0.012
        };
        for _ in 0..18 {
            self.pde_solver.step(&params, dt);
        }
    }

    /// Zero-allocation `2048`-sample symbolic `CompiledExpr` frame evaluation + `rustfft` conditioning.
    pub(crate) fn evaluate_formula_frame_zero_alloc(
        &mut self,
        compiled: &CompiledExpr,
        params: &[f64],
        morph: f64,
        max_harmonic: usize,
    ) {
        let inv_half = 2.0 / (WAVETABLE_FRAME_SAMPLES as f64);
        for (i, slot) in self.active_frame.iter_mut().enumerate() {
            let x = -1.0 + (i as f64) * inv_half;
            let raw = compiled.eval_2d(x, morph, params);
            *slot = if raw.is_finite() {
                raw.clamp(-64.0, 64.0) as f32
            } else {
                0.0
            };
        }
        self.spectral
            .condition_frame_in_place(&mut self.active_frame, max_harmonic, 0.98);
        self.spectral.extract_periodic_wave_and_spectrum(
            &self.active_frame,
            &mut self.periodic_real,
            &mut self.periodic_imag,
            &mut self.harmonic_mags,
        );
        self.last_dc_mean = compute_mean(&self.active_frame);
        self.last_nyquist_bin = max_harmonic;
    }

    /// Zero-allocation `2048`-sample 2D PDE bilinear cross-section evaluation + `rustfft` conditioning.
    pub(crate) fn evaluate_pde_slice_frame_zero_alloc(
        &mut self,
        morph: f64,
        angle_rad: f64,
        max_harmonic: usize,
    ) {
        let (sin_a, cos_a) = angle_rad.sin_cos();
        let sweep_offset = -0.34 + 0.68 * morph.clamp(0.0, 1.0);
        let inv_n = 1.0 / (WAVETABLE_FRAME_SAMPLES as f64);
        for i in 0..WAVETABLE_FRAME_SAMPLES {
            let s = (i as f64) * inv_n; // [0, 1)
            let along = -0.42 + 0.84 * s;
            let xn = (0.5 + along * cos_a - sweep_offset * sin_a).clamp(0.03, 0.97);
            let yn = (0.5 + along * sin_a + sweep_offset * cos_a).clamp(0.03, 0.97);
            let window = (std::f64::consts::PI * s).sin();
            let u = self.pde_solver.sample_displacement_bilinear(xn, yn) * window;
            self.active_frame[i] = if u.is_finite() { u as f32 } else { 0.0 };
        }
        self.spectral
            .condition_frame_in_place(&mut self.active_frame, max_harmonic, 0.98);
        self.spectral.extract_periodic_wave_and_spectrum(
            &self.active_frame,
            &mut self.periodic_real,
            &mut self.periodic_imag,
            &mut self.harmonic_mags,
        );
        self.last_dc_mean = compute_mean(&self.active_frame);
        self.last_nyquist_bin = max_harmonic;
    }

    /// Refreshes start/end morph ghost curves and the active morph frame.
    pub(crate) fn refresh_single_preview(
        &mut self,
        state: &WavetableStudioState,
        expr_bar: &MathExpressionInputBar,
    ) {
        let max_h = max_nyquist_harmonic(
            state.sample_rate.hz(),
            state.bandlimit_fundamental_hz as f32,
        );
        self.eval_mode_at_morph(state, expr_bar, 0.0, max_h);
        self.keyframe_start.copy_from_slice(&self.active_frame);
        self.eval_mode_at_morph(state, expr_bar, 1.0, max_h);
        self.keyframe_end.copy_from_slice(&self.active_frame);
        self.eval_mode_at_morph(state, expr_bar, state.morph_pos, max_h);
    }

    fn eval_mode_at_morph(
        &mut self,
        state: &WavetableStudioState,
        expr_bar: &MathExpressionInputBar,
        morph: f64,
        max_h: usize,
    ) {
        match state.source_mode {
            WavetableSourceMode::FormulaExpr => {
                if let Some(compiled) = expr_bar.active_compiled() {
                    let params = expr_bar.param_slice();
                    self.evaluate_formula_frame_zero_alloc(compiled, params, morph, max_h);
                } else {
                    self.active_frame.fill(0.0);
                    self.harmonic_mags.fill(0.0);
                }
            }
            WavetableSourceMode::PdeWaveSlice => {
                let angle_rad = state.pde_slice_angle_deg.to_radians();
                self.evaluate_pde_slice_frame_zero_alloc(morph, angle_rad, max_h);
            }
        }
    }

    /// Synthesizes all `1..=256` frames into `self.bank_samples` and encodes `.wav` with `clm ` + `uhWT`.
    pub(crate) fn encode_bank_wav(
        &mut self,
        state: &WavetableStudioState,
        expr_bar: &MathExpressionInputBar,
    ) -> Vec<u8> {
        let frames = usize::from(state.frame_count.clamp(1, 256));
        let total_samples = frames * WAVETABLE_FRAME_SAMPLES;
        if self.bank_samples.len() < total_samples {
            self.bank_samples.resize(total_samples, 0.0);
        }
        let max_h = max_nyquist_harmonic(
            state.sample_rate.hz(),
            state.bandlimit_fundamental_hz as f32,
        );
        let saved_active = self.active_frame;
        let denom = (frames.saturating_sub(1)).max(1) as f64;
        for f_idx in 0..frames {
            let morph = if frames == 1 {
                state.morph_pos
            } else {
                (f_idx as f64) / denom
            };
            self.eval_mode_at_morph(state, expr_bar, morph, max_h);
            let dst = f_idx * WAVETABLE_FRAME_SAMPLES;
            self.bank_samples[dst..dst + WAVETABLE_FRAME_SAMPLES]
                .copy_from_slice(&self.active_frame);
        }
        self.active_frame = saved_active;
        self.spectral.extract_periodic_wave_and_spectrum(
            &self.active_frame,
            &mut self.periodic_real,
            &mut self.periodic_imag,
            &mut self.harmonic_mags,
        );

        encode_wavetable_wav(
            state.sample_rate,
            state.sample_format,
            WAVETABLE_FRAME_SAMPLES as u16,
            frames as u16,
            &self.bank_samples[..total_samples],
        )
    }
}

fn compute_mean(frame: &[f32; WAVETABLE_FRAME_SAMPLES]) -> f32 {
    let mut sum = 0.0_f32;
    for &s in frame {
        sum += s;
    }
    sum / (WAVETABLE_FRAME_SAMPLES as f32)
}
