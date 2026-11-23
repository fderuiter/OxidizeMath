#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use eframe::egui;
use math_explorer::pure_math::analysis::pde::wave_2d::{
    apply_biconvex_lens, apply_circular_drumhead_mask, apply_multi_slit_barrier, apply_point_clamp,
    apply_sponge_layer, BoundaryMode2D, Wave2DMedium, Wave2DParams, Wave2DSolver,
};

use super::{ChladniWaveTankState, ChladniWaveTankTool, ClampLayout, TankPreset, GRID_SIZE};
use crate::accessibility::announce_status;
use crate::audio::frame::LIMITER_CEILING_LINEAR;
use crate::audio::studio_preview::trigger_studio_ir_strike_from_gesture;
use crate::audio::wav::encode_stereo_ir_wav;
use crate::audio::wavetable::condition_stereo_ir_buffers;
use crate::web_io::PlatformInstant;

pub(crate) const IR_DURATION_MIN_SECS: f64 = 0.5;
pub(crate) const IR_DURATION_MAX_SECS: f64 = 4.0;
pub(crate) const PICKUP_COORD_MIN: f64 = 0.04;
pub(crate) const PICKUP_COORD_MAX: f64 = 0.96;
pub(crate) const PLATE_IR_GRID: usize = 36;
pub(crate) const ACOUSTIC_IR_GRID: usize = 64;
pub(crate) const COOPERATIVE_IR_BUDGET_MS: f64 = 6.0;
const SLICE_CHUNK_SAMPLES: usize = 64;

pub(super) const fn default_pickup_left() -> [f64; 2] {
    [0.32, 0.58]
}

pub(super) const fn default_pickup_right() -> [f64; 2] {
    [0.68, 0.42]
}

pub(super) const fn default_ir_duration_secs() -> f64 {
    1.5
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PickupDragTarget {
    #[default]
    None,
    Left,
    Right,
}

pub(crate) struct ChladniIrStudio {
    pub(crate) plate_solver: Wave2DSolver,
    pub(crate) wave_solver: Wave2DSolver,
    pub(crate) left_buf: Vec<f32>,
    pub(crate) right_buf: Vec<f32>,
    pub(crate) active_params: Wave2DParams,
    pub(crate) dt_per_sample: f64,
    pub(crate) use_plate_grid: bool,
    pub(crate) render_cursor: usize,
    pub(crate) render_total: usize,
    pub(crate) is_rendering: bool,
    pub(crate) pending_download: bool,
    pub(crate) last_slice_ms: f64,
    pub(crate) active_drag: PickupDragTarget,
    pub(crate) last_rendered_wav: Option<Vec<u8>>,
    pub(crate) cached_state_rev: u64,
}

impl Default for ChladniIrStudio {
    fn default() -> Self {
        Self {
            plate_solver: Wave2DSolver::new_unit_square(PLATE_IR_GRID),
            wave_solver: Wave2DSolver::new_unit_square(ACOUSTIC_IR_GRID),
            left_buf: Vec::new(),
            right_buf: Vec::new(),
            active_params: Wave2DParams::default(),
            dt_per_sample: 0.01,
            use_plate_grid: true,
            render_cursor: 0,
            render_total: 0,
            is_rendering: false,
            pending_download: false,
            last_slice_ms: 0.0,
            active_drag: PickupDragTarget::None,
            last_rendered_wav: None,
            cached_state_rev: 0,
        }
    }
}

pub(super) fn are_ir_fields_valid(s: &ChladniWaveTankState) -> bool {
    let valid_coord = |p: [f64; 2]| {
        p[0].is_finite()
            && (PICKUP_COORD_MIN..=PICKUP_COORD_MAX).contains(&p[0])
            && p[1].is_finite()
            && (PICKUP_COORD_MIN..=PICKUP_COORD_MAX).contains(&p[1])
    };
    valid_coord(s.pickup_left)
        && valid_coord(s.pickup_right)
        && s.ir_duration_secs.is_finite()
        && (IR_DURATION_MIN_SECS..=IR_DURATION_MAX_SECS).contains(&s.ir_duration_secs)
}

pub(super) fn clamp_ir_fields(s: &mut ChladniWaveTankState) {
    for (coord, def) in [
        (&mut s.pickup_left, default_pickup_left()),
        (&mut s.pickup_right, default_pickup_right()),
    ] {
        coord[0] = if coord[0].is_finite() {
            coord[0].clamp(PICKUP_COORD_MIN, PICKUP_COORD_MAX)
        } else {
            def[0]
        };
        coord[1] = if coord[1].is_finite() {
            coord[1].clamp(PICKUP_COORD_MIN, PICKUP_COORD_MAX)
        } else {
            def[1]
        };
    }
    s.ir_duration_secs = if s.ir_duration_secs.is_finite() {
        s.ir_duration_secs
            .clamp(IR_DURATION_MIN_SECS, IR_DURATION_MAX_SECS)
    } else {
        default_ir_duration_secs()
    };
}

pub(super) fn configure_solver_masks(solver: &mut Wave2DSolver, state: &ChladniWaveTankState) {
    let w = solver.width();
    let h = solver.height();
    let scale = (w as f64) / (GRID_SIZE as f64);
    solver.wall_mask.fill(1.0);
    solver.sponge_mask.fill(0.0);
    solver.wave_speed_mask.fill(1.0);

    if state.preset == TankPreset::CircularDrumhead {
        apply_circular_drumhead_mask(&mut solver.wall_mask, w, h, 0.45);
    }
    if state.boundary == BoundaryMode2D::Sponge {
        let sponge_w = ((12.0 * scale).round() as usize).max(3);
        apply_sponge_layer(&mut solver.sponge_mask, w, h, sponge_w, 1.8);
    }
    apply_scaled_clamp_pins(solver, state, scale);
    if state.barrier_enabled {
        let bx =
            (((state.barrier_x as f64) * scale).round() as usize).clamp(2, w.saturating_sub(4));
        let thick = ((2.0 * scale).round() as usize).max(1);
        let sw = ((state.slit_width as f64) * scale).round().max(1.0) as usize;
        let sep = (state.slit_separation * scale).max(2.0);
        apply_multi_slit_barrier(
            &mut solver.wall_mask,
            w,
            h,
            bx,
            thick,
            state.slit_count,
            sw,
            sep,
        );
    }
    if state.lens_enabled {
        let cx = (((44.0 * scale).round()) as usize).clamp(4, w.saturating_sub(4));
        let ap = (state.lens_aperture * scale).max(4.0);
        let ht = (8.0 * scale).max(2.0);
        apply_biconvex_lens(
            &mut solver.wave_speed_mask,
            w,
            h,
            cx,
            ap,
            ht,
            state.refractive_index,
        );
    }
}

fn apply_scaled_clamp_pins(solver: &mut Wave2DSolver, state: &ChladniWaveTankState, scale: f64) {
    let (w, h) = (solver.width(), solver.height());
    let (pins, radius): (&[(f64, f64)], f64) = match state.clamp_layout {
        ClampLayout::None => (&[], 0.0),
        ClampLayout::CenterPin => (&[(0.5, 0.5)], 2.0),
        ClampLayout::QuadSymmetry => (&[(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)], 1.8),
        ClampLayout::EdgeMidpoints => (&[(0.5, 0.04), (0.5, 0.96), (0.04, 0.5), (0.96, 0.5)], 2.2),
    };
    let r_scaled = (radius * scale).max(0.95);
    for &(cx, cy) in pins {
        apply_point_clamp(&mut solver.wall_mask, w, h, cx, cy, r_scaled);
    }
    let custom_r = (1.8 * scale).max(0.95);
    for &[cx, cy] in &state.custom_clamps {
        apply_point_clamp(&mut solver.wall_mask, w, h, cx, cy, custom_r);
    }
}

impl ChladniIrStudio {
    /// Prepares pre-allocated buffers and strikes the PDE domain for deterministic IR capture.
    pub(crate) fn prepare_ir_capture(
        &mut self,
        state: &ChladniWaveTankState,
        total_samples: usize,
    ) {
        let samples = total_samples.max(64);
        if self.left_buf.len() < samples {
            self.left_buf.resize(samples, 0.0);
        }
        if self.right_buf.len() < samples {
            self.right_buf.resize(samples, 0.0);
        }
        self.left_buf[..samples].fill(0.0);
        self.right_buf[..samples].fill(0.0);

        self.use_plate_grid = state.medium == Wave2DMedium::BiharmonicPlate
            && !state.barrier_enabled
            && !state.lens_enabled;
        if self.use_plate_grid {
            self.prepare_plate_ir_solver(state);
        } else {
            self.prepare_acoustic_ir_solver(state);
        }
        self.render_cursor = 0;
        self.render_total = samples;
        self.is_rendering = true;
    }

    fn prepare_plate_ir_solver(&mut self, state: &ChladniWaveTankState) {
        let sr = state.ir_sample_rate.hz() as f64;
        self.plate_solver.reset_all();
        configure_solver_masks(&mut self.plate_solver, state);
        let d = state.plate_stiffness.max(1.0e-5);
        let h = self.plate_solver.dx();
        let dt_max = (0.18 * h.powi(4) / d).sqrt();
        let pitch_scale = (d / 0.0004).powf(0.20).clamp(0.55, 1.8);
        self.dt_per_sample = (0.94 * dt_max * pitch_scale) * (48_000.0 / sr);
        let pde_per_sec = self.dt_per_sample * sr;
        let t60_decay = (2.2 + 10.0 * state.damping) / pde_per_sec.max(1.0e-6);
        self.active_params = Wave2DParams {
            medium: Wave2DMedium::BiharmonicPlate,
            boundary: state.boundary,
            wave_speed: 0.0,
            plate_stiffness: d,
            damping: t60_decay,
        };
        self.plate_solver
            .inject_gaussian_pulse(state.driver_x, state.driver_y, 0.055, 1.0);
    }

    fn prepare_acoustic_ir_solver(&mut self, state: &ChladniWaveTankState) {
        let sr = state.ir_sample_rate.hz() as f64;
        let cell_spacing = if state.barrier_enabled
            || state.lens_enabled
            || state.boundary == BoundaryMode2D::Sponge
        {
            1.0
        } else {
            1.0 / ((ACOUSTIC_IR_GRID - 1) as f64)
        };
        if (self.wave_solver.dx() - cell_spacing).abs() > 1.0e-12 {
            self.wave_solver = Wave2DSolver::new(
                ACOUSTIC_IR_GRID,
                ACOUSTIC_IR_GRID,
                cell_spacing,
                cell_spacing,
            );
        } else {
            self.wave_solver.reset_all();
        }
        configure_solver_masks(&mut self.wave_solver, state);
        let c = state.wave_speed.max(0.25);
        let dt_max = 0.66 * cell_spacing / c;
        self.dt_per_sample = (0.90 * dt_max) * (48_000.0 / sr);
        let pde_per_sec = self.dt_per_sample * sr;
        let t60_decay = (1.8 + 8.0 * state.damping) / pde_per_sec.max(1.0e-6);
        self.active_params = Wave2DParams {
            medium: Wave2DMedium::AcousticWave,
            boundary: state.boundary,
            wave_speed: c,
            plate_stiffness: 0.0,
            damping: t60_decay,
        };
        let sigma = if cell_spacing >= 0.5 { 0.022 } else { 0.04 };
        self.wave_solver
            .inject_gaussian_pulse(state.driver_x, state.driver_y, sigma, 1.0);
    }

    /// Zero-allocation inner pickup stepping loop suitable for `VerifiedEngine::run_verified`.
    pub(crate) fn step_pickup_frames_zero_alloc(
        &mut self,
        pickup_left: [f64; 2],
        pickup_right: [f64; 2],
        max_frames: usize,
    ) -> usize {
        let remaining = self.render_total.saturating_sub(self.render_cursor);
        let count = max_frames.min(remaining);
        let solver = if self.use_plate_grid {
            &mut self.plate_solver
        } else {
            &mut self.wave_solver
        };
        for offset in 0..count {
            let idx = self.render_cursor + offset;
            solver.step(&self.active_params, self.dt_per_sample);
            self.left_buf[idx] =
                solver.sample_displacement_bilinear(pickup_left[0], pickup_left[1]) as f32;
            self.right_buf[idx] =
                solver.sample_displacement_bilinear(pickup_right[0], pickup_right[1]) as f32;
        }
        self.render_cursor += count;
        if self.render_cursor >= self.render_total {
            self.is_rendering = false;
        }
        count
    }
}

impl ChladniWaveTankTool {
    pub(super) fn start_cooperative_ir_export(&mut self, trigger_download: bool) {
        let total_samples =
            ((self.state.ir_duration_secs * (self.state.ir_sample_rate.hz() as f64)).round()
                as usize)
                .clamp(22_050, 192_000);
        self.ir_studio
            .prepare_ir_capture(&self.state, total_samples);
        self.ir_studio.pending_download = trigger_download;
    }

    pub(super) fn advance_cooperative_ir_slice(&mut self, ctx: &egui::Context) {
        if !self.ir_studio.is_rendering {
            return;
        }
        let slice_start = PlatformInstant::now();
        let pl = self.state.pickup_left;
        let pr = self.state.pickup_right;
        while self.ir_studio.is_rendering {
            self.ir_studio
                .step_pickup_frames_zero_alloc(pl, pr, SLICE_CHUNK_SAMPLES);
            if slice_start.elapsed().as_secs_f64() * 1_000.0 >= COOPERATIVE_IR_BUDGET_MS {
                break;
            }
        }
        self.ir_studio.last_slice_ms = slice_start.elapsed().as_secs_f64() * 1_000.0;
        if self.ir_studio.is_rendering {
            ctx.request_repaint();
        } else {
            self.finalize_rendered_ir();
        }
    }

    pub(crate) fn render_complete_ir_blocking(&mut self, duration_secs: f64) -> Vec<u8> {
        let samples = ((duration_secs.clamp(0.1, IR_DURATION_MAX_SECS)
            * (self.state.ir_sample_rate.hz() as f64))
            .round() as usize)
            .max(256);
        self.ir_studio.prepare_ir_capture(&self.state, samples);
        self.ir_studio.pending_download = false;
        self.ir_studio.step_pickup_frames_zero_alloc(
            self.state.pickup_left,
            self.state.pickup_right,
            samples,
        );
        self.finalize_rendered_ir();
        self.ir_studio.last_rendered_wav.clone().unwrap_or_default()
    }

    fn finalize_rendered_ir(&mut self) {
        let n = self.ir_studio.render_total;
        condition_stereo_ir_buffers(
            &mut self.ir_studio.left_buf[..n],
            &mut self.ir_studio.right_buf[..n],
            LIMITER_CEILING_LINEAR,
        );
        let wav_bytes = encode_stereo_ir_wav(
            self.state.ir_sample_rate,
            self.state.ir_format,
            &self.ir_studio.left_buf[..n],
            &self.ir_studio.right_buf[..n],
        );
        self.ir_studio.cached_state_rev = self.state_rev;
        let filename = format!(
            "chladni_{:?}_{}hz.wav",
            self.state.preset,
            self.state.ir_sample_rate.hz()
        )
        .to_lowercase();
        if self.ir_studio.pending_download {
            self.ir_studio.pending_download = false;
            save_ir_wav_file(&filename, wav_bytes.clone());
            announce_status(&format!("Exported stereo PDE impulse response {filename}."));
        }
        self.ir_studio.last_rendered_wav = Some(wav_bytes);
    }

    pub(super) fn audition_strike_from_user_gesture(&mut self) {
        self.trigger_mallet_pulse(self.state.driver_x, self.state.driver_y, 0.045, 1.0, 1.0);
        if self.ir_studio.cached_state_rev != self.state_rev
            || self.ir_studio.render_total == 0
            || self.ir_studio.is_rendering
        {
            let preview_secs = self.state.ir_duration_secs.clamp(0.5, 1.25);
            let _ = self.render_complete_ir_blocking(preview_secs);
        }
        let n = self.ir_studio.render_total;
        if let Err(err) = trigger_studio_ir_strike_from_gesture(
            self.state.ir_sample_rate.hz(),
            &self.ir_studio.left_buf[..n],
            &self.ir_studio.right_buf[..n],
            0.85,
        ) {
            announce_status(&format!("Strike preview unavailable: {err}"));
        } else {
            announce_status("Playing stereo PDE impulse response strike.");
        }
    }
}

fn save_ir_wav_file(filename: &str, wav_bytes: Vec<u8>) {
    #[cfg(not(all(test, not(target_arch = "wasm32"))))]
    {
        let (tx, _rx) = std::sync::mpsc::channel();
        let config = crate::web_io::SaveAsDialogConfig {
            filter_name: "WAV Impulse Response (*.wav)",
            extensions: &["wav"],
            fallback_filename: "chladni_impulse_response.wav",
            write_error_prefix: "Failed to write IR WAV",
        };
        crate::web_io::spawn_save_binary_file_dialog(tx, config, filename, wav_bytes);
    }
    #[cfg(all(test, not(target_arch = "wasm32")))]
    {
        let _ = (filename, wav_bytes);
    }
}
