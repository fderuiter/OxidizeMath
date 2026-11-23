#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use std::f64::consts::TAU;

use eframe::egui;
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::analysis::pde::wave_2d::{
    analytic_clamped_square_omega, apply_biconvex_lens, apply_circular_drumhead_mask,
    apply_multi_slit_barrier, apply_point_clamp, apply_sponge_layer, BoundaryMode2D, Wave2DMedium,
    Wave2DParams, Wave2DSolver,
};

use super::{ChladniWaveTankState, ChladniWaveTankTool, ClampLayout, TankPreset, GRID_SIZE};
use crate::audio::frame::{SonificationFrame, VoiceSpec, WaveformPreset, LIMITER_CEILING_LINEAR};
use crate::audio::scala::ScalaScale;
use crate::audio::wav::encode_wav_stereo_pcm16;

/// Physical side length ($L = 5.0\text{ cm}$) of the acoustic plate resonator used for IR capture.
pub(crate) const CHLADNI_ACOUSTIC_LENGTH_METERS: f64 = 0.05;

/// Grid resolution ($21 \times 21$) balancing `< 2%` Leissa `(1,1)` accuracy with fast browser export.
pub(crate) const IR_GRID_SIZE: usize = 21;

/// Default normalized left stereo pickup coordinate `(x, y)`.
pub(crate) const fn default_pickup_left() -> [f64; 2] {
    [0.31, 0.43]
}

/// Default normalized right stereo pickup coordinate `(x, y)`.
pub(crate) const fn default_pickup_right() -> [f64; 2] {
    [0.69, 0.57]
}

/// Active pointer drag target on the 2D Chladni / wave tank canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum CanvasDragTarget {
    #[default]
    None,
    Driver,
    PickupLeft,
    PickupRight,
}

/// Pre-allocated workspace for zero-allocation 2D Verlet impulse-response rendering.
pub(crate) struct ChladniImpulseWorkspace {
    pub(crate) solver: Wave2DSolver,
}

impl ChladniImpulseWorkspace {
    /// Allocates a physical `grid_size x grid_size` resonator of side `length_meters`.
    pub(crate) fn new(grid_size: usize, length_meters: f64) -> Self {
        let n = grid_size.max(8);
        let h = length_meters.max(1.0e-3) / ((n - 1) as f64);
        Self {
            solver: Wave2DSolver::new(n, n, h, h),
        }
    }

    /// Resets wave buffers, rebuilds scaled masks, and injects the initial mallet/wave impulse.
    pub(crate) fn prepare(&mut self, state: &ChladniWaveTankState, custom_clamps: &[[f64; 2]]) {
        self.solver.reset_all();
        configure_domain_masks(&mut self.solver, state, custom_clamps);
        let w = self.solver.width();
        let center_idx = flatten_2d_index(w / 2, self.solver.height() / 2, w);
        let center_free = self.solver.wall_mask[center_idx] > 0.5;
        if state.medium == Wave2DMedium::BiharmonicPlate && center_free {
            let cx = (0.44 + 0.18 * (state.driver_x - 0.5)).clamp(0.18, 0.82);
            let cy = (0.46 + 0.18 * (state.driver_y - 0.5)).clamp(0.18, 0.82);
            self.solver.inject_gaussian_pulse(cx, cy, 0.18, 1.0);
        } else if state.medium == Wave2DMedium::BiharmonicPlate {
            self.solver
                .inject_gaussian_pulse(state.driver_x, state.driver_y, 0.10, 1.0);
        } else {
            self.solver
                .inject_gaussian_pulse(state.driver_x, state.driver_y, 0.06, 1.0);
        }
    }
}

/// Applies drumhead, sponge, nodal clamps, barriers, and acoustic lenses scaled to `solver`.
pub(crate) fn configure_domain_masks(
    solver: &mut Wave2DSolver,
    state: &ChladniWaveTankState,
    custom_clamps: &[[f64; 2]],
) {
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
        let thick = ((12.0 * scale).round() as usize).max(2);
        apply_sponge_layer(&mut solver.sponge_mask, w, h, thick, 1.8);
    }
    apply_scaled_clamp_pins(&mut solver.wall_mask, w, h, scale, state, custom_clamps);
    if state.barrier_enabled {
        let bx = ((state.barrier_x as f64) * scale).round() as usize;
        let bw = ((2.0 * scale).round() as usize).max(1);
        let sw = (((state.slit_width as f64) * scale).round() as usize).max(1);
        apply_multi_slit_barrier(
            &mut solver.wall_mask,
            w,
            h,
            bx.clamp(2, w.saturating_sub(4)),
            bw,
            state.slit_count,
            sw,
            (state.slit_separation * scale).max(2.0),
        );
    }
    if state.lens_enabled {
        let lx = (((44.0 * scale).round()) as usize).clamp(3, w.saturating_sub(4));
        apply_biconvex_lens(
            &mut solver.wave_speed_mask,
            w,
            h,
            lx,
            (state.lens_aperture * scale).max(3.0),
            (8.0 * scale).max(1.5),
            state.refractive_index,
        );
    }
}

fn apply_scaled_clamp_pins(
    wall_mask: &mut [f64],
    w: usize,
    h: usize,
    scale: f64,
    state: &ChladniWaveTankState,
    custom_clamps: &[[f64; 2]],
) {
    let (pins, radius): (&[(f64, f64)], f64) = match state.clamp_layout {
        ClampLayout::None => (&[], 0.0),
        ClampLayout::CenterPin => (&[(0.5, 0.5)], 2.0),
        ClampLayout::QuadSymmetry => (&[(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)], 1.8),
        ClampLayout::EdgeMidpoints => (&[(0.5, 0.04), (0.5, 0.96), (0.04, 0.5), (0.96, 0.5)], 2.2),
    };
    let r_scaled = (radius * scale).max(0.6);
    for &(cx, cy) in pins {
        apply_point_clamp(wall_mask, w, h, cx, cy, r_scaled);
    }
    let custom_r = (1.8 * scale).max(0.6);
    for &[cx, cy] in custom_clamps {
        apply_point_clamp(wall_mask, w, h, cx, cy, custom_r);
    }
}

/// Bilinearly samples a `width x height` scalar grid at normalized `[0.0, 1.0]^2` coordinates.
pub(crate) fn sample_bilinear(field: &[f64], width: usize, height: usize, pos: [f64; 2]) -> f64 {
    if width < 2 || height < 2 || field.len() < width * height {
        return 0.0;
    }
    let fx = pos[0].clamp(0.0, 1.0) * ((width - 1) as f64);
    let fy = pos[1].clamp(0.0, 1.0) * ((height - 1) as f64);
    let x0 = (fx.floor() as usize).min(width - 2);
    let y0 = (fy.floor() as usize).min(height - 2);
    let tx = fx - (x0 as f64);
    let ty = fy - (y0 as f64);

    let v00 = field[flatten_2d_index(x0, y0, width)];
    let v10 = field[flatten_2d_index(x0 + 1, y0, width)];
    let v01 = field[flatten_2d_index(x0, y0 + 1, width)];
    let v11 = field[flatten_2d_index(x0 + 1, y0 + 1, width)];
    let bottom = v00 + tx * (v10 - v00);
    let top = v01 + tx * (v11 - v01);
    let val = bottom + ty * (top - bottom);
    if val.is_finite() {
        val
    } else {
        0.0
    }
}

/// Steps `workspace` into `out_stereo` with **zero dynamic heap allocations**, advancing
/// the 2D Verlet PDE every 4 audio frames (`11.025`/`12.0 kHz`) with linear pickup interpolation.
pub(crate) fn render_chladni_impulse_into(
    workspace: &mut ChladniImpulseWorkspace,
    ir_params: &Wave2DParams,
    sample_rate: u32,
    pickup_left: [f64; 2],
    pickup_right: [f64; 2],
    out_stereo: &mut [f32],
) -> usize {
    const STRIDE: usize = 4;
    let frames = out_stereo.len() / 2;
    if frames == 0 {
        return 0;
    }
    let dt_sample = 1.0 / f64::from(sample_rate.max(8_000));
    let w = workspace.solver.width();
    let h = workspace.solver.height();
    let disp0 = workspace.solver.displacement();
    let mut prev_l = sample_bilinear(disp0, w, h, pickup_left) as f32;
    let mut prev_r = sample_bilinear(disp0, w, h, pickup_right) as f32;
    let mut peak = 0.0_f32;
    let mut written = 0_usize;

    while written < frames {
        let block = (frames - written).min(STRIDE);
        workspace.solver.step(ir_params, (block as f64) * dt_sample);
        let disp = workspace.solver.displacement();
        let next_l = sample_bilinear(disp, w, h, pickup_left) as f32;
        let next_r = sample_bilinear(disp, w, h, pickup_right) as f32;
        let inv_block = 1.0 / (block as f32);
        for k in 0..block {
            let t = ((k + 1) as f32) * inv_block;
            let l = prev_l + t * (next_l - prev_l);
            let r = prev_r + t * (next_r - prev_r);
            peak = peak.max(l.abs()).max(r.abs());
            let dst = (written + k) * 2;
            out_stereo[dst] = l;
            out_stereo[dst + 1] = r;
        }
        prev_l = next_l;
        prev_r = next_r;
        written += block;
    }

    normalize_and_fade_tail(out_stereo, frames, sample_rate, peak);
    frames
}

fn normalize_and_fade_tail(out_stereo: &mut [f32], frames: usize, sample_rate: u32, peak: f32) {
    let scale = if peak > 1.0e-12 && peak.is_finite() {
        (LIMITER_CEILING_LINEAR * 0.95) / peak
    } else {
        0.0
    };
    let tail_len = ((sample_rate as usize) / 200).clamp(1, (frames / 4).max(1));
    let tail_start = frames.saturating_sub(tail_len);
    for idx in 0..frames {
        let env = if idx >= tail_start && tail_len > 1 {
            let frac = ((idx - tail_start) as f32) / ((tail_len - 1) as f32);
            0.5 * (1.0 + (std::f32::consts::PI * frac).cos())
        } else {
            1.0
        };
        let gain = scale * env;
        out_stereo[idx * 2] =
            (out_stereo[idx * 2] * gain).clamp(-LIMITER_CEILING_LINEAR, LIMITER_CEILING_LINEAR);
        out_stereo[idx * 2 + 1] =
            (out_stereo[idx * 2 + 1] * gain).clamp(-LIMITER_CEILING_LINEAR, LIMITER_CEILING_LINEAR);
    }
}

impl ChladniWaveTankTool {
    pub(super) fn impulse_pde_params(&self) -> Wave2DParams {
        let (wave_speed, plate_stiffness, damping) = match self.state.medium {
            Wave2DMedium::BiharmonicPlate => (
                self.state.wave_speed,
                self.state.plate_stiffness.max(1.0e-5),
                (self.state.damping * 10.0).clamp(0.35, 8.0),
            ),
            Wave2DMedium::AcousticWave => (
                self.state.wave_speed.max(0.15) * 12.0,
                0.0,
                (self.state.damping * 40.0).clamp(0.8, 12.0),
            ),
        };
        Wave2DParams {
            medium: self.state.medium,
            boundary: self.state.boundary,
            wave_speed,
            plate_stiffness,
            damping,
        }
    }

    pub(super) fn modal_frequencies_hz(&self) -> [f64; 6] {
        let pin_lift = 1.0 + 0.04 * (self.custom_clamps.len().min(8) as f64);
        if self.state.medium == Wave2DMedium::BiharmonicPlate {
            let d = self.state.plate_stiffness.max(1.0e-5);
            let pairs = match self.state.clamp_layout {
                ClampLayout::None => [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)],
                ClampLayout::CenterPin => [(1, 2), (2, 2), (2, 3), (1, 4), (2, 4), (3, 4)],
                ClampLayout::QuadSymmetry | ClampLayout::EdgeMidpoints => {
                    [(2, 2), (2, 3), (3, 3), (2, 4), (3, 4), (4, 4)]
                }
            };
            pairs.map(|(m, n)| {
                let w = analytic_clamped_square_omega(m, n, CHLADNI_ACOUSTIC_LENGTH_METERS, d);
                (w * pin_lift / TAU).clamp(24.0, 6_000.0)
            })
        } else {
            let base = (self.state.wave_speed.max(0.2) * 12.0 * pin_lift
                / (2.0 * CHLADNI_ACOUSTIC_LENGTH_METERS))
                .clamp(55.0, 880.0);
            [1.0, 1.593_3, 2.135_5, 2.295_4, 2.653_1, 2.917_3]
                .map(|r| (base * r).clamp(24.0, 6_000.0))
        }
    }

    pub(super) fn live_sonification_frame(&self) -> SonificationFrame {
        let e_field = self.solver.energy_field();
        let el = sample_bilinear(e_field, GRID_SIZE, GRID_SIZE, self.state.pickup_left).max(0.0);
        let er = sample_bilinear(e_field, GRID_SIZE, GRID_SIZE, self.state.pickup_right).max(0.0);
        let total_e = el + er;
        let pan = if total_e > 1.0e-9 {
            ((er - el) / total_e).clamp(-0.85, 0.85) as f32
        } else {
            0.0
        };
        let level = (total_e.sqrt() * 2.4).clamp(0.08, 0.78) as f32;
        let freqs = self.modal_frequencies_hz();

        let mut frame = SonificationFrame::silent();
        frame.waveform = WaveformPreset::QuantumModal;
        frame.master_gain = level;
        let weights = [0.42_f32, 0.25, 0.18, 0.12];
        for (i, &w) in weights.iter().enumerate() {
            let side = if i % 2 == 0 { -0.35 } else { 0.35 };
            frame.voices[i] = VoiceSpec {
                freq_hz: freqs[i] as f32,
                gain: (level * w).clamp(0.02, 0.85),
                pan: (pan + side).clamp(-0.95, 0.95),
                ..VoiceSpec::silent()
            };
        }
        frame.active_voices = 4;
        frame.sanitize()
    }

    pub(super) fn handle_canvas_pointer(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        rect: egui::Rect,
    ) {
        if rect.width() <= 1.0 || rect.height() <= 1.0 {
            return;
        }
        if response.drag_started() || response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                self.select_canvas_drag_target(ctx, rect, pos);
            }
        }
        if response.clicked() || response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                self.apply_canvas_pointer_drag(ctx, response.clicked(), rect, pos);
            }
        }
        if response.drag_stopped() {
            self.finish_canvas_pointer_drag(ctx);
        }
    }

    fn select_canvas_drag_target(
        &mut self,
        ctx: &egui::Context,
        rect: egui::Rect,
        pos: egui::Pos2,
    ) {
        let shift = ctx.input(|i| i.modifiers.shift);
        let to_scr = |pt: [f64; 2]| {
            egui::Pos2::new(
                rect.left() + (pt[0] as f32) * rect.width(),
                rect.bottom() - (pt[1] as f32) * rect.height(),
            )
        };
        let dl = pos.distance(to_scr(self.state.pickup_left));
        let dr = pos.distance(to_scr(self.state.pickup_right));
        self.active_drag = if !shift && dl <= 14.0 && dl <= dr {
            CanvasDragTarget::PickupLeft
        } else if !shift && dr <= 14.0 {
            CanvasDragTarget::PickupRight
        } else {
            CanvasDragTarget::Driver
        };
    }

    fn apply_canvas_pointer_drag(
        &mut self,
        ctx: &egui::Context,
        clicked: bool,
        rect: egui::Rect,
        pos: egui::Pos2,
    ) {
        let nx = (((pos.x - rect.left()) / rect.width()) as f64).clamp(0.02, 0.98);
        let ny = ((1.0 - (pos.y - rect.top()) / rect.height()) as f64).clamp(0.02, 0.98);
        if ctx.input(|i| i.modifiers.shift) && clicked && self.custom_clamps.len() < 16 {
            self.custom_clamps.push([nx, ny]);
            self.rebuild_masks();
        } else {
            match self.active_drag {
                CanvasDragTarget::PickupLeft => self.state.pickup_left = [nx, ny],
                CanvasDragTarget::PickupRight => self.state.pickup_right = [nx, ny],
                CanvasDragTarget::Driver | CanvasDragTarget::None => {
                    self.state.driver_x = nx;
                    self.state.driver_y = ny;
                    if clicked {
                        self.solver.inject_gaussian_pulse(nx, ny, 0.04, 0.65);
                    }
                }
            }
        }
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    fn finish_canvas_pointer_drag(&mut self, ctx: &egui::Context) {
        match self.active_drag {
            CanvasDragTarget::PickupLeft => crate::accessibility::announce_live_status(
                ctx,
                format!(
                    "Left pickup moved to ({:.2}, {:.2})",
                    self.state.pickup_left[0], self.state.pickup_left[1]
                ),
            ),
            CanvasDragTarget::PickupRight => crate::accessibility::announce_live_status(
                ctx,
                format!(
                    "Right pickup moved to ({:.2}, {:.2})",
                    self.state.pickup_right[0], self.state.pickup_right[1]
                ),
            ),
            _ => {}
        }
        self.active_drag = CanvasDragTarget::None;
    }

    pub(super) fn export_chladni_wav(
        &self,
        sample_rate: u32,
        duration_secs: f32,
    ) -> Result<Vec<u8>, String> {
        let cfg = crate::audio::OfflineRenderConfig {
            sample_rate,
            duration_secs,
            master_volume: 0.85,
        }
        .sanitized();
        let total_frames = cfg.total_stereo_frames();
        let mut ws = ChladniImpulseWorkspace::new(IR_GRID_SIZE, CHLADNI_ACOUSTIC_LENGTH_METERS);
        ws.prepare(&self.state, &self.custom_clamps);
        let ir_params = self.impulse_pde_params();
        let mut stereo = vec![0.0_f32; total_frames * 2];
        render_chladni_impulse_into(
            &mut ws,
            &ir_params,
            cfg.sample_rate,
            self.state.pickup_left,
            self.state.pickup_right,
            &mut stereo,
        );
        Ok(encode_wav_stereo_pcm16(cfg.sample_rate, &stereo))
    }

    pub(super) fn export_chladni_midi(&self, duration_secs: f32) -> Result<Vec<u8>, String> {
        let freqs = self.modal_frequencies_hz();
        let decay = self.impulse_pde_params().damping * 0.5;
        let title = format!("Chladni {}", self.state.preset.label());
        Ok(crate::audio::render_frames_midi(
            &title,
            duration_secs,
            move |_step, t_sec| {
                let mut frame = SonificationFrame::silent();
                frame.waveform = WaveformPreset::QuantumModal;
                frame.master_gain = 0.78;
                let base_weights = [0.75_f64, 0.52, 0.38, 0.28];
                for (i, &w) in base_weights.iter().enumerate() {
                    let mode_decay = (-decay * (1.0 + 0.35 * (i as f64)) * t_sec).exp();
                    frame.voices[i] = VoiceSpec {
                        freq_hz: freqs[i] as f32,
                        gain: (w * mode_decay) as f32,
                        pan: if i % 2 == 0 { -0.4 } else { 0.4 },
                        ..VoiceSpec::silent()
                    };
                }
                frame.active_voices = 4;
                frame.sanitize()
            },
        ))
    }

    pub(super) fn export_chladni_scala(&self) -> Result<(String, String), String> {
        let freqs = self.modal_frequencies_hz();
        let f0 = freqs[0].max(1.0);
        let mut ratios = Vec::with_capacity(6);
        for &f in &freqs[1..] {
            ratios.push(f / f0);
        }
        let desc = format!("Chladni {} modal tuning", self.state.preset.label());
        let scale = ScalaScale::from_modal_ratios("chladni_modal", &desc, &ratios);
        let scl = scale.to_scl_string();
        let kbm = scale.to_kbm_string(60, f0.clamp(55.0, 880.0));
        Ok((scl, kbm))
    }
}
