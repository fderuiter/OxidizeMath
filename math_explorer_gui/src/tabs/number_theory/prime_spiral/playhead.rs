//! Interactive quadratic ray ($4n^2 + bn + c$) and spiral sweep playhead UI,
//! keyboard navigation, pointer steering, and polyrhythmic sonification for [`PrimeSpiralWidget`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui::{self, Pos2, Rect, Sense, Stroke, Vec2};
use math_commons::math_kernel::types::flatten_2d_index;

use super::PrimeSpiralWidget;
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleHoverText, AccessibleTheoryHover,
};
use crate::app::SemanticPalette;
use crate::audio::frame::MAX_VOICES;
use crate::audio::mappings::{ulam_spiral_frame, UlamPolySample};
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use crate::web_io::PlatformInstant;

/// Minimum quadratic linear coefficient $b$ in $4n^2 + bn + c$.
pub const MIN_RAY_B: i64 = -40;
/// Maximum quadratic linear coefficient $b$ in $4n^2 + bn + c$.
pub const MAX_RAY_B: i64 = 40;
/// Default Euler quadratic coefficient $b = 2$ ($4n^2 + 2n + 41$).
pub const DEFAULT_RAY_B: i64 = 2;

/// Minimum quadratic constant offset $c$ in $4n^2 + bn + c$.
pub const MIN_RAY_C: i64 = 1;
/// Maximum quadratic constant offset $c$ in $4n^2 + bn + c$.
pub const MAX_RAY_C: i64 = 200;
/// Default Euler prime-generating constant $c = 41$.
pub const DEFAULT_RAY_C: i64 = 41;

/// Minimum Ulam playhead tempo in beats per minute.
pub const MIN_TEMPO_BPM: f64 = 30.0;
/// Maximum Ulam playhead tempo in beats per minute.
pub const MAX_TEMPO_BPM: f64 = 360.0;
/// Default Ulam playhead tempo in beats per minute.
pub const DEFAULT_TEMPO_BPM: f64 = 120.0;

/// Maximum cursor step index along a quadratic ray or spiral sweep.
pub const MAX_ULAM_CURSOR_STEP: usize = 512;

const PARALLEL_RAY_OFFSETS: [i64; MAX_VOICES] = [0, 2, -2, 4, -4, 6, -6, 8];

/// Active trajectory mode for the Ulam prime spiral playhead.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum UlamPlayheadMode {
    /// Steps $n = 0, 1, 2, \dots$ along parallel Euler quadratics $4n^2 + bn + (c + 2k)$.
    #[default]
    QuadraticRay,
    /// Steps $m = 1, 2, 3, \dots$ sequentially along the rectangular Ulam spiral lattice.
    SpiralSweep,
}

/// Evaluates $f(n) = 4n^2 + bn + c$, clamping below `1` to `1`.
#[inline]
#[must_use]
pub fn eval_ulam_quadratic(n: usize, b: i64, c: i64) -> u64 {
    let n_i = n as i64;
    let val = 4_i64
        .saturating_mul(n_i)
        .saturating_mul(n_i)
        .saturating_add(b.saturating_mul(n_i))
        .saturating_add(c);
    val.max(1) as u64
}

impl PrimeSpiralWidget {
    /// Evaluates up to 8 stack-resident `UlamPolySample` voices at the current playhead cursor.
    #[must_use]
    pub fn sample_playhead_voices(&self) -> [UlamPolySample; MAX_VOICES] {
        let mut out = [UlamPolySample::default(); MAX_VOICES];
        match self.playhead_mode {
            UlamPlayheadMode::QuadraticRay => {
                for (idx, &offset) in PARALLEL_RAY_OFFSETS.iter().enumerate() {
                    let c_k = (self.ray_c + offset).max(if self.ray_c >= 4 { 4 } else { 1 });
                    let val = eval_ulam_quadratic(self.cursor_step, self.ray_b, c_k);
                    let is_p = self.is_prime_fast(val);
                    let gap = self.quadratic_prime_gap(self.cursor_step, self.ray_b, c_k);
                    out[idx] = UlamPolySample {
                        value: val,
                        is_prime: is_p,
                        prime_gap: gap,
                    };
                }
            }
            UlamPlayheadMode::SpiralSweep => {
                let base_m = (self.cursor_step as u64) + 1;
                for (idx, slot) in out.iter_mut().enumerate() {
                    let m = base_m + (idx as u64) * 2;
                    *slot = UlamPolySample {
                        value: m,
                        is_prime: self.is_prime_fast(m),
                        prime_gap: self.spiral_prime_gap(m),
                    };
                }
            }
        }
        out
    }

    /// Synthesizes the current zero-allocation `SonificationFrame` for `PrimeSpiralWidget`.
    #[must_use]
    pub fn current_sonification_frame(&self) -> crate::audio::SonificationFrame {
        let samples = self.sample_playhead_voices();
        ulam_spiral_frame(&samples, self.tempo_bpm as f32, self.pulse_gate)
    }

    fn quadratic_prime_gap(&self, step: usize, b: i64, c: i64) -> u32 {
        let max_back = step.min(64);
        for back in 1..=max_back {
            let prev = eval_ulam_quadratic(step - back, b, c);
            if self.is_prime_fast(prev) {
                return back as u32;
            }
        }
        (max_back.max(1)) as u32
    }

    fn spiral_prime_gap(&self, m: u64) -> u32 {
        let max_back = m.saturating_sub(1).min(128);
        for back in 1..=max_back {
            if self.is_prime_fast(m - back) {
                return back as u32;
            }
        }
        (max_back.max(1)) as u32
    }

    pub(crate) fn step_playhead_delta(&mut self, delta: isize) {
        let max_step = self.max_visible_step();
        let next = (self.cursor_step as isize + delta).rem_euclid((max_step + 1) as isize) as usize;
        self.cursor_step = next.min(MAX_ULAM_CURSOR_STEP);
        self.pulse_gate = true;
        self.beat_accum_sec = 0.0;
    }

    pub(crate) fn shift_secondary_axis(&mut self, delta: i64) {
        match self.playhead_mode {
            UlamPlayheadMode::QuadraticRay => {
                self.ray_c = (self.ray_c + delta).clamp(MIN_RAY_C, MAX_RAY_C);
            }
            UlamPlayheadMode::SpiralSweep => {
                let stride = (self.grid_size.max(10) / 5) as isize;
                self.step_playhead_delta((delta as isize) * stride);
            }
        }
        self.pulse_gate = true;
    }

    fn max_visible_step(&self) -> usize {
        let max_n = (self.grid_size * self.grid_size).max(16);
        match self.playhead_mode {
            UlamPlayheadMode::QuadraticRay => {
                (((max_n as f64) * 0.25).sqrt().floor() as usize).clamp(4, MAX_ULAM_CURSOR_STEP)
            }
            UlamPlayheadMode::SpiralSweep => max_n.saturating_sub(1).min(MAX_ULAM_CURSOR_STEP),
        }
    }

    pub(super) fn advance_playhead_clock(&mut self, ctx: &egui::Context) {
        let now = PlatformInstant::now();
        let dt = self.last_instant.map_or(0.016_f64, |prev| {
            prev.elapsed().as_secs_f64().clamp(0.0, 0.1)
        });
        self.last_instant = Some(now);
        if !self.is_playing {
            return;
        }
        let beat_period = (60.0 / self.tempo_bpm.clamp(MIN_TEMPO_BPM, MAX_TEMPO_BPM)).max(0.05);
        self.beat_accum_sec += dt;
        if self.beat_accum_sec >= beat_period {
            self.beat_accum_sec %= beat_period;
            let max_step = self.max_visible_step();
            self.cursor_step = (self.cursor_step + 1) % (max_step + 1);
            self.pulse_gate = true;
        } else {
            self.pulse_gate = self.beat_accum_sec < beat_period * 0.55;
        }
        ctx.request_repaint();
    }

    pub(super) fn render_playhead_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Ulam Spiral Playhead");
        ui.separator();
        if let Some(action) =
            PlaybackTransport::in_thread("ulam_spiral_transport", self.name(), self.is_playing)
                .show(ui)
        {
            match action {
                PlaybackAction::Play => {
                    self.is_playing = true;
                    self.pulse_gate = true;
                    self.beat_accum_sec = 0.0;
                    self.last_instant = Some(PlatformInstant::now());
                }
                PlaybackAction::Pause => self.is_playing = false,
                PlaybackAction::Step => self.step_playhead_delta(1),
                PlaybackAction::Reset => {
                    self.is_playing = false;
                    self.cursor_step = 0;
                    self.ray_b = DEFAULT_RAY_B;
                    self.ray_c = DEFAULT_RAY_C;
                    self.pulse_gate = true;
                }
            }
        }
        ui.separator();
        self.render_mode_and_polynomial_sliders(ui);
        ui.separator();
        let frame = self.current_sonification_frame();
        crate::audio::render_sonification_panel(ui, "prime_spiral", &frame, false);
        ui.separator();
        self.render_cursor_readout(ui);
    }

    fn render_mode_and_polynomial_sliders(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.playhead_mode,
                UlamPlayheadMode::QuadraticRay,
                "Quadratic Ray 4n²+bn+c",
            );
            ui.selectable_value(
                &mut self.playhead_mode,
                UlamPlayheadMode::SpiralSweep,
                "Spiral Sweep",
            );
        });
        if ui
            .add(
                egui::Slider::new(
                    &mut self.grid_size,
                    10..=math_commons::registry::MAX_GRID_SIZE,
                )
                .text("Grid Size"),
            )
            .changed()
        {
            self.regenerate_texture(ui.ctx());
        }
        ui.add(
            egui::Slider::new(&mut self.tempo_bpm, MIN_TEMPO_BPM..=MAX_TEMPO_BPM)
                .text("Tempo (BPM)"),
        );
        if self.playhead_mode == UlamPlayheadMode::QuadraticRay {
            ui.add(egui::Slider::new(&mut self.ray_b, MIN_RAY_B..=MAX_RAY_B).text("Ray b"));
            ui.add(egui::Slider::new(&mut self.ray_c, MIN_RAY_C..=MAX_RAY_C).text("Ray c"));
            self.render_euler_presets(ui);
        }
        let max_step = self.max_visible_step();
        self.cursor_step = self.cursor_step.min(max_step);
        ui.add(egui::Slider::new(&mut self.cursor_step, 0..=max_step).text("Cursor Step"));
    }

    fn render_euler_presets(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            for (label, b, c) in [
                ("Euler 4n²+2n+41", 2_i64, 41_i64),
                ("4n²-2n+41", -2_i64, 41_i64),
                ("Composite 4n²+4n+4", 4_i64, 4_i64),
            ] {
                if ui
                    .small_button(label)
                    .accessible_hover_text("Select quadratic diagonal preset on Ulam spiral")
                    .clicked()
                {
                    self.ray_b = b;
                    self.ray_c = c;
                    self.cursor_step = 0;
                    self.pulse_gate = true;
                }
            }
        });
    }

    fn render_cursor_readout(&self, ui: &mut egui::Ui) {
        let voices = self.sample_playhead_voices();
        let primary = voices[0];
        let prime_count = voices.iter().filter(|v| v.is_prime).count();
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let tag = if primary.is_prime {
            "PRIME"
        } else {
            "composite"
        };
        let col = if primary.is_prime {
            palette.success
        } else {
            palette.muted_text
        };
        ui.colored_label(
            col,
            format!(
                "Cursor m = {} ({}) | gap = {} | parallel primes = {}/{}",
                primary.value, tag, primary.prime_gap, prime_count, MAX_VOICES
            ),
        );
    }

    pub(super) fn render_spiral_canvas(&mut self, ui: &mut egui::Ui) {
        let avail = ui.available_size().max(Vec2::splat(200.0));
        let side = avail.x.min(avail.y).max(180.0);
        let (raw_resp, painter) = ui.allocate_painter(Vec2::splat(side), Sense::click_and_drag());
        let primary = self.sample_playhead_voices()[0];
        let alt_label = format!(
            "Ulam Prime Spiral visualization ({0}x{0} grid)",
            self.grid_size
        );
        let value_desc = format!(
            "Step {}, value {}, prime={}, tempo {:.0} BPM",
            self.cursor_step, primary.value, primary.is_prime, self.tempo_bpm
        );
        register_canvas_accessibility(
            ui,
            &raw_resp,
            egui::accesskit::Role::GraphicsDocument,
            &alt_label,
            Some(&value_desc),
        );
        let response = raw_resp.accessible_theory_hover(self);
        lock_canvas_arrow_keys(ui, &response);

        if let Some(tex) = &self.texture {
            painter.image(
                tex.id(),
                response.rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        let palette = SemanticPalette::from_visuals(ui.visuals());
        self.paint_playhead_overlay(&painter, response.rect, &palette);
        painter.rect_stroke(
            response.rect,
            0.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Inside,
        );
        paint_canvas_focus_ring(ui, &response);
        self.handle_canvas_interactions(ui, &response);
    }

    fn paint_playhead_overlay(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        palette: &SemanticPalette,
    ) {
        let size = self.grid_size.max(2) as f32;
        let to_screen = |m: u64| -> Option<Pos2> {
            let &(px, py) = self.coord_lookup.get(m as usize)?;
            if m == 0 || m > (self.grid_size * self.grid_size) as u64 {
                return None;
            }
            Some(Pos2::new(
                rect.left() + ((f32::from(px) + 0.5) / size) * rect.width(),
                rect.top() + ((f32::from(py) + 0.5) / size) * rect.height(),
            ))
        };

        if self.playhead_mode == UlamPlayheadMode::QuadraticRay {
            let max_step = self.max_visible_step();
            let mut prev: Option<Pos2> = None;
            for step in 0..=max_step {
                let m = eval_ulam_quadratic(step, self.ray_b, self.ray_c);
                if let Some(pt) = to_screen(m) {
                    if let Some(p0) = prev {
                        painter.line_segment([p0, pt], Stroke::new(1.4_f32, palette.accent));
                    }
                    prev = Some(pt);
                }
            }
        }

        for (idx, sample) in self.sample_playhead_voices().iter().enumerate() {
            if let Some(pt) = to_screen(sample.value) {
                let rad = if idx == 0 { 5.5 } else { 3.2 };
                let col = if idx == 0 {
                    palette.warning
                } else if sample.is_prime {
                    palette.success
                } else {
                    palette.info
                };
                painter.circle_stroke(pt, rad, Stroke::new(1.8_f32, col));
            }
        }
    }

    fn handle_canvas_interactions(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if response.clicked() || response.dragged() {
            response.request_focus();
            if let Some(pos) = response.interact_pointer_pos() {
                self.steer_playhead_to_pointer(pos, response.rect);
            }
        }
        if response.has_focus() {
            self.focused_canvas = Some(response.id);
            let mut moved = false;
            ui.input(|i| {
                if i.key_pressed(egui::Key::ArrowRight) {
                    self.step_playhead_delta(1);
                    moved = true;
                } else if i.key_pressed(egui::Key::ArrowLeft) {
                    self.step_playhead_delta(-1);
                    moved = true;
                } else if i.key_pressed(egui::Key::ArrowUp) {
                    self.shift_secondary_axis(1);
                    moved = true;
                } else if i.key_pressed(egui::Key::ArrowDown) {
                    self.shift_secondary_axis(-1);
                    moved = true;
                }
            });
            if moved {
                let m = self.sample_playhead_voices()[0].value;
                announce_live_status(ui.ctx(), format!("Ulam playhead at {m}"));
            }
        } else {
            self.focused_canvas = None;
        }
    }

    fn steer_playhead_to_pointer(&mut self, pos: Pos2, rect: Rect) {
        let size = self.grid_size.max(2);
        if self.spiral_grid.len() != size * size {
            return;
        }
        let ux = ((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 0.999_9);
        let uy = ((pos.y - rect.top()) / rect.height().max(1.0)).clamp(0.0, 0.999_9);
        let px = ((ux * (size as f32)) as usize).min(size - 1);
        let py = ((uy * (size as f32)) as usize).min(size - 1);
        let m = u64::from(self.spiral_grid[flatten_2d_index(px, py, size)]);
        if m == 0 {
            return;
        }
        match self.playhead_mode {
            UlamPlayheadMode::SpiralSweep => {
                self.cursor_step = (m.saturating_sub(1) as usize).min(MAX_ULAM_CURSOR_STEP);
            }
            UlamPlayheadMode::QuadraticRay => {
                let n = (((m as f64) * 0.25).sqrt().floor() as usize).min(self.max_visible_step());
                let base = 4_i64 * (n as i64) * (n as i64) + self.ray_b * (n as i64);
                self.cursor_step = n;
                self.ray_c = ((m as i64) - base).clamp(MIN_RAY_C, MAX_RAY_C);
            }
        }
        self.pulse_gate = true;
    }
}

// [cite:mmwave_radiotherapy_setup]
