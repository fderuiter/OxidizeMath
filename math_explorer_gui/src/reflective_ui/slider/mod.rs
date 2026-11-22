#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Animated parameter slider widget with unclamped soft entry, finite hard bounds,
//! GeoGebra-parity sweep repeat modes, and a keyboard-accessible `⚙` settings popover.

pub mod popover;
pub mod state;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub use state::{active_sweep_count, SweepFrameFlags};
pub use state::{
    begin_pane_sweep_scope, has_active_sweeps, has_sweep_mutation, live_active_sweep_count,
    live_sweep_mutation_count, load_slider_settings, load_slider_state, merge_pane_sweep_flags,
    reset_frame_sweep_flags, resolve_scoped_slider_id, save_slider_settings, save_slider_state,
    AnimatedSliderState, SweepMode,
};

use crate::accessibility::{announce_status, AccessibleHoverText};
use eframe::egui;
use popover::{render_settings_popover, SliderPopoverContext};
use state::record_slider_frame;
use std::ops::RangeInclusive;

const DEFAULT_HARD_MIN: f64 = -1_000_000.0;
const DEFAULT_HARD_MAX: f64 = 1_000_000.0;

/// An interactive parameter slider coupled with play/pause animation controls and a `⚙` settings popover.
pub struct AnimatedSlider<'a> {
    pub(crate) value: &'a mut f64,
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) hard_min: f64,
    pub(crate) hard_max: f64,
    pub(crate) step: f64,
    pub(crate) label: &'a str,
    pub(crate) id_salt: Option<egui::Id>,
    pub(crate) speed: Option<f64>,
    pub(crate) default_sweep_mode: SweepMode,
}

impl<'a> AnimatedSlider<'a> {
    /// Creates an `AnimatedSlider` from a value reference and initial soft `[min, max]` bounds.
    pub fn new(value: &'a mut f64, range: RangeInclusive<f64>) -> Self {
        let raw_start = sanitize_endpoint(*range.start(), -10.0);
        let raw_end = sanitize_endpoint(*range.end(), 10.0);
        let min = raw_start.min(raw_end);
        let max = raw_start.max(raw_end);
        let hard_min = DEFAULT_HARD_MIN.min(min);
        let hard_max = DEFAULT_HARD_MAX.max(max);
        Self {
            value,
            min,
            max,
            hard_min,
            hard_max,
            step: 0.0,
            label: "",
            id_salt: None,
            speed: None,
            default_sweep_mode: SweepMode::Oscillate,
        }
    }

    /// Sets the label text for the slider.
    pub fn text(mut self, label: &'a str) -> Self {
        self.label = label;
        self
    }

    /// Sets the ID salt used for storing animation state and soft bounds in `egui` memory.
    pub fn id_salt(mut self, salt: impl std::hash::Hash) -> Self {
        self.id_salt = Some(egui::Id::new(salt));
        self
    }

    /// Sets the default increment step size (`0.0` for continuous).
    pub fn step_by(mut self, step: f64) -> Self {
        self.step = if step.is_finite() && step > 0.0 {
            step
        } else {
            0.0
        };
        self
    }

    /// Sets custom animation sweep speed in parameter units per second.
    pub fn speed(mut self, speed: f64) -> Self {
        if speed.is_finite() && speed > 0.0 {
            self.speed = Some(speed);
        }
        self
    }

    /// Sets the default animation repeat mode (`Oscillate`, `Increasing`, `Decreasing`, or `Once`).
    pub fn sweep_mode(mut self, mode: SweepMode) -> Self {
        self.default_sweep_mode = mode;
        self
    }

    /// Configures finite safety bounds enforced on manual numeric entry and custom soft bounds.
    pub fn hard_bounds(mut self, bounds: RangeInclusive<f64>) -> Self {
        let a = sanitize_endpoint(*bounds.start(), DEFAULT_HARD_MIN);
        let b = sanitize_endpoint(*bounds.end(), DEFAULT_HARD_MAX);
        self.hard_min = a.min(b);
        self.hard_max = a.max(b);
        self.min = self.min.clamp(self.hard_min, self.hard_max);
        self.max = self.max.clamp(self.hard_min, self.hard_max);
        if self.min > self.max {
            std::mem::swap(&mut self.min, &mut self.max);
        }
        self
    }

    fn advance_animation(
        &mut self,
        ui: &mut egui::Ui,
        state: &mut AnimatedSliderState,
        eff_min: f64,
        eff_max: f64,
        speed: f64,
    ) -> bool {
        let dt = f64::from(ui.input(|i| i.stable_dt)).min(0.1);
        let range = (eff_max - eff_min).abs();
        if range <= 0.0 || dt <= 0.0 || !speed.is_finite() || speed <= 0.0 {
            return false;
        }

        apply_sweep_step(self.value, state, eff_min, eff_max, speed * dt);
        *self.value = self.value.clamp(self.hard_min, self.hard_max);
        ui.ctx().request_repaint();
        true
    }
}

fn sanitize_endpoint(v: f64, fallback: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

fn apply_sweep_step(
    value: &mut f64,
    state: &mut AnimatedSliderState,
    eff_min: f64,
    eff_max: f64,
    delta: f64,
) {
    match state.sweep_mode {
        SweepMode::Oscillate => {
            if state.direction == 0.0 {
                state.direction = 1.0;
            }
            *value += delta * state.direction;
            if *value >= eff_max {
                *value = eff_max;
                state.direction = -1.0;
            } else if *value <= eff_min {
                *value = eff_min;
                state.direction = 1.0;
            }
            *value = value.clamp(eff_min, eff_max);
        }
        SweepMode::Increasing => {
            state.direction = 1.0;
            if *value >= eff_max {
                *value = eff_min;
            } else {
                *value = (*value + delta).max(eff_min);
                if *value > eff_max {
                    *value = eff_min;
                }
            }
        }
        SweepMode::Decreasing => {
            state.direction = -1.0;
            if *value <= eff_min {
                *value = eff_max;
            } else {
                *value = (*value - delta).min(eff_max);
                if *value < eff_min {
                    *value = eff_max;
                }
            }
        }
        SweepMode::Once => {
            state.direction = 1.0;
            *value = (*value + delta).max(eff_min);
            if *value >= eff_max {
                *value = eff_max;
                state.playing = false;
            }
        }
    }
}

fn format_play_label(label: &str, playing: bool) -> String {
    match (playing, label.is_empty()) {
        (true, true) => "Pause animation".to_string(),
        (true, false) => format!("Pause animation for {label}"),
        (false, true) => "Play animation".to_string(),
        (false, false) => format!("Play animation for {label}"),
    }
}

fn dispatch_play_status(ui: &egui::Ui, id: egui::Id, label: &str, now_playing: bool) {
    let announcement = format_play_label(label, !now_playing);
    let next_label = format_play_label(label, now_playing);
    announce_status(&announcement);
    ui.ctx().data_mut(|d| {
        d.insert_temp(egui::Id::new("aria_live_message"), announcement);
        d.insert_temp(egui::Id::new("aria_live_dispatched"), true);
        d.insert_temp(
            id.with("play_button_info"),
            egui::WidgetInfo::selected(
                egui::WidgetType::Button,
                ui.is_enabled(),
                now_playing,
                next_label,
            ),
        );
    });
}

fn render_play_button(
    ui: &mut egui::Ui,
    id: egui::Id,
    label: &str,
    state: &mut AnimatedSliderState,
    value: &mut f64,
    eff_min: f64,
    eff_max: f64,
) -> bool {
    let button_text = if state.playing { "⏸" } else { "▶" };
    let button_label = format_play_label(label, state.playing);
    let play_info = egui::WidgetInfo::selected(
        egui::WidgetType::Button,
        ui.is_enabled(),
        state.playing,
        &button_label,
    );
    let play_btn_resp = ui.button(button_text);
    play_btn_resp.widget_info({
        let play_info = play_info.clone();
        move || play_info.clone()
    });
    ui.ctx().data_mut(|d| {
        d.insert_temp(id.with("play_button_info"), play_info);
    });
    let play_btn = play_btn_resp.accessible_hover_text(&button_label);

    if play_btn.clicked() {
        state.playing = !state.playing;
        if state.playing && matches!(state.sweep_mode, SweepMode::Once) && *value >= eff_max {
            *value = eff_min;
        }
        dispatch_play_status(ui, id, label, state.playing);
        return true;
    }
    false
}

fn render_core_slider(
    ui: &mut egui::Ui,
    value: &mut f64,
    ctx: &SliderPopoverContext<'_>,
    settings: &state::AnimatedSliderSettings,
) -> egui::Response {
    let (eff_min, eff_max) = ctx.resolved_bounds(settings);
    let step = ctx.resolved_step(settings);
    let prev_finite = if value.is_finite() {
        value.clamp(ctx.hard_min, ctx.hard_max)
    } else {
        eff_min.clamp(ctx.hard_min, ctx.hard_max)
    };
    let incoming_clamped = !value.is_finite() || (*value - prev_finite).abs() > f64::EPSILON;
    *value = prev_finite;

    let avail = ui.available_width();
    if avail.is_finite() && avail < 160.0 {
        let default_w = ui.spacing().slider_width;
        if default_w > 38.0 {
            ui.spacing_mut().slider_width = (avail - 52.0).clamp(38.0, default_w);
        }
    }

    let mut slider = egui::Slider::new(value, eff_min..=eff_max)
        .clamping(egui::SliderClamping::Never)
        .custom_parser(|s| s.trim().parse::<f64>().ok().filter(|v| v.is_finite()));
    if !ctx.label.is_empty() {
        slider = slider.text(ctx.label);
    }
    if step > 0.0 {
        slider = slider.step_by(step);
    }

    let mut resp = ui.add(slider);
    if incoming_clamped {
        resp.mark_changed();
    }
    if !value.is_finite() {
        *value = prev_finite;
    } else {
        let clamped = value.clamp(ctx.hard_min, ctx.hard_max);
        if (*value - clamped).abs() > f64::EPSILON {
            *value = clamped;
            resp.mark_changed();
        }
    }
    resp
}

impl<'a> AnimatedSlider<'a> {
    fn popover_context(&self, id: egui::Id) -> SliderPopoverContext<'a> {
        SliderPopoverContext {
            id,
            label: self.label,
            default_min: self.min,
            default_max: self.max,
            hard_min: self.hard_min,
            hard_max: self.hard_max,
            default_step: self.step,
            default_speed: self.speed,
            default_mode: self.default_sweep_mode,
        }
    }
}

impl<'a> egui::Widget for AnimatedSlider<'a> {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        let id = resolve_scoped_slider_id(ui, self.id_salt, self.label);
        let mut state = load_slider_state(ui.ctx(), id, self.default_sweep_mode);
        let mut settings = load_slider_settings(ui.ctx(), id);
        let was_playing = state.playing;
        let mut changed = false;
        let popover_ctx = self.popover_context(id);

        let mut response = ui
            .horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = ui.spacing().item_spacing.x.min(4.0);
                let (init_min, init_max) = popover_ctx.resolved_bounds(&settings);
                let play_clicked = render_play_button(
                    ui, id, self.label, &mut state, self.value, init_min, init_max,
                );
                changed |= play_clicked;
                let _ = render_settings_popover(ui, &popover_ctx, &mut state, &mut settings);
                let slider_resp = render_core_slider(ui, self.value, &popover_ctx, &settings);
                changed |= slider_resp.changed();
                if !play_clicked
                    && (slider_resp.dragged() || slider_resp.lost_focus())
                    && was_playing
                {
                    state.playing = false;
                    dispatch_play_status(ui, id, self.label, false);
                }
                slider_resp
            })
            .inner;

        let (eff_min, eff_max) = popover_ctx.resolved_bounds(&settings);
        let eff_speed = popover_ctx.resolved_speed(&settings, eff_min, eff_max);
        let animated =
            state.playing && self.advance_animation(ui, &mut state, eff_min, eff_max, eff_speed);
        if changed || animated {
            response.mark_changed();
        }
        record_slider_frame(ui.ctx(), state.playing, animated);
        save_slider_state(ui.ctx(), id, state);
        save_slider_settings(ui.ctx(), id, settings);
        response
    }
}
