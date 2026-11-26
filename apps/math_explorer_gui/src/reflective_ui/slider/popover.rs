#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{AnimatedSliderSettings, AnimatedSliderState, SweepMode};
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::framework::{bounded_drag_value, enforce_ordered_drag_range};
use eframe::egui;

const POPOVER_WIDTH: f32 = 172.0;
const MIN_SOFT_SPAN: f64 = 1e-6;

/// Immutable defaults and hard domain bounds passed into the `⚙` slider configuration popover.
#[derive(Clone, Copy)]
pub(super) struct SliderPopoverContext<'a> {
    pub id: egui::Id,
    pub label: &'a str,
    pub default_min: f64,
    pub default_max: f64,
    pub hard_min: f64,
    pub hard_max: f64,
    pub default_step: f64,
    pub default_speed: Option<f64>,
    pub default_mode: SweepMode,
}

impl<'a> SliderPopoverContext<'a> {
    fn subject(&self) -> String {
        if self.label.is_empty() {
            "slider".to_string()
        } else {
            self.label.to_string()
        }
    }

    pub(super) fn resolved_bounds(&self, settings: &AnimatedSliderSettings) -> (f64, f64) {
        let mut lo = settings
            .min
            .unwrap_or(self.default_min)
            .clamp(self.hard_min, self.hard_max);
        let mut hi = settings
            .max
            .unwrap_or(self.default_max)
            .clamp(self.hard_min, self.hard_max);
        if lo > hi {
            std::mem::swap(&mut lo, &mut hi);
        }
        (lo, hi)
    }

    pub(super) fn resolved_step(&self, settings: &AnimatedSliderSettings) -> f64 {
        settings.step.unwrap_or(self.default_step).max(0.0)
    }

    pub(super) fn resolved_speed(
        &self,
        settings: &AnimatedSliderSettings,
        eff_min: f64,
        eff_max: f64,
    ) -> f64 {
        let span = (eff_max - eff_min).abs();
        let fallback = self
            .default_speed
            .unwrap_or_else(|| if span > 0.0 { span / 5.0 } else { 1.0 });
        settings.speed.unwrap_or(fallback).max(1e-4)
    }
}

/// Renders the keyboard-accessible `⚙` button and range/speed/mode popover.
pub(super) fn render_settings_popover(
    ui: &mut egui::Ui,
    ctx_cfg: &SliderPopoverContext<'_>,
    state: &mut AnimatedSliderState,
    settings: &mut AnimatedSliderSettings,
) -> bool {
    let popup_id = ctx_cfg.id.with("settings_popup");
    let was_open = egui::Popup::is_id_open(ui.ctx(), popup_id);
    let gear_label = if ctx_cfg.label.is_empty() {
        "Configure slider range and animation".to_string()
    } else {
        format!("Configure range and animation for {}", ctx_cfg.label)
    };

    let gear_info = egui::WidgetInfo::selected(
        egui::WidgetType::Button,
        ui.is_enabled(),
        was_open || settings.has_overrides(),
        &gear_label,
    );
    let gear_resp = ui.add(egui::Button::new("⚙").small());
    gear_resp.widget_info({
        let gear_info = gear_info.clone();
        move || gear_info.clone()
    });
    ui.ctx().data_mut(|d| {
        d.insert_temp(ctx_cfg.id.with("gear_button_info"), gear_info);
    });
    let gear_resp = gear_resp.accessible_hover_text(&gear_label);

    if gear_resp.clicked() {
        let msg = if was_open {
            format!("Closed slider settings for {}", ctx_cfg.subject())
        } else {
            format!("Opened slider settings for {}", ctx_cfg.subject())
        };
        announce_live_status(ui.ctx(), msg);
    }

    let mut edited = false;
    egui::Popup::from_toggle_button_response(&gear_resp)
        .id(popup_id)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .width(POPOVER_WIDTH)
        .show(|ui| {
            ui.set_max_width(POPOVER_WIDTH);
            edited |= render_popover_contents(ui, ctx_cfg, state, settings);
        });
    edited
}

fn render_popover_contents(
    ui: &mut egui::Ui,
    ctx_cfg: &SliderPopoverContext<'_>,
    state: &mut AnimatedSliderState,
    settings: &mut AnimatedSliderSettings,
) -> bool {
    let mut edited = false;
    ui.label(egui::RichText::new("Slider Settings").strong().small());

    edited |= render_bounds_rows(ui, ctx_cfg, settings);
    edited |= render_step_and_speed_rows(ui, ctx_cfg, settings);
    edited |= render_sweep_mode_row(ui, ctx_cfg, state);

    ui.separator();
    ui.horizontal(|ui| {
        if ui
            .small_button("Reset defaults")
            .accessible_hover_text("Reset slider bounds, step, speed, and sweep mode")
            .clicked()
        {
            *settings = AnimatedSliderSettings::default();
            state.sweep_mode = ctx_cfg.default_mode;
            state.direction = if matches!(ctx_cfg.default_mode, SweepMode::Decreasing) {
                -1.0
            } else {
                1.0
            };
            announce_live_status(
                ui.ctx(),
                format!("Reset slider settings for {}", ctx_cfg.subject()),
            );
            edited = true;
        }
    });
    edited
}

fn render_bounds_rows(
    ui: &mut egui::Ui,
    ctx_cfg: &SliderPopoverContext<'_>,
    settings: &mut AnimatedSliderSettings,
) -> bool {
    let (mut min_val, mut max_val) = ctx_cfg.resolved_bounds(settings);
    let hard_span = (ctx_cfg.hard_max - ctx_cfg.hard_min).abs();
    let drag_speed = (hard_span / 200.0).clamp(0.001, 10.0);
    let min_span = MIN_SOFT_SPAN.min(hard_span * 0.5);

    let mut min_changed = false;
    let mut max_changed = false;
    ui.horizontal(|ui| {
        ui.label("Min:");
        min_changed = ui
            .add(bounded_drag_value(
                &mut min_val,
                ctx_cfg.hard_min..=ctx_cfg.hard_max,
                drag_speed,
            ))
            .accessible_hover_text("Minimum soft slider bound")
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Max:");
        max_changed = ui
            .add(bounded_drag_value(
                &mut max_val,
                ctx_cfg.hard_min..=ctx_cfg.hard_max,
                drag_speed,
            ))
            .accessible_hover_text("Maximum soft slider bound")
            .changed();
    });

    if min_changed || max_changed {
        if hard_span > 0.0 {
            enforce_ordered_drag_range(
                &mut min_val,
                &mut max_val,
                ctx_cfg.hard_min,
                ctx_cfg.hard_max,
                min_span,
                min_changed,
                max_changed,
            );
        }
        settings.min = Some(min_val);
        settings.max = Some(max_val);
        announce_live_status(
            ui.ctx(),
            format!(
                "Updated {} bounds to [{min_val:.3}, {max_val:.3}]",
                ctx_cfg.subject()
            ),
        );
        return true;
    }
    false
}

fn render_step_and_speed_rows(
    ui: &mut egui::Ui,
    ctx_cfg: &SliderPopoverContext<'_>,
    settings: &mut AnimatedSliderSettings,
) -> bool {
    let (eff_min, eff_max) = ctx_cfg.resolved_bounds(settings);
    let span = (eff_max - eff_min).abs().max(1e-3);
    let mut step_val = ctx_cfg.resolved_step(settings);
    let mut speed_val = ctx_cfg.resolved_speed(settings, eff_min, eff_max);
    let mut edited = false;

    ui.horizontal(|ui| {
        ui.label("Step:");
        if ui
            .add(bounded_drag_value(
                &mut step_val,
                0.0..=span,
                (span / 100.0).max(0.001),
            ))
            .accessible_hover_text("Slider increment step (0 for continuous)")
            .changed()
        {
            settings.step = Some(step_val.max(0.0));
            announce_live_status(
                ui.ctx(),
                format!("Updated {} step to {step_val:.3}", ctx_cfg.subject()),
            );
            edited = true;
        }
    });

    let max_speed = (span * 20.0).max(100.0);
    ui.horizontal(|ui| {
        ui.label("Speed:");
        if ui
            .add(bounded_drag_value(
                &mut speed_val,
                1e-4..=max_speed,
                (span / 20.0).max(0.01),
            ))
            .accessible_hover_text("Animation sweep speed in units per second")
            .changed()
        {
            settings.speed = Some(speed_val.max(1e-4));
            announce_live_status(
                ui.ctx(),
                format!("Updated {} speed to {speed_val:.3}", ctx_cfg.subject()),
            );
            edited = true;
        }
    });
    edited
}

fn render_sweep_mode_row(
    ui: &mut egui::Ui,
    ctx_cfg: &SliderPopoverContext<'_>,
    state: &mut AnimatedSliderState,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Repeat:");
        egui::ComboBox::from_id_salt(ctx_cfg.id.with("sweep_mode_combo"))
            .width(104.0)
            .selected_text(state.sweep_mode.label())
            .show_ui(ui, |ui| {
                for mode in SweepMode::ALL {
                    let selected = state.sweep_mode == mode;
                    if ui.selectable_label(selected, mode.label()).clicked() && !selected {
                        state.sweep_mode = mode;
                        state.direction = if matches!(mode, SweepMode::Decreasing) {
                            -1.0
                        } else {
                            1.0
                        };
                        announce_live_status(
                            ui.ctx(),
                            format!(
                                "Sweep mode set to {} for {}",
                                mode.label(),
                                ctx_cfg.subject()
                            ),
                        );
                        changed = true;
                    }
                }
            });
    });
    changed
}
