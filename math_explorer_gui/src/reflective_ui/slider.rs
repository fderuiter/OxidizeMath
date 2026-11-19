#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::{announce_status, AccessibleHoverText};
use eframe::egui;
use std::ops::RangeInclusive;

/// Persistent animated slider state per widget ID.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimatedSliderState {
    /// Whether parameter animation is currently playing.
    pub playing: bool,
    /// Sweep direction (+1.0 for forward, -1.0 for reverse).
    pub direction: f64,
}

impl Default for AnimatedSliderState {
    fn default() -> Self {
        Self {
            playing: false,
            direction: 1.0,
        }
    }
}

/// An interactive parameter slider coupled with play/pause animation controls.
pub struct AnimatedSlider<'a> {
    value: &'a mut f64,
    min: f64,
    max: f64,
    step: f64,
    label: &'a str,
    id_salt: Option<egui::Id>,
    speed: Option<f64>,
}

impl<'a> AnimatedSlider<'a> {
    /// Creates an `AnimatedSlider` from a value reference and range bounds.
    pub fn new(value: &'a mut f64, range: RangeInclusive<f64>) -> Self {
        let min = (*range.start()).min(*range.end());
        let max = (*range.start()).max(*range.end());
        Self {
            value,
            min,
            max,
            step: 0.0,
            label: "",
            id_salt: None,
            speed: None,
        }
    }

    /// Sets the label text for the slider.
    pub fn text(mut self, label: &'a str) -> Self {
        self.label = label;
        self
    }

    /// Sets the ID salt used for storing animation state in egui memory.
    pub fn id_salt(mut self, salt: impl std::hash::Hash) -> Self {
        self.id_salt = Some(egui::Id::new(salt));
        self
    }

    /// Sets step size for slider.
    #[cfg(test)]
    pub fn step_by(mut self, step: f64) -> Self {
        self.step = step;
        self
    }

    /// Sets custom animation sweep speed per second.
    #[cfg(test)]
    pub fn speed(mut self, speed: f64) -> Self {
        self.speed = Some(speed);
        self
    }

    fn advance_animation(&mut self, ui: &mut egui::Ui, state: &mut AnimatedSliderState) -> bool {
        let dt = ui.input(|i| i.stable_dt as f64).min(0.1);
        let range = (self.max - self.min).abs();
        let speed = self
            .speed
            .unwrap_or_else(|| if range > 0.0 { range / 5.0 } else { 1.0 });

        if state.direction == 0.0 {
            state.direction = 1.0;
        }

        if range > 0.0 && dt > 0.0 {
            *self.value += speed * dt * state.direction;
            if *self.value >= self.max {
                *self.value = self.max;
                state.direction = -1.0;
            } else if *self.value <= self.min {
                *self.value = self.min;
                state.direction = 1.0;
            }
            *self.value = self.value.clamp(self.min, self.max);
            ui.ctx().request_repaint();
            true
        } else {
            false
        }
    }
}

impl<'a> egui::Widget for AnimatedSlider<'a> {
    #[allow(clippy::too_many_lines)]
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        let id = match self.id_salt {
            Some(raw_id) => {
                let scope_key = ui.ctx().data(|d| {
                    let pane = d.get_temp::<usize>(egui::Id::new("CURRENT_RENDERING_PANE"));
                    let fw = d.get_temp::<u64>(egui::Id::new("SCOPED_ANIMATED_SLIDER_OWNER"));
                    (pane, fw)
                });
                if scope_key.0.is_some() || scope_key.1.is_some() {
                    ui.make_persistent_id((raw_id, scope_key.0, scope_key.1))
                } else {
                    raw_id
                }
            }
            None => ui.make_persistent_id(self.label),
        };
        let mut state: AnimatedSliderState =
            ui.ctx().data_mut(|d| d.get_temp(id)).unwrap_or_default();
        let was_playing = state.playing;

        let mut changed = false;

        let response = ui
            .horizontal(|ui| {
                let button_text = if state.playing { "⏸" } else { "▶" };
                let button_label = if state.playing {
                    if self.label.is_empty() {
                        "Pause animation".to_string()
                    } else {
                        format!("Pause animation for {}", self.label)
                    }
                } else if self.label.is_empty() {
                    "Play animation".to_string()
                } else {
                    format!("Play animation for {}", self.label)
                };

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
                    changed = true;
                    let (announcement, next_label) = if state.playing {
                        (
                            if self.label.is_empty() {
                                "Play animation".to_string()
                            } else {
                                format!("Play animation for {}", self.label)
                            },
                            if self.label.is_empty() {
                                "Pause animation".to_string()
                            } else {
                                format!("Pause animation for {}", self.label)
                            },
                        )
                    } else {
                        (
                            if self.label.is_empty() {
                                "Pause animation".to_string()
                            } else {
                                format!("Pause animation for {}", self.label)
                            },
                            if self.label.is_empty() {
                                "Play animation".to_string()
                            } else {
                                format!("Play animation for {}", self.label)
                            },
                        )
                    };
                    announce_status(&announcement);
                    ui.ctx().data_mut(|d| {
                        d.insert_temp(egui::Id::new("aria_live_message"), announcement);
                        d.insert_temp(egui::Id::new("aria_live_dispatched"), true);
                        d.insert_temp(
                            id.with("play_button_info"),
                            egui::WidgetInfo::selected(
                                egui::WidgetType::Button,
                                ui.is_enabled(),
                                state.playing,
                                next_label,
                            ),
                        );
                    });
                }

                let mut slider = egui::Slider::new(self.value, self.min..=self.max);
                if !self.label.is_empty() {
                    slider = slider.text(self.label);
                }
                if self.step > 0.0 {
                    slider = slider.step_by(self.step);
                }

                let slider_resp = ui.add(slider);
                if slider_resp.changed() {
                    changed = true;
                }

                if play_btn.clicked() {
                    // Manual button click takes precedence.
                } else if (slider_resp.dragged() || slider_resp.lost_focus()) && was_playing {
                    state.playing = false;
                    let announcement = if self.label.is_empty() {
                        "Pause animation".to_string()
                    } else {
                        format!("Pause animation for {}", self.label)
                    };
                    let next_label = if self.label.is_empty() {
                        "Play animation".to_string()
                    } else {
                        format!("Play animation for {}", self.label)
                    };
                    announce_status(&announcement);
                    ui.ctx().data_mut(|d| {
                        d.insert_temp(egui::Id::new("aria_live_message"), announcement);
                        d.insert_temp(egui::Id::new("aria_live_dispatched"), true);
                        d.insert_temp(
                            id.with("play_button_info"),
                            egui::WidgetInfo::selected(
                                egui::WidgetType::Button,
                                ui.is_enabled(),
                                false,
                                next_label,
                            ),
                        );
                    });
                }

                slider_resp
            })
            .response;

        if state.playing && self.advance_animation(ui, &mut state) {
            changed = true;
        }

        ui.ctx().data_mut(|d| d.insert_temp(id, state));

        let mut response = response;
        if changed {
            response.mark_changed();
        }

        response
    }
}

#[cfg(test)]
#[path = "slider_tests.rs"]
mod tests;
