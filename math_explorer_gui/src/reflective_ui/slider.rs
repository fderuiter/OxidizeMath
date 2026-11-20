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

/// Per-frame animated slider sweep counters stored in `egui::Context` temporary memory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SweepFrameFlags {
    /// Cumulative egui pass number when these flags were recorded.
    pub pass_nr: Option<u64>,
    /// Number of `AnimatedSlider` widgets actively playing during the frame.
    pub active_sweep_count: usize,
    /// Number of `AnimatedSlider` widgets that mutated bound parameters during the frame.
    pub mutation_count: u64,
}

const ROOT_SWEEP_FLAGS_KEY: &str = "ANIMATED_SLIDER_SWEEP_FLAGS";
const LIVE_SWEEP_FLAGS_KEY: &str = "ANIMATED_SLIDER_LIVE_SWEEP_FLAGS";
const PANE_SWEEP_FLAGS_KEY: &str = "ANIMATED_SLIDER_PANE_SWEEP_FLAGS";

/// Resets root and live frame sweep flags at the start of an application frame.
pub fn reset_frame_sweep_flags(ctx: &egui::Context) {
    let flags = SweepFrameFlags {
        pass_nr: Some(ctx.cumulative_pass_nr()),
        active_sweep_count: 0,
        mutation_count: 0,
    };
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(ROOT_SWEEP_FLAGS_KEY), flags);
        d.insert_temp(egui::Id::new(LIVE_SWEEP_FLAGS_KEY), flags);
    });
}

/// Starts a per-pane sweep accumulation scope prior to rendering `pane_idx`.
pub fn begin_pane_sweep_scope(ctx: &egui::Context, pane_idx: usize) {
    let flags = SweepFrameFlags {
        pass_nr: Some(ctx.cumulative_pass_nr()),
        active_sweep_count: 0,
        mutation_count: 0,
    };
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new((PANE_SWEEP_FLAGS_KEY, pane_idx)), flags);
    });
}

/// Merges per-pane sweep flags recorded while rendering `pane_idx` into the root `egui::Context`.
pub fn merge_pane_sweep_flags(ctx: &egui::Context, pane_idx: usize) {
    let pass_nr = ctx.cumulative_pass_nr();
    ctx.data_mut(|d| {
        let Some(pane_flags) =
            d.remove_temp::<SweepFrameFlags>(egui::Id::new((PANE_SWEEP_FLAGS_KEY, pane_idx)))
        else {
            return;
        };
        let root_id = egui::Id::new(ROOT_SWEEP_FLAGS_KEY);
        let mut root = d.get_temp::<SweepFrameFlags>(root_id).unwrap_or_default();
        if root.pass_nr != Some(pass_nr) {
            root = SweepFrameFlags {
                pass_nr: Some(pass_nr),
                active_sweep_count: 0,
                mutation_count: 0,
            };
        }
        root.active_sweep_count = root
            .active_sweep_count
            .saturating_add(pane_flags.active_sweep_count);
        root.mutation_count = root
            .mutation_count
            .saturating_add(pane_flags.mutation_count);
        d.insert_temp(root_id, root);
    });
}

fn read_valid_flags(ctx: &egui::Context, id: egui::Id) -> SweepFrameFlags {
    let current_pass = ctx.cumulative_pass_nr();
    ctx.data(|d| d.get_temp::<SweepFrameFlags>(id))
        .filter(|f| {
            f.pass_nr
                .is_some_and(|p| current_pass <= p.saturating_add(1))
        })
        .unwrap_or_default()
}

/// Returns the merged count of actively playing `AnimatedSlider` sweeps in the root context.
pub fn active_sweep_count(ctx: &egui::Context) -> usize {
    read_valid_flags(ctx, egui::Id::new(ROOT_SWEEP_FLAGS_KEY)).active_sweep_count
}

/// Returns `true` when at least one `AnimatedSlider` sweep is active on the current frame.
pub fn has_active_sweeps(ctx: &egui::Context) -> bool {
    active_sweep_count(ctx) > 0 || live_active_sweep_count(ctx) > 0
}

/// Returns the merged count of `AnimatedSlider` sweep mutations in the root context.
pub fn frame_sweep_mutation_count(ctx: &egui::Context) -> u64 {
    read_valid_flags(ctx, egui::Id::new(ROOT_SWEEP_FLAGS_KEY)).mutation_count
}

/// Returns `true` when at least one `AnimatedSlider` sweep mutated a bound value this frame.
pub fn has_sweep_mutation(ctx: &egui::Context) -> bool {
    frame_sweep_mutation_count(ctx) > 0
}

/// Returns the live intra-pass count of actively playing `AnimatedSlider` sweeps.
pub fn live_active_sweep_count(ctx: &egui::Context) -> usize {
    read_valid_flags(ctx, egui::Id::new(LIVE_SWEEP_FLAGS_KEY)).active_sweep_count
}

/// Returns the live intra-pass count of `AnimatedSlider` parameter mutations.
pub fn live_sweep_mutation_count(ctx: &egui::Context) -> u64 {
    read_valid_flags(ctx, egui::Id::new(LIVE_SWEEP_FLAGS_KEY)).mutation_count
}

fn record_slider_frame(ctx: &egui::Context, playing: bool, mutated: bool) {
    let pass_nr = ctx.cumulative_pass_nr();
    let add_active = usize::from(playing);
    let add_mut = u64::from(mutated);
    ctx.data_mut(|d| {
        let live_id = egui::Id::new(LIVE_SWEEP_FLAGS_KEY);
        let mut live = d.get_temp::<SweepFrameFlags>(live_id).unwrap_or_default();
        if live.pass_nr != Some(pass_nr) {
            live = SweepFrameFlags {
                pass_nr: Some(pass_nr),
                active_sweep_count: 0,
                mutation_count: 0,
            };
        }
        live.active_sweep_count = live.active_sweep_count.saturating_add(add_active);
        live.mutation_count = live.mutation_count.saturating_add(add_mut);
        d.insert_temp(live_id, live);

        let target_id = d
            .get_temp::<usize>(egui::Id::new("CURRENT_RENDERING_PANE"))
            .map(|p| egui::Id::new((PANE_SWEEP_FLAGS_KEY, p)))
            .filter(|pane_id| d.get_temp::<SweepFrameFlags>(*pane_id).is_some())
            .unwrap_or_else(|| egui::Id::new(ROOT_SWEEP_FLAGS_KEY));
        let mut flags = d.get_temp::<SweepFrameFlags>(target_id).unwrap_or_default();
        if flags.pass_nr != Some(pass_nr) {
            flags = SweepFrameFlags {
                pass_nr: Some(pass_nr),
                active_sweep_count: 0,
                mutation_count: 0,
            };
        }
        flags.active_sweep_count = flags.active_sweep_count.saturating_add(add_active);
        flags.mutation_count = flags.mutation_count.saturating_add(add_mut);
        d.insert_temp(target_id, flags);
    });
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

        let swept_this_frame = state.playing && self.advance_animation(ui, &mut state);
        if swept_this_frame {
            changed = true;
        }

        record_slider_frame(ui.ctx(), state.playing, swept_this_frame);
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
