#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui::{self, Align2, Color32, CornerRadius, Stroke, Vec2};
use egui_plot::{Callout, PlotPoint, PlotUi};
use serde::{Deserialize, Serialize};

/// Maximum custom callout annotations stored per tool.
pub const MAX_CALLOUTS: usize = 30;

fn default_true() -> bool {
    true
}

/// Stored annotation payload for plot coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalloutData {
    /// Unique identifier for this callout.
    pub id: usize,
    /// Target coordinates in plot space `[x, y]`.
    pub target_point: [f64; 2],
    /// Screen pixel displacement `[dx, dy]` from the target coordinate.
    pub label_offset: [f32; 2],
    /// Annotation label string.
    pub text: String,
    /// Optional semantic tag for programmatic simulation features.
    #[serde(default)]
    pub tag: Option<String>,
    /// Visibility state of the callout.
    #[serde(default = "default_true")]
    pub visible: bool,
    /// Optional custom highlight RGB color.
    #[serde(default)]
    pub color_rgb: Option<[u8; 3]>,
}

impl CalloutData {
    /// Create a new callout annotation targeting a plot point.
    pub fn new(id: usize, target_point: [f64; 2], text: impl Into<String>) -> Self {
        Self {
            id,
            target_point,
            label_offset: [20.0, -20.0],
            text: text.into(),
            tag: None,
            visible: true,
            color_rgb: None,
        }
    }

    /// Set screen pixel label offset.
    pub fn with_offset(mut self, offset: [f32; 2]) -> Self {
        self.label_offset = offset;
        self
    }

    /// Set semantic tag identifier.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }
}

/// Framework manager for custom user callouts and automated simulation annotations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalloutManager {
    callouts: Vec<CalloutData>,
    next_id: usize,
    #[serde(skip)]
    dragged_id: Option<usize>,
}

impl Default for CalloutManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CalloutManager {
    /// Instantiate an empty callout manager.
    pub fn new() -> Self {
        Self {
            callouts: Vec::new(),
            next_id: 1,
            dragged_id: None,
        }
    }

    /// Number of active callout annotations.
    pub fn len(&self) -> usize {
        self.callouts.len()
    }

    /// Returns true if no callout annotations exist.
    pub fn is_empty(&self) -> bool {
        self.callouts.is_empty()
    }

    /// Read-only slice of all registered callouts.
    pub fn callouts(&self) -> &[CalloutData] {
        &self.callouts
    }

    /// Mutable slice of all registered callouts.
    pub fn callouts_mut(&mut self) -> &mut [CalloutData] {
        &mut self.callouts
    }

    /// Add a callout annotation at the given coordinates.
    pub fn add_callout(
        &mut self,
        point: [f64; 2],
        text: impl Into<String>,
        offset: [f32; 2],
    ) -> usize {
        if self.callouts.len() >= MAX_CALLOUTS {
            self.callouts.remove(0);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.callouts.push(CalloutData {
            id,
            target_point: point,
            label_offset: offset,
            text: text.into(),
            tag: None,
            visible: true,
            color_rgb: None,
        });
        id
    }

    /// Register or update an automated domain feature callout identified by tag.
    pub fn set_auto_callout(
        &mut self,
        tag: &str,
        point: [f64; 2],
        text: impl Into<String>,
        offset: [f32; 2],
    ) {
        let text_str = text.into();
        if let Some(existing) = self
            .callouts
            .iter_mut()
            .find(|c| c.tag.as_deref() == Some(tag))
        {
            existing.target_point = point;
            existing.text = text_str;
            if self.dragged_id != Some(existing.id) {
                existing.label_offset = offset;
            }
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.callouts.push(CalloutData {
                id,
                target_point: point,
                label_offset: offset,
                text: text_str,
                tag: Some(tag.to_string()),
                visible: true,
                color_rgb: None,
            });
        }
    }

    /// Update the screen label offset for a callout by ID.
    pub fn update_offset(&mut self, id: usize, offset: [f32; 2]) -> bool {
        if let Some(callout) = self.callouts.iter_mut().find(|c| c.id == id) {
            callout.label_offset = offset;
            true
        } else {
            false
        }
    }

    /// Remove a callout annotation by ID.
    pub fn remove_callout(&mut self, id: usize) -> bool {
        if let Some(pos) = self.callouts.iter().position(|c| c.id == id) {
            self.callouts.remove(pos);
            true
        } else {
            false
        }
    }

    /// Clear all callout annotations.
    pub fn clear(&mut self) {
        self.callouts.clear();
        self.dragged_id = None;
    }

    /// Clear only automated feature annotations while preserving manual user annotations.
    pub fn clear_auto(&mut self) {
        self.callouts.retain(|c| c.tag.is_none());
    }

    /// Toggle visibility state of a callout.
    pub fn toggle_visibility(&mut self, id: usize) {
        if let Some(callout) = self.callouts.iter_mut().find(|c| c.id == id) {
            callout.visible = !callout.visible;
        }
    }

    /// Render stored callouts onto the plot canvas and process frame dragging.
    #[allow(clippy::too_many_lines)]
    pub fn render_on_plot(&mut self, plot_ui: &mut PlotUi) {
        // Drain context menu canvas creations for this plot
        for pending in egui_plot::drain_pending_callouts(plot_ui.plot_id()) {
            self.add_callout(pending.target_point, pending.text, pending.label_offset);
        }

        let ctx = plot_ui.ctx().clone();
        let pointer_down = ctx.input(|i| i.pointer.primary_down());
        let pointer_pos = ctx.input(|i| i.pointer.latest_pos());
        let drag_delta = ctx.input(|i| i.pointer.delta());

        if !pointer_down && self.dragged_id.is_some() {
            self.dragged_id = None;
            ctx.data_mut(|d| {
                d.remove_temp::<bool>(plot_ui.plot_id().with("dragging_callout"));
            });
        }

        let visuals = ctx.style().visuals.clone();
        let palette = crate::app::SemanticPalette::from_visuals(&visuals);
        let text_color = visuals.text_color();
        let frame_fill = visuals.window_fill;
        let frame_stroke_color = palette.graph_edge;
        let frame_stroke = Stroke::new(1.0_f32, frame_stroke_color);
        let leader_stroke = Stroke::new(1.0_f32, frame_stroke_color);

        for callout in &mut self.callouts {
            if !callout.visible {
                continue;
            }

            let target_point = PlotPoint::new(callout.target_point[0], callout.target_point[1]);
            let target_pos = plot_ui.screen_from_plot(target_point);
            let offset = Vec2::new(callout.label_offset[0], callout.label_offset[1]);
            let anchor_pos = target_pos + offset;

            let char_width = 7.0_f32;
            let approx_width = (callout.text.chars().count() as f32 * char_width).max(40.0) + 14.0;
            let frame_size = Vec2::new(approx_width, 22.0);
            let frame_rect = Align2::CENTER_CENTER.anchor_size(anchor_pos, frame_size);

            if let Some(pos) = pointer_pos {
                if frame_rect.contains(pos) && self.dragged_id.is_none() {
                    ctx.set_cursor_icon(egui::CursorIcon::Grab);
                    if ctx.input(|i| i.pointer.primary_pressed()) {
                        self.dragged_id = Some(callout.id);
                        ctx.data_mut(|d| {
                            d.insert_temp(plot_ui.plot_id().with("dragging_callout"), true);
                        });
                    }
                }
            }

            if self.dragged_id == Some(callout.id) {
                ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
                callout.label_offset[0] += drag_delta.x;
                callout.label_offset[1] += drag_delta.y;
                ctx.data_mut(|d| {
                    d.insert_temp(plot_ui.plot_id().with("dragging_callout"), true);
                });
            }

            let mut item = Callout::new(target_point, &callout.text)
                .offset(callout.label_offset)
                .leader_stroke(leader_stroke)
                .frame(frame_fill, frame_stroke)
                .corner_radius(CornerRadius::same(4))
                .text_color(text_color);

            if let Some(rgb) = callout.color_rgb {
                let custom_color = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                item = item
                    .leader_stroke(Stroke::new(1.5_f32, custom_color))
                    .frame_stroke(Stroke::new(1.5_f32, custom_color));
            }

            plot_ui.callout(item);
        }
    }

    /// Render sidebar overlay UI controls for managing annotations.
    #[allow(clippy::too_many_lines)]
    pub fn render_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.heading("📍 Plot Callout Annotations");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if !self.is_empty() && ui.button("Clear All").clicked() {
                            self.clear();
                            changed = true;
                        }
                    });
                });

                if self.is_empty() {
                    ui.label(
                        egui::RichText::new("Right-click on plot to add callout annotations.")
                            .small()
                            .weak(),
                    );
                } else {
                    let mut to_remove = None;
                    for callout in &mut self.callouts {
                        ui.horizontal(|ui| {
                            let eye = if callout.visible { "👁" } else { "🙈" };
                            if ui.button(eye).clicked() {
                                callout.visible = !callout.visible;
                                changed = true;
                            }
                            let label = if let Some(tag) = &callout.tag {
                                format!("{}: {}", tag, callout.text)
                            } else {
                                callout.text.clone()
                            };
                            ui.label(egui::RichText::new(label).small());
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.button("❌").clicked() {
                                        to_remove = Some(callout.id);
                                    }
                                },
                            );
                        });
                    }
                    if let Some(id) = to_remove {
                        self.remove_callout(id);
                        changed = true;
                    }
                }
            });
        });
        changed
    }
}

#[cfg(test)]
#[path = "callout_overlay_tests.rs"]
mod callout_overlay_tests;
