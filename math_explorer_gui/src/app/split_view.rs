#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::MathExplorerApp;
use eframe::egui;

/// Layout mode for the main simulation viewport container.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitLayoutMode {
    /// Single viewport rendering one active module.
    #[default]
    Single,
    /// Dual viewport split horizontally (left and right panes).
    Horizontal,
    /// Dual viewport split vertically (top and bottom panes).
    Vertical,
}

impl SplitLayoutMode {
    /// Returns the human-readable display label for the layout mode.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Single => "Single View",
            Self::Horizontal => "Horizontal Split",
            Self::Vertical => "Vertical Split",
        }
    }
}

/// Geometry representation of split pane viewports and divider bar.
struct SplitGeometry {
    rect_primary: egui::Rect,
    rect_divider: egui::Rect,
    rect_secondary: egui::Rect,
}

/// Persistent state manager for dual-pane split view layout composition.
#[derive(Default)]
pub struct SplitViewState {
    primary_ctx: egui::Context,
    secondary_ctx: egui::Context,
    is_dragging_divider: bool,
}

impl SplitViewState {
    /// Creates a new split view state instance with isolated contexts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes the rectangular viewports and divider bar bounding box.
    fn compute_geometry(
        available: egui::Rect,
        mode: SplitLayoutMode,
        ratio: f32,
    ) -> Option<SplitGeometry> {
        let clamped = ratio.clamp(0.20, 0.80);
        let thickness = 6.0;

        match mode {
            SplitLayoutMode::Horizontal => {
                let div_x = available.min.x + available.width() * clamped;
                Some(SplitGeometry {
                    rect_primary: egui::Rect::from_min_max(
                        available.min,
                        egui::pos2(div_x - thickness * 0.5, available.max.y),
                    ),
                    rect_divider: egui::Rect::from_min_max(
                        egui::pos2(div_x - thickness * 0.5, available.min.y),
                        egui::pos2(div_x + thickness * 0.5, available.max.y),
                    ),
                    rect_secondary: egui::Rect::from_min_max(
                        egui::pos2(div_x + thickness * 0.5, available.min.y),
                        available.max,
                    ),
                })
            }
            SplitLayoutMode::Vertical => {
                let div_y = available.min.y + available.height() * clamped;
                Some(SplitGeometry {
                    rect_primary: egui::Rect::from_min_max(
                        available.min,
                        egui::pos2(available.max.x, div_y - thickness * 0.5),
                    ),
                    rect_divider: egui::Rect::from_min_max(
                        egui::pos2(available.min.x, div_y - thickness * 0.5),
                        egui::pos2(available.max.x, div_y + thickness * 0.5),
                    ),
                    rect_secondary: egui::Rect::from_min_max(
                        egui::pos2(available.min.x, div_y + thickness * 0.5),
                        available.max,
                    ),
                })
            }
            SplitLayoutMode::Single => None,
        }
    }

    /// Handles pointer interaction and paints the styled divider bar.
    fn update_divider(
        &mut self,
        app: &mut MathExplorerApp,
        ctx: &egui::Context,
        available: egui::Rect,
        divider_rect: egui::Rect,
    ) {
        let hover_pos = ctx.input(|i| i.pointer.hover_pos());
        let is_hovered = hover_pos.is_some_and(|pos| divider_rect.contains(pos));
        let is_down = ctx.input(|i| i.pointer.primary_down());

        if is_hovered && ctx.input(|i| i.pointer.any_pressed()) {
            self.is_dragging_divider = true;
        }
        if !is_down {
            self.is_dragging_divider = false;
        }

        if self.is_dragging_divider {
            let delta = ctx.input(|i| i.pointer.delta());
            let ratio_delta = match app.split_layout_mode {
                SplitLayoutMode::Horizontal => delta.x / available.width(),
                SplitLayoutMode::Vertical => delta.y / available.height(),
                SplitLayoutMode::Single => 0.0,
            };
            app.split_ratio = (app.split_ratio + ratio_delta).clamp(0.20, 0.80);
        }

        if is_hovered || self.is_dragging_divider {
            ctx.set_cursor_icon(match app.split_layout_mode {
                SplitLayoutMode::Horizontal => egui::CursorIcon::ResizeColumn,
                SplitLayoutMode::Vertical => egui::CursorIcon::ResizeRow,
                SplitLayoutMode::Single => egui::CursorIcon::Default,
            });
        }

        let painter = ctx.layer_painter(egui::LayerId::background());
        let color = if self.is_dragging_divider {
            egui::Color32::from_rgb(90, 150, 255)
        } else if is_hovered {
            egui::Color32::from_rgb(140, 140, 150)
        } else {
            egui::Color32::from_gray(50)
        };
        painter.rect_filled(divider_rect, 0.0, color);

        let center = divider_rect.center();
        let handle_color = if self.is_dragging_divider || is_hovered {
            egui::Color32::WHITE
        } else {
            egui::Color32::from_gray(130)
        };
        let (p1, p2) = if app.split_layout_mode == SplitLayoutMode::Horizontal {
            (
                center - egui::vec2(0.0, 16.0),
                center + egui::vec2(0.0, 16.0),
            )
        } else {
            (
                center - egui::vec2(16.0, 0.0),
                center + egui::vec2(16.0, 0.0),
            )
        };
        painter.line_segment([p1, p2], egui::Stroke::new(2.0_f32, handle_color));
    }

    /// Dispatches input events to the primary and secondary viewport sub-contexts.
    fn dispatch_inputs(
        &self,
        ctx: &egui::Context,
        geom: &SplitGeometry,
    ) -> (egui::RawInput, egui::RawInput) {
        let input = ctx.input(|i| i.clone());
        let mut r1 = egui::RawInput {
            screen_rect: Some(geom.rect_primary),
            time: Some(input.time),
            predicted_dt: input.predicted_dt,
            modifiers: input.modifiers,
            ..Default::default()
        };
        let mut r2 = egui::RawInput {
            screen_rect: Some(geom.rect_secondary),
            time: Some(input.time),
            predicted_dt: input.predicted_dt,
            modifiers: input.modifiers,
            ..Default::default()
        };

        if !self.is_dragging_divider {
            for event in &input.events {
                Self::route_event(event, &input, geom, &mut r1, &mut r2);
            }
        }

        (r1, r2)
    }

    /// Routes an individual UI event to the appropriate pane input buffer.
    fn route_event(
        event: &egui::Event,
        input: &egui::InputState,
        geom: &SplitGeometry,
        r1: &mut egui::RawInput,
        r2: &mut egui::RawInput,
    ) {
        match event {
            egui::Event::PointerMoved(pos) | egui::Event::PointerButton { pos, .. } => {
                if geom.rect_primary.contains(*pos) {
                    r1.events.push(event.clone());
                } else if geom.rect_secondary.contains(*pos) {
                    r2.events.push(event.clone());
                }
            }
            egui::Event::MouseWheel { .. } | egui::Event::Zoom(_) => {
                if let Some(pos) = input.pointer.latest_pos() {
                    if geom.rect_primary.contains(pos) {
                        r1.events.push(event.clone());
                    } else if geom.rect_secondary.contains(pos) {
                        r2.events.push(event.clone());
                    }
                }
            }
            egui::Event::Key { .. } | egui::Event::Text(_) | egui::Event::Paste(_) => {
                if let Some(pos) = input.pointer.latest_pos() {
                    if geom.rect_secondary.contains(pos) {
                        r2.events.push(event.clone());
                    } else {
                        r1.events.push(event.clone());
                    }
                } else {
                    r1.events.push(event.clone());
                }
            }
            _ => {
                r1.events.push(event.clone());
                r2.events.push(event.clone());
            }
        }
    }

    /// Executes both tab module viewports and returns their output passes.
    fn execute_panes(
        &mut self,
        app: &mut MathExplorerApp,
        frame: &mut eframe::Frame,
        r1: egui::RawInput,
        r2: egui::RawInput,
    ) -> (egui::FullOutput, egui::FullOutput) {
        let count = app.tabs.len();
        if count == 0 {
            return (egui::FullOutput::default(), egui::FullOutput::default());
        }
        if app.selected_tab >= count {
            app.selected_tab = 0;
        }
        if app.secondary_tab >= count || (count > 1 && app.secondary_tab == app.selected_tab) {
            app.secondary_tab = (app.selected_tab + 1) % count;
        }

        let p_idx = app.selected_tab;
        let s_idx = app.secondary_tab;

        if p_idx < s_idx {
            let (first, second) = app.tabs.split_at_mut(s_idx);
            let out1 = self.primary_ctx.run(r1, |c| first[p_idx].show(c, frame));
            let out2 = self.secondary_ctx.run(r2, |c| second[0].show(c, frame));
            (out1, out2)
        } else if p_idx > s_idx {
            let (first, second) = app.tabs.split_at_mut(p_idx);
            let out1 = self.primary_ctx.run(r1, |c| second[0].show(c, frame));
            let out2 = self.secondary_ctx.run(r2, |c| first[s_idx].show(c, frame));
            (out1, out2)
        } else {
            let out1 = self.primary_ctx.run(r1, |c| app.tabs[p_idx].show(c, frame));
            (out1, egui::FullOutput::default())
        }
    }

    /// Renders dual simulation viewports according to the active split layout configuration.
    pub fn render_split(
        &mut self,
        app: &mut MathExplorerApp,
        ctx: &egui::Context,
        frame: &mut eframe::Frame,
    ) {
        let available = ctx.available_rect();
        if available.width() < 20.0 || available.height() < 20.0 {
            return;
        }

        let Some(geom) = Self::compute_geometry(available, app.split_layout_mode, app.split_ratio)
        else {
            return;
        };

        self.update_divider(app, ctx, available, geom.rect_divider);

        self.primary_ctx.set_style((*ctx.style()).clone());
        self.secondary_ctx.set_style((*ctx.style()).clone());

        let (r1, r2) = self.dispatch_inputs(ctx, &geom);
        let (out1, out2) = self.execute_panes(app, frame, r1, r2);

        if !out1.textures_delta.is_empty() || !out2.textures_delta.is_empty() {
            let main_tex = ctx.tex_manager();
            let mut writer = main_tex.write();
            for (id, delta) in out1.textures_delta.set {
                writer.set(id, delta);
            }
            for (id, delta) in out2.textures_delta.set {
                writer.set(id, delta);
            }
        }

        let painter = ctx.layer_painter(egui::LayerId::background());

        let mut painter1 = painter.clone();
        painter1.set_clip_rect(geom.rect_primary);
        painter1.extend(out1.shapes.into_iter().map(|cs| cs.shape));

        let mut painter2 = painter.clone();
        painter2.set_clip_rect(geom.rect_secondary);
        painter2.extend(out2.shapes.into_iter().map(|cs| cs.shape));

        egui::CentralPanel::default().show(ctx, |_ui| {});

        if self.primary_ctx.has_requested_repaint() || self.secondary_ctx.has_requested_repaint() {
            ctx.request_repaint();
        }
    }
}
