#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::controller::{AtlasTeleportEvent, ParameterAtlasController};
use super::{AtlasAxisPair, RegimeCartographer};
use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
    AccessibleHoverText,
};
use eframe::egui;
use egui_plot::{HLine, Line, LineStyle, Plot, PlotImage, PlotPoint, PlotPoints, Points, VLine};
use std::sync::Arc;

impl ParameterAtlasController {
    /// Renders the Atlas Mode toggle button and returns `true` if visibility changed.
    pub fn show_toggle_button(&mut self, ui: &mut egui::Ui) -> bool {
        let label = if self.is_visible() {
            "🗺 Hide 2D Parameter Atlas"
        } else {
            "🗺 2D Parameter Atlas"
        };
        let resp = ui.selectable_label(self.is_visible(), label);
        let resp = resp.accessible_hover_text(
            "Toggle progressive 2D parameter regime atlas with hover previews and click-to-teleport.",
        );
        if resp.clicked() {
            let next = !self.is_visible();
            self.set_visible(next);
            let msg = if next {
                "2D Parameter Atlas opened"
            } else {
                "2D Parameter Atlas closed"
            };
            crate::accessibility::announce_live_status(ui.ctx(), msg);
            return true;
        }
        false
    }

    fn draw_atlas_toolbar(&mut self, ui: &mut egui::Ui, high_contrast: bool) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal_wrapped(|ui| {
            ui.strong("2D Regime Atlas");
            if self.axis_pair_options().len() > 1 {
                let current_idx = self.selected_pair_index();
                let (xk, yk) = self.axis_keys();
                egui::ComboBox::from_id_salt("atlas_axis_pair_selector")
                    .selected_text(format!("{xk} vs {yk}"))
                    .show_ui(ui, |ui| {
                        let options = self.axis_pair_options().to_vec();
                        for (idx, (ox, oy)) in options.into_iter().enumerate() {
                            if ui
                                .selectable_label(idx == current_idx, format!("{ox} vs {oy}"))
                                .clicked()
                            {
                                self.select_axis_pair_index(idx);
                            }
                        }
                    });
            }
            let status_txt = if self.is_converged() {
                "1x Converged".to_string()
            } else {
                format!("Refining {}x…", self.current_stride())
            };
            let status_col = if self.is_converged() {
                palette.success
            } else {
                palette.info
            };
            ui.colored_label(status_col, status_txt);
            ui.label("• Click or press Enter to teleport");
        });

        if let Some(cartographer) = &self.active_cartographer {
            let entries = cartographer.legend_entries(high_contrast);
            if !entries.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    for (label, swatch) in entries {
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                        ui.painter().rect_filled(rect, 2.0, swatch);
                        ui.label(label);
                    }
                });
            }
        }
    }

    fn step_keyboard_cursor(
        &mut self,
        ui: &egui::Ui,
        axes: &AtlasAxisPair,
        active_x: f64,
        active_y: f64,
    ) -> Option<bool> {
        let mut cur = self
            .keyboard_cursor
            .unwrap_or([axes.x.clamp(active_x), axes.y.clamp(active_y)]);
        let dx = (axes.x.span() * 0.025).max(axes.x.constraint.step.max(1e-6));
        let dy = (axes.y.span() * 0.025).max(axes.y.constraint.step.max(1e-6));
        let mut moved = false;
        let mut activate = false;

        ui.input(|i| {
            for (key, axis_idx, delta) in [
                (egui::Key::ArrowLeft, 0usize, -dx),
                (egui::Key::ArrowRight, 0usize, dx),
                (egui::Key::ArrowDown, 1usize, -dy),
                (egui::Key::ArrowUp, 1usize, dy),
            ] {
                if i.key_pressed(key) {
                    cur[axis_idx] = if axis_idx == 0 {
                        axes.x.clamp(cur[0] + delta)
                    } else {
                        axes.y.clamp(cur[1] + delta)
                    };
                    moved = true;
                }
            }
            if i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Space) {
                activate = true;
            }
        });

        if moved || self.keyboard_cursor.is_none() {
            self.keyboard_cursor = Some(cur);
        }
        (moved || activate).then_some(activate)
    }

    fn draw_atlas_plot_items(
        plot_ui: &mut egui_plot::PlotUi<'_>,
        axes: &AtlasAxisPair,
        palette: &crate::app::SemanticPalette,
        tex_id: Option<egui::TextureId>,
        contour: Option<(&'static str, Vec<[f64; 2]>)>,
        clamped: [f64; 2],
        kb_cursor: Option<[f64; 2]>,
    ) -> Option<[f64; 2]> {
        if let Some(tid) = tex_id {
            plot_ui.image(PlotImage::new(
                "atlas_raster",
                tid,
                PlotPoint::new(axes.x.midpoint(), axes.y.midpoint()),
                [axes.x.span() as f32, axes.y.span() as f32],
            ));
        }
        if let Some((contour_label, pts)) = contour {
            plot_ui.line(
                Line::new(contour_label, PlotPoints::from(pts))
                    .color(palette.warning)
                    .width(2.0_f32)
                    .style(LineStyle::Dashed { length: 6.0 }),
            );
        }
        plot_ui.hline(
            HLine::new("Active Y", clamped[1])
                .color(palette.accent)
                .width(1.5_f32),
        );
        plot_ui.vline(
            VLine::new("Active X", clamped[0])
                .color(palette.accent)
                .width(1.5_f32),
        );
        plot_ui.points(
            Points::new("Active Parameters", vec![clamped])
                .radius(5.0_f32)
                .color(palette.accent),
        );
        if let Some(kb) = kb_cursor {
            plot_ui.points(
                Points::new("Cursor", vec![kb])
                    .radius(4.0_f32)
                    .color(palette.info),
            );
        }
        plot_ui
            .response()
            .hovered()
            .then(|| plot_ui.pointer_coordinate())
            .flatten()
            .map(|pt| [axes.x.clamp(pt.x), axes.y.clamp(pt.y)])
    }

    fn render_atlas_plot(
        &self,
        ui: &mut egui::Ui,
        plot_id_salt: &str,
        cartographer: &dyn RegimeCartographer,
        clamped: [f64; 2],
    ) -> (egui::Response, Option<[f64; 2]>) {
        let axes = cartographer.axes();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let tex_id = self.texture.as_ref().map(egui::TextureHandle::id);
        let contour = cartographer.overlay_contour();
        let kb_cursor = self.keyboard_cursor;
        let mut hover_coord: Option<[f64; 2]> = None;

        let plot_resp = Plot::new(format!("{plot_id_salt}_parameter_atlas"))
            .x_axis_label(axes.x.label)
            .y_axis_label(axes.y.label)
            .view_aspect(1.25)
            .height(240.0)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .include_x(axes.x.constraint.min)
            .include_x(axes.x.constraint.max)
            .include_y(axes.y.constraint.min)
            .include_y(axes.y.constraint.max)
            .show(ui, |plot_ui| {
                hover_coord = Self::draw_atlas_plot_items(
                    plot_ui, &axes, &palette, tex_id, contour, clamped, kb_cursor,
                );
            });

        (plot_resp.response, hover_coord)
    }

    fn handle_atlas_interaction(
        &mut self,
        ui: &mut egui::Ui,
        raw_resp: egui::Response,
        cartographer: &dyn RegimeCartographer,
        clamped: [f64; 2],
        hover_coord: Option<[f64; 2]>,
    ) -> Option<AtlasTeleportEvent> {
        let ctx = ui.ctx().clone();
        let axes = cartographer.axes();
        let focus_resp = raw_resp.interact(egui::Sense::click());
        let active_cell = cartographer.evaluate_cell(clamped[0], clamped[1]);
        let a11y_value = format!(
            "{} = {:.3}, {} = {:.3}, {} ({})",
            axes.x.label,
            clamped[0],
            axes.y.label,
            clamped[1],
            cartographer.regime_label(active_cell.regime),
            cartographer.primary_metric_label()
        );
        register_canvas_accessibility(
            ui,
            &focus_resp,
            egui::accesskit::Role::Canvas,
            "2D Parameter Regime Atlas",
            Some(&a11y_value),
        );
        lock_canvas_arrow_keys(ui, &focus_resp);
        paint_canvas_focus_ring(ui, &focus_resp);

        let mut teleport = None;
        if focus_resp.clicked() {
            focus_resp.request_focus();
            if let Some([hx, hy]) = hover_coord {
                teleport = self.teleport_to(Some(&ctx), hx, hy);
            }
        } else if focus_resp.has_focus() {
            if let Some(activate) = self.step_keyboard_cursor(ui, &axes, clamped[0], clamped[1]) {
                if let Some([kx, ky]) = self.keyboard_cursor {
                    if activate {
                        teleport = self.teleport_to(Some(&ctx), kx, ky);
                    } else if let Some(cell) = self.lookup_or_evaluate_cell(kx, ky) {
                        let tip = cartographer.format_hover_tooltip(kx, ky, cell);
                        crate::accessibility::announce_live_status(&ctx, tip.replace('\n', " — "));
                    }
                }
            }
        }

        let preview = hover_coord.or_else(|| {
            focus_resp
                .has_focus()
                .then_some(self.keyboard_cursor)
                .flatten()
        });
        if let Some([px, py]) = preview {
            if let Some(cell) = self.lookup_or_evaluate_cell(px, py) {
                let tooltip = cartographer.format_hover_tooltip(px, py, cell);
                ui.small(tooltip.replace('\n', "  |  "));
                let _ = focus_resp.on_hover_text(tooltip);
            }
        }
        teleport
    }

    /// Renders the interactive 2D regime atlas panel when visible and returns any click/keyboard teleport.
    pub fn show_panel<F>(
        &mut self,
        ui: &mut egui::Ui,
        plot_id_salt: &str,
        active_x: f64,
        active_y: f64,
        context_sig: u64,
        build_cartographer: F,
    ) -> Option<AtlasTeleportEvent>
    where
        F: FnOnce((&'static str, &'static str)) -> Arc<dyn RegimeCartographer>,
    {
        if !self.is_visible() {
            return None;
        }
        let ctx = ui.ctx().clone();
        let high_contrast = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        self.sync_and_poll(&ctx, context_sig, high_contrast, build_cartographer);

        let cartographer = self.active_cartographer.clone()?;
        let axes = cartographer.axes();
        let clamped = [axes.x.clamp(active_x), axes.y.clamp(active_y)];

        self.draw_atlas_toolbar(ui, high_contrast);
        let (raw_resp, hover_coord) =
            self.render_atlas_plot(ui, plot_id_salt, cartographer.as_ref(), clamped);
        self.handle_atlas_interaction(ui, raw_resp, cartographer.as_ref(), clamped, hover_coord)
    }
}
