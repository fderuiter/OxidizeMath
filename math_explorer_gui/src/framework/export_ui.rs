use super::*;
use eframe::egui;

/// Pane toolbar width threshold below which `Export` and `Presets` collapse into overflow menus.
pub const EXPORT_BAR_OVERFLOW_BREAKPOINT: f32 = 960.0;

fn wrapped_colored_label(ui: &mut egui::Ui, color: egui::Color32, text: String) {
    ui.add(egui::Label::new(egui::RichText::new(text).color(color)).wrap());
}

pub(super) fn export_action_button(
    ui: &mut egui::Ui,
    enabled: bool,
    label: &str,
    tip: &str,
    close_on_click: bool,
) -> bool {
    let clicked = ui
        .add_enabled(enabled, egui::Button::new(label))
        .accessible_hover_text(tip)
        .clicked();
    if clicked && close_on_click {
        ui.close();
    }
    clicked
}

impl SimulationFramework {
    pub(super) fn show_export_action_bar(&mut self, ui: &mut egui::Ui, id_source: &str) {
        if self.active_tool.is_none() {
            return;
        }

        let ctx = ui.ctx().clone();
        self.poll_preset_import_and_sync(&ctx);

        let bar_id = scoped_panel_id(ui, format!("{id_source}_export_bar"));
        egui::TopBottomPanel::top(bar_id)
            .min_height(32.0)
            .show_inside(ui, |ui| {
                self.render_export_bar_row(ui, &ctx);
            });
    }

    fn render_export_bar_row(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let bar_width = ui.available_width();
        let compact = bar_width < EXPORT_BAR_OVERFLOW_BREAKPOINT;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);
            let file_slug = if compact {
                self.render_compact_export_menu(ui)
            } else {
                self.render_dataset_and_figure_buttons(ui)
            };
            self.render_action_bar_export_status(ctx, ui);
            if !compact {
                ui.separator();
            }
            self.render_starter_gallery_toggle(ctx, ui);
            self.render_theory_portal_toggle(ctx, ui);
            if compact {
                self.render_compact_preset_controls(ui, &file_slug, bar_width >= 420.0);
            } else {
                self.render_preset_action_bar_controls(ui, &file_slug);
            }
        });
    }

    fn active_tool_slug_and_caps(&self) -> (String, ExportCapabilities) {
        let (tool_name, caps) = if let Some(tool) = &self.active_tool {
            (tool.name(), tool.export_capabilities())
        } else {
            ("simulation", ExportCapabilities::none())
        };
        (tool_name.to_lowercase().replace(' ', "_"), caps)
    }

    fn render_export_action_items(
        &mut self,
        ui: &mut egui::Ui,
        caps: ExportCapabilities,
        close_on_click: bool,
    ) {
        let ds_tip = |tip| {
            if caps.dataset {
                tip
            } else {
                caps.dataset_disabled_reason
            }
        };
        if export_action_button(
            ui,
            caps.dataset,
            "Export CSV",
            ds_tip("Export dataset points to CSV file"),
            close_on_click,
        ) {
            self.queue_export(ExportFormat::Csv);
        }
        if export_action_button(
            ui,
            caps.dataset,
            "Copy CSV",
            ds_tip("Copy dataset points as CSV to clipboard"),
            close_on_click,
        ) {
            self.queue_clipboard_export();
        }
        if export_action_button(
            ui,
            caps.dataset,
            "Export JSON",
            ds_tip("Export dataset points to JSON file"),
            close_on_click,
        ) {
            self.queue_export(ExportFormat::Json);
        }
        let fig_tip = if caps.figure {
            "Configure and render high-resolution PNG figure"
        } else {
            caps.figure_disabled_reason
        };
        if export_action_button(ui, caps.figure, "Export PNG Image", fig_tip, close_on_click) {
            self.open_export_modal_for_active_tool();
        }
        self.render_3d_export_action_items(ui, caps, close_on_click);
        self.render_shader_export_items(ui, caps, close_on_click);
    }

    fn render_compact_export_menu(&mut self, ui: &mut egui::Ui) -> String {
        let (file_slug, caps) = self.active_tool_slug_and_caps();
        let resp = ui.menu_button("📊 Export ▾", |menu_ui| {
            self.render_export_action_items(menu_ui, caps, true);
        });
        resp.response
            .accessible_hover_text("Open dataset and figure export menu");
        file_slug
    }

    fn render_dataset_and_figure_buttons(&mut self, ui: &mut egui::Ui) -> String {
        ui.label(egui::RichText::new("📊 Export:").strong());
        let (file_slug, caps) = self.active_tool_slug_and_caps();
        self.render_export_action_items(ui, caps, false);
        file_slug
    }

    fn render_starter_gallery_toggle(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        if ui
            .button("🎨 Starter Gallery")
            .accessible_hover_text("Toggle starter template gallery overlay")
            .clicked()
        {
            self.show_starter_gallery = !self.show_starter_gallery;
            ctx.data_mut(|d| {
                d.insert_temp(
                    egui::Id::new("SHOW_STARTER_GALLERY"),
                    self.show_starter_gallery,
                )
            });
        }
    }

    fn render_theory_portal_toggle(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let btn = ui
            .button("📖 Theory")
            .accessible_hover_text("Toggle Theory Context Portal");
        super::overlay_sync::record_default_theory_portal_toggle(ctx, btn.id);
        if btn.clicked() {
            self.show_theory_portal = !self.show_theory_portal;
            super::overlay_sync::record_theory_portal_toggle(ctx, btn.id, self.show_theory_portal);
        }
    }

    fn render_action_bar_export_status(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        self.poll_action_bar_clipboard_status(ctx);

        let now = ctx.input(|i| i.time);
        let mut clear_status = false;
        let mut dismissed_by_user = false;

        if let Some(ref mut status) = self.export_queue.status {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            if !status.is_error {
                let exp = *status.expires_at_time.get_or_insert(now + 3.0);
                if now > 0.0 && now >= exp {
                    clear_status = true;
                } else {
                    let remaining = (exp - now).max(0.05);
                    ctx.request_repaint_after(std::time::Duration::from_secs_f64(remaining));
                    if !self.show_export_modal {
                        wrapped_colored_label(ui, palette.success, format!("✓ {}", status.message));
                        if ui
                            .small_button("✖")
                            .accessible_hover_text("Dismiss notification")
                            .clicked()
                        {
                            clear_status = true;
                            dismissed_by_user = true;
                        }
                    }
                }
            } else if !self.show_export_modal {
                let err_color = if self.export_status_severity == ExportStatusSeverity::Warning {
                    palette.warning
                } else {
                    palette.error
                };
                wrapped_colored_label(ui, err_color, format!("⚠️ {}", status.message));
                if ui
                    .small_button("✖")
                    .accessible_hover_text("Dismiss warning")
                    .clicked()
                {
                    clear_status = true;
                    dismissed_by_user = true;
                }
            }
        }

        if clear_status {
            self.export_queue.status = None;
            if dismissed_by_user {
                self.export_status_message = None;
            }
        }
    }

    fn export_active_preset_json(&mut self, file_slug: &str) {
        if let Some(tool) = &self.active_tool {
            self.preset_dialog
                .export_preset_result(tool.export_preset_json(), file_slug);
        }
    }

    fn render_reset_parameters_button(&mut self, ui: &mut egui::Ui, close_on_click: bool) {
        if ui
            .button("↻ Reset Parameters")
            .accessible_hover_text("Restore default baseline parameters")
            .clicked()
        {
            self.reset_parameters();
            crate::accessibility::announce_status("Parameters reset to baseline defaults");
            ui.ctx().request_repaint();
            if close_on_click {
                ui.close();
            }
        }
    }

    fn render_preset_toasts(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref msg) = self.preset_dialog.toast_info {
            ui.separator();
            wrapped_colored_label(ui, palette.success, format!("ℹ️ {msg}"));
            if ui
                .small_button("✖")
                .accessible_hover_text("Dismiss notification")
                .clicked()
            {
                self.preset_dialog.toast_info = None;
            }
        }
        if let Some(ref err) = self.preset_dialog.toast_error {
            ui.separator();
            wrapped_colored_label(ui, palette.error, format!("⚠️ {err}"));
            if ui
                .small_button("✖")
                .accessible_hover_text("Dismiss warning")
                .clicked()
            {
                self.preset_dialog.toast_error = None;
            }
        }
    }

    fn render_preset_json_buttons(
        &mut self,
        ui: &mut egui::Ui,
        file_slug: &str,
        close_on_click: bool,
    ) {
        if export_action_button(
            ui,
            true,
            "Import Preset JSON",
            "Import parameter preset from a JSON file",
            close_on_click,
        ) {
            self.preset_dialog.trigger_import();
        }
        if export_action_button(
            ui,
            true,
            "Export Preset JSON",
            "Export current parameter preset to a JSON file",
            close_on_click,
        ) {
            self.export_active_preset_json(file_slug);
        }
        if export_action_button(
            ui,
            true,
            "🔗 Copy Share Link",
            "Copy a shareable simulation link to the clipboard",
            close_on_click,
        ) {
            let ctx = ui.ctx().clone();
            self.copy_active_share_link(&ctx);
        }
    }

    fn render_compact_preset_controls(
        &mut self,
        ui: &mut egui::Ui,
        file_slug: &str,
        inline_reset: bool,
    ) {
        let resp = ui.menu_button("⚙ Presets ▾", |menu_ui| {
            self.render_preset_json_buttons(menu_ui, file_slug, true);
            if !inline_reset {
                menu_ui.separator();
                self.render_reset_parameters_button(menu_ui, true);
            }
        });
        resp.response
            .accessible_hover_text("Open parameter presets menu");
        if inline_reset {
            self.render_reset_parameters_button(ui, false);
        }
        self.render_preset_toasts(ui);
    }

    fn render_preset_action_bar_controls(&mut self, ui: &mut egui::Ui, file_slug: &str) {
        ui.separator();
        ui.label(egui::RichText::new("⚙ Presets:").strong());
        self.render_preset_json_buttons(ui, file_slug, false);
        self.render_reset_parameters_button(ui, false);
        self.render_preset_toasts(ui);
    }
}
