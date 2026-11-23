//! Theory Context Portal side-panel rendering, dismissal, and citation links for [`super::SimulationFramework`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use eframe::egui;

pub(crate) fn format_theory_context_key(key: &str) -> String {
    match key.trim() {
        "g_na" => "Sodium Conductance".to_string(),
        "g_k" => "Potassium Conductance".to_string(),
        "g_l" => "Leak Conductance".to_string(),
        "i_ext" => "External Stimulus Current".to_string(),
        "potential_type" => "Potential Profile".to_string(),
        "dt" => "Time Step".to_string(),
        "params" => "Model Parameters".to_string(),
        "viscosity" => "Kinematic Viscosity".to_string(),
        other if other.contains('_') => other
            .split('_')
            .filter(|s| !s.is_empty())
            .map(|w| {
                let mut chars = w.chars();
                match chars.next() {
                    Some(c) => format!("{}{}", c.to_uppercase(), chars.as_str()),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        other => {
            let mut chars = other.chars();
            match chars.next() {
                Some(c) if c.is_lowercase() => format!("{}{}", c.to_uppercase(), chars.as_str()),
                _ => other.to_string(),
            }
        }
    }
}

impl SimulationFramework {
    pub(super) fn show_theory_portal(&mut self, ui: &mut egui::Ui, id_source: &str) {
        if !self.show_theory_portal {
            return;
        }
        let ctx = ui.ctx().clone();
        self.refresh_default_theory_toggle(&ctx, id_source);
        let panel_id = scoped_panel_id(ui, format!("{id_source}_theory_portal"));
        let mut close_clicked = false;
        PaneSidePanel::right(panel_id)
            .resizable(true)
            .show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    let heading = match &self.active_tool {
                        Some(tool) => format!("Theory: {}", tool.name()),
                        None => "Theory Context Portal".to_string(),
                    };
                    ui.heading(heading);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button("Close")
                            .accessible_hover_text("Close Theory Context Portal")
                            .clicked()
                        {
                            close_clicked = true;
                        }
                    });
                });
                ui.separator();
                if let Some(tool) = self.active_tool.as_deref() {
                    render_active_tool_theory_body(self, ui, tool);
                }
            });
        let can_escape = !self.show_export_modal
            && !self.show_starter_gallery
            && !egui::Popup::is_any_open(&ctx)
            && ctx.memory(|m| m.top_modal_layer().is_none());
        let escape_dismissed = can_escape
            && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
        if close_clicked || escape_dismissed {
            self.close_theory_portal_and_restore_focus(&ctx, id_source);
        }
    }
}

fn render_active_tool_theory_body(
    framework: &SimulationFramework,
    ui: &mut egui::Ui,
    tool: &dyn InteractiveTool,
) {
    use crate::accessibility::AccessibleHoverText;
    use crate::reflective_ui::render_theory_summary_with_export;
    let theory = tool.theory();
    let citation_text = theory.theory_citation();
    scientific_metadata::citation_registry::CitationRegistry::register(
        tool.name().to_string(),
        citation_text.clone(),
    );
    let theory_desc = theory.theory_description();
    let latex_eq = theory.theory_latex_equation();
    render_theory_summary_with_export(ui, &theory_desc, latex_eq.as_deref())
        .accessible_hover_text("Theoretical background description");
    ui.separator();
    ui.heading("Citations");
    render_theory_summary_with_export(ui, &citation_text, None)
        .accessible_hover_text("Academic citations");
    super::theory_links::render_portal_links_for_framework(framework, ui, tool, &citation_text);
    let mut entries: Vec<(String, String)> = theory
        .available_descriptions()
        .into_iter()
        .filter(|(k, _)| !k.trim().eq_ignore_ascii_case("default"))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    if !entries.is_empty() {
        ui.separator();
        ui.heading("Additional Context");
        for (key, desc) in entries {
            let label = format_theory_context_key(&key);
            let context_entry = format!("{label}: {desc}");
            render_theory_summary_with_export(ui, &context_entry, None)
                .accessible_hover_text(format!("Additional context for {label}"));
        }
    }
}
