#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::{announce_status, AccessibleHoverText};
use crate::app::MathExplorerApp;
use eframe::egui;

/// Academic domains used for categorizing simulation modules in the persistent top navigation bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AcademicDomain {
    PureMath,
    Physics,
    AppliedMath,
    Biology,
    Climate,
    Ai,
}

impl AcademicDomain {
    /// Ordered list of all academic domains.
    pub const ALL: &'static [AcademicDomain] = &[
        AcademicDomain::PureMath,
        AcademicDomain::Physics,
        AcademicDomain::AppliedMath,
        AcademicDomain::Biology,
        AcademicDomain::Climate,
        AcademicDomain::Ai,
    ];

    /// Returns the human-readable display name for the academic domain.
    pub fn name(&self) -> &'static str {
        match self {
            AcademicDomain::PureMath => "Pure Math",
            AcademicDomain::Physics => "Physics",
            AcademicDomain::AppliedMath => "Applied Math",
            AcademicDomain::Biology => "Biology",
            AcademicDomain::Climate => "Climate",
            AcademicDomain::Ai => "AI",
        }
    }

    /// Parses an academic domain from string identifier or display name.
    pub fn from_str(s: &str) -> Option<Self> {
        match s
            .trim()
            .to_lowercase()
            .replace([' ', '_', '&', '-'], "")
            .as_str()
        {
            "puremath" | "pure" => Some(AcademicDomain::PureMath),
            "physics" => Some(AcademicDomain::Physics),
            "appliedmath" | "applied" => Some(AcademicDomain::AppliedMath),
            "biology" | "bio" => Some(AcademicDomain::Biology),
            "climate" => Some(AcademicDomain::Climate),
            "ai" | "artificialintelligence" => Some(AcademicDomain::Ai),
            _ => None,
        }
    }
}

/// Selects an active tab by index, synchronizing primary pane and emitting accessibility announcements.
pub fn select_tab(app: &mut MathExplorerApp, ctx: &egui::Context, tab_idx: usize) {
    if tab_idx < app.tabs.len() {
        app.selected_tab = tab_idx;
        if !app.pane_tabs.is_empty() {
            app.pane_tabs[0] = tab_idx;
        }
        let tab_name = app.tabs[tab_idx].name();
        let status_msg = format!("Switched to tab {}", tab_name);
        announce_status(&status_msg);
        ctx.data_mut(|d| {
            d.insert_temp(egui::Id::new("aria_live_message"), status_msg);
        });
    }
}

/// Renders the persistent categorized domain switcher bar below the main menu.
pub fn render_domain_switcher_bar(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;

        let active_domain = app
            .tabs
            .get(app.selected_tab)
            .and_then(|t| AcademicDomain::from_str(t.academic_domain()));
        let active_tab_name = app
            .tabs
            .get(app.selected_tab)
            .map(|t| t.name())
            .unwrap_or("");

        for domain in AcademicDomain::ALL {
            render_domain_button(app, ctx, ui, *domain, active_domain);
        }

        render_active_path_label(ui, active_domain, active_tab_name);
    });
}

fn render_domain_button(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
    domain: AcademicDomain,
    active_domain: Option<AcademicDomain>,
) {
    let modules: Vec<(usize, &'static str)> = app
        .tabs
        .iter()
        .enumerate()
        .filter_map(|(idx, tab)| {
            if AcademicDomain::from_str(tab.academic_domain()) == Some(domain) {
                Some((idx, tab.name()))
            } else {
                None
            }
        })
        .collect();

    // Adapt cleanly when users disable optional Cargo features
    if modules.is_empty() {
        return;
    }

    let is_active_domain = active_domain == Some(domain);
    let domain_label = if is_active_domain {
        egui::RichText::new(format!("● {}", domain.name()))
            .strong()
            .color(ui.visuals().selection.stroke.color)
    } else {
        egui::RichText::new(domain.name())
    };

    let menu_resp = ui.menu_button(domain_label, |ui| {
        ui.set_min_width(180.0);
        for (tab_idx, tab_name) in &modules {
            let is_active_module = *tab_idx == app.selected_tab;
            let item_label = if is_active_module {
                egui::RichText::new(format!("✓ {}", tab_name)).strong()
            } else {
                egui::RichText::new(*tab_name)
            };

            let resp = ui
                .selectable_label(is_active_module, item_label)
                .accessible_hover_text(format!("Open {} simulation", tab_name));
            if resp.clicked() {
                select_tab(app, ctx, *tab_idx);
                ui.close();
            }
        }
    });

    let is_clicked = menu_resp.response.clicked();
    menu_resp
        .response
        .accessible_hover_text(format!("{} domain simulations", domain.name()));

    // Direct switch for single-module domains on domain button click
    if modules.len() == 1 && is_clicked {
        select_tab(app, ctx, modules[0].0);
    }
}

fn render_active_path_label(
    ui: &mut egui::Ui,
    active_domain: Option<AcademicDomain>,
    active_tab_name: &str,
) {
    if let Some(active_dom) = active_domain {
        if ui.available_width() > 200.0 {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "Active: {} › {}",
                        active_dom.name(),
                        active_tab_name
                    ))
                    .weak(),
                );
            });
        }
    }
}
