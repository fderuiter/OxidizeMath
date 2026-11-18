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

    /// Returns the display name prefixed with an active selection indicator.
    pub const fn active_name(&self) -> &'static str {
        match self {
            AcademicDomain::PureMath => "● Pure Math",
            AcademicDomain::Physics => "● Physics",
            AcademicDomain::AppliedMath => "● Applied Math",
            AcademicDomain::Biology => "● Biology",
            AcademicDomain::Climate => "● Climate",
            AcademicDomain::Ai => "● AI",
        }
    }

    /// Parses an academic domain from string identifier or display name without heap allocation.
    pub fn from_str(s: &str) -> Option<Self> {
        let trimmed = s.trim();
        match trimmed {
            "Pure Math" => return Some(AcademicDomain::PureMath),
            "Physics" => return Some(AcademicDomain::Physics),
            "Applied Math" => return Some(AcademicDomain::AppliedMath),
            "Biology" => return Some(AcademicDomain::Biology),
            "Climate" => return Some(AcademicDomain::Climate),
            "AI" => return Some(AcademicDomain::Ai),
            _ => {}
        }
        let norm = trimmed
            .bytes()
            .filter(|b| !matches!(b, b' ' | b'_' | b'&' | b'-'))
            .map(|b| b.to_ascii_lowercase());
        if norm.clone().eq(*b"puremath") || norm.clone().eq(*b"pure") {
            Some(AcademicDomain::PureMath)
        } else if norm.clone().eq(*b"physics") {
            Some(AcademicDomain::Physics)
        } else if norm.clone().eq(*b"appliedmath") || norm.clone().eq(*b"applied") {
            Some(AcademicDomain::AppliedMath)
        } else if norm.clone().eq(*b"biology") || norm.clone().eq(*b"bio") {
            Some(AcademicDomain::Biology)
        } else if norm.clone().eq(*b"climate") {
            Some(AcademicDomain::Climate)
        } else if norm.clone().eq(*b"ai") || norm.eq(*b"artificialintelligence") {
            Some(AcademicDomain::Ai)
        } else {
            None
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

/// Assigns a simulation module to a specific viewport pane, synchronizing state and emitting accessibility announcements.
pub fn select_pane_tab(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    pane_idx: usize,
    tab_idx: usize,
) {
    if pane_idx < app.pane_tabs.len() && tab_idx < app.tabs.len() {
        app.set_pane_tab(pane_idx, tab_idx);
        let tab_name = app.tabs[tab_idx].name();
        let pane_label = app
            .viewport_layout
            .pane_labels()
            .get(pane_idx)
            .copied()
            .unwrap_or("Pane");
        let status_msg = format!("Assigned {} to {}", tab_name, pane_label);
        announce_status(&status_msg);
        ctx.data_mut(|d| {
            d.insert_temp(egui::Id::new("aria_live_message"), status_msg);
        });
    }
}

pub(crate) fn attach_hover_hint(
    resp: egui::Response,
    hint_fn: impl FnOnce() -> String,
) -> egui::Response {
    if resp.hovered() || resp.has_focus() {
        resp.accessible_hover_text(hint_fn())
    } else {
        let dismiss_id = resp.id.with("tooltip_dismissed");
        if resp
            .ctx
            .data(|d| d.get_temp::<bool>(dismiss_id))
            .unwrap_or(false)
        {
            resp.ctx.data_mut(|d| d.insert_temp(dismiss_id, false));
        }
        resp
    }
}

/// Renders hierarchical academic domain submenus for selecting a simulation module.
///
/// Returns `Some(tab_idx)` when a simulation module is clicked.
pub fn render_domain_module_submenus(
    ui: &mut egui::Ui,
    tabs: &[Box<dyn crate::tabs::ExplorerTab>],
    active_tab_idx: usize,
) -> Option<usize> {
    let active_domain = tabs
        .get(active_tab_idx)
        .and_then(|t| AcademicDomain::from_str(t.academic_domain()));
    let mut selected_idx = None;

    for &domain in AcademicDomain::ALL {
        if let Some(chosen) =
            render_single_domain_submenu(ui, tabs, domain, active_domain, active_tab_idx)
        {
            selected_idx = Some(chosen);
        }
    }

    selected_idx
}

fn render_single_domain_submenu(
    ui: &mut egui::Ui,
    tabs: &[Box<dyn crate::tabs::ExplorerTab>],
    domain: AcademicDomain,
    active_domain: Option<AcademicDomain>,
    active_tab_idx: usize,
) -> Option<usize> {
    let has_modules = tabs
        .iter()
        .any(|t| AcademicDomain::from_str(t.academic_domain()) == Some(domain));
    if !has_modules {
        return None;
    }

    let domain_label = if active_domain == Some(domain) {
        domain.active_name()
    } else {
        domain.name()
    };

    let mut chosen_idx = None;
    let menu_resp = ui.menu_button(domain_label, |sub_ui| {
        sub_ui.set_min_width(180.0);
        for (tab_idx, tab) in tabs.iter().enumerate() {
            if AcademicDomain::from_str(tab.academic_domain()) != Some(domain) {
                continue;
            }
            let tab_name = tab.name();
            let is_active = tab_idx == active_tab_idx;
            let resp = if is_active {
                sub_ui.selectable_label(true, ("✓", tab_name))
            } else {
                sub_ui.selectable_label(false, tab_name)
            };
            let resp = attach_hover_hint(resp, || format!("Open {} simulation", tab_name));
            if resp.clicked() {
                chosen_idx = Some(tab_idx);
                sub_ui.close();
            }
        }
    });

    attach_hover_hint(menu_resp.response, || {
        format!("{} domain simulations", domain.name())
    });

    chosen_idx
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
    let mut first_module_idx = None;
    let mut module_count = 0usize;
    for (idx, tab) in app.tabs.iter().enumerate() {
        if AcademicDomain::from_str(tab.academic_domain()) == Some(domain) {
            if first_module_idx.is_none() {
                first_module_idx = Some(idx);
            }
            module_count += 1;
        }
    }

    // Adapt cleanly when users disable optional Cargo features
    if module_count == 0 {
        return;
    }

    let is_active_domain = active_domain == Some(domain);
    let domain_label = if is_active_domain {
        egui::RichText::new(domain.active_name())
            .strong()
            .color(ui.visuals().selection.stroke.color)
    } else {
        egui::RichText::new(domain.name())
    };

    let mut chosen_tab = None;
    let menu_resp = ui.menu_button(domain_label, |ui| {
        ui.set_min_width(180.0);
        for (tab_idx, tab) in app.tabs.iter().enumerate() {
            if AcademicDomain::from_str(tab.academic_domain()) != Some(domain) {
                continue;
            }
            let tab_name = tab.name();
            let is_active_module = tab_idx == app.selected_tab;
            let resp = if is_active_module {
                ui.selectable_label(true, ("✓", tab_name))
            } else {
                ui.selectable_label(false, tab_name)
            };
            let resp = attach_hover_hint(resp, || format!("Open {} simulation", tab_name));
            if resp.clicked() {
                chosen_tab = Some(tab_idx);
                ui.close();
            }
        }
    });

    if let Some(tab_idx) = chosen_tab {
        select_tab(app, ctx, tab_idx);
    }

    let is_clicked = menu_resp.response.clicked();
    attach_hover_hint(menu_resp.response, || {
        format!("{} domain simulations", domain.name())
    });

    // Direct switch for single-module domains on domain button click
    if module_count == 1 && is_clicked {
        if let Some(first_idx) = first_module_idx {
            select_tab(app, ctx, first_idx);
        }
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
