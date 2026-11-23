//! Accessible WCAG 2.2 AA maturity tier badges and tool tier resolution for [`super::SimulationFramework`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{InteractiveTool, SimulationFramework, ToolMetadata};
use crate::accessibility::AccessibleHoverText;
use crate::app::theme::ThemeMode;
use crate::app::SemanticPalette;
use eframe::egui;
use oxidize_core::traceability::MaturityRegistry;
use scientific_metadata::theory::MaturityTier;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Foreground, background, and border colors for rendering a [`MaturityTier`] pill badge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaturityBadgeColors {
    /// Opaque sRGB pill fill guaranteeing `>= 4.5:1` WCAG 2.2 AA contrast with [`Self::fg_text`].
    pub bg_fill: egui::Color32,
    /// High-contrast foreground label color derived from [`SemanticPalette`].
    pub fg_text: egui::Color32,
    /// Graphical border stroke color (`>= 3.0:1` against panel surfaces).
    pub border_stroke: egui::Color32,
}

/// Resolves theme-aware [`MaturityBadgeColors`] from `visuals` for `tier` with zero allocations.
#[must_use]
pub fn badge_colors_for_tier(visuals: &egui::Visuals, tier: MaturityTier) -> MaturityBadgeColors {
    let mode = ThemeMode::from_visuals(visuals);
    let palette = SemanticPalette::for_theme(mode);
    match (mode, tier) {
        (ThemeMode::Dark, MaturityTier::Certified) => MaturityBadgeColors {
            bg_fill: egui::Color32::from_rgb(14, 48, 26),
            fg_text: palette.success,
            border_stroke: palette.success,
        },
        (ThemeMode::Dark, MaturityTier::Validated) => MaturityBadgeColors {
            bg_fill: egui::Color32::from_rgb(12, 38, 66),
            fg_text: palette.info,
            border_stroke: palette.info,
        },
        (ThemeMode::Dark, MaturityTier::Experimental) => MaturityBadgeColors {
            bg_fill: egui::Color32::from_rgb(52, 36, 8),
            fg_text: palette.warning,
            border_stroke: palette.warning,
        },
        (ThemeMode::Light, MaturityTier::Certified) => MaturityBadgeColors {
            bg_fill: egui::Color32::from_rgb(232, 248, 237),
            fg_text: palette.success,
            border_stroke: palette.success,
        },
        (ThemeMode::Light, MaturityTier::Validated) => MaturityBadgeColors {
            bg_fill: egui::Color32::from_rgb(232, 243, 255),
            fg_text: palette.info,
            border_stroke: palette.info,
        },
        (ThemeMode::Light, MaturityTier::Experimental) => MaturityBadgeColors {
            bg_fill: egui::Color32::from_rgb(255, 246, 226),
            fg_text: palette.warning,
            border_stroke: palette.warning,
        },
        (ThemeMode::HighContrast, MaturityTier::Certified) => MaturityBadgeColors {
            bg_fill: egui::Color32::BLACK,
            fg_text: palette.success,
            border_stroke: palette.success,
        },
        (ThemeMode::HighContrast, MaturityTier::Validated) => MaturityBadgeColors {
            bg_fill: egui::Color32::BLACK,
            fg_text: palette.info,
            border_stroke: palette.info,
        },
        (ThemeMode::HighContrast, MaturityTier::Experimental) => MaturityBadgeColors {
            bg_fill: egui::Color32::BLACK,
            fg_text: palette.warning,
            border_stroke: palette.warning,
        },
    }
}

/// Returns the non-color-only bracketed badge text for `tier` (`compact` selects `[CERT]` vs `[CERTIFIED]`).
#[inline]
#[must_use]
pub const fn badge_display_text(tier: MaturityTier, compact: bool) -> &'static str {
    if compact {
        match tier {
            MaturityTier::Experimental => "[EXP]",
            MaturityTier::Validated => "[VAL]",
            MaturityTier::Certified => "[CERT]",
        }
    } else {
        match tier {
            MaturityTier::Experimental => "[EXPERIMENTAL]",
            MaturityTier::Validated => "[VALIDATED]",
            MaturityTier::Certified => "[CERTIFIED]",
        }
    }
}

/// Returns a concise qualification explanation string for `tier` displayed in portals and tooltips.
#[inline]
#[must_use]
pub const fn tier_qualification_summary(tier: MaturityTier) -> &'static str {
    match tier {
        MaturityTier::Certified => {
            "Deterministic theory_verification! contract, zero-allocation proof, and SemVer signature lock."
        }
        MaturityTier::Validated => {
            "Seeded deterministic execution, golden reference fixtures, and >= 0.40 assertion density."
        }
        MaturityTier::Experimental => {
            "Exploratory prototype tier; fluid API evolution permitted prior to formal promotion."
        }
    }
}

/// Renders an accessible pill badge for `tier` with `SemanticPalette` colors, `AccessKit` node metadata,
/// and polite screen-reader hover/focus live-region support via [`AccessibleHoverText`].
pub fn render_maturity_badge(
    ui: &mut egui::Ui,
    tier: MaturityTier,
    compact: bool,
    subject: &str,
) -> egui::Response {
    let colors = badge_colors_for_tier(ui.visuals(), tier);
    let badge_text = badge_display_text(tier, compact);
    let a11y_label = if subject.trim().is_empty() {
        tier.screen_reader_label().to_string()
    } else {
        format!("{subject} — {}", tier.screen_reader_label())
    };

    let inner = egui::Frame::NONE
        .fill(colors.bg_fill)
        .stroke(egui::Stroke::new(1.0_f32, colors.border_stroke))
        .corner_radius(4.0_f32)
        .inner_margin(egui::Margin::symmetric(5, 1))
        .show(ui, |ui| {
            let rich = egui::RichText::new(badge_text)
                .color(colors.fg_text)
                .monospace()
                .small()
                .strong();
            ui.add(egui::Label::new(rich).extend().selectable(false))
        });

    let response = inner.response;
    let enabled = ui.is_enabled();
    let widget_label = a11y_label.clone();
    response.widget_info(move || {
        egui::WidgetInfo::labeled(egui::WidgetType::Label, enabled, &widget_label)
    });
    ui.ctx().accesskit_node_builder(response.id, |builder| {
        builder.set_role(egui::accesskit::Role::Label);
        builder.set_label(a11y_label.as_str());
        builder.set_value(tier.as_str());
    });
    response.accessible_hover_text(a11y_label)
}

/// Resolves the [`MaturityTier`] for `(tool_id, domain)` combining an optional `TheoryDescribable` override
/// with the embedded `traceability.toml` [`MaturityRegistry`].
#[must_use]
pub fn resolve_tool_maturity_tier(
    tool_id: &str,
    domain: &str,
    declared: Option<MaturityTier>,
) -> MaturityTier {
    if let Some(tier) = declared {
        if tier != MaturityTier::Experimental {
            return tier;
        }
    }
    let dom_clean = domain.trim();
    if dom_clean.eq_ignore_ascii_case("climate") || dom_clean.eq_ignore_ascii_case("favoritism") {
        return MaturityTier::Experimental;
    }

    let registry = MaturityRegistry::embedded();
    let mut matched: Vec<MaturityTier> = super::theory_links::tool_module_tokens(tool_id)
        .into_iter()
        .filter_map(|tok| registry.modules.get(&tok).copied())
        .collect();

    if matched.is_empty() {
        matched = super::theory_links::domain_module_tokens(dom_clean)
            .into_iter()
            .filter_map(|tok| registry.modules.get(&tok).copied())
            .collect();
    }

    matched
        .into_iter()
        .min()
        .unwrap_or(MaturityTier::Experimental)
}

/// Resolves the active tool's [`MaturityTier`] inside [`SimulationFramework`].
#[must_use]
pub(crate) fn active_framework_tool_tier(
    framework: &SimulationFramework,
    tool: &dyn InteractiveTool,
) -> MaturityTier {
    let active_meta: Option<&ToolMetadata> = framework
        .selected_tool_index
        .and_then(|idx| framework.available_tools.get(idx).copied());
    let tool_id = active_meta.map_or_else(|| tool.name(), |m| m.id);
    let domain = active_meta.map_or_else(
        || {
            if framework.domain.is_empty() {
                tool.domain()
            } else {
                framework.domain.as_str()
            }
        },
        |m| m.domain,
    );
    resolve_tool_maturity_tier(tool_id, domain, Some(tool.theory().maturity_tier()))
}

impl ToolMetadata {
    /// Resolves and caches the [`MaturityTier`] for this registered [`ToolMetadata`] descriptor.
    #[must_use]
    pub fn maturity_tier(&self) -> MaturityTier {
        static TIER_CACHE: OnceLock<Mutex<HashMap<(&'static str, &'static str), MaturityTier>>> =
            OnceLock::new();
        let cache = TIER_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Ok(guard) = cache.lock() {
            if let Some(&cached) = guard.get(&(self.id, self.domain)) {
                return cached;
            }
        }
        let declared = (self.build)().theory().maturity_tier();
        let resolved = resolve_tool_maturity_tier(self.id, self.domain, Some(declared));
        if let Ok(mut guard) = cache.lock() {
            guard.insert((self.id, self.domain), resolved);
        }
        resolved
    }
}
