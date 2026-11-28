//! Accessible Tier 5 interval enclosure and Tier 6 Lean 4 theorem-binding panels for [`super::TraceabilityTab`].

use super::web_snapshot::{lean_proof_url, VerificationCertificate};
use crate::accessibility::AccessibleHoverText;
use crate::framework::maturity_badge::badge_colors_for_tier;
use eframe::egui;
use oxidize_core::traceability::LeanProofBinding;
use scientific_metadata::theory::MaturityTier;
use std::collections::BTreeMap;

/// Returns `true` iff `module_name` has a kernel-verified, non-empty theorem binding in a passing certificate.
#[must_use]
pub fn is_proof_kernel_verified(cert: &VerificationCertificate, module_name: &str) -> bool {
    if !cert.passed {
        return false;
    }
    match cert.lean4_proof_bindings.get(module_name) {
        Some(binding) => {
            binding.kernel_verified
                && !binding.rust_modules.is_empty()
                && !binding.theorems.is_empty()
        }
        None => false,
    }
}

/// Renders an accessible WCAG 2.2 AA status pill (`[Kernel Verified]` or `[Unverified]`) for `module_name`.
pub fn render_proof_status_badge(
    ui: &mut egui::Ui,
    kernel_verified: bool,
    module_name: &str,
) -> egui::Response {
    let (tier, badge_text, status_value, a11y_detail) = if kernel_verified {
        (
            MaturityTier::Certified,
            "[Kernel Verified]",
            "Kernel Verified",
            "Kernel Verified by Lean 4.14.0 (lake build + axiom-clean environment)",
        )
    } else {
        (
            MaturityTier::Experimental,
            "[Unverified]",
            "Unverified",
            "Unverified (missing passing Lean 4 kernel certificate or theorem binding)",
        )
    };
    let colors = badge_colors_for_tier(ui.visuals(), tier);
    let a11y_label = format!("{module_name} — {a11y_detail}");

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
        builder.set_value(status_value);
    });
    response.accessible_hover_text(a11y_label)
}

fn render_single_lean_proof_entry(
    ui: &mut egui::Ui,
    cert: &VerificationCertificate,
    live_proof_bindings: &BTreeMap<String, LeanProofBinding>,
    module_name: &str,
    sha256: &str,
) {
    let kernel_verified = is_proof_kernel_verified(cert, module_name);
    let binding = cert
        .lean4_proof_bindings
        .get(module_name)
        .or_else(|| live_proof_bindings.get(module_name));
    let rust_modules: &[String] = binding.map_or(&[], |b| b.rust_modules.as_slice());
    let theorems: &[String] = binding.map_or(&[], |b| b.theorems.as_slice());
    let short_sha = sha256.get(..16).unwrap_or(sha256);
    let linked_mods = if rust_modules.is_empty() {
        "unbound".to_string()
    } else {
        rust_modules.join(", ")
    };

    ui.horizontal_wrapped(|ui| {
        render_proof_status_badge(ui, kernel_verified, module_name);
        if let Some(url) = lean_proof_url(module_name) {
            ui.hyperlink_to(module_name, url);
        } else {
            ui.label(module_name);
        }
        ui.label("Rust Modules:");
        ui.monospace(&linked_mods);
        ui.monospace(format!("sha256:{short_sha}…"));
    });

    let thm_header = format!("Theorem Bindings ({})", theorems.len());
    ui.indent(("lean_proof_indent", module_name), |ui| {
        egui::CollapsingHeader::new(thm_header)
            .id_salt(("lean_proof_theorems", module_name))
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.strong("Linked Rust Modules:");
                    ui.monospace(&linked_mods);
                });
                if theorems.is_empty() {
                    ui.monospace("No bound Lean theorems recorded.");
                } else {
                    ui.horizontal_wrapped(|ui| {
                        ui.strong("Compiled Theorems:");
                        for theorem_name in theorems {
                            ui.monospace(theorem_name);
                        }
                    });
                }
            });
    });
}

/// Renders Tier 5 outward-rounded interval enclosures and Tier 6 Lean 4 kernel proof bindings.
pub(super) fn render_interval_and_proof_details(
    ui: &mut egui::Ui,
    cert: &VerificationCertificate,
    live_proof_bindings: &BTreeMap<String, LeanProofBinding>,
) {
    if !cert.interval_bounds.is_empty() {
        let header = format!(
            "Tier 5 Outward-Rounded Interval Enclosures ({})",
            cert.interval_bounds.len()
        );
        egui::CollapsingHeader::new(header)
            .default_open(true)
            .show(ui, |ui| {
                for (kernel, bounds) in &cert.interval_bounds {
                    let width = (bounds[1] - bounds[0]).max(0.0);
                    ui.monospace(format!(
                        "{kernel}: [{:.12e}, {:.12e}] (w={width:.2e})",
                        bounds[0], bounds[1]
                    ));
                }
            });
    }
    if !cert.lean4_proof_digests.is_empty() {
        let header = format!(
            "Tier 6 Lean 4 Formal Proof Certificates ({})",
            cert.lean4_proof_digests.len()
        );
        egui::CollapsingHeader::new(header)
            .default_open(true)
            .show(ui, |ui| {
                for (module_name, sha256) in &cert.lean4_proof_digests {
                    render_single_lean_proof_entry(
                        ui,
                        cert,
                        live_proof_bindings,
                        module_name,
                        sha256,
                    );
                }
            });
    }
}

// [cite:essay]
