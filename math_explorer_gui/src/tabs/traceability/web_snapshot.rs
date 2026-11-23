#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// theory_verification!

use eframe::egui;
pub use oxidize_core::traceability::{
    TraceabilityManifestPayload, TraceabilityReport, VerificationCertificate,
};
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

/// Document-relative path to the precomputed traceability manifest in `dist/`.
#[cfg(any(target_arch = "wasm32", test))]
pub const MANIFEST_REL_URL: &str = "./traceability_manifest.json";

/// Canonical GitHub repository tree base for `.tex` paper sources.
const GITHUB_PAPERS_BASE: &str = "https://github.com/fderuiter/OxidizeMath/blob/main/papers";

/// Per-paper coverage entry rendered inside [`super::TraceabilityTab`].
#[derive(Clone)]
pub(super) struct PaperStatus {
    pub name: String,
    pub linked_code: Vec<String>,
}

/// Parses a `traceability_manifest.json` payload into a [`TraceabilityReport`] and optional [`VerificationCertificate`].
pub fn parse_manifest_json(
    raw: &str,
) -> Result<(TraceabilityReport, Option<VerificationCertificate>), String> {
    serde_json::from_str::<TraceabilityManifestPayload>(raw)
        .map(|payload| (payload.report, payload.certificate))
        .map_err(|err| format!("Invalid traceability_manifest.json: {err}"))
}

/// Parses a `Verification_Certificate.json` payload into a [`VerificationCertificate`].
#[cfg(not(target_arch = "wasm32"))]
pub fn parse_certificate_json(raw: &str) -> Result<VerificationCertificate, String> {
    serde_json::from_str::<VerificationCertificate>(raw)
        .map_err(|err| format!("Invalid Verification_Certificate.json: {err}"))
}

/// Validates that a web fetch path is document-relative (`./...`) without parent traversal.
#[cfg(any(target_arch = "wasm32", test))]
pub fn is_relative_same_origin_url(url: &str) -> bool {
    url.starts_with("./")
        && !url.contains('\\')
        && !url.contains("://")
        && !url.split('/').any(|seg| seg == "..")
}

/// Returns the canonical GitHub `.tex` source link for a tracked academic paper.
pub fn paper_tex_url(name: &str) -> Option<String> {
    let clean = name.strip_prefix("papers/").unwrap_or(name);
    if clean.starts_with("spec:") || clean.starts_with("registry:") || clean.is_empty() {
        return None;
    }
    let stem = clean
        .strip_suffix(".tex")
        .or_else(|| clean.strip_suffix(".pdf"))
        .unwrap_or(clean);
    (!stem.is_empty()).then(|| format!("{GITHUB_PAPERS_BASE}/{stem}.tex"))
}

/// Loads `Verification_Certificate.json` from a local native directory when available.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_local_certificate(base_dir: &str) -> Option<VerificationCertificate> {
    let candidate = if base_dir.is_empty() {
        std::path::PathBuf::from("Verification_Certificate.json")
    } else {
        std::path::Path::new(base_dir).join("Verification_Certificate.json")
    };
    let raw = std::fs::read_to_string(candidate).ok()?;
    parse_certificate_json(&raw).ok()
}

/// Returns the compile-time precomputed traceability report and certificate for packaged desktop runs.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn embedded_traceability(
) -> Result<(TraceabilityReport, Option<VerificationCertificate>), String> {
    static CACHE: OnceLock<Result<(TraceabilityReport, Option<VerificationCertificate>), String>> =
        OnceLock::new();
    CACHE
        .get_or_init(|| {
            const MANIFEST_JSON: &str =
                include_str!(concat!(env!("OUT_DIR"), "/traceability_manifest.json"));
            let (report, embedded_cert) = parse_manifest_json(MANIFEST_JSON)?;
            let cert = embedded_cert.or_else(|| {
                const CERT_JSON: &str = include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../Verification_Certificate.json"
                ));
                parse_certificate_json(CERT_JSON).ok()
            });
            Ok((report, cert))
        })
        .clone()
}

/// Fetches `./traceability_manifest.json` via [`oxidize_core::vfs::WasmVfs`] on `wasm32`.
#[cfg(target_arch = "wasm32")]
pub async fn fetch_web_traceability(
) -> Result<(TraceabilityReport, Option<VerificationCertificate>), String> {
    use oxidize_core::vfs::{VirtualFileSystem, WasmVfs};

    if !is_relative_same_origin_url(MANIFEST_REL_URL) {
        return Err(format!(
            "Refusing non-relative manifest URL: {MANIFEST_REL_URL}"
        ));
    }
    let raw = WasmVfs
        .read_to_string(MANIFEST_REL_URL)
        .await
        .map_err(|err| format!("Failed loading {MANIFEST_REL_URL}: {err}"))?;
    parse_manifest_json(&raw)
}

/// Renders precomputed or local `Verification_Certificate.json` metrics in the health summary.
pub fn render_certificate_metrics(
    ui: &mut egui::Ui,
    cert: Option<&VerificationCertificate>,
    palette: &crate::app::SemanticPalette,
) {
    let Some(cert) = cert else {
        return;
    };
    ui.separator();
    let (status_label, color) = if cert.passed {
        ("Verification Certificate: Passed", palette.success)
    } else {
        ("Verification Certificate: Failed", palette.error)
    };
    ui.label(egui::RichText::new(status_label).color(color).strong());
    ui.label(format!(
        "Native Execution Coverage: {:.1}% | WASM Path Coverage: {:.1}%",
        cert.native_execution_coverage_pct, cert.wasm_path_coverage_pct
    ));
    ui.label(format!(
        "Assertion Density: {:.2} total | {:.2} verified | {:.2} unverified",
        cert.total_assertion_density,
        cert.verified_modules_density,
        cert.unverified_modules_density
    ));
}

/// Renders both the hosted PDF link and canonical GitHub `.tex` source fallback link.
pub fn render_paper_links(
    ui: &mut egui::Ui,
    pdf_url: Option<&str>,
    tex_url: Option<&str>,
    pdf_label: &str,
) {
    if pdf_url.is_none() && tex_url.is_none() {
        return;
    }
    ui.horizontal(|ui| {
        if let Some(url) = pdf_url {
            ui.hyperlink_to(pdf_label, url);
        }
        if let Some(src_url) = tex_url {
            ui.hyperlink_to("📝 View .tex Source", src_url);
        }
    });
}

/// Renders the scrollable `Ready` traceability summary, diagnostics, and paper matrix.
pub(super) fn render_ready_report(
    ui: &mut egui::Ui,
    papers_map: &HashMap<String, PaperStatus>,
    orphaned_papers: &[String],
    unlinked_code: &[String],
    invalid_links: &[(String, String)],
    certificate: Option<&VerificationCertificate>,
    paper_pdf_fn: fn(&str) -> Option<String>,
) {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    egui::ScrollArea::vertical().show(ui, |ui| {
        egui::CollapsingHeader::new("System Health & Alignment")
            .default_open(true)
            .show(ui, |ui| {
                let full_parity = !papers_map.is_empty()
                    && orphaned_papers.is_empty()
                    && unlinked_code.is_empty()
                    && invalid_links.is_empty();
                if full_parity {
                    ui.label(egui::RichText::new("100% Parity Achieved").color(palette.success));
                } else {
                    ui.label(egui::RichText::new("Partial Alignment").color(palette.warning));
                }
                ui.label(format!("Total Papers: {}", papers_map.len()));
                render_certificate_metrics(ui, certificate, &palette);
            });

        render_diagnostics(
            ui,
            orphaned_papers,
            unlinked_code,
            invalid_links,
            &palette,
            paper_pdf_fn,
        );
        render_matrix_section(ui, papers_map, &palette, paper_pdf_fn);
    });
}

fn render_diagnostics(
    ui: &mut egui::Ui,
    orphaned_papers: &[String],
    unlinked_code: &[String],
    invalid_links: &[(String, String)],
    palette: &crate::app::SemanticPalette,
    paper_pdf_fn: fn(&str) -> Option<String>,
) {
    if !orphaned_papers.is_empty() {
        ui.add_space(10.0);
        ui.heading(egui::RichText::new("Orphaned Papers (No Code)").color(palette.error));
        for p in orphaned_papers {
            ui.horizontal(|ui| {
                ui.label(p);
                let pdf = paper_pdf_fn(p);
                let tex = paper_tex_url(p);
                render_paper_links(ui, pdf.as_deref(), tex.as_deref(), "📄 View PDF");
            });
        }
    }
    if !unlinked_code.is_empty() {
        ui.add_space(10.0);
        ui.heading(egui::RichText::new("Unlinked Code Modules (No Paper)").color(palette.error));
        for c in unlinked_code {
            ui.label(c);
        }
    }
    if !invalid_links.is_empty() {
        ui.add_space(10.0);
        ui.heading(egui::RichText::new("Naming Mismatches / Invalid Cites").color(palette.error));
        for (c, p) in invalid_links {
            ui.label(format!("{c} cites invalid paper: {p}"));
        }
    }
}

fn render_matrix_section(
    ui: &mut egui::Ui,
    papers_map: &HashMap<String, PaperStatus>,
    palette: &crate::app::SemanticPalette,
    paper_pdf_fn: fn(&str) -> Option<String>,
) {
    ui.add_space(10.0);
    ui.heading("Traceability Matrix");

    let mut papers: Vec<_> = papers_map.values().collect();
    papers.sort_by(|a, b| a.name.cmp(&b.name));

    for p in papers {
        ui.collapsing(&p.name, |ui| {
            let pdf = paper_pdf_fn(&p.name);
            let tex = paper_tex_url(&p.name);
            render_paper_links(ui, pdf.as_deref(), tex.as_deref(), "📄 View PDF Online");
            if p.linked_code.is_empty() {
                ui.label(egui::RichText::new("No implementation found").color(palette.error));
            } else {
                for code in &p.linked_code {
                    ui.label(code);
                }
            }
        });
    }
}
