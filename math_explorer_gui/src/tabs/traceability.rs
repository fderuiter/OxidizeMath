use crate::tabs::ExplorerTab;
use eframe::egui;
use std::collections::HashMap;

#[path = "traceability/presets.rs"]
mod presets_impl;
pub use presets_impl::*;
#[path = "traceability/scanner.rs"]
mod scanner;
#[path = "traceability/web_snapshot.rs"]
mod web_snapshot;

#[cfg(all(test, not(target_arch = "wasm32")))]
use scanner::execute_workspace_scan;
pub(crate) use web_snapshot::embedded_traceability_report;
use web_snapshot::PaperStatus;
pub use web_snapshot::{paper_tex_url, render_paper_links};

pub struct TraceabilityTab {
    papers: HashMap<String, PaperStatus>,
    orphaned_papers: Vec<String>,
    unlinked_code: Vec<String>,
    invalid_links: Vec<(String, String)>,
    repo_path: Option<String>,
    user_picked_repo: bool,
    report_rx: Option<
        std::sync::mpsc::Receiver<Result<oxidize_core::traceability::TraceabilityReport, String>>,
    >,
    cert_rx: Option<std::sync::mpsc::Receiver<Option<web_snapshot::VerificationCertificate>>>,
    certificate: Option<web_snapshot::VerificationCertificate>,
    pub(crate) has_initial_scan: bool,
    pub state: TraceabilityScanState,
}

impl Default for TraceabilityTab {
    fn default() -> Self {
        Self {
            papers: HashMap::new(),
            orphaned_papers: Vec::new(),
            unlinked_code: Vec::new(),
            invalid_links: Vec::new(),
            repo_path: None,
            user_picked_repo: false,
            report_rx: None,
            cert_rx: None,
            certificate: None,
            has_initial_scan: false,
            state: TraceabilityScanState::Idle,
        }
    }
}

#[allow(missing_docs)]
impl TraceabilityTab {
    #[cfg(test)]
    pub fn retry(&mut self) {
        self.refresh();
    }

    #[cfg(test)]
    pub fn error_message(&self) -> Option<&str> {
        self.state.error()
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn has_workspace_header(manifest: &str) -> bool {
        scanner::has_workspace_header(manifest)
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn canonical_workspace_root() -> Option<std::path::PathBuf> {
        scanner::canonical_workspace_root()
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn join_relative_segments(base: &std::path::Path, rel: &str) -> std::path::PathBuf {
        scanner::join_relative_segments(base, rel)
    }

    pub(crate) fn paper_url_for_target(name: &str, is_wasm: bool) -> Option<String> {
        let trimmed = name.trim();
        let clean = trimmed.strip_prefix("papers/").unwrap_or(trimmed);
        if clean.is_empty() || clean.starts_with("spec:") || clean.starts_with("registry:") {
            return None;
        }
        let base = if is_wasm {
            "."
        } else {
            "https://fderuiter.github.io/OxidizeMath"
        };
        if let Some(stem) = clean.strip_suffix(".tex") {
            (!stem.is_empty()).then(|| format!("{base}/{stem}.pdf"))
        } else if let Some(stem) = clean.strip_suffix(".pdf") {
            (!stem.is_empty()).then(|| format!("{base}/{stem}.pdf"))
        } else {
            Some(format!("{base}/{clean}.pdf"))
        }
    }

    pub fn paper_url(name: &str) -> Option<String> {
        Self::paper_url_for_target(name, cfg!(target_arch = "wasm32"))
    }

    #[cfg(test)]
    pub fn online_document_url(&self, name: &str) -> Option<String> {
        let clean = name.strip_prefix("papers/").unwrap_or(name);
        let is_tracked = self.papers.keys().any(|k| {
            let k_clean = k.strip_prefix("papers/").unwrap_or(k.as_str());
            k == name || k_clean == clean
        }) || self.orphaned_papers.iter().any(|p| {
            let p_clean = p.strip_prefix("papers/").unwrap_or(p.as_str());
            p == name || p_clean == clean
        });
        is_tracked.then(|| Self::paper_url(name)).flatten()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn select_directory(&mut self, path: &std::path::Path) {
        match scanner::validate_picked_repo_path(path) {
            Ok(normalized) => {
                self.repo_path = Some(normalized);
                self.user_picked_repo = true;
                self.refresh();
            }
            Err(err) => {
                self.clear_results();
                self.repo_path = None;
                self.user_picked_repo = false;
                self.report_rx = None;
                self.cert_rx = None;
                self.has_initial_scan = true;
                self.state = TraceabilityScanState::Failed(err);
            }
        }
    }

    fn clear_results(&mut self) {
        self.papers.clear();
        self.orphaned_papers.clear();
        self.unlinked_code.clear();
        self.invalid_links.clear();
        self.certificate = None;
    }

    pub fn refresh(&mut self) {
        self.has_initial_scan = true;
        self.clear_results();
        self.repo_path = self
            .repo_path
            .as_deref()
            .and_then(|raw| scanner::revalidate_repo_path(raw, self.user_picked_repo));
        if self.repo_path.is_none() {
            self.user_picked_repo = false;
        }
        self.state = TraceabilityScanState::Scanning;

        let (tx, rx) = std::sync::mpsc::channel();
        let (cert_tx, cert_rx) = std::sync::mpsc::channel();
        self.report_rx = Some(rx);
        self.cert_rx = Some(cert_rx);

        #[cfg(not(target_arch = "wasm32"))]
        scanner::dispatch_native_refresh(self.repo_path.clone(), tx, cert_tx);

        #[cfg(target_arch = "wasm32")]
        // theory_verification!
        wasm_bindgen_futures::spawn_local(async move {
            match web_snapshot::fetch_web_traceability().await {
                Ok((report, cert)) => {
                    let _ = cert_tx.send(cert);
                    let _ = tx.send(Ok(report));
                }
                Err(err) => {
                    let _ = tx.send(Err(err));
                }
            }
        });
    }

    fn apply_scan_report(&mut self, report: oxidize_core::traceability::TraceabilityReport) {
        self.clear_results();
        self.orphaned_papers = report
            .orphaned_papers
            .into_iter()
            .filter(|p| !p.starts_with("spec:") && !p.starts_with("registry:"))
            .collect();
        self.unlinked_code = report.unlinked_code;
        self.invalid_links = report.invalid_links;

        for (paper_name, linked_code) in report.paper_coverage {
            if !paper_name.starts_with("spec:") && !paper_name.starts_with("registry:") {
                self.papers.insert(
                    paper_name.clone(),
                    PaperStatus {
                        name: paper_name,
                        linked_code,
                    },
                );
            }
        }

        if self.papers.is_empty() && self.orphaned_papers.is_empty() {
            self.state = TraceabilityScanState::Failed(
                "No academic papers (.tex) were discovered in the repository.".to_string(),
            );
        } else {
            self.state = TraceabilityScanState::Ready;
        }
    }

    fn poll_scan_channels(&mut self) {
        let Some(rx) = &self.report_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(report)) => {
                self.report_rx = None;
                self.apply_scan_report(report);
                self.certificate = self
                    .cert_rx
                    .as_ref()
                    .and_then(|crx| crx.try_recv().ok())
                    .flatten();
                self.cert_rx = None;
            }
            Ok(Err(err)) => {
                self.state = TraceabilityScanState::Failed(err);
                self.report_rx = None;
                self.cert_rx = None;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.state = TraceabilityScanState::Failed(
                    "Worker thread disconnected unexpectedly".to_string(),
                );
                self.report_rx = None;
                self.cert_rx = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
    }

    fn render_portal_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.heading("Theory-to-Code Traceability Portal");
            crate::framework::render_workspace_history_buttons(ui);
            if ui.button("Refresh").clicked() {
                self.refresh();
            }
            self.render_share_link_button(ui);
            #[cfg(not(target_arch = "wasm32"))]
            if ui.button("Select Codebase Directory...").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    self.select_directory(&path);
                }
            }

            if let Some(path) = &self.repo_path {
                ui.label(format!("Scanning: {}", path));
            } else {
                ui.label("Scanning: default workspace");
            }
        });
        ui.separator();
    }

    fn render_failed_banner(&mut self, ui: &mut egui::Ui, err: &str) {
        let err_color = crate::app::SemanticPalette::from_visuals(ui.visuals()).error;
        let mut retry_clicked = false;
        #[cfg(not(target_arch = "wasm32"))]
        let mut picked_dir: Option<std::path::PathBuf> = None;

        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new(format!("⚠️ Scan Failed: {}", err))
                        .color(err_color)
                        .strong(),
                );
                ui.horizontal(|ui| {
                    if ui.button("Retry").clicked() {
                        retry_clicked = true;
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.button("Select Codebase Directory...").clicked() {
                        picked_dir = rfd::FileDialog::new().pick_folder();
                    }
                });
            });
        });

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = picked_dir {
            self.select_directory(&path);
            return;
        }
        if retry_clicked {
            self.refresh();
        }
    }
}

impl ExplorerTab for TraceabilityTab {
    fn name(&self) -> &'static str {
        "Traceability Portal"
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.repo_path.hash(&mut h);
        h.finish().max(1)
    }

    fn export_url_preset_json(&self) -> Result<Option<String>, String> {
        self.to_compact_url_json().map(Some)
    }

    fn import_url_preset_json(
        &mut self,
        tool_id: Option<&str>,
        preset_json: Option<&str>,
    ) -> Result<(), String> {
        if let Some(req_tool) = tool_id {
            if !crate::url_state::matches_tab_slug(req_tool, self.url_slug(), self.name()) {
                return Err(format!(
                    "Tab '{}' does not support sub-tool '{req_tool}'",
                    self.name()
                ));
            }
        }
        if let Some(json_str) = preset_json {
            crate::url_state::validate_url_preset_json(json_str)?;
            self.from_json_str(json_str)?;
        }
        Ok(())
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&TraceabilityState {
            repo_path: self.repo_path.clone(),
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        let validated = serde_json::from_str::<TraceabilityState>(state)
            .ok()
            .and_then(|s| s.repo_path)
            .as_deref()
            .and_then(scanner::validate_session_repo_path);
        self.user_picked_repo = false;
        if self.repo_path != validated {
            self.repo_path = validated;
            self.report_rx = None;
            self.cert_rx = None;
            self.has_initial_scan = false;
            self.state = TraceabilityScanState::Idle;
        }
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_scan_channels();
        if !self.has_initial_scan {
            self.refresh();
        }
        if self.state.is_scanning() {
            ui.ctx().request_repaint();
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_portal_toolbar(ui);
            match self.state.clone() {
                TraceabilityScanState::Scanning => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Scanning repository via Virtual File System...");
                    });
                }
                TraceabilityScanState::Failed(err) => self.render_failed_banner(ui, &err),
                TraceabilityScanState::Idle => {
                    ui.label("Traceability scan is idle. Click Refresh to scan.");
                }
                TraceabilityScanState::Ready => web_snapshot::render_ready_report(
                    ui,
                    &self.papers,
                    &self.orphaned_papers,
                    &self.unlinked_code,
                    &self.invalid_links,
                    self.certificate.as_ref(),
                    Self::paper_url,
                ),
            }
        });
    }
}

#[cfg(test)]
#[path = "traceability/tests.rs"]
mod tests;

// [cite:essay]
