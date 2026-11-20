use crate::tabs::ExplorerTab;
use eframe::egui;
use std::collections::HashMap;

#[path = "traceability/manifest.rs"]
mod manifest;
#[path = "traceability/scanner.rs"]
mod scanner;

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TraceabilityScanState {
    #[default]
    Idle,
    Scanning,
    Ready,
    Failed(String),
}

#[allow(missing_docs)]
impl TraceabilityScanState {
    pub fn is_scanning(&self) -> bool {
        matches!(self, Self::Scanning)
    }

    #[cfg(test)]
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    #[cfg(test)]
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }

    #[cfg(test)]
    pub fn error(&self) -> Option<&str> {
        match self {
            Self::Failed(err) => Some(err.as_str()),
            _ => None,
        }
    }
}

impl std::fmt::Display for TraceabilityScanState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Scanning => write!(f, "Scanning"),
            Self::Ready => write!(f, "Ready"),
            Self::Failed(err) => write!(f, "Failed: {}", err),
        }
    }
}

pub struct TraceabilityTab {
    papers: HashMap<String, PaperStatus>,
    orphaned_papers: Vec<String>,
    unlinked_code: Vec<String>,
    invalid_links: Vec<(String, String)>,
    repo_path: Option<String>,
    report_rx: Option<
        std::sync::mpsc::Receiver<
            Result<oxidize_core::traceability::TraceabilityReport, String>,
        >,
    >,
    pub state: TraceabilityScanState,
}

#[derive(Clone)]
struct PaperStatus {
    name: String,
    linked_code: Vec<String>,
}

impl Default for TraceabilityTab {
    fn default() -> Self {
        Self {
            papers: HashMap::new(),
            orphaned_papers: Vec::new(),
            unlinked_code: Vec::new(),
            invalid_links: Vec::new(),
            repo_path: None,
            report_rx: None,
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

    pub fn paper_url(name: &str) -> Option<String> {
        const BASE: &str = "https://fderuiter.github.io/OxidizeMath";
        let clean = name.strip_prefix("papers/").unwrap_or(name);
        if let Some(stem) = clean.strip_suffix(".tex") {
            Some(format!("{BASE}/{stem}.pdf"))
        } else if clean.ends_with(".pdf") {
            Some(format!("{BASE}/{clean}"))
        } else if !clean.starts_with("spec:") && !clean.starts_with("registry:") {
            Some(format!("{BASE}/{clean}.pdf"))
        } else {
            None
        }
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
                self.refresh();
            }
            Err(err) => {
                self.clear_results();
                self.repo_path = None;
                self.report_rx = None;
                self.state = TraceabilityScanState::Failed(err);
            }
        }
    }

    fn clear_results(&mut self) {
        self.papers.clear();
        self.orphaned_papers.clear();
        self.unlinked_code.clear();
        self.invalid_links.clear();
    }

    fn load_embedded_manifest(&mut self) {
        match manifest::embedded_manifest() {
            Ok(embedded) => {
                self.orphaned_papers = embedded.orphaned_papers.clone();
                self.unlinked_code = embedded.unlinked_code.clone();
                self.invalid_links = embedded.invalid_links.clone();
                for entry in &embedded.papers {
                    self.papers.insert(
                        entry.name.clone(),
                        PaperStatus {
                            name: entry.name.clone(),
                            linked_code: entry.linked_code.clone(),
                        },
                    );
                }
                if self.papers.is_empty() && self.orphaned_papers.is_empty() {
                    self.state = TraceabilityScanState::Failed(
                        "Embedded traceability manifest contains zero papers.".to_string(),
                    );
                } else {
                    self.state = TraceabilityScanState::Ready;
                }
            }
            Err(err) => {
                self.state = TraceabilityScanState::Failed(err);
            }
        }
    }

    pub fn refresh(&mut self) {
        self.clear_results();
        self.report_rx = None;
        self.repo_path = self
            .repo_path
            .as_deref()
            .and_then(scanner::revalidate_active_repo_path);

        if let Some(repo_dir) = self.repo_path.clone() {
            self.state = TraceabilityScanState::Scanning;
            let (tx, rx) = std::sync::mpsc::channel();
            self.report_rx = Some(rx);
            scanner::spawn_live_scan(repo_dir, tx);
            return;
        }

        self.load_embedded_manifest();
    }

    fn apply_live_report(&mut self, report: oxidize_core::traceability::TraceabilityReport) {
        self.clear_results();
        self.orphaned_papers = report.orphaned_papers;
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

    fn poll_scan_report(&mut self) {
        let Some(rx) = &self.report_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(report)) => {
                self.report_rx = None;
                self.apply_live_report(report);
            }
            Ok(Err(err)) => {
                self.state = TraceabilityScanState::Failed(err);
                self.report_rx = None;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.state = TraceabilityScanState::Failed(
                    "Worker thread disconnected unexpectedly".to_string(),
                );
                self.report_rx = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
    }

    fn render_header(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Theory-to-Code Traceability Portal");
            if ui.button("Refresh").clicked() {
                self.refresh();
            }

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

    fn render_health_section(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        egui::CollapsingHeader::new("System Health & Alignment")
            .default_open(true)
            .show(ui, |ui| {
                let aligned = !self.papers.is_empty()
                    && self.orphaned_papers.is_empty()
                    && self.unlinked_code.is_empty()
                    && self.invalid_links.is_empty();
                let (text, color) = if aligned {
                    ("100% Parity Achieved", palette.success)
                } else {
                    ("Partial Alignment", palette.warning)
                };
                ui.label(egui::RichText::new(text).color(color));
                ui.label(format!("Total Papers: {}", self.papers.len()));

                if self.repo_path.is_none() {
                    if let Ok(embedded) = manifest::embedded_manifest() {
                        if let Some(cert) = &embedded.certificate {
                            ui.label(format!(
                                "Native Coverage: {:.1}% | WASM Path Coverage: {:.1}% | Verified Density: {:.2} asserts/fn",
                                cert.native_execution_coverage_pct,
                                cert.wasm_path_coverage_pct,
                                cert.verified_modules_density
                            ));
                        }
                    }
                }
            });
    }

    fn render_discrepancies(
        &self,
        ui: &mut egui::Ui,
        palette: &crate::app::SemanticPalette,
    ) {
        if !self.orphaned_papers.is_empty() {
            ui.add_space(10.0);
            ui.heading(egui::RichText::new("Orphaned Papers (No Code)").color(palette.error));
            for p in &self.orphaned_papers {
                ui.horizontal(|ui| {
                    ui.label(p);
                    if let Some(url) = Self::paper_url(p) {
                        ui.hyperlink_to("📄 View PDF", &url);
                    }
                });
            }
        }

        if !self.unlinked_code.is_empty() {
            ui.add_space(10.0);
            ui.heading(
                egui::RichText::new("Unlinked Code Modules (No Paper)").color(palette.error),
            );
            for c in &self.unlinked_code {
                ui.label(c);
            }
        }

        if !self.invalid_links.is_empty() {
            ui.add_space(10.0);
            ui.heading(egui::RichText::new("Naming Mismatches / Invalid Cites").color(palette.error));
            for (c, p) in &self.invalid_links {
                ui.label(format!("{} cites invalid paper: {}", c, p));
            }
        }
    }

    fn render_matrix(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        ui.add_space(10.0);
        ui.heading("Traceability Matrix");

        let mut papers: Vec<_> = self.papers.values().collect();
        papers.sort_by(|a, b| a.name.cmp(&b.name));

        for p in papers {
            ui.collapsing(&p.name, |ui| {
                if let Some(url) = Self::paper_url(&p.name) {
                    ui.horizontal(|ui| {
                        ui.hyperlink_to("📄 View PDF Online", &url);
                    });
                }
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
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TraceabilityState {
    repo_path: Option<String>,
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
        if self.repo_path != validated {
            self.repo_path = validated;
            self.report_rx = None;
            self.state = TraceabilityScanState::Idle;
        }
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if matches!(self.state, TraceabilityScanState::Idle) {
            self.refresh();
        }
        self.poll_scan_report();

        if self.state.is_scanning() {
            ui.ctx().request_repaint();
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_header(ui);

            match self.state.clone() {
                TraceabilityScanState::Scanning => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Scanning repository via Virtual File System...");
                    });
                    return;
                }
                TraceabilityScanState::Failed(err) => {
                    self.render_failed_banner(ui, &err);
                    return;
                }
                TraceabilityScanState::Idle => {
                    ui.label("Traceability scan is idle. Click Refresh to scan.");
                    return;
                }
                TraceabilityScanState::Ready => {}
            }

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.render_health_section(ui, &palette);
                self.render_discrepancies(ui, &palette);
                self.render_matrix(ui, &palette);
            });
        });
    }
}

#[cfg(test)]
#[path = "traceability/tests.rs"]
mod tests;

// [cite:essay]
