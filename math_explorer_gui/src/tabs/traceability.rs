use crate::tabs::ExplorerTab;
use eframe::egui;
use oxidize_core::vfs::VirtualFileSystem;
use std::collections::HashMap;

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TraceabilityScanState {
    #[default]
    Idle,
    Scanning,
    Ready,
    Failed(String),
}

#[allow(missing_docs, dead_code)]
impl TraceabilityScanState {
    #[allow(missing_docs)]
    pub fn is_idle(&self) -> bool {
        matches!(self, Self::Idle)
    }

    #[allow(missing_docs)]
    pub fn is_scanning(&self) -> bool {
        matches!(self, Self::Scanning)
    }

    #[allow(missing_docs)]
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    #[allow(missing_docs)]
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }

    #[allow(missing_docs)]
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
    invalid_links: Vec<(String, String)>, // code_file, paper_name
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
        let mut tab = Self {
            papers: HashMap::new(),
            orphaned_papers: Vec::new(),
            unlinked_code: Vec::new(),
            invalid_links: Vec::new(),
            repo_path: None,
            report_rx: None,
            state: TraceabilityScanState::Idle,
        };
        tab.refresh();
        tab
    }
}

#[allow(missing_docs, dead_code)]
impl TraceabilityTab {
    #[allow(missing_docs)]
    pub fn retry(&mut self) {
        self.refresh();
    }

    #[allow(missing_docs)]
    pub fn state(&self) -> &TraceabilityScanState {
        &self.state
    }

    #[allow(missing_docs)]
    pub fn error_message(&self) -> Option<&str> {
        self.state.error()
    }

    #[allow(missing_docs)]
    pub fn paper_url(name: &str) -> Option<String> {
        let clean = name.strip_prefix("papers/").unwrap_or(name);
        if let Some(stem) = clean.strip_suffix(".tex") {
            Some(format!(
                "https://fderuiter.github.io/OxidizeMath/{}.pdf",
                stem
            ))
        } else if clean.ends_with(".pdf") {
            Some(format!("https://fderuiter.github.io/OxidizeMath/{}", clean))
        } else if !clean.starts_with("spec:") && !clean.starts_with("registry:") {
            Some(format!(
                "https://fderuiter.github.io/OxidizeMath/{}.pdf",
                clean
            ))
        } else {
            None
        }
    }

    #[allow(missing_docs)]
    pub fn online_document_url(&self, name: &str) -> Option<String> {
        Self::paper_url(name)
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn refresh(&mut self) {
        self.papers.clear();
        self.orphaned_papers.clear();
        self.unlinked_code.clear();
        self.invalid_links.clear();
        self.state = TraceabilityScanState::Scanning;

        let base_dir = match &self.repo_path {
            Some(p) => format!("{}/", p),
            None => "".to_string(),
        };

        // Scan the standard directories
        let code_dirs = [
            format!("{}math_explorer/src", base_dir),
            format!("{}math_explorer_gui/src/tabs", base_dir),
        ];

        let papers_dir = format!("{}papers", base_dir);
        let crate_base = format!("{}crates", base_dir);

        let (tx, rx) = std::sync::mpsc::channel();
        self.report_rx = Some(rx);

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            futures::executor::block_on(async move {
                let vfs = oxidize_core::vfs::DefaultVfs;
                let engine = oxidize_core::traceability::TraceabilityEngine::new(vfs);
                
                let mut crate_dirs = Vec::new();
                if let Ok(crates) = engine.vfs.list_dir(&crate_base).await {
                    for crate_name in crates {
                        crate_dirs.push(format!("{}/{}/src", crate_base, crate_name));
                    }
                }

                let mut all_dirs: Vec<&str> = code_dirs.iter().map(|s| s.as_str()).collect();
                for dir in &crate_dirs {
                    all_dirs.push(dir.as_str());
                }

                let res = match engine.scan_repository(&all_dirs, &papers_dir, false).await {
                    Ok(report) => Ok(report),
                    Err(e) => Err(e.to_string()),
                };
                let _ = tx.send(res);
            });
        });

        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            let vfs = oxidize_core::vfs::WasmVfs;
            let engine = oxidize_core::traceability::TraceabilityEngine::new(vfs);
            
            let mut crate_dirs = Vec::new();
            if let Ok(crates) = engine.vfs.list_dir(&crate_base).await {
                for crate_name in crates {
                    crate_dirs.push(format!("{}/{}/src", crate_base, crate_name));
                }
            }

            let mut all_dirs: Vec<&str> = code_dirs.iter().map(|s| s.as_str()).collect();
            for dir in &crate_dirs {
                all_dirs.push(dir.as_str());
            }

            let res = match engine.scan_repository(&all_dirs, &papers_dir, false).await {
                Ok(report) => Ok(report),
                Err(e) => Err(e.to_string()),
            };
            let _ = tx.send(res);
        });
    }
}

impl ExplorerTab for TraceabilityTab {
    fn name(&self) -> &'static str {
        "Traceability Portal"
    }

    fn save_state(&self) -> Option<String> {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct TraceabilityState {
            repo_path: Option<String>,
        }

        serde_json::to_string(&TraceabilityState {
            repo_path: self.repo_path.clone(),
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct TraceabilityState {
            repo_path: Option<String>,
        }

        if let Ok(s) = serde_json::from_str::<TraceabilityState>(state) {
            self.repo_path = s.repo_path;
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(rx) = &self.report_rx {
            match rx.try_recv() {
                Ok(Ok(report)) => {
                    self.orphaned_papers = report.orphaned_papers;
                    self.unlinked_code = report.unlinked_code;
                    self.invalid_links = report.invalid_links;

                    for (paper_name, linked_code) in report.paper_coverage {
                        self.papers.insert(
                            paper_name.clone(),
                            PaperStatus {
                                name: paper_name,
                                linked_code,
                            },
                        );
                    }
                    self.state = TraceabilityScanState::Ready;
                    self.report_rx = None;
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

        if self.state.is_scanning() {
            ctx.request_repaint();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Theory-to-Code Traceability Portal");
                if ui.button("Refresh").clicked() {
                    self.refresh();
                }
                
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if ui.button("Select Codebase Directory...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                            self.repo_path = Some(path.display().to_string());
                            self.refresh();
                        }
                    }
                }
                
                if let Some(path) = &self.repo_path {
                    ui.label(format!("Scanning: {}", path));
                } else {
                    ui.label("Scanning: default workspace");
                }
            });
            ui.separator();

            let mut retry_clicked = false;
            let mut new_path_picked = None;
            match &self.state {
                TraceabilityScanState::Scanning => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Scanning repository via Virtual File System...");
                    });
                    return;
                }
                TraceabilityScanState::Failed(err) => {
                    let err_color = crate::app::SemanticPalette::from_visuals(ui.visuals()).error;
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
                                {
                                    if ui.button("Select Codebase Directory...").clicked() {
                                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                            new_path_picked = Some(path.display().to_string());
                                        }
                                    }
                                }
                            });
                        });
                    });
                    if let Some(path) = new_path_picked {
                        self.repo_path = Some(path);
                        self.refresh();
                    } else if retry_clicked {
                        self.refresh();
                    }
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
                ui.collapsing("System Health & Alignment", |ui| {
                    if self.orphaned_papers.is_empty()
                        && self.unlinked_code.is_empty()
                        && self.invalid_links.is_empty()
                    {
                        ui.label(
                            egui::RichText::new("100% Parity Achieved").color(palette.success),
                        );
                    } else {
                        ui.label(egui::RichText::new("Partial Alignment").color(palette.warning));
                    }
                    ui.label(format!("Total Papers: {}", self.papers.len()));
                });

                if !self.orphaned_papers.is_empty() {
                    ui.add_space(10.0);
                    ui.heading(
                        egui::RichText::new("Orphaned Papers (No Code)").color(palette.error),
                    );
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
                        egui::RichText::new("Unlinked Code Modules (No Paper)")
                            .color(palette.error),
                    );
                    for c in &self.unlinked_code {
                        ui.label(c);
                    }
                }

                if !self.invalid_links.is_empty() {
                    ui.add_space(10.0);
                    ui.heading(
                        egui::RichText::new("Naming Mismatches / Invalid Cites")
                            .color(palette.error),
                    );
                    for (c, p) in &self.invalid_links {
                        ui.label(format!("{} cites invalid paper: {}", c, p));
                    }
                }

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
                            ui.label(
                                egui::RichText::new("No implementation found").color(palette.error),
                            );
                        } else {
                            for code in &p.linked_code {
                                ui.label(code);
                            }
                        }
                    });
                }
            });
        });
    }
}

#[cfg(test)]
#[path = "traceability/tests.rs"]
mod tests;

// [cite:essay]
