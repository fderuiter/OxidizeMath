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

    fn has_traversal_sequence(raw: &str) -> bool {
        raw.contains('\0')
            || raw.split(['/', '\\']).any(|seg| seg == "..")
            || std::path::Path::new(raw)
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn has_workspace_header(manifest: &str) -> bool {
        manifest.lines().any(|line| {
            let h = line.split('#').next().unwrap_or("").trim();
            let k = h.strip_prefix('[').and_then(|s| s.strip_suffix(']'));
            k.is_some_and(|s| {
                let key = s.trim();
                !s.starts_with('[')
                    && !s.ends_with(']')
                    && (key == "workspace"
                        || key
                            .strip_prefix("workspace")
                            .is_some_and(|rest| rest.trim_start().starts_with('.')))
            })
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn canonical_workspace_root() -> Option<std::path::PathBuf> {
        let cwd = std::env::current_dir().ok()?.canonicalize().ok()?;
        cwd.ancestors()
            .take(64)
            .find(|dir| {
                std::fs::read_to_string(dir.join("Cargo.toml"))
                    .is_ok_and(|m| Self::has_workspace_header(&m))
            })
            .map(std::path::Path::to_path_buf)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn validate_repo_path(raw: &str) -> Option<String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() || Self::has_traversal_sequence(trimmed) {
            return None;
        }
        let canonical_root = Self::canonical_workspace_root()?;
        let candidate = std::path::Path::new(trimmed);
        let resolved = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            canonical_root.join(candidate)
        };
        let canonical_candidate = resolved.canonicalize().ok()?;
        if !canonical_candidate.is_dir() || !canonical_candidate.starts_with(&canonical_root) {
            return None;
        }
        let normalized = oxidize_core::path_utils::normalize_path(&canonical_candidate);
        (!normalized.is_empty()).then_some(normalized)
    }

    #[cfg(target_arch = "wasm32")]
    fn validate_repo_path(raw: &str) -> Option<String> {
        let _ = Self::has_traversal_sequence(raw);
        None
    }

    pub fn refresh(&mut self) {
        self.papers.clear();
        self.orphaned_papers.clear();
        self.unlinked_code.clear();
        self.invalid_links.clear();
        self.repo_path = self.repo_path.as_deref().and_then(Self::validate_repo_path);
        self.state = TraceabilityScanState::Scanning;

        let base_dir = match &self.repo_path {
            Some(p) => format!("{}/", p),
            None => String::new(),
        };

        let (tx, rx) = std::sync::mpsc::channel();
        self.report_rx = Some(rx);

        #[cfg(all(test, not(target_arch = "wasm32")))]
        if base_dir.is_empty() {
            static DEFAULT_SCAN_CACHE: std::sync::OnceLock<
                Result<oxidize_core::traceability::TraceabilityReport, String>,
            > = std::sync::OnceLock::new();
            let cached = DEFAULT_SCAN_CACHE
                .get_or_init(|| {
                    futures::executor::block_on(execute_workspace_scan(
                        oxidize_core::vfs::DefaultVfs,
                        String::new(),
                    ))
                })
                .clone();
            let _ = tx.send(cached);
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            let res = futures::executor::block_on(execute_workspace_scan(
                oxidize_core::vfs::DefaultVfs,
                base_dir,
            ));
            let _ = tx.send(res);
        });

        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            let res = execute_workspace_scan(oxidize_core::vfs::WasmVfs, base_dir).await;
            let _ = tx.send(res);
        });
    }
}

async fn execute_workspace_scan<V: VirtualFileSystem>(
    vfs: V,
    base_dir: String,
) -> Result<oxidize_core::traceability::TraceabilityReport, String> {
    let code_dirs = [
        format!("{base_dir}math_explorer/src"),
        format!("{base_dir}math_explorer_gui/src/tabs"),
    ];
    let papers_dir = format!("{base_dir}papers");
    let crate_base = format!("{base_dir}crates");
    let engine = oxidize_core::traceability::TraceabilityEngine::new(vfs);

    let mut crate_dirs = Vec::new();
    if let Ok(crates) = engine.vfs.list_dir(&crate_base).await {
        for crate_name in crates {
            crate_dirs.push(format!("{crate_base}/{crate_name}/src"));
        }
    }

    let mut all_dirs: Vec<&str> = code_dirs.iter().map(String::as_str).collect();
    all_dirs.extend(crate_dirs.iter().map(String::as_str));
    engine
        .scan_repository(&all_dirs, &papers_dir, false)
        .await
        .map_err(|e| e.to_string())
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
        if let Ok(s) = serde_json::from_str::<TraceabilityState>(state) {
            self.repo_path = s.repo_path.as_deref().and_then(Self::validate_repo_path);
        } else {
            self.repo_path = None;
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
                if ui.button("Select Codebase Directory...").clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        self.repo_path = Self::validate_repo_path(&path.to_string_lossy());
                        self.refresh();
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
            #[cfg(not(target_arch = "wasm32"))]
            let mut new_path_picked: Option<String> = None;
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
                                if ui.button("Select Codebase Directory...").clicked() {
                                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                        new_path_picked = Some(path.display().to_string());
                                    }
                                }
                            });
                        });
                    });
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(path) = new_path_picked {
                        self.repo_path = Self::validate_repo_path(&path);
                        self.refresh();
                        return;
                    }
                    if retry_clicked {
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
