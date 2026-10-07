use crate::tabs::ExplorerTab;
use eframe::egui;
use oxidize_core::vfs::VirtualFileSystem;
use std::collections::HashMap;

pub struct TraceabilityTab {
    papers: HashMap<String, PaperStatus>,
    orphaned_papers: Vec<String>,
    unlinked_code: Vec<String>,
    invalid_links: Vec<(String, String)>, // code_file, paper_name
    repo_path: Option<String>,
    report_rx: Option<std::sync::mpsc::Receiver<oxidize_core::traceability::TraceabilityReport>>,
    is_loading: bool,
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
            is_loading: false,
        };
        tab.refresh();
        tab
    }
}

impl TraceabilityTab {
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn refresh(&mut self) {
        self.papers.clear();
        self.orphaned_papers.clear();
        self.unlinked_code.clear();
        self.invalid_links.clear();
        self.is_loading = true;

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

                if let Ok(report) = engine.scan_repository(&all_dirs, &papers_dir, false).await {
                    let _ = tx.send(report);
                }
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

            if let Ok(report) = engine.scan_repository(&all_dirs, &papers_dir, false).await {
                let _ = tx.send(report);
            }
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
            if let Ok(report) = rx.try_recv() {
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
                self.is_loading = false;
                self.report_rx = None;
            } else if self.is_loading {
                ctx.request_repaint();
            }
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

            if self.is_loading {
                ui.spinner();
                ui.label("Scanning repository via Virtual File System...");
                return;
            }

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.collapsing("System Health & Alignment", |ui| {
                    if self.orphaned_papers.is_empty()
                        && self.unlinked_code.is_empty()
                        && self.invalid_links.is_empty()
                    {
                        ui.label(
                            egui::RichText::new("100% Parity Achieved").color(egui::Color32::GREEN),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new("Partial Alignment").color(egui::Color32::YELLOW),
                        );
                    }
                    ui.label(format!("Total Papers: {}", self.papers.len()));
                });

                if !self.orphaned_papers.is_empty() {
                    ui.add_space(10.0);
                    ui.heading(
                        egui::RichText::new("Orphaned Papers (No Code)").color(egui::Color32::RED),
                    );
                    for p in &self.orphaned_papers {
                        ui.label(p);
                    }
                }

                if !self.unlinked_code.is_empty() {
                    ui.add_space(10.0);
                    ui.heading(
                        egui::RichText::new("Unlinked Code Modules (No Paper)")
                            .color(egui::Color32::RED),
                    );
                    for c in &self.unlinked_code {
                        ui.label(c);
                    }
                }

                if !self.invalid_links.is_empty() {
                    ui.add_space(10.0);
                    ui.heading(
                        egui::RichText::new("Naming Mismatches / Invalid Cites")
                            .color(egui::Color32::RED),
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
                        if p.linked_code.is_empty() {
                            ui.label(
                                egui::RichText::new("No implementation found")
                                    .color(egui::Color32::RED),
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
// [cite:essay]
