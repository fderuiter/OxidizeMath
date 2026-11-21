mod analysis;

use crate::vfs::VirtualFileSystem;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[allow(missing_docs)]
pub struct TraceabilityReport {
    #[allow(missing_docs)]
    pub scanned_files: usize,
    #[allow(missing_docs)]
    pub invalid_links: Vec<(String, String)>,
    #[allow(missing_docs)]
    pub orphaned_papers: Vec<String>,
    #[allow(missing_docs)]
    pub unlinked_code: Vec<String>,
    #[allow(missing_docs)]
    pub unverified_modules: Vec<String>,
    #[allow(missing_docs)]
    pub paper_coverage: HashMap<String, Vec<String>>,
    #[allow(missing_docs)]
    pub total_funcs: usize,
    #[allow(missing_docs)]
    pub total_asserts: usize,
    #[allow(missing_docs)]
    pub verified_funcs: usize,
    #[allow(missing_docs)]
    pub verified_asserts: usize,
    #[allow(missing_docs)]
    pub invalid_tiers: Vec<String>,
    #[allow(missing_docs)]
    pub vacuous_bypasses: Vec<String>,
    #[allow(missing_docs)]
    pub semantic_integrity_status: HashMap<String, String>,
}

#[allow(missing_docs)]
pub struct TraceabilityEngine<V: VirtualFileSystem> {
    #[allow(missing_docs)]
    pub vfs: V,
}

impl<V: VirtualFileSystem> TraceabilityEngine<V> {
    #[allow(missing_docs)]
    pub fn new(vfs: V) -> Self {
        Self { vfs }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(missing_docs)]
    pub fn verify_module_registered(module_name: &str) -> bool {
        let vfs = crate::vfs::DefaultVfs;
        let mut registry_content =
            futures::executor::block_on(vfs.read_to_string("traceability.toml"))
                .unwrap_or_default();
        if registry_content.is_empty() {
            registry_content =
                futures::executor::block_on(vfs.read_to_string("../../traceability.toml"))
                    .unwrap_or_default();
        }
        analysis::is_module_in_registry_toml(&registry_content, module_name)
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    #[allow(missing_docs)]
    pub fn verify_module_registered(module_name: &str) -> bool {
        const EMBEDDED_REGISTRY: &str = include_str!("../../../traceability.toml");
        analysis::is_module_in_registry_toml(EMBEDDED_REGISTRY, module_name)
    }

    /// Extract citations matching the regex `\[cite:([a-zA-Z0-9_.-]+)\]` manually to avoid heavy regex compilation.
    pub fn extract_citations(content: &str) -> Vec<String> {
        let mut cites = Vec::new();
        let mut search_idx = 0;
        while let Some(start) = content[search_idx..].find("[cite:") {
            let real_start = search_idx + start + 6;
            if let Some(end) = content[real_start..].find(']') {
                let paper_name = content[real_start..real_start + end].to_string();
                if paper_name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
                    && !paper_name.is_empty()
                {
                    cites.push(paper_name);
                }
                search_idx = real_start + end + 1;
            } else {
                break;
            }
        }
        cites
    }

    async fn scan_papers_dir(
        &self,
        papers_dir: &str,
        valid_papers: &mut HashSet<String>,
        report: &mut TraceabilityReport,
    ) -> Result<(), std::io::Error> {
        let normalized_papers_dir = crate::path_utils::normalize_path(papers_dir);
        let entries = self.vfs.list_dir(&normalized_papers_dir).await?;
        for name in entries {
            if name.ends_with(".tex") {
                valid_papers.insert(name.clone());
                report.paper_coverage.insert(name, Vec::new());
            }
        }
        Ok(())
    }

    async fn read_registry_table(&self, papers_dir: &str) -> Result<toml::Table, std::io::Error> {
        let norm_papers = crate::path_utils::normalize_path(papers_dir);
        let mut candidates = Vec::new();
        if let Some((parent, _)) = norm_papers.trim_end_matches('/').rsplit_once('/')
            && !parent.is_empty()
        {
            candidates.push(format!("{parent}/traceability.toml"));
        }
        candidates.push("traceability.toml".to_string());
        #[cfg(not(target_arch = "wasm32"))]
        {
            candidates.push("../traceability.toml".to_string());
            candidates.push("../../traceability.toml".to_string());
        }
        candidates.dedup();

        let mut last_err = None;
        for candidate in candidates {
            match self.vfs.read_to_string(&candidate).await {
                Ok(content) => {
                    return content.parse::<toml::Table>().map_err(|err| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Invalid TOML in {candidate}: {err}"),
                        )
                    });
                }
                Err(err) => last_err = Some(err),
            }
        }
        Err(last_err.unwrap_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "traceability.toml not found")
        }))
    }

    async fn read_optional_file(&self, path: &str) -> Result<Option<String>, std::io::Error> {
        match self.vfs.read_to_string(path).await {
            Ok(content) => Ok(Some(content)),
            Err(err)
                if matches!(
                    err.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                ) =>
            {
                Ok(None)
            }
            Err(err) => Err(err),
        }
    }

    async fn gather_active_files(
        &self,
        code_dirs: &[&str],
        active_files: &mut HashSet<String>,
    ) -> Result<(), std::io::Error> {
        for dir in code_dirs {
            let normalized = crate::path_utils::normalize_path(dir);
            let lib = crate::path_utils::join_and_normalize(&normalized, "lib.rs");
            let main = crate::path_utils::join_and_normalize(&normalized, "main.rs");
            let mut has_entrypoint = false;

            if self.read_optional_file(&lib).await?.is_some() {
                self.parse_module_tree(&lib, active_files).await?;
                has_entrypoint = true;
            }
            if self.read_optional_file(&main).await?.is_some() {
                self.parse_module_tree(&main, active_files).await?;
                has_entrypoint = true;
            }
            if !has_entrypoint {
                self.gather_files_fallback(&normalized, active_files)
                    .await?;
            }
        }
        Ok(())
    }

    async fn gather_files_fallback(
        &self,
        start_dir: &str,
        active_files: &mut HashSet<String>,
    ) -> Result<(), std::io::Error> {
        let mut stack = vec![start_dir.to_string()];
        while let Some(current_dir) = stack.pop() {
            let entries = match self.vfs.list_dir(&current_dir).await {
                Ok(entries) => entries,
                Err(err)
                    if matches!(
                        err.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                    ) =>
                {
                    continue;
                }
                Err(err) => return Err(err),
            };
            for name in entries {
                let path = crate::path_utils::join_and_normalize(&current_dir, &name);
                if name.ends_with(".rs") {
                    active_files.insert(path);
                } else if !name.contains('.') {
                    stack.push(path);
                }
            }
        }
        Ok(())
    }

    #[allow(missing_docs)]
    pub async fn scan_repository(
        &self,
        code_dirs: &[&str],
        papers_dir: &str,
        auto_fix: bool,
    ) -> Result<TraceabilityReport, std::io::Error> {
        let mut valid_papers = HashSet::new();
        let mut report = TraceabilityReport::default();

        self.scan_papers_dir(papers_dir, &mut valid_papers, &mut report)
            .await?;

        let registry_table = self.read_registry_table(papers_dir).await?;
        let mut registered_modules = HashSet::new();
        let mut registry_links = HashMap::new();
        analysis::apply_registry_table(
            &registry_table,
            &valid_papers,
            &mut registered_modules,
            &mut registry_links,
            &mut report,
        );

        let mut active_files = HashSet::new();
        self.gather_active_files(code_dirs, &mut active_files)
            .await?;

        let mut code_files: Vec<String> = active_files.into_iter().collect();
        code_files.sort();

        self.process_code_files(
            code_files,
            &registered_modules,
            &registry_links,
            &mut report,
            auto_fix,
        )
        .await?;

        analysis::populate_paper_semantic_status(&valid_papers, &registry_links, &mut report);
        analysis::finalize_report_orphans(&mut report);

        Ok(report)
    }

    async fn process_code_files(
        &self,
        code_files: Vec<String>,
        registered_modules: &HashSet<String>,
        registry_links: &HashMap<String, String>,
        report: &mut TraceabilityReport,
        auto_fix: bool,
    ) -> Result<(), std::io::Error> {
        for file in code_files {
            self.process_single_file(file, registered_modules, registry_links, report, auto_fix)
                .await?;
        }
        Ok(())
    }

    async fn process_single_file(
        &self,
        file: String,
        registered_modules: &HashSet<String>,
        registry_links: &HashMap<String, String>,
        report: &mut TraceabilityReport,
        auto_fix: bool,
    ) -> Result<(), std::io::Error> {
        let norm_file = crate::path_utils::normalize_path(&file);
        let mut content = self.vfs.read_to_string(&norm_file).await?;
        report.scanned_files += 1;
        let is_module = norm_file.ends_with("mod.rs")
            || norm_file.contains("/tabs/")
            || norm_file.ends_with("lib.rs");

        let citations = Self::extract_citations(&content);
        let (file_modified, final_citations) = analysis::fix_and_collect_citations(
            &norm_file,
            &mut content,
            citations,
            registered_modules,
            registry_links,
            report,
            auto_fix,
        );

        if file_modified {
            self.vfs
                .write_to_file(&norm_file, content.as_bytes())
                .await?;
        }

        analysis::analyze_file_ast(
            &norm_file,
            &content,
            is_module,
            &final_citations,
            registered_modules,
            report,
        );
        Ok(())
    }

    #[allow(clippy::double_must_use)]
    #[async_recursion::async_recursion(?Send)]
    async fn parse_module_tree(
        &self,
        file_path: &str,
        active_files: &mut HashSet<String>,
    ) -> Result<(), std::io::Error> {
        let norm_file_path = crate::path_utils::normalize_path(file_path);
        if active_files.contains(&norm_file_path) {
            return Ok(());
        }
        let content = self.vfs.read_to_string(&norm_file_path).await?;
        active_files.insert(norm_file_path.clone());
        let Ok(ast) = syn::parse_file(&content) else {
            return Ok(());
        };
        let mut visitor = crate::ast_visitor::AstVisitor::new();
        syn::visit::Visit::visit_file(&mut visitor, &ast);

        let dir_path = module_child_dir(&norm_file_path);
        for submodule in visitor.active_submodules {
            let path1 = crate::path_utils::join_and_normalize(&dir_path, format!("{submodule}.rs"));
            let path2 =
                crate::path_utils::join_and_normalize(&dir_path, format!("{submodule}/mod.rs"));

            if self.read_optional_file(&path1).await?.is_some() {
                self.parse_module_tree(&path1, active_files).await?;
            } else if self.read_optional_file(&path2).await?.is_some() {
                self.parse_module_tree(&path2, active_files).await?;
            }
        }
        Ok(())
    }
}

fn module_child_dir(norm_file_path: &str) -> String {
    let mut dir_parts: Vec<&str> = norm_file_path.split('/').collect();
    let is_mod_rs = norm_file_path.ends_with("mod.rs")
        || norm_file_path.ends_with("lib.rs")
        || norm_file_path.ends_with("main.rs");
    if is_mod_rs {
        dir_parts.pop();
    } else if let Some(stem) = dir_parts.pop().map(|s| s.strip_suffix(".rs").unwrap_or(s)) {
        dir_parts.push(stem);
    }
    dir_parts.join("/")
}

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use super::TraceabilityEngine;
    use crate::vfs::DefaultVfs;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_traceability_verifies_embedded_registry_modules() {
        assert!(TraceabilityEngine::<DefaultVfs>::verify_module_registered(
            "quantum"
        ));
        assert!(TraceabilityEngine::<DefaultVfs>::verify_module_registered(
            "pure_math"
        ));
        assert!(!TraceabilityEngine::<DefaultVfs>::verify_module_registered(
            "unregistered_module_xyz"
        ));
    }
}
