use crate::vfs::VirtualFileSystem;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
        if let Ok(value) = registry_content.parse::<toml::Table>()
            && let Some(links) = value.get("links").and_then(|v| v.as_table())
            && let Some(paper) = links.get(module_name)
        {
            return paper.as_str().is_some();
        }
        false
    }

    #[cfg(target_arch = "wasm32")]
    pub fn verify_module_registered(_module_name: &str) -> bool {
        true
    }

    async fn verify_and_link_registry(
        &self,
        valid_papers: &HashSet<String>,
        report: &mut TraceabilityReport,
    ) {
        if let Ok(registry_content) = self.vfs.read_to_string("traceability.toml").await
            && let Ok(value) = registry_content.parse::<toml::Table>()
            && let Some(links) = value.get("links").and_then(|v| v.as_table())
        {
            for (module_name, paper_val) in links {
                if let Some(paper_name) = paper_val.as_str() {
                    let paper_name_string = paper_name.to_string();
                    if valid_papers.contains(&paper_name_string)
                        || paper_name_string.starts_with("spec:")
                        || paper_name_string.starts_with("registry:")
                    {
                        if !valid_papers.contains(&paper_name_string) {
                            report
                                .paper_coverage
                                .insert(paper_name_string.clone(), Vec::new());
                        }
                        if let Some(linked) = report.paper_coverage.get_mut(&paper_name_string) {
                            linked.push(module_name.clone());
                        }
                    } else {
                        report
                            .invalid_links
                            .push((module_name.clone(), paper_name_string));
                    }
                } else {
                    report
                        .invalid_links
                        .push((module_name.clone(), "INVALID_FORMAT".to_string()));
                }
            }
        }
    }

    /// Extract citations matching the regex `\[cite:([a-zA-Z0-9_.-]+)\]` manually to avoid heavy regex compilation.
    pub fn extract_citations(content: &str) -> Vec<String> {
        let mut cites = Vec::new();
        let mut search_idx = 0;
        while let Some(start) = content[search_idx..].find("[cite:") {
            let real_start = search_idx + start + 6;
            if let Some(end) = content[real_start..].find("]") {
                let paper_name = content[real_start..real_start + end].to_string();
                // basic regex validation: [a-zA-Z0-9_.-]+
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
    ) {
        let normalized_papers_dir = crate::path_utils::normalize_path(papers_dir);
        if let Ok(entries) = self.vfs.list_dir(&normalized_papers_dir).await {
            for name in entries {
                if name.ends_with(".tex") {
                    let paper_name = name.clone();
                    valid_papers.insert(paper_name.clone());
                    report.paper_coverage.insert(paper_name, Vec::new());
                }
            }
        }
    }

    async fn load_registry_modules(
        &self,
        registered_modules: &mut HashSet<String>,
        registry_links: &mut HashMap<String, String>,
    ) {
        if let Ok(registry_content) = self.vfs.read_to_string("traceability.toml").await
            && let Ok(value) = registry_content.parse::<toml::Table>()
            && let Some(links) = value.get("links").and_then(|v| v.as_table())
        {
            for (module_name, paper_val) in links {
                registered_modules.insert(module_name.clone());
                if let Some(paper_str) = paper_val.as_str() {
                    registry_links.insert(module_name.clone(), paper_str.to_string());
                }
            }
        }
    }

    async fn gather_active_files(&self, code_dirs: &[&str], active_files: &mut HashSet<String>) {
        for dir in code_dirs {
            let normalized = crate::path_utils::normalize_path(dir);
            let lib = crate::path_utils::join_and_normalize(&normalized, "lib.rs");
            let main = crate::path_utils::join_and_normalize(&normalized, "main.rs");
            let mut has_entrypoint = false;

            if self.vfs.read_to_string(&lib).await.is_ok() {
                self.parse_module_tree(&lib, active_files).await;
                has_entrypoint = true;
            }
            if self.vfs.read_to_string(&main).await.is_ok() {
                self.parse_module_tree(&main, active_files).await;
                has_entrypoint = true;
            }

            if !has_entrypoint {
                self.gather_files_fallback(&normalized, active_files).await;
            }
        }
    }

    async fn gather_files_fallback(&self, start_dir: &str, active_files: &mut HashSet<String>) {
        let mut stack = vec![start_dir.to_string()];
        while let Some(current_dir) = stack.pop() {
            if let Ok(entries) = self.vfs.list_dir(&current_dir).await {
                for name in entries {
                    let path = crate::path_utils::join_and_normalize(&current_dir, &name);
                    if name.ends_with(".rs") {
                        active_files.insert(path);
                    } else if !name.contains('.') {
                        stack.push(path);
                    }
                }
            }
        }
    }

    fn finalize_report_orphans(report: &mut TraceabilityReport) {
        for (name, linked_code) in &report.paper_coverage {
            if linked_code.is_empty() {
                report.orphaned_papers.push(name.clone());
            }
        }
        report.orphaned_papers.sort();
        report.unlinked_code.sort();
        report.invalid_links.sort();
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
            .await;

        self.verify_and_link_registry(&valid_papers, &mut report)
            .await;

        let mut registered_modules = HashSet::new();
        let mut registry_links = HashMap::new();
        self.load_registry_modules(&mut registered_modules, &mut registry_links)
            .await;

        let mut active_files = HashSet::new();
        self.gather_active_files(code_dirs, &mut active_files).await;

        let mut code_files: Vec<String> = active_files.into_iter().collect();
        code_files.sort();

        self.process_code_files(
            code_files,
            &registered_modules,
            &registry_links,
            &mut report,
            auto_fix,
        )
        .await;

        for paper in &valid_papers {
            let model_name = paper.strip_suffix(".tex").unwrap_or(paper);
            if !report.semantic_integrity_status.contains_key(model_name) {
                let mut is_verified = false;
                for (module, target_paper) in &registry_links {
                    let target_clean = target_paper.strip_suffix(".tex").unwrap_or(target_paper);
                    if target_paper == paper || target_clean == model_name {
                        if let Some(status) = report.semantic_integrity_status.get(module) {
                            if status == "Verified" {
                                is_verified = true;
                                break;
                            }
                        }
                    }
                }
                if is_verified {
                    report
                        .semantic_integrity_status
                        .insert(model_name.to_string(), "Verified".to_string());
                } else {
                    report
                        .semantic_integrity_status
                        .insert(model_name.to_string(), "Unverified".to_string());
                }
            }
        }

        Self::finalize_report_orphans(&mut report);

        Ok(report)
    }

    async fn process_code_files(
        &self,
        code_files: Vec<String>,
        registered_modules: &HashSet<String>,
        registry_links: &HashMap<String, String>,
        report: &mut TraceabilityReport,
        auto_fix: bool,
    ) {
        for file in code_files {
            self.process_single_file(file, registered_modules, registry_links, report, auto_fix)
                .await;
        }
    }

    async fn process_single_file(
        &self,
        file: String,
        registered_modules: &HashSet<String>,
        registry_links: &HashMap<String, String>,
        report: &mut TraceabilityReport,
        auto_fix: bool,
    ) {
        report.scanned_files += 1;
        let is_module =
            file.ends_with("mod.rs") || file.contains("/tabs/") || file.ends_with("lib.rs");

        if let Ok(mut content) = self.vfs.read_to_string(&file).await {
            let citations = Self::extract_citations(&content);
            let (file_modified, final_citations) = Self::fix_and_collect_citations(
                &file,
                &mut content,
                citations,
                registered_modules,
                registry_links,
                report,
                auto_fix,
            );

            if file_modified {
                let _ = self.vfs.write_to_file(&file, content.as_bytes()).await;
            }

            Self::analyze_file_ast(
                &file,
                &content,
                is_module,
                &final_citations,
                registered_modules,
                report,
            );
        }
    }

    fn fix_and_collect_citations(
        file: &str,
        content: &mut String,
        citations: Vec<String>,
        registered_modules: &HashSet<String>,
        registry_links: &HashMap<String, String>,
        report: &mut TraceabilityReport,
        auto_fix: bool,
    ) -> (bool, Vec<String>) {
        let mut file_modified = false;
        let mut final_citations = Vec::new();

        for cite in citations {
            if !registered_modules.contains(&cite) {
                if auto_fix {
                    let mut best_key = None;
                    let mut sorted_registry: Vec<_> = registry_links.iter().collect();
                    sorted_registry.sort_by_key(|(k, _)| *k);

                    for (key, val) in sorted_registry {
                        let base_val = val.replace(".tex", "").replace("spec:", "");
                        if base_val == cite || val == &cite {
                            best_key = Some(key.clone());
                            if file.contains(key) {
                                break;
                            }
                        }
                    }
                    if let Some(key) = best_key {
                        let old_tag = format!("[cite:{}]", cite);
                        let new_tag = format!("[cite:{}]", key);
                        *content = content.replace(&old_tag, &new_tag);
                        file_modified = true;
                        final_citations.push(key);
                    } else {
                        report.invalid_links.push((file.to_string(), cite.clone()));
                    }
                } else {
                    report.invalid_links.push((file.to_string(), cite.clone()));
                }
            } else {
                final_citations.push(cite.clone());
            }
        }

        (file_modified, final_citations)
    }

    fn analyze_file_ast(
        file: &str,
        content: &str,
        is_module: bool,
        final_citations: &[String],
        registered_modules: &HashSet<String>,
        report: &mut TraceabilityReport,
    ) {
        if let Ok(ast) = syn::parse_file(content) {
            let mut visitor = crate::ast_visitor::AstVisitor::new();
            visitor.verified_modules.extend(final_citations.to_vec());
            syn::visit::Visit::visit_file(&mut visitor, &ast);

            report.total_funcs += visitor.total_funcs;
            report.total_asserts += visitor.total_asserts;
            report.verified_funcs += visitor.verified_funcs;
            report.verified_asserts += visitor.verified_asserts;

            let mut all_citations = final_citations.to_vec();
            all_citations.extend(visitor.verified_modules.clone());
            all_citations.sort();
            all_citations.dedup();

            for cite in all_citations {
                if visitor.semantic_integrity_funcs > 0 || visitor.verified_funcs > 0 {
                    report
                        .semantic_integrity_status
                        .insert(cite.clone(), "Verified".to_string());
                } else if !report.semantic_integrity_status.contains_key(&cite) {
                    report
                        .semantic_integrity_status
                        .insert(cite.clone(), "Unverified".to_string());
                }
            }

            if visitor.has_vacuous_bypass && file.contains("domain_ai") {
                report.vacuous_bypasses.push(file.to_string());
            }

            for (module, tier) in &visitor.module_tiers {
                if file.contains("pure_math") && tier != "Deterministic" {
                    report
                        .invalid_tiers
                        .push(format!("{} in {} used tier {}", module, file, tier));
                }
            }

            if is_module {
                for module in &visitor.verified_modules {
                    if !registered_modules.contains(module) {
                        report.unlinked_code.push(file.to_string());
                    }
                }
            }

            if visitor.total_funcs > 0
                && visitor.verified_funcs == 0
                && visitor.verified_modules.is_empty()
                && !visitor.opted_out
            {
                report.unverified_modules.push(file.to_string());
            }
        }
    }

    #[allow(clippy::double_must_use)]
    #[async_recursion::async_recursion(?Send)]
    async fn parse_module_tree(&self, file_path: &str, active_files: &mut HashSet<String>) {
        if active_files.contains(file_path) {
            return;
        }
        if let Ok(content) = self.vfs.read_to_string(file_path).await {
            active_files.insert(file_path.to_string());
            if let Ok(ast) = syn::parse_file(&content) {
                let mut visitor = crate::ast_visitor::AstVisitor::new();
                syn::visit::Visit::visit_file(&mut visitor, &ast);

                let mut dir_parts: Vec<&str> = file_path.split('/').collect();
                let is_mod_rs = file_path.ends_with("mod.rs")
                    || file_path.ends_with("lib.rs")
                    || file_path.ends_with("main.rs");
                if is_mod_rs {
                    dir_parts.pop();
                } else if let Some(stem) =
                    dir_parts.pop().map(|s| s.strip_suffix(".rs").unwrap_or(s))
                {
                    dir_parts.push(stem);
                }

                let dir_path = dir_parts.join("/");

                for submodule in visitor.active_submodules {
                    let path1 = crate::path_utils::join_and_normalize(
                        &dir_path,
                        format!("{}.rs", submodule),
                    );
                    let path2 = crate::path_utils::join_and_normalize(
                        &dir_path,
                        format!("{}/mod.rs", submodule),
                    );

                    if self.vfs.read_to_string(&path1).await.is_ok() {
                        self.parse_module_tree(&path1, active_files).await;
                    } else if self.vfs.read_to_string(&path2).await.is_ok() {
                        self.parse_module_tree(&path2, active_files).await;
                    }
                }
            }
        }
    }
}
