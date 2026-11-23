mod analysis;
mod cache;
mod module_tree;
mod runner;

pub use cache::{AstCache, CachedRustFile};
pub use runner::{
    check_and_print_traceability_errors, discover_workspace_code_dirs,
    print_traceability_dashboard, run_workspace_traceability,
    run_workspace_traceability_with_cache,
};

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

    async fn load_cached_optional(
        &self,
        norm_path: &str,
        cache: &mut AstCache,
    ) -> Result<bool, std::io::Error> {
        let key = AstCache::normalize_key(norm_path);
        if cache.get(&key).is_some() {
            return Ok(true);
        }
        if let Some(content) = self.read_optional_file(norm_path).await? {
            cache.insert_source(key, content);
            return Ok(true);
        }
        Ok(false)
    }

    async fn gather_active_files(
        &self,
        code_dirs: &[&str],
        active_files: &mut HashSet<String>,
        cache: &mut AstCache,
    ) -> Result<(), std::io::Error> {
        for dir in code_dirs {
            let normalized = crate::path_utils::normalize_path(dir);
            let mut has_entrypoint = false;
            for entry_name in ["lib.rs", "main.rs", "mod.rs"] {
                let entry = crate::path_utils::join_and_normalize(&normalized, entry_name);
                if self.load_cached_optional(&entry, cache).await? {
                    self.parse_module_tree(&entry, active_files, cache).await?;
                    has_entrypoint = true;
                }
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
        let mut cache = AstCache::new();
        self.scan_repository_with_cache(code_dirs, papers_dir, auto_fix, &mut cache)
            .await
    }

    /// Scans `code_dirs` and `papers_dir` while reusing and populating pre-parsed ASTs in `cache`.
    pub async fn scan_repository_with_cache(
        &self,
        code_dirs: &[&str],
        papers_dir: &str,
        auto_fix: bool,
        cache: &mut AstCache,
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
        self.gather_active_files(code_dirs, &mut active_files, cache)
            .await?;

        let mut code_files: Vec<String> = active_files.into_iter().collect();
        code_files.sort();

        self.process_code_files(
            code_files,
            &registered_modules,
            &registry_links,
            &mut report,
            auto_fix,
            cache,
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
        cache: &mut AstCache,
    ) -> Result<(), std::io::Error> {
        for file in code_files {
            self.process_single_file(
                file,
                registered_modules,
                registry_links,
                report,
                auto_fix,
                cache,
            )
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
        cache: &mut AstCache,
    ) -> Result<(), std::io::Error> {
        let norm_file = crate::path_utils::normalize_path(&file);
        let key = AstCache::normalize_key(&norm_file);
        if cache.get(&key).is_none() {
            let content = self.vfs.read_to_string(&norm_file).await?;
            cache.insert_source(key.clone(), content);
        }

        report.scanned_files += 1;
        let is_module = norm_file.ends_with("mod.rs")
            || norm_file.contains("/tabs/")
            || norm_file.ends_with("lib.rs");

        let citations = cache
            .get(&key)
            .map(|entry| Self::extract_citations(&entry.content))
            .unwrap_or_default();

        let final_citations = if auto_fix {
            let mut updated = cache
                .get(&key)
                .map(|entry| entry.content.clone())
                .unwrap_or_default();
            let (file_modified, resolved) = analysis::fix_and_collect_citations(
                &norm_file,
                &mut updated,
                citations,
                registered_modules,
                registry_links,
                report,
                true,
            );
            if file_modified {
                self.vfs
                    .write_to_file(&norm_file, updated.as_bytes())
                    .await?;
                cache.insert_source(key.clone(), updated);
            }
            resolved
        } else {
            analysis::collect_valid_citations(&norm_file, citations, registered_modules, report)
        };

        let ast = cache.get(&key).and_then(|entry| entry.ast.as_ref());
        analysis::analyze_parsed_ast(
            &norm_file,
            ast,
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
        cache: &mut AstCache,
    ) -> Result<(), std::io::Error> {
        let norm_file_path = crate::path_utils::normalize_path(file_path);
        if active_files.contains(&norm_file_path) {
            return Ok(());
        }
        let key = AstCache::normalize_key(&norm_file_path);
        if cache.get(&key).is_none() {
            let content = self.vfs.read_to_string(&norm_file_path).await?;
            cache.insert_source(key.clone(), content);
        }
        active_files.insert(norm_file_path.clone());

        let submodules =
            module_tree::collect_external_submodules(cache.get(&key).and_then(|e| e.ast.as_ref()));
        let parent_dir = module_tree::module_parent_dir(&norm_file_path);
        let dir_path = module_tree::module_child_dir(&norm_file_path);
        for submodule in submodules {
            if let Some(rel_path) = submodule.custom_path {
                let custom = crate::path_utils::join_and_normalize(
                    &parent_dir,
                    rel_path.trim_start_matches("./"),
                );
                if self.load_cached_optional(&custom, cache).await? {
                    self.parse_module_tree(&custom, active_files, cache).await?;
                }
                continue;
            }
            let path1 =
                crate::path_utils::join_and_normalize(&dir_path, format!("{}.rs", submodule.ident));
            let path2 = crate::path_utils::join_and_normalize(
                &dir_path,
                format!("{}/mod.rs", submodule.ident),
            );

            if self.load_cached_optional(&path1, cache).await? {
                self.parse_module_tree(&path1, active_files, cache).await?;
            } else if self.load_cached_optional(&path2, cache).await? {
                self.parse_module_tree(&path2, active_files, cache).await?;
            }
        }
        Ok(())
    }
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
