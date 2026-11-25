mod analysis;
#[cfg(feature = "ast-analysis")]
mod cache;
#[cfg(feature = "ast-analysis")]
mod debt_floors;
mod manifest;
pub mod maturity;
#[cfg(feature = "ast-analysis")]
mod module_tree;
pub mod proof_digest;
#[cfg(feature = "ast-analysis")]
pub mod qualification;
#[cfg(feature = "ast-analysis")]
mod runner;

#[cfg(feature = "ast-analysis")]
pub use cache::{AstCache, CachedRustFile};
#[cfg(feature = "ast-analysis")]
pub use debt_floors::{
    DensityFloors, DensityFloorsError, load_workspace_density_floors, parse_density_floors_toml,
};
pub use manifest::{TraceabilityManifestPayload, VerificationCertificate};
pub use maturity::{EXPECTED_WORKSPACE_CRATES, MaturityRegistry};
pub use proof_digest::{
    CANONICAL_PYRAMID_TIERS, CERTIFIED_SOLVER_COUNT, collect_lean4_proof_digests, sha256_hex,
    strip_lean_comments, validate_lean_proof_source,
};
#[cfg(feature = "ast-analysis")]
pub use qualification::{
    MaturityEvidence, MaturityGateReport, VALIDATED_MIN_ASSERTION_DENSITY,
    discover_maturity_scan_files, evaluate_maturity_with_registry_and_cache,
    evaluate_workspace_maturity, evaluate_workspace_maturity_with_cache,
};
#[cfg(feature = "ast-analysis")]
pub use runner::{
    WorkspaceAstMetrics, check_and_print_traceability_errors,
    collect_workspace_ast_metrics_with_cache, discover_workspace_code_dirs,
    format_traceability_errors, format_traceability_errors_with_floors,
    print_traceability_dashboard, run_workspace_traceability,
    run_workspace_traceability_with_cache,
};
pub use scientific_metadata::theory::MaturityTier;

use crate::vfs::VirtualFileSystem;
#[cfg(feature = "ast-analysis")]
use std::collections::HashSet;
use std::collections::{BTreeMap, HashMap};

/// Summary report produced by scanning workspace Rust modules against `traceability.toml` and `papers/*.tex`.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TraceabilityReport {
    /// Total active Rust source files scanned.
    pub scanned_files: usize,
    /// `(source, target)` pairs referencing non-existent registry modules or LaTeX papers.
    pub invalid_links: Vec<(String, String)>,
    /// Manuscript `.tex` files unreferenced by any active entry in `traceability.toml`.
    pub orphaned_papers: Vec<String>,
    /// Formatted module paths lacking `[cite:...]` traceability annotations.
    pub unlinked_code: Vec<String>,
    /// Detailed `(module_path, file_path, line)` tuples for unlinked modules.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unlinked_module_details: Vec<(String, String, usize)>,
    /// Module files containing unverified functions without an explicit opt-out.
    pub unverified_modules: Vec<String>,
    /// Mapping from manuscript `.tex` filename to citing source files.
    pub paper_coverage: HashMap<String, Vec<String>>,
    /// Total functions discovered across scanned files.
    pub total_funcs: usize,
    /// Total assertions discovered across scanned files.
    pub total_asserts: usize,
    /// Total verified functions across scanned files.
    pub verified_funcs: usize,
    /// Total verified assertions across scanned files.
    pub verified_asserts: usize,
    /// Registry entries declaring an invalid verification tier mismatch.
    pub invalid_tiers: Vec<String>,
    /// Files where stochastic verification blocks used vacuous zero-initialized inputs.
    pub vacuous_bypasses: Vec<String>,
    /// Semantic integrity status per manuscript `.tex` file.
    pub semantic_integrity_status: HashMap<String, String>,
    /// Source line numbers for invalid `[cite:...]` links.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub invalid_link_lines: BTreeMap<String, usize>,
    /// First unverified `(fn_name, line)` span per unverified module file.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unverified_module_spans: BTreeMap<String, (String, usize)>,
    /// Source line numbers for vacuous stochastic bypasses.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub vacuous_bypass_spans: BTreeMap<String, usize>,
    /// Per-file `[verified_funcs, total_funcs]` counts.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub file_verified_counts: BTreeMap<String, [usize; 2]>,
    /// Explicit crate-level maturity tiers loaded from `[crates]` in `traceability.toml`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub crate_maturity_tiers: BTreeMap<String, String>,
    /// Explicit module-level maturity tiers loaded from `[modules]` in `traceability.toml`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub module_maturity_tiers: BTreeMap<String, String>,
    /// Registered modules with non-vacuous deterministic `theory_verification!` contracts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deterministic_contract_modules: Vec<String>,
    /// Registered modules with `VerifiedAllocator` or `VerifiedEngine::run_verified` proofs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub zero_alloc_modules: Vec<String>,
    /// Per-tier maturity violations detected during AST traceability scanning.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub maturity_violations: Vec<String>,
    /// Per-tier verification macro invocation counts across the 6-tier scientific verification pyramid.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tier_counts: BTreeMap<String, usize>,
}

/// Traceability scanner parameterized over a [`VirtualFileSystem`] backend.
pub struct TraceabilityEngine<V: VirtualFileSystem> {
    /// Underlying virtual file system used to read manifests, papers, and Rust sources.
    pub vfs: V,
}

impl<V: VirtualFileSystem> TraceabilityEngine<V> {
    /// Creates a new [`TraceabilityEngine`] backed by `vfs`.
    pub fn new(vfs: V) -> Self {
        Self { vfs }
    }

    /// Returns `true` when `module_name` is registered in the workspace `traceability.toml`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn verify_module_registered(module_name: &str) -> bool {
        let mut registry_content = std::fs::read_to_string("traceability.toml").unwrap_or_default();
        if registry_content.is_empty()
            && let Some(root) = crate::workspace::find_workspace_root()
        {
            registry_content =
                std::fs::read_to_string(root.join("traceability.toml")).unwrap_or_default();
        }
        analysis::is_module_in_registry_toml(&registry_content, module_name)
    }

    /// Returns `true` when `module_name` is registered in the embedded `traceability.toml`.
    #[cfg(target_arch = "wasm32")]
    pub fn verify_module_registered(module_name: &str) -> bool {
        const EMBEDDED_REGISTRY: &str = include_str!("../../../traceability.toml");
        analysis::is_module_in_registry_toml(EMBEDDED_REGISTRY, module_name)
    }

    /// Extract citations matching the regex `\[cite:([a-zA-Z0-9_.-]+)\]` manually to avoid heavy regex compilation.
    pub fn extract_citations(content: &str) -> Vec<String> {
        analysis::extract_citations_with_lines(content)
            .into_iter()
            .map(|(cite, _)| cite)
            .collect()
    }
}

#[cfg(feature = "ast-analysis")]
impl<V: VirtualFileSystem> TraceabilityEngine<V> {
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

    pub(super) async fn load_cached_optional(
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

    /// Discovers workspace code directories (`math_explorer/src`, `math_explorer_gui/src/tabs`,
    /// `crates/*/src`, and grouped `crates/*/*/src`) and the `papers` directory relative to `base_dir`.
    pub async fn discover_workspace_dirs(&self, base_dir: &str) -> (Vec<String>, String) {
        let prefix = if base_dir.is_empty() || base_dir.ends_with('/') {
            base_dir.to_string()
        } else {
            format!("{base_dir}/")
        };
        let mut code_dirs = vec![
            format!("{prefix}math_explorer/src"),
            format!("{prefix}math_explorer_gui/src/tabs"),
        ];
        let crate_base = format!("{prefix}crates");
        if let Ok(mut crates) = self.vfs.list_dir(&crate_base).await {
            crates.sort();
            for crate_name in crates {
                if crate_name.starts_with('.') {
                    continue;
                }
                let crate_dir = format!("{crate_base}/{crate_name}");
                if let Ok(mut sub_entries) = self.vfs.list_dir(&crate_dir).await
                    && !sub_entries.iter().any(|e| e == "src" || e == "Cargo.toml")
                {
                    sub_entries.sort();
                    for sub in sub_entries {
                        if !sub.starts_with('.') {
                            code_dirs.push(format!("{crate_dir}/{sub}/src"));
                        }
                    }
                } else {
                    code_dirs.push(format!("{crate_dir}/src"));
                }
            }
        }
        (code_dirs, format!("{prefix}papers"))
    }

    /// Scans all discovered workspace code directories and `papers` under `base_dir`.
    pub async fn scan_workspace(
        &self,
        base_dir: &str,
        auto_fix: bool,
    ) -> Result<TraceabilityReport, std::io::Error> {
        let (code_dirs, papers_dir) = self.discover_workspace_dirs(base_dir).await;
        let all_dirs: Vec<&str> = code_dirs.iter().map(String::as_str).collect();
        self.scan_repository(&all_dirs, &papers_dir, auto_fix).await
    }

    /// Scans `code_dirs` and `papers_dir` using a fresh single-pass [`AstCache`].
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
        if cache
            .get(&key)
            .and_then(|entry| entry.ast.as_ref())
            .is_some_and(|ast| crate::ast_visitor::attrs_require_cfg_test(&ast.attrs))
        {
            return Ok(());
        }

        report.scanned_files += 1;
        let is_module = norm_file.ends_with("mod.rs")
            || norm_file.contains("/tabs/")
            || norm_file.ends_with("lib.rs");

        let citations = cache
            .get(&key)
            .map(|entry| analysis::extract_citations_with_lines(&entry.content))
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
