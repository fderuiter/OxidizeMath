//! Serializable manifest envelope and verification certificate schemas for traceability snapshots.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Kernel verification status, bound Rust modules, and theorem names for a Lean 4 proof module.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeanProofBinding {
    /// Whether `leanprover/lean4:v4.14.0` kernel-checked this proof with zero banned axioms.
    #[serde(default)]
    pub kernel_verified: bool,
    /// Sorted workspace Rust module keys bound to this proof via `formal_proof_verification!`.
    #[serde(default)]
    pub rust_modules: Vec<String>,
    /// Sorted Lean 4 theorem identifiers verified and bound by Rust modules.
    #[serde(default)]
    pub theorems: Vec<String>,
}

/// Summary report produced by scanning workspace Rust modules against `traceability.toml` and `papers/*.tex`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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
    /// AST-discovered Lean 4 proof bindings (`proof.lean` -> linked Rust modules and theorem names).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub lean4_proof_bindings: BTreeMap<String, LeanProofBinding>,
}

/// Deserializable schema matching `Verification_Certificate.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerificationCertificate {
    /// Percentage of native Rust lines executed under `cargo llvm-cov`.
    pub native_execution_coverage_pct: f64,
    /// Percentage of `wasm32`-gated AST paths exercised by `#[wasm_bindgen_test]`.
    pub wasm_path_coverage_pct: f64,
    /// Overall assertion-to-function ratio across scanned workspace crates.
    pub total_assertion_density: f64,
    /// Assertion-to-function ratio across `#[verified]` modules.
    pub verified_modules_density: f64,
    /// Assertion-to-function ratio across unverified modules.
    pub unverified_modules_density: f64,
    /// Overall assertion-to-function ratio across non-test production functions.
    #[serde(default)]
    pub total_production_assertion_density: f64,
    /// Assertion-to-function ratio across `#[verified]` non-test production functions.
    #[serde(default)]
    pub verified_production_assertion_density: f64,
    /// Assertion-to-function ratio across test-scoped functions and verification macros.
    #[serde(default)]
    pub test_assertion_density: f64,
    /// Whether all workspace verification thresholds passed.
    pub passed: bool,
    /// Count of numerical solvers certified by Method of Manufactured Solutions (Tier 1).
    #[serde(default)]
    pub certified_solvers: usize,
    /// Maximum IEEE-754 outward-rounded interval enclosure width across certified kernels (Tier 5).
    #[serde(default)]
    pub max_interval_width: f64,
    /// Per-tier pass/invocation counts across all 6 scientific verification pyramid tiers.
    #[serde(default)]
    pub tier_counts: BTreeMap<String, usize>,
    /// Representative outward-rounded `[lo, hi]` interval enclosures for certified kernels (Tier 5).
    #[serde(default)]
    pub interval_bounds: BTreeMap<String, [f64; 2]>,
    /// Deterministic SHA-256 digests of committed `sorry`-free Lean 4 proof modules (Tier 6).
    #[serde(default)]
    pub lean4_proof_digests: BTreeMap<String, String>,
    /// Kernel verification status, bound Rust modules, and theorem names per Lean 4 proof module.
    #[serde(default)]
    pub lean4_proof_bindings: BTreeMap<String, LeanProofBinding>,
}

/// Deserializable web and desktop manifest envelope flattening [`TraceabilityReport`] with optional certificate metadata.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TraceabilityManifestPayload {
    /// Workspace traceability report containing paper coverage, AST counts, and semantic integrity status.
    #[serde(flatten)]
    pub report: TraceabilityReport,
    /// Optional precomputed verification certificate embedded alongside the traceability report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub certificate: Option<VerificationCertificate>,
}

#[cfg(test)]
mod core_tests {
    use super::{TraceabilityManifestPayload, VerificationCertificate};
    use crate::traceability::{TraceabilityEngine, TraceabilityReport};
    use crate::vfs::DefaultVfs;

    #[test]
    fn manifest_payload_default_and_citation_extraction_work_without_ast_feature() {
        let payload = TraceabilityManifestPayload {
            report: TraceabilityReport {
                scanned_files: 2,
                ..TraceabilityReport::default()
            },
            certificate: Some(VerificationCertificate {
                native_execution_coverage_pct: 80.0,
                wasm_path_coverage_pct: 75.0,
                total_assertion_density: 0.65,
                verified_modules_density: 0.55,
                unverified_modules_density: 0.70,
                total_production_assertion_density: 0.04,
                verified_production_assertion_density: 0.12,
                test_assertion_density: 3.05,
                passed: true,
                certified_solvers: 55,
                max_interval_width: 1e-14,
                tier_counts: Default::default(),
                interval_bounds: Default::default(),
                lean4_proof_digests: Default::default(),
                lean4_proof_bindings: Default::default(),
            }),
        };
        assert_eq!(payload.report.scanned_files, 2);
        assert!(payload.certificate.as_ref().is_some_and(|c| c.passed));

        let sample = concat!("// [cite", ":pure_math] [cite", ":ai_01]");
        let cites = TraceabilityEngine::<DefaultVfs>::extract_citations(sample);
        assert_eq!(cites, vec!["pure_math", "ai_01"]);
    }
}

#[cfg(all(test, feature = "ast-analysis"))]
mod tests {
    use super::{TraceabilityManifestPayload, VerificationCertificate};
    use crate::traceability::{TraceabilityEngine, TraceabilityReport};
    use crate::vfs::VirtualFileSystem;
    use std::collections::HashMap;
    use std::future::Future;
    use std::pin::Pin;

    struct MemoryVfs {
        files: HashMap<String, String>,
        dirs: HashMap<String, Vec<String>>,
    }

    impl VirtualFileSystem for MemoryVfs {
        fn read_to_string(
            &self,
            path: &str,
        ) -> Pin<Box<dyn Future<Output = Result<String, std::io::Error>> + '_>> {
            let res = self.files.get(path).cloned().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, format!("missing {path}"))
            });
            Box::pin(async move { res })
        }

        fn write_to_file(
            &self,
            _path: &str,
            _content: &[u8],
        ) -> Pin<Box<dyn Future<Output = Result<(), std::io::Error>> + '_>> {
            Box::pin(async { Ok(()) })
        }

        fn list_dir(
            &self,
            path: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, std::io::Error>> + '_>> {
            let res = self.dirs.get(path).cloned().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, format!("missing dir {path}"))
            });
            Box::pin(async move { res })
        }
    }

    #[test]
    fn manifest_payload_roundtrips_with_and_without_certificate() {
        let mut report = TraceabilityReport {
            scanned_files: 3,
            total_funcs: 5,
            total_asserts: 8,
            verified_funcs: 2,
            verified_asserts: 4,
            ..TraceabilityReport::default()
        };
        report
            .paper_coverage
            .insert("essay.tex".to_string(), vec!["pure_math".to_string()]);

        let bare = TraceabilityManifestPayload {
            report: report.clone(),
            certificate: None,
        };
        let encoded_bare = toml::to_string(&bare).expect("serialize bare payload");
        assert!(!encoded_bare.contains("certificate"));
        let decoded_bare: TraceabilityManifestPayload =
            toml::from_str(&encoded_bare).expect("deserialize bare payload");
        assert_eq!(decoded_bare, bare);

        let mut tier_counts = std::collections::BTreeMap::new();
        tier_counts.insert("Tier1_Convergence".to_string(), 55);
        let mut interval_bounds = std::collections::BTreeMap::new();
        interval_bounds.insert(
            "bessel_j0_unit".to_string(),
            [0.7651976865579664, 0.7651976865579668],
        );
        let mut lean4_proof_digests = std::collections::BTreeMap::new();
        lean4_proof_digests.insert(
            "IntervalSoundness.lean".to_string(),
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
        );
        let with_cert = TraceabilityManifestPayload {
            report,
            certificate: Some(VerificationCertificate {
                native_execution_coverage_pct: 85.0,
                wasm_path_coverage_pct: 72.5,
                total_assertion_density: 0.76,
                verified_modules_density: 0.61,
                unverified_modules_density: 0.83,
                total_production_assertion_density: 0.04,
                verified_production_assertion_density: 0.13,
                test_assertion_density: 3.10,
                passed: true,
                certified_solvers: 55,
                max_interval_width: 4.44e-16,
                tier_counts,
                interval_bounds,
                lean4_proof_digests,
                lean4_proof_bindings: Default::default(),
            }),
        };
        let encoded_cert = toml::to_string(&with_cert).expect("serialize certified payload");
        let decoded_cert: TraceabilityManifestPayload =
            toml::from_str(&encoded_cert).expect("deserialize certified payload");
        assert_eq!(decoded_cert, with_cert);
    }

    #[test]
    fn engine_discover_and_scan_workspace_populates_ast_metrics() {
        let mut files = HashMap::new();
        files.insert(
            "traceability.toml".to_string(),
            "[links]\ndemo_mod = \"demo.tex\"\n".to_string(),
        );
        files.insert(
            "crates/demo/src/lib.rs".to_string(),
            concat!(
                "// [cite",
                ":demo_mod]\n#[verified]\npub fn compute(x: i32) -> i32 { assert!(x >= 0); x + 1 }\n"
            )
            .to_string(),
        );
        let mut dirs = HashMap::new();
        dirs.insert("papers".to_string(), vec!["demo.tex".to_string()]);
        dirs.insert(
            "crates".to_string(),
            vec![
                "zeta".to_string(),
                "02_engines".to_string(),
                "demo".to_string(),
            ],
        );
        dirs.insert(
            "crates/02_engines".to_string(),
            vec!["engine_linalg".to_string()],
        );

        let engine = TraceabilityEngine::new(MemoryVfs { files, dirs });
        let (code_dirs, papers_dir) =
            futures::executor::block_on(engine.discover_workspace_dirs(""));
        assert_eq!(papers_dir, "papers");
        assert_eq!(
            code_dirs,
            vec![
                "crates/02_engines/engine_linalg/src",
                "crates/demo/src",
                "crates/zeta/src",
            ]
        );

        let report = futures::executor::block_on(engine.scan_workspace("", false))
            .expect("scan_workspace succeeds");
        assert_eq!(report.scanned_files, 1);
        assert_eq!(report.total_funcs, 1);
        assert_eq!(report.total_asserts, 1);
        assert_eq!(report.verified_funcs, 1);
        assert_eq!(report.verified_asserts, 1);
        assert!(report.orphaned_papers.is_empty());
        assert_eq!(
            report
                .semantic_integrity_status
                .get("demo")
                .map(String::as_str),
            Some("Verified")
        );
    }
}
