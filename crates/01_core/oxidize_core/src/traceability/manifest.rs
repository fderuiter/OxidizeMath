//! Serializable manifest envelope and verification certificate schemas for traceability snapshots.

use super::TraceabilityReport;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
