//! Serializable manifest envelope and verification certificate schemas for traceability snapshots.

use super::TraceabilityReport;
use serde::{Deserialize, Serialize};

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
    /// Whether all workspace verification thresholds passed.
    pub passed: bool,
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

        let with_cert = TraceabilityManifestPayload {
            report,
            certificate: Some(VerificationCertificate {
                native_execution_coverage_pct: 85.0,
                wasm_path_coverage_pct: 72.5,
                total_assertion_density: 0.76,
                verified_modules_density: 0.61,
                unverified_modules_density: 0.83,
                passed: true,
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
            vec!["zeta".to_string(), "demo".to_string()],
        );

        let engine = TraceabilityEngine::new(MemoryVfs { files, dirs });
        let (code_dirs, papers_dir) =
            futures::executor::block_on(engine.discover_workspace_dirs(""));
        assert_eq!(papers_dir, "papers");
        assert_eq!(
            code_dirs,
            vec![
                "math_explorer/src",
                "math_explorer_gui/src/tabs",
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
