#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub(super) struct EmbeddedPaperEntry {
    pub(super) name: String,
    pub(super) linked_code: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub(super) struct EmbeddedCertificate {
    pub(super) native_execution_coverage_pct: f64,
    pub(super) wasm_path_coverage_pct: f64,
    pub(super) total_assertion_density: f64,
    pub(super) verified_modules_density: f64,
    pub(super) unverified_modules_density: f64,
    pub(super) passed: bool,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub(super) struct EmbeddedTraceabilityManifest {
    pub(super) papers: Vec<EmbeddedPaperEntry>,
    pub(super) orphaned_papers: Vec<String>,
    pub(super) unlinked_code: Vec<String>,
    pub(super) invalid_links: Vec<(String, String)>,
    pub(super) certificate: Option<EmbeddedCertificate>,
}

const EMBEDDED_MANIFEST_JSON: &str =
    include_str!(concat!(env!("OUT_DIR"), "/traceability_manifest.json"));

static PARSED_MANIFEST: OnceLock<Result<EmbeddedTraceabilityManifest, String>> = OnceLock::new();

pub(super) fn embedded_manifest() -> Result<&'static EmbeddedTraceabilityManifest, String> {
    PARSED_MANIFEST
        .get_or_init(|| {
            serde_json::from_str::<EmbeddedTraceabilityManifest>(EMBEDDED_MANIFEST_JSON)
                .map_err(|err| format!("Failed to parse embedded traceability manifest: {err}"))
        })
        .as_ref()
        .map_err(Clone::clone)
}
