use super::*;
use crate::utils::test_utils::TempTestDir;
use std::fs;
use std::path::Path;

fn write_crate(root: &Path, dir: &str, name: &str, tier: &str, extra_toml: &str) {
    let crate_dir = root.join(dir);
    fs::create_dir_all(crate_dir.join("src")).unwrap();
    let manifest = format!(
        "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[package.metadata.oxidize]\ntier = \"{tier}\"\n\n{extra_toml}\n"
    );
    fs::write(crate_dir.join("Cargo.toml"), manifest).unwrap();
    fs::write(crate_dir.join("src/lib.rs"), "").unwrap();
}

fn write_root_workspace(root: &Path, members: &[&str]) {
    let quoted: Vec<String> = members.iter().map(|m| format!("    \"{m}\",")).collect();
    let ws = format!("[workspace]\nmembers = [\n{}\n]\n", quoted.join("\n"));
    fs::write(root.join("Cargo.toml"), ws).unwrap();
}

#[test]
fn test_live_workspace_satisfies_strict_five_tier_dag_and_feature_closures() {
    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let violations = collect_dag_violations(&ws_root);
    assert!(
        violations.is_empty(),
        "Live workspace must have zero DAG or feature-closure violations, got:\n{}",
        violations.join("\n")
    );
}

#[test]
fn test_rejects_sideways_peer_domain_edge_with_remediation_hint() {
    let tmp = TempTestDir::new("dag_gate_peer_domain");
    write_root_workspace(
        &tmp,
        &[
            "crates/03_domains/domain_applied",
            "crates/03_domains/domain_physics",
        ],
    );
    write_crate(
        &tmp,
        "crates/03_domains/domain_applied",
        "domain_applied",
        "03_domains",
        "[dependencies]",
    );
    write_crate(
        &tmp,
        "crates/03_domains/domain_physics",
        "domain_physics",
        "03_domains",
        "[dependencies]\ndomain_applied = { path = \"../domain_applied\" }",
    );

    let violations = collect_dag_violations(&tmp);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("sideways peer-domain edge"));
    assert!(violations[0].contains("source tier `03_domains`"));
    assert!(violations[0].contains("target tier `03_domains`"));
    assert!(violations[0].contains("Remediation:"));
}

#[test]
fn test_rejects_upward_pure_to_engine_edge_and_runtime_to_tooling_edge() {
    let tmp = TempTestDir::new("dag_gate_upward_and_tooling");
    write_root_workspace(
        &tmp,
        &[
            "crates/02_engines/engine_linalg",
            "crates/03_domains/pure_math",
            "apps/xtask",
        ],
    );
    write_crate(
        &tmp,
        "apps/xtask",
        "xtask",
        "apps",
        "[dependencies]\npure_math = { path = \"../../crates/03_domains/pure_math\" }",
    );
    write_crate(
        &tmp,
        "crates/02_engines/engine_linalg",
        "engine_linalg",
        "02_engines",
        "[dev-dependencies]\nxtask = { path = \"../../../apps/xtask\" }",
    );
    write_crate(
        &tmp,
        "crates/03_domains/pure_math",
        "pure_math",
        "03_domains",
        "[dependencies]\nengine_linalg = { path = \"../../02_engines/engine_linalg\" }",
    );

    let violations = collect_dag_violations(&tmp);
    assert_eq!(
        violations.len(),
        2,
        "Expected upward + tooling violations: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|v| v.contains("source tier `01_pure`")
                && v.contains("target tier `02_engines`")
                && v.contains("Remediation:"))
    );
    assert!(
        violations
            .iter()
            .any(|v| v.contains("target tier `developer_tooling`") && v.contains("Remediation:"))
    );
}

#[test]
fn test_rejects_cross_domain_feature_chaining_and_unclassified_crate() {
    let tmp = TempTestDir::new("dag_gate_feature_chain");
    write_root_workspace(
        &tmp,
        &[
            "crates/03_domains/domain_ai",
            "crates/03_domains/domain_climate",
            "crates/04_bindings/math_explorer",
            "mystery_crate",
        ],
    );
    write_crate(
        &tmp,
        "crates/03_domains/domain_ai",
        "domain_ai",
        "03_domains",
        "[dependencies]",
    );
    write_crate(
        &tmp,
        "crates/03_domains/domain_climate",
        "domain_climate",
        "03_domains",
        "[dependencies]",
    );
    write_crate(
        &tmp,
        "crates/04_bindings/math_explorer",
        "math_explorer",
        "04_bindings",
        "[features]\nai = [\"dep:domain_ai\"]\nclimate = [\"ai\", \"dep:domain_climate\"]\n\n[dependencies]\ndomain_ai = { path = \"../../03_domains/domain_ai\", optional = true }\ndomain_climate = { path = \"../../03_domains/domain_climate\", optional = true }",
    );
    fs::create_dir_all(tmp.join("mystery_crate/src")).unwrap();
    fs::write(
        tmp.join("mystery_crate/Cargo.toml"),
        "[package]\nname = \"mystery_crate\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    fs::write(tmp.join("mystery_crate/src/lib.rs"), "").unwrap();

    let violations = collect_dag_violations(&tmp);
    assert_eq!(violations.len(), 2, "Got: {violations:?}");
    assert!(violations.iter().any(|v| {
        v.contains("--no-default-features --features climate")
            && v.contains("domain_ai, domain_climate")
            && v.contains("Remediation:")
    }));
    assert!(
        violations
            .iter()
            .any(|v| v.contains("mystery_crate") && v.contains("no recognized tier classification"))
    );
}
