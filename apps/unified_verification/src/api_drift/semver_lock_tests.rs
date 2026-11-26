use super::*;
use oxidize_core::traceability::MaturityRegistry;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};

static SEMVER_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn temp_semver_ws(label: &str) -> PathBuf {
    let id = SEMVER_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = env::temp_dir().join(format!(
        "oxidize_semver_lock_{}_{}_{}",
        label,
        std::process::id(),
        id
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_ws_file(root: &Path, rel: &str, content: &str) {
    let path = root.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

fn sample_maturity_and_baseline() -> (MaturityRegistry, BTreeMap<String, Vec<ApiItem>>) {
    let registry_toml = r#"
[links]
cera_framework = "cera_framework.tex"
climate = "graph_parameters_rust.tex"
math_commons = "essay.tex"

[semver_lock]
certified_baseline_version = "0.4.0"

[crates]
domain_climate = "Experimental"
math_commons = "Certified"

[modules]
cera_framework = "Experimental"
climate = "Experimental"
math_commons = "Certified"
"#;
    let reg = MaturityRegistry::parse_toml_str(registry_toml);
    let mut baseline = BTreeMap::new();
    baseline.insert(
        "crates/domain_climate/src/climate/cera.rs".to_string(),
        vec![ApiItem {
            signature: "pub fn prototype_v1 (_ : f64) -> f64".to_string(),
            deprecation_expires: None,
        }],
    );
    baseline.insert(
        "crates/math_commons/src/fit/mod.rs".to_string(),
        vec![ApiItem {
            signature: "pub fn fit_model (_ : f64) -> f64".to_string(),
            deprecation_expires: None,
        }],
    );
    (reg, baseline)
}

#[test]
fn experimental_fluid_evolution_passes_api_drift() {
    let (reg, baseline) = sample_maturity_and_baseline();
    let mut exp_evolved = baseline.clone();
    exp_evolved.insert(
        "crates/domain_climate/src/climate/cera.rs".to_string(),
        vec![
            ApiItem {
                signature: "pub fn prototype_v2 (_ : f64 , _ : usize) -> f64".to_string(),
                deprecation_expires: None,
            },
            ApiItem {
                signature: "pub struct CeraConfig".to_string(),
                deprecation_expires: None,
            },
        ],
    );
    exp_evolved.insert(
        "crates/domain_climate/src/climate/new_probe.rs".to_string(),
        vec![ApiItem {
            signature: "pub fn probe () -> f64".to_string(),
            deprecation_expires: None,
        }],
    );
    let (passed_exp, exp_msgs) =
        collect_api_drift_messages_with_registry(&exp_evolved, &baseline, Some(&reg), "2026-11-23");
    assert!(
        passed_exp,
        "Experimental API evolution must pass check-api-drift, got: {exp_msgs:?}"
    );
    assert!(
        exp_msgs
            .iter()
            .any(|m| m.contains("Experimental API evolution permitted"))
    );
}

#[test]
fn certified_breaking_removal_fails_semver_lock() {
    let (reg, baseline) = sample_maturity_and_baseline();
    let mut cert_broken = baseline.clone();
    cert_broken.insert(
        "crates/math_commons/src/fit/mod.rs".to_string(),
        vec![ApiItem {
            signature: "pub fn fit_model (_ : f32) -> f32".to_string(),
            deprecation_expires: None,
        }],
    );
    let (passed_cert, cert_msgs) =
        collect_api_drift_messages_with_registry(&cert_broken, &baseline, Some(&reg), "2026-11-23");
    assert!(!passed_cert);
    assert!(
        cert_msgs
            .iter()
            .any(|m| m.contains("SemVer Lock Violation [Certified]")),
        "Expected SemVer Lock Violation on Certified removal, got: {cert_msgs:?}"
    );
}

#[test]
fn regenerate_baseline_rejects_certified_breaking_change_without_semver_bump() {
    let ws = temp_semver_ws("regen_lock");
    write_ws_file(
        &ws,
        "Cargo.toml",
        "[workspace]\nmembers = [\"crates/math_commons\"]\n\n[workspace.package]\nversion = \"0.4.0\"\n",
    );
    write_ws_file(
        &ws,
        "traceability.toml",
        "[links]\nmath_commons = \"essay.tex\"\n\n[semver_lock]\ncertified_baseline_version = \"0.4.0\"\n\n[crates]\nmath_commons = \"Certified\"\n\n[modules]\nmath_commons = \"Certified\"\n",
    );
    write_ws_file(
        &ws,
        "crates/math_commons/src/lib.rs",
        "pub fn stable_solver(x: f64) -> f64 { x }\n",
    );
    assert!(regenerate_baseline_in_dir(&ws, false).is_ok());

    // Additive signature on Certified module succeeds without SemVer bump.
    write_ws_file(
        &ws,
        "crates/math_commons/src/lib.rs",
        "pub fn stable_solver(x: f64) -> f64 { x }\npub fn extra_helper() -> usize { 1 }\n",
    );
    assert!(regenerate_baseline_in_dir(&ws, false).is_ok());

    // Breaking signature change on Certified module fails while version stays 0.4.0!
    write_ws_file(
        &ws,
        "crates/math_commons/src/lib.rs",
        "pub fn stable_solver(x: f32) -> f32 { x }\n",
    );
    let err = regenerate_baseline_in_dir(&ws, false)
        .expect_err("Certified breaking removal must fail without SemVer bump");
    assert!(
        err.iter()
            .any(|e| e.contains("SemVer Lock Violation [Certified]"))
    );

    // Bumping [workspace.package].version to 0.5.0 unlocks regenerate-baseline and syncs traceability.toml.
    write_ws_file(
        &ws,
        "Cargo.toml",
        "[workspace]\nmembers = [\"crates/math_commons\"]\n\n[workspace.package]\nversion = \"0.5.0\"\n",
    );
    assert!(regenerate_baseline_in_dir(&ws, false).is_ok());
    let synced_toml = fs::read_to_string(ws.join("traceability.toml")).unwrap();
    assert!(synced_toml.contains("certified_baseline_version = \"0.5.0\""));

    let _ = fs::remove_dir_all(&ws);
}
