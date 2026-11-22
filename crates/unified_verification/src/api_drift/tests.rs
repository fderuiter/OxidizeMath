use super::visitor::find_date_pattern;
use super::*;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};

static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn make_temp_workspace(label: &str) -> PathBuf {
    let id = TEMP_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = env::temp_dir().join(format!(
        "oxidize_api_drift_{}_{}_{}",
        label,
        std::process::id(),
        id
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_file(root: &Path, rel: &str, content: &str) {
    let path = root.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

#[test]
fn test_date_pattern_parsing() {
    assert_eq!(
        find_date_pattern("expires = 2026-12-31"),
        Some("2026-12-31".to_string())
    );
    assert_eq!(find_date_pattern("short"), None);
    assert_eq!(find_date_pattern("no-date-here-at-all"), None);
}

#[test]
fn test_api_item_deserialization() {
    let json_omitted = r#"{"signature": "fn test_function()"}"#;
    let item: ApiItem = serde_json::from_str(json_omitted).unwrap();
    assert_eq!(item.signature, "fn test_function()");
    assert_eq!(item.deprecation_expires, None);

    let json_null = r#"{"signature": "fn null_func()", "deprecation_expires": null}"#;
    let item_null: ApiItem = serde_json::from_str(json_null).unwrap();
    assert_eq!(item_null.signature, "fn null_func()");
    assert_eq!(item_null.deprecation_expires, None);

    let json_detailed = r#"{"signature": "fn old_func()", "deprecation_expires": "2026-11-01"}"#;
    let item2: ApiItem = serde_json::from_str(json_detailed).unwrap();
    assert_eq!(item2.signature, "fn old_func()");
    assert_eq!(item2.deprecation_expires, Some("2026-11-01".to_string()));

    assert!(serde_json::from_str::<ApiItem>(r#""fn legacy()""#).is_err());
}

#[test]
fn test_module_visibility_and_cfg_test_exclusion() {
    let ws = make_temp_workspace("visibility");
    write_file(
        &ws,
        "crates/demo/src/lib.rs",
        r#"
mod hidden;
pub mod open;
#[cfg(test)]
pub mod test_only;

#[cfg(feature = "gpu")]
pub use open::gpu_helper;

#[cfg(test)]
pub fn helper_for_tests() {}
"#,
    );
    write_file(
        &ws,
        "crates/demo/src/hidden.rs",
        "pub fn secret_internal() {}\npub struct HiddenType;\n",
    );
    write_file(
        &ws,
        "crates/demo/src/test_only.rs",
        "pub fn should_not_be_scanned() {}\n",
    );
    write_file(
        &ws,
        "crates/demo/src/open/mod.rs",
        "pub const SCALE: f64 = 2.0;\npub fn gpu_helper() {}\n",
    );

    let apis = extract_apis_in_dir(&ws);
    assert!(!apis.contains_key("crates/demo/src/hidden.rs"));
    assert!(!apis.contains_key("crates/demo/src/test_only.rs"));

    let lib_sigs: Vec<&str> = apis["crates/demo/src/lib.rs"]
        .iter()
        .map(|i| i.signature.as_str())
        .collect();
    assert!(
        lib_sigs
            .iter()
            .any(|s| s.contains("pub use open :: gpu_helper"))
    );
    assert!(!lib_sigs.iter().any(|s| s.contains("helper_for_tests")));

    let _ = fs::remove_dir_all(&ws);
}

#[test]
fn test_private_submodule_facade_resolution() {
    let ws = make_temp_workspace("facade");
    write_file(
        &ws,
        "crates/demo/src/lib.rs",
        "mod facade;\npub use facade::ExportedModel;\n",
    );
    write_file(
        &ws,
        "crates/demo/src/facade.rs",
        r#"
pub struct ExportedModel;
impl ExportedModel {
    pub fn compute(&self, x: f64) -> f64 { x }
    fn private_method(&self) {}
}
pub fn unexported_helper() {}
"#,
    );

    let apis = extract_apis_in_dir(&ws);
    let lib_sigs: Vec<&str> = apis["crates/demo/src/lib.rs"]
        .iter()
        .map(|i| i.signature.as_str())
        .collect();
    assert!(
        lib_sigs
            .iter()
            .any(|s| s.contains("pub use facade :: ExportedModel"))
    );

    let facade_sigs: Vec<&str> = apis["crates/demo/src/facade.rs"]
        .iter()
        .map(|i| i.signature.as_str())
        .collect();
    assert!(
        facade_sigs
            .iter()
            .any(|s| s.contains("pub struct ExportedModel"))
    );
    assert!(facade_sigs.iter().any(|s| s.contains("pub fn compute")));
    assert!(!facade_sigs.iter().any(|s| s.contains("private_method")));
    assert!(!facade_sigs.iter().any(|s| s.contains("unexported_helper")));

    let _ = fs::remove_dir_all(&ws);
}

#[test]
fn test_drift_fails_on_method_trait_const_and_reexport_changes() {
    let ws = make_temp_workspace("mutations");
    let initial = r#"
pub const MAX_STEPS: usize = 16;
pub static GLOBAL_FLAG: bool = false;
pub use std::collections::BTreeMap as OrderedMap;

pub trait Solver {
    type Output;
    fn step(&mut self, dt: f64) -> Self::Output;
}

pub struct Engine;
impl Engine {
    pub fn run(&self, n: usize) -> usize { n }
}
"#;
    write_file(&ws, "math_explorer/src/lib.rs", initial);
    let baseline = extract_apis_in_dir(&ws);
    assert!(check_api_drift_against(&baseline, &baseline, "2026-11-19"));

    for mutated in [
        initial.replace("pub fn run(&self, n: usize)", "pub fn run(&self, n: u64)"),
        initial.replace("fn step(&mut self, dt: f64)", "fn step(&mut self, dt: f32)"),
        initial.replace(
            "pub const MAX_STEPS: usize = 16;",
            "pub const MAX_STEPS: usize = 32;",
        ),
        initial.replace("OrderedMap", "SortedMap"),
    ] {
        write_file(&ws, "math_explorer/src/lib.rs", &mutated);
        let current = extract_apis_in_dir(&ws);
        assert!(
            !check_api_drift_against(&current, &baseline, "2026-11-19"),
            "Mutation must fail API drift check:\n{mutated}"
        );
    }

    let _ = fs::remove_dir_all(&ws);
}

#[test]
fn test_deprecation_expiration_and_deletion_bypass() {
    let base_item = ApiItem {
        signature: "pub fn legacy ()".to_string(),
        deprecation_expires: Some("2026-01-01".to_string()),
    };
    let mut baseline = BTreeMap::new();
    baseline.insert("crates/a/src/lib.rs".to_string(), vec![base_item.clone()]);

    // Past expiration while still present -> fails
    assert!(!check_api_drift_against(&baseline, &baseline, "2026-11-19"));

    // Deleted after expiration -> allowed bypass
    let current = BTreeMap::new();
    assert!(check_api_drift_against(&current, &baseline, "2026-11-19"));
}

#[test]
fn test_check_api_drift_runs_and_verifies_workspace_invariants() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let passed = check_api_drift();
    assert!(
        passed,
        "Public API drift should match baseline in repository"
    );

    let apis = extract_apis();
    let math_explorer_items = apis
        .get("math_explorer/src/lib.rs")
        .expect("math_explorer/src/lib.rs must be tracked in baseline");
    assert_eq!(math_explorer_items.len(), 24);
    assert!(apis.contains_key("math_explorer/src/error.rs"));
    assert!(apis.contains_key("math_explorer/src/prelude.rs"));

    for private_mod in [
        "crates/verified_engine_macros/src/crate_paths.rs",
        "crates/verified_engine_macros/src/embed_theory.rs",
        "crates/verified_engine_macros/src/latex_parser.rs",
        "crates/verified_engine_macros/src/semantic_integrity.rs",
        "crates/verified_engine_macros/src/theory.rs",
        "crates/verified_engine_macros/src/verified.rs",
    ] {
        assert!(
            !apis.contains_key(private_mod),
            "Unreachable private submodule must be excluded: {private_mod}"
        );
    }

    assert!(
        apis.contains_key(
            "crates/domain_applied/src/applied/game_theory/evolutionary/replicator.rs"
        )
    );
    assert!(apis.contains_key("crates/pure_math/src/pure_math/statistics/markov/dtmc/types.rs"));
}
