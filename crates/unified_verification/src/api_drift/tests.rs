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
            "pub const MAX_STEPS: u64 = 16;",
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
    assert_eq!(math_explorer_items.len(), 23);
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

#[test]
fn test_api_drift_header_precedes_diff_lines_and_prints_adr_hint() {
    let mut baseline = BTreeMap::new();
    baseline.insert(
        "crates/demo/src/lib.rs".to_string(),
        vec![ApiItem {
            signature: "pub fn old_api (_ : usize)".to_string(),
            deprecation_expires: None,
        }],
    );
    let mut current = BTreeMap::new();
    current.insert(
        "crates/demo/src/lib.rs".to_string(),
        vec![ApiItem {
            signature: "pub fn new_api (_ : f64)".to_string(),
            deprecation_expires: None,
        }],
    );

    let (passed, lines) = collect_api_drift_messages(&current, &baseline, "2026-11-19");
    assert!(!passed);
    let header_idx = lines
        .iter()
        .position(|l| l == "API Drift detected in crates/demo/src/lib.rs:")
        .expect("drift header must be emitted");
    let plus_idx = lines
        .iter()
        .position(|l| l.starts_with("  + "))
        .expect("added signature line must be emitted");
    let minus_idx = lines
        .iter()
        .position(|l| l.starts_with("  - "))
        .expect("removed signature line must be emitted");
    assert!(
        header_idx < plus_idx && header_idx < minus_idx,
        "Header must precede `+` and `-` diff lines: {lines:?}"
    );
    let joined = lines.join("\n");
    assert!(joined.contains("cargo run -p xtask -- regenerate-baseline"));
    assert!(joined.contains("docs/adr/"));
}

#[test]
fn test_private_param_renames_and_const_expr_changes_pass_drift() {
    let ws = make_temp_workspace("norm_params_consts");
    let v1 = r#"
pub const MAX_STEPS: usize = 16;
pub static GLOBAL_FLAG: bool = false;

pub trait Integrator<T: Clone> {
    const ORDER: usize = 4;
    fn advance(&mut self, dt: f64, (x, y): (f64, usize)) -> (f64, T)
    where
        T: Send;
}

pub struct State(pub f64);
impl State {
    pub const DEFAULT_TOL: f64 = 1e-6;
    pub fn into_raw(self) -> f64 { self.0 }
    pub fn scale(&self, factor: f64, cb: fn(old_v: f64) -> f64) -> f64 { cb(self.0 * factor) }
    pub fn reset(&mut self, next_val: f64) { self.0 = next_val; }
}
"#;
    let v2 = r#"
pub const MAX_STEPS: usize = 32 * 2;
pub static GLOBAL_FLAG: bool = true;

pub trait Integrator<T: Clone> {
    const ORDER: usize = 8;
    fn advance(&mut self, step_size: f64, coords: (f64, usize)) -> (f64, T)
    where
        T: Send;
}

pub struct State(pub f64);
impl State {
    pub const DEFAULT_TOL: f64 = 1e-9;
    pub fn into_raw(mut self) -> f64 { self.0 += 0.0; self.0 }
    pub fn scale(&self, alpha: f64, transform: fn(new_v: f64) -> f64) -> f64 { transform(self.0 * alpha) }
    pub fn reset(&mut self, target: f64) { self.0 = target; }
}
"#;
    write_file(&ws, "math_explorer/src/lib.rs", v1);
    let baseline = extract_apis_in_dir(&ws);

    write_file(&ws, "math_explorer/src/lib.rs", v2);
    let renamed = extract_apis_in_dir(&ws);
    assert_eq!(renamed, baseline);
    assert!(check_api_drift_against(&renamed, &baseline, "2026-11-19"));

    // Receiver mutability, tuple element types, and generic bounds must still fail drift
    for breaking in [
        v2.replace("pub fn scale(&self", "pub fn scale(&mut self"),
        v2.replace("coords: (f64, usize)", "coords: (f64, u64)"),
        v2.replace("Integrator<T: Clone>", "Integrator<T: Copy>"),
    ] {
        write_file(&ws, "math_explorer/src/lib.rs", &breaking);
        let broken = extract_apis_in_dir(&ws);
        assert!(!check_api_drift_against(&broken, &baseline, "2026-11-19"));
    }

    let _ = fs::remove_dir_all(&ws);
}
