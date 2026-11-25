use super::*;
use crate::utils::test_utils::TempTestDir;
use std::fs;

#[test]
fn test_public_debug_accepts_derived_and_manual_debug_and_ignores_private() {
    let dir = TempTestDir::new("public_debug_ok");
    let src = dir.join("crates/demo/src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("lib.rs"),
        r#"
mod hidden;
mod facade;
pub use facade::ManualType;

#[derive(Clone, Debug, PartialEq)]
pub struct PlainConfig {
    pub dt: f64,
}

#[cfg_attr(not(test), derive(Debug))]
pub struct CfgAttrDebugStruct;

#[derive(Debug)]
pub enum Mode {
    Fast,
}

struct PrivateHelper;
"#,
    )
    .unwrap();
    fs::write(src.join("hidden.rs"), "pub struct UnexportedNoDebug;\n").unwrap();
    fs::write(
        src.join("facade.rs"),
        r#"
pub struct ManualType;
impl core::fmt::Debug for ManualType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("ManualType")
    }
}
pub struct InternalFacadeOnly;
"#,
    )
    .unwrap();

    let mut cache = AstCache::new();
    let violations = collect_crate_public_debug_violations(&src.join("lib.rs"), &dir, &mut cache);
    assert!(
        violations.is_empty(),
        "Expected zero violations, got {violations:?}"
    );
}

#[test]
fn test_public_debug_detects_missing_struct_and_enum_debug() {
    let dir = TempTestDir::new("public_debug_fail");
    let src = dir.join("crates/bad/src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("lib.rs"),
        "pub struct NoDebugStruct;\npub enum NoDebugEnum { A }\n",
    )
    .unwrap();

    let mut cache = AstCache::new();
    let violations = collect_crate_public_debug_violations(&src.join("lib.rs"), &dir, &mut cache);
    assert_eq!(violations.len(), 2);
    assert!(
        violations
            .iter()
            .any(|v| v.contains("pub struct NoDebugStruct"))
    );
    assert!(
        violations
            .iter()
            .any(|v| v.contains("pub enum NoDebugEnum"))
    );
}

#[test]
fn test_workspace_crates_pass_public_debug_gate() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let root = crate::manifest_cache::workspace_root_dir();
    let mut cache = AstCache::new();
    assert!(check_public_debug_in_dir_with_cache(&root, &mut cache));
}
