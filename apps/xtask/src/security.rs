use crate::gates::{resolve_workspace_root, run_cmd_in_path, run_verification_with_args};
use std::fs;
use std::path::Path;
use std::process::exit;

pub const MATH_CRATES: &[(&str, &str)] = &[
    ("math_commons", "crates/math_commons"),
    ("pure_math", "crates/pure_math"),
    ("domain_ai", "crates/domain_ai"),
    ("domain_applied", "crates/domain_applied"),
    ("domain_biology", "crates/domain_biology"),
    ("domain_climate", "crates/domain_climate"),
    ("domain_epidemiology", "crates/domain_epidemiology"),
    ("domain_physics", "crates/domain_physics"),
];

pub(crate) fn collect_tier5_cargo_test_args(ws_root: &Path) -> Result<Vec<String>, String> {
    let mut args = vec!["test".to_string()];
    for &(pkg_name, rel_dir) in MATH_CRATES {
        let tests_dir = ws_root.join(rel_dir).join("tests");
        let entries = fs::read_dir(&tests_dir)
            .map_err(|err| format!("Failed to read {}: {err}", tests_dir.display()))?;
        let mut targets = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("rs")
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                && stem.starts_with("security_")
            {
                targets.push(stem.to_string());
            }
        }
        targets.sort();
        if targets.is_empty() {
            return Err(format!(
                "Math crate `{pkg_name}` ({rel_dir}) has no Tier 5 `tests/security_*.rs` targets"
            ));
        }
        args.push("-p".to_string());
        args.push(pkg_name.to_string());
        for target in targets {
            args.push("--test".to_string());
            args.push(target);
        }
    }
    Ok(args)
}

pub fn check_security(args: &[String]) {
    println!("=== Check Supply-Chain Policy & Tier 5 Security Suites ===");
    let extra: Vec<&str> = args.iter().map(String::as_str).collect();
    run_verification_with_args("check-security", &extra);

    let ws_root = resolve_workspace_root();
    let cargo_args = collect_tier5_cargo_test_args(&ws_root).unwrap_or_else(|err| {
        eprintln!("Error: {err}");
        exit(1);
    });
    let arg_refs: Vec<&str> = cargo_args.iter().map(String::as_str).collect();
    run_cmd_in_path("cargo", &arg_refs, &ws_root);
}
