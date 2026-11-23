use regex::Regex;
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Serialize)]
pub struct IntegrityReport {
    pub native_execution_coverage_pct: f64,
    pub wasm_path_coverage_pct: f64,
    pub total_assertion_density: f64,
    pub verified_modules_density: f64,
    pub unverified_modules_density: f64,
    pub passed: bool,
}

pub fn get_llvm_cov_output() -> std::process::Output {
    let mut cmd = Command::new("cargo");
    cmd.args(["llvm-cov", "--all-features", "--workspace", "--json"]);
    cmd.output().expect("Failed to run cargo llvm-cov")
}

pub fn collect_rs_files() -> Vec<std::path::PathBuf> {
    crate::utils::collect_rs_files_filtered(&[Path::new(".")], &["target", ".git"])
}

pub fn parse_coverage(cov_json: &serde_json::Value) -> (f64, f64) {
    let mut native_lines_total = 0.0;
    let mut native_lines_covered = 0.0;

    if let Some(files) = cov_json
        .get("data")
        .and_then(|d| d.as_array())
        .and_then(|a| a.first())
        .and_then(|data| data.get("files"))
        .and_then(|f| f.as_array())
    {
        for f in files {
            if let Some(summary) = f.get("summary").and_then(|s| s.get("lines")) {
                native_lines_total += summary.get("count").and_then(|c| c.as_f64()).unwrap_or(0.0);
                native_lines_covered += summary
                    .get("covered")
                    .and_then(|c| c.as_f64())
                    .unwrap_or(0.0);
            }
        }
    }
    (native_lines_total, native_lines_covered)
}

pub struct FileMetrics {
    pub wasm_paths: usize,
    pub wasm_covered: usize,
    pub total_funcs: f64,
    pub total_asserts: f64,
    pub verified_funcs: f64,
    pub verified_asserts: f64,
    pub unverified_funcs: f64,
    pub unverified_asserts: f64,
    pub opt_outs: Vec<(String, String)>,
    pub orphaned_wasm_tests: Vec<String>,
}

struct ReportPatterns {
    assert_re: Regex,
    fn_re: Regex,
    opt_out_re: Regex,
    legacy_re: Regex,
}

impl ReportPatterns {
    fn new() -> Self {
        Self {
            assert_re: Regex::new(r"\b(assert|assert_eq|assert_ne|debug_assert)!\s*\(").unwrap(),
            fn_re: Regex::new(r"\bfn\s+\w+\s*(?:<[^>]*>)?\s*\(").unwrap(),
            opt_out_re: Regex::new(
                r#"#\[(?:verified_engine::)?verified\(opt_out\s*=\s*"([^"]+)"\)\]"#,
            )
            .unwrap(),
            legacy_re: Regex::new(r"\bfn\s+([a-zA-Z0-9_]*_legacy)\b").unwrap(),
        }
    }
}

fn normalize_report_path(filepath: &Path) -> String {
    let norm = crate::whitelist::lossy_slash_path(filepath);
    norm.strip_prefix("./").unwrap_or(&norm).to_string()
}

fn analyze_file_content(
    filepath: &Path,
    content: &str,
    patterns: &ReportPatterns,
    m: &mut FileMetrics,
) {
    let is_verified =
        content.contains("#[verified") || content.contains("#[verified_engine::verified");
    let funcs = patterns.fn_re.find_iter(content).count() as f64;
    let asserts = patterns.assert_re.find_iter(content).count() as f64;
    let norm = normalize_report_path(filepath);

    for cap in patterns.opt_out_re.captures_iter(content) {
        m.opt_outs.push((norm.clone(), cap[1].to_string()));
    }
    for cap in patterns.legacy_re.captures_iter(content) {
        m.opt_outs
            .push((norm.clone(), format!("Legacy function call: {}", &cap[1])));
    }

    m.total_funcs += funcs;
    m.total_asserts += asserts;
    if is_verified {
        m.verified_funcs += funcs;
        m.verified_asserts += asserts;
    } else {
        m.unverified_funcs += funcs;
        m.unverified_asserts += asserts;
    }

    let (file_wasm_paths, file_wasm_covered) =
        crate::wasm_coverage::count_file_wasm_coverage(filepath, content);
    m.wasm_paths += file_wasm_paths;
    m.wasm_covered += file_wasm_covered;

    for fn_name in crate::wasm_coverage::find_orphaned_wasm_unit_tests(filepath, content) {
        m.orphaned_wasm_tests.push(format!(
            "Orphaned wasm32 #[test] in {norm}: `{fn_name}` (use #[wasm_bindgen_test])"
        ));
    }
}

pub fn analyze_files(rs_files: &[std::path::PathBuf]) -> FileMetrics {
    let mut m = FileMetrics {
        wasm_paths: 0,
        wasm_covered: 0,
        total_funcs: 0.0,
        total_asserts: 0.0,
        verified_funcs: 0.0,
        verified_asserts: 0.0,
        unverified_funcs: 0.0,
        unverified_asserts: 0.0,
        opt_outs: Vec::new(),
        orphaned_wasm_tests: Vec::new(),
    };
    let patterns = ReportPatterns::new();

    for filepath in rs_files {
        if let Ok(content) = fs::read_to_string(filepath) {
            analyze_file_content(filepath, &content, &patterns, &mut m);
        }
    }
    m.opt_outs.sort();
    m.orphaned_wasm_tests.sort();
    m
}

pub fn check_unverified_modules(members: &[String]) -> Vec<String> {
    let mut unverified_modules = Vec::new();
    for fm in members {
        if fm == "egui_plot" {
            continue;
        }
        let mod_dir = Path::new(fm).join("src");
        if mod_dir.exists() {
            let has_theory = crate::utils::collect_rs_files_filtered(&[&mod_dir], &["target"])
                .into_iter()
                .filter_map(|path| fs::read_to_string(path).ok())
                .any(|content| {
                    content.contains("theory_verification!")
                        || content.contains("stochastic_signature_verification!")
                });
            if !has_theory {
                unverified_modules.push(format!("Unverified Module: {}", fm));
            }
        }
    }
    unverified_modules.sort();
    unverified_modules
}

fn save_report(report: &IntegrityReport, native_cov: f64) {
    if cfg!(test) {
        return;
    }
    let json_report = serde_json::to_string_pretty(&report).unwrap();
    fs::write("Verification_Certificate.json", json_report).unwrap();
    println!("Generated Verification_Certificate.json");

    if let Ok(mut readme) = fs::read_to_string("README.md") {
        let badge_color = if native_cov >= 90.0 {
            "brightgreen"
        } else {
            "yellow"
        };
        let new_badge = format!(
            "![Coverage](https://img.shields.io/badge/coverage-{:.1}%25-{})",
            native_cov, badge_color
        );
        let badge_re = Regex::new(r"!\[Coverage\]\(.*?\)").unwrap();
        readme = badge_re.replace(&readme, new_badge.as_str()).to_string();
        fs::write("README.md", readme).unwrap();
        println!("README static badge updated with dynamic metric.");
    }
}

fn ratio_or_zero(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn print_debt_section(m: &FileMetrics, debt_items: &[String]) {
    let mut all_debt =
        Vec::with_capacity(m.opt_outs.len() + m.orphaned_wasm_tests.len() + debt_items.len());
    for (file, reason) in &m.opt_outs {
        all_debt.push(format!("{} bypassed: '{}'", file, reason));
    }
    all_debt.extend(m.orphaned_wasm_tests.iter().cloned());
    all_debt.extend(debt_items.iter().cloned());

    if !all_debt.is_empty() {
        println!("\n--- Integrity Debt ---");
        for (idx, item) in all_debt.iter().enumerate() {
            println!("[{}] {}", idx + 1, item);
        }
    }
}

pub fn print_report(
    native_cov_pct: f64,
    wasm_cov_pct: f64,
    m: &FileMetrics,
    debt_items: &[String],
) -> bool {
    let total_density = ratio_or_zero(m.total_asserts, m.total_funcs);
    let verified_density = ratio_or_zero(m.verified_asserts, m.verified_funcs);
    let unverified_density = ratio_or_zero(m.unverified_asserts, m.unverified_funcs);

    println!("\n--- Integrity Report ---");
    println!("Native Execution Coverage: {:.2}%", native_cov_pct);
    println!("WASM Path Coverage: {:.2}%", wasm_cov_pct);

    println!("\n--- High-Integrity Dashboard ---");
    println!("Total Assertion Density: {:.2} asserts/fn", total_density);
    println!(
        "Verified Modules Density: {:.2} asserts/fn",
        verified_density
    );
    println!(
        "Unverified Modules Density: {:.2} asserts/fn",
        unverified_density
    );

    let mut passed = true;
    if verified_density < 0.4 {
        println!(
            "\n[!] Assertion Density Failure: Verified modules have a density of {:.2} asserts/fn, which is below the minimum required 0.4 asserts/fn.",
            verified_density
        );
        passed = false;
    }

    print_debt_section(m, debt_items);

    if !m.orphaned_wasm_tests.is_empty() {
        println!(
            "\n[!] Orphaned wasm32 #[test] Failure: {} #[cfg(target_arch = \"wasm32\")] #[test] item(s) detected; convert to #[wasm_bindgen_test].",
            m.orphaned_wasm_tests.len()
        );
        passed = false;
    }

    if native_cov_pct < 50.0 {
        println!(
            "Coverage {:.2}% is below threshold of 50.0%",
            native_cov_pct
        );
        passed = false;
    }

    let report = IntegrityReport {
        native_execution_coverage_pct: native_cov_pct,
        wasm_path_coverage_pct: wasm_cov_pct,
        total_assertion_density: total_density,
        verified_modules_density: verified_density,
        unverified_modules_density: unverified_density,
        passed,
    };

    save_report(&report, native_cov_pct);
    passed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_coverage_empty_and_valid() {
        let (tot, cov) = parse_coverage(&serde_json::json!({}));
        assert_eq!(tot, 0.0);
        assert_eq!(cov, 0.0);

        let data = serde_json::json!({
            "data": [{
                "files": [
                    {
                        "summary": {
                            "lines": {
                                "count": 100,
                                "covered": 80
                            }
                        }
                    },
                    {
                        "summary": {
                            "lines": {
                                "count": 50,
                                "covered": 20
                            }
                        }
                    }
                ]
            }]
        });
        let (tot2, cov2) = parse_coverage(&data);
        assert_eq!(tot2, 150.0);
        assert_eq!(cov2, 100.0);
    }

    #[test]
    fn test_print_report_threshold_enforcement() {
        let metrics = FileMetrics {
            wasm_paths: 0,
            wasm_covered: 0,
            total_funcs: 10.0,
            total_asserts: 10.0,
            verified_funcs: 5.0,
            verified_asserts: 5.0,
            unverified_funcs: 5.0,
            unverified_asserts: 5.0,
            opt_outs: Vec::new(),
            orphaned_wasm_tests: Vec::new(),
        };

        // Coverage below 50.0% should fail
        let passed_low = print_report(49.9, 100.0, &metrics, &[]);
        assert!(!passed_low, "Coverage below 50.0% must fail");

        // Coverage at or above 50.0% with valid density should pass
        let passed_high = print_report(55.0, 100.0, &metrics, &[]);
        assert!(passed_high, "Coverage at or above 50.0% should pass");
        let _ = fs::remove_file("Verification_Certificate.json");
    }

    #[test]
    fn test_collect_rs_files_and_unverified_modules() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let files = collect_rs_files();
        assert!(!files.is_empty());
        let mut saw_egui_plot = false;
        for f in &files {
            assert!(f.extension().and_then(|s| s.to_str()) == Some("rs"));
            let p_str = crate::whitelist::lossy_slash_path(f);
            assert!(!p_str.contains("/target/"));
            if p_str.contains("egui_plot/src/file_io.rs") {
                saw_egui_plot = true;
            }
        }
        if Path::new("egui_plot/src/file_io.rs").is_file() {
            assert!(
                saw_egui_plot,
                "collect_rs_files must include egui_plot sources for WASM coverage"
            );
        }

        let unverified =
            check_unverified_modules(&["egui_plot".to_string(), "crates/pure_math".to_string()]);
        assert!(
            !unverified
                .iter()
                .any(|u| u.contains("egui_plot") || u.contains("crates/pure_math"))
        );
    }

    fn empty_file_metrics() -> FileMetrics {
        FileMetrics {
            wasm_paths: 0,
            wasm_covered: 0,
            total_funcs: 0.0,
            total_asserts: 0.0,
            verified_funcs: 0.0,
            verified_asserts: 0.0,
            unverified_funcs: 0.0,
            unverified_asserts: 0.0,
            opt_outs: Vec::new(),
            orphaned_wasm_tests: Vec::new(),
        }
    }

    #[test]
    fn test_wasm_coverage_ignores_theory_and_attribute_comments() {
        let patterns = ReportPatterns::new();
        let comment_only_src = r#"
            #[cfg(target_arch = "wasm32")]
            // theory_verification!
            // stochastic_signature_verification!
            // #[wasm_bindgen_test]
            fn mark_boot_initialized() {
                let _marker = "theory_verification! #[wasm_bindgen_test]";
            }

            #[cfg(target_arch = "wasm32")]
            // theory_verification!
            fn handle_webgl_failure() {}
        "#;
        let mut m = empty_file_metrics();
        analyze_file_content(
            Path::new("math_explorer_gui/src/comment_only_probe.rs"),
            comment_only_src,
            &patterns,
            &mut m,
        );
        assert_eq!(m.wasm_paths, 2);
        assert_eq!(
            m.wasm_covered, 0,
            "// theory_verification! comments must never count as covered WASM paths"
        );
    }

    #[test]
    fn test_wasm_coverage_counts_real_wasm_bindgen_tests() {
        let patterns = ReportPatterns::new();
        let tested_src = r#"
            #[cfg(target_arch = "wasm32")]
            pub fn browser_helper() -> u32 { 42 }

            #[cfg(all(test, target_arch = "wasm32"))]
            mod wasm_tests {
                use wasm_bindgen_test::wasm_bindgen_test;

                #[wasm_bindgen_test]
                fn verifies_browser_helper() {
                    assert_eq!(super::browser_helper(), 42);
                }
            }
        "#;
        let mut m = empty_file_metrics();
        analyze_file_content(
            Path::new("math_explorer_gui/src/tested_probe.rs"),
            tested_src,
            &patterns,
            &mut m,
        );
        assert_eq!(m.wasm_paths, 1);
        assert_eq!(m.wasm_covered, 1);
    }

    #[test]
    fn test_report_rejects_orphaned_wasm32_unit_tests() {
        let patterns = ReportPatterns::new();
        let bad_src = r#"
            #[cfg(target_arch = "wasm32")]
            #[test]
            fn orphaned_browser_test() {
                assert_eq!(1, 1);
            }
        "#;
        let mut m = empty_file_metrics();
        m.verified_funcs = 2.0;
        m.verified_asserts = 2.0;
        analyze_file_content(
            Path::new("egui_plot/src/probe.rs"),
            bad_src,
            &patterns,
            &mut m,
        );
        assert_eq!(m.orphaned_wasm_tests.len(), 1);
        assert!(!print_report(85.0, 100.0, &m, &[]));
    }
}
