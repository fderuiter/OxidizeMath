use regex::Regex;
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::process::Command;
use walkdir::WalkDir;

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
    let mut rs_files = Vec::new();
    for entry in WalkDir::new(".").into_iter().filter_map(|e| e.ok()) {
        let path_str = entry.path().to_string_lossy().replace("\\", "/");
        if path_str.contains("/target/")
            || path_str.contains("/.git/")
            || path_str.contains("/egui_plot/")
            || path_str.starts_with("target/")
            || path_str.starts_with(".git/")
            || path_str.starts_with("egui_plot/")
        {
            continue;
        }
        if entry.file_type().is_file()
            && entry.path().extension().and_then(|s| s.to_str()) == Some("rs")
        {
            rs_files.push(entry.path().to_path_buf());
        }
    }
    rs_files.sort();
    rs_files
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
    };

    let assert_re = Regex::new(r"\b(assert|assert_eq|assert_ne|debug_assert)!\s*\(").unwrap();
    let fn_re = Regex::new(r"\bfn\s+\w+\s*(?:<[^>]*>)?\s*\(").unwrap();
    let opt_out_re =
        Regex::new(r#"#\[(?:verified_engine::)?verified\(opt_out\s*=\s*"([^"]+)"\)\]"#).unwrap();
    let wasm_re =
        Regex::new(r#"#\[cfg\(target_arch\s*=\s*"wasm32"\)\]\s*(.*?)(?:#\[cfg|$)"#).unwrap();
    let legacy_re = Regex::new(r"\bfn\s+([a-zA-Z0-9_]*_legacy)\b").unwrap();

    for filepath in rs_files {
        if let Ok(content) = fs::read_to_string(filepath) {
            let is_verified =
                content.contains("#[verified") || content.contains("#[verified_engine::verified");

            let funcs = fn_re.find_iter(&content).count() as f64;
            let asserts = assert_re.find_iter(&content).count() as f64;

            for cap in opt_out_re.captures_iter(&content) {
                let mut norm = filepath.to_string_lossy().replace("\\", "/");
                if norm.starts_with("./") {
                    norm = norm[2..].to_string();
                }
                m.opt_outs.push((norm, cap[1].to_string()));
            }
            for cap in legacy_re.captures_iter(&content) {
                let mut norm = filepath.to_string_lossy().replace("\\", "/");
                if norm.starts_with("./") {
                    norm = norm[2..].to_string();
                }
                m.opt_outs
                    .push((norm, format!("Legacy function call: {}", &cap[1])));
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

            let content_str = content.replace("\r\n", "\n").replace("\n", " ");
            for cap in wasm_re.captures_iter(&content_str) {
                m.wasm_paths += 1;
                let block = &cap[1];
                if block.contains("theory_verification!")
                    || content.contains("theory_verification!")
                    || block.contains("stochastic_signature_verification!")
                    || content.contains("stochastic_signature_verification!")
                    || block.contains("empirical_verification!")
                    || content.contains("empirical_verification!")
                {
                    m.wasm_covered += 1;
                }
            }
        }
    }
    m.opt_outs.sort();
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
            let mut has_theory = false;
            for entry in WalkDir::new(mod_dir).into_iter().filter_map(|e| e.ok()) {
                if entry.file_type().is_file()
                    && entry.path().extension().and_then(|s| s.to_str()) == Some("rs")
                {
                    if let Ok(content) = fs::read_to_string(entry.path()) {
                        if content.contains("theory_verification!")
                            || content.contains("stochastic_signature_verification!")
                            || content.contains("empirical_verification!")
                        {
                            has_theory = true;
                            break;
                        }
                    }
                }
            }
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

pub fn print_report(
    native_cov_pct: f64,
    wasm_cov_pct: f64,
    m: &FileMetrics,
    debt_items: &[String],
) -> bool {
    let total_density = if m.total_funcs > 0.0 {
        m.total_asserts / m.total_funcs
    } else {
        0.0
    };
    let verified_density = if m.verified_funcs > 0.0 {
        m.verified_asserts / m.verified_funcs
    } else {
        0.0
    };
    let unverified_density = if m.unverified_funcs > 0.0 {
        m.unverified_asserts / m.unverified_funcs
    } else {
        0.0
    };

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

    let mut all_debt = Vec::new();
    for (file, reason) in &m.opt_outs {
        all_debt.push(format!("{} bypassed: '{}'", file, reason));
    }
    for item in debt_items {
        all_debt.push(item.clone());
    }

    if !all_debt.is_empty() {
        println!("\n--- Integrity Debt ---");
        for (idx, item) in all_debt.iter().enumerate() {
            println!("[{}] {}", idx + 1, item);
        }
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
        for f in &files {
            assert!(f.extension().and_then(|s| s.to_str()) == Some("rs"));
            let p_str = f.to_string_lossy().replace("\\", "/");
            assert!(!p_str.contains("/target/"));
            assert!(!p_str.contains("/egui_plot/"));
        }

        let unverified = check_unverified_modules(&["crates/pure_math".to_string()]);
        // pure_math has theory_verification! so it should not be unverified
        assert!(!unverified.iter().any(|u| u.contains("crates/pure_math")));
    }
}
