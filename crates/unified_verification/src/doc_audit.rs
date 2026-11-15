//! Automated Crate Documentation Verification Gate
//!
//! Enforces active documentation standards across workspace crates:
//! 1. Every workspace crate in `Cargo.toml` contains a localized `README.md`.
//! 2. Prohibited placeholder strings (e.g., `Legacy crate.`) are detected and rejected.
//! 3. Content thresholds (minimum line count and mandatory architectural headings) are enforced.
//! 4. Crate root entrypoints (`src/lib.rs` or `src/main.rs`) include localized documentation
//!    via `#![doc = include_str!("../README.md")]`.
//! 5. Documented allowlist and staging flag support for transitional legacy crates.

use regex::Regex;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

/// Default allowlist of legacy or application workspace crates exempt from
/// standardized architectural headings while awaiting manual authoring.
pub const DEFAULT_LEGACY_ALLOWLIST: &[&str] = &[
    "crates/diagnostics",
    "egui_plot",
    "apps/xtask",
    "math_explorer",
    "math_explorer_gui",
];

/// Minimum required lines for a standard crate README.
pub const MIN_README_LINES: usize = 30;

/// Mandatory architectural section headings for standard library crates.
pub const MANDATORY_HEADINGS: &[(&str, &str)] = &[
    ("## 1. Overview", "## 1. Overview & Architecture"),
    (
        "## 2. Developer Instructions",
        "## 2. Developer Instructions",
    ),
];

/// Prohibited placeholder patterns that indicate unauthored stubs.
pub const PROHIBITED_PLACEHOLDERS: &[&str] = &[
    "Legacy crate.",
    "Legacy crate",
    "Placeholder crate",
    "Placeholder documentation",
    "Under construction",
    "TODO: Add documentation",
    "TBD: Document",
];

#[derive(Deserialize, Default, Debug)]
struct WhitelistConfig {
    doc_verification: Option<DocVerificationConfig>,
}

#[derive(Deserialize, Default, Debug)]
struct DocVerificationConfig {
    allowlist: Option<Vec<String>>,
}

/// Audit report containing all found documentation violations.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct AuditReport {
    pub violations: Vec<String>,
}

impl AuditReport {
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
        }
    }

    pub fn is_success(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn add(&mut self, violation: String) {
        self.violations.push(violation);
    }

    pub fn extend(&mut self, violations: Vec<String>) {
        self.violations.extend(violations);
    }
}

/// Loads configured allowlist from `verification_whitelist.toml` if available,
/// merging with `DEFAULT_LEGACY_ALLOWLIST`.
pub fn load_allowlist(whitelist_path: &Path) -> HashSet<String> {
    let mut allowlist: HashSet<String> = DEFAULT_LEGACY_ALLOWLIST
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    if let Ok(content) = fs::read_to_string(whitelist_path) {
        if let Ok(config) = toml::from_str::<WhitelistConfig>(&content) {
            if let Some(doc_cfg) = config.doc_verification {
                if let Some(crates) = doc_cfg.allowlist {
                    for c in crates {
                        allowlist.insert(c);
                    }
                }
            }
        }
    }

    allowlist
}

/// Checks an entrypoint file (`lib.rs` or `main.rs`) for the required active doc attribute.
pub fn check_entrypoint_attribute(entry_path: &Path) -> Result<(), String> {
    let content = fs::read_to_string(entry_path)
        .map_err(|e| format!("{}: Could not read entrypoint: {}", entry_path.display(), e))?;

    let doc_regex = Regex::new(r#"^#!\[doc\s*=\s*include_str!\(\s*"(\.\./)?README\.md"\s*\)\]"#)
        .expect("Valid regex");

    let has_doc_attr = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.starts_with("//") && !l.starts_with("/*") && !l.starts_with('*'))
        .any(|l| doc_regex.is_match(l));

    if !has_doc_attr {
        return Err(format!(
            "{}: Missing required crate-level documentation attribute '#![doc = include_str!(\"../README.md\")]'",
            entry_path.display()
        ));
    }

    Ok(())
}

/// Scans a crate README file for prohibited placeholder strings.
pub fn check_prohibited_placeholders(readme_path: &Path, content: &str) -> Vec<String> {
    let mut violations = Vec::new();

    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line_trimmed = line.trim();

        for &placeholder in PROHIBITED_PLACEHOLDERS {
            if line_trimmed.contains(placeholder) {
                violations.push(format!(
                    "{}:{}: Prohibited placeholder string found: \"{}\"",
                    readme_path.display(),
                    line_num,
                    placeholder
                ));
                break;
            }
        }
    }

    violations
}

/// Checks minimum line count and required architectural headings.
pub fn check_content_thresholds(
    readme_path: &Path,
    content: &str,
    is_allowlisted: bool,
) -> Vec<String> {
    if is_allowlisted {
        return Vec::new();
    }

    let mut violations = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let line_count = lines.len();

    if line_count < MIN_README_LINES {
        violations.push(format!(
            "{}: README.md content below minimum threshold (has {} lines, minimum required is {})",
            readme_path.display(),
            line_count,
            MIN_README_LINES
        ));
    }

    for (prefix, full_name) in MANDATORY_HEADINGS {
        let has_heading = lines.iter().any(|l| l.trim().starts_with(prefix));
        if !has_heading {
            violations.push(format!(
                "{}: README.md missing mandatory architectural heading '{}'",
                readme_path.display(),
                full_name
            ));
        }
    }

    violations
}

/// Performs documentation audit on a single workspace crate directory.
pub fn audit_crate(crate_name: &str, crate_dir: &Path, is_allowlisted: bool) -> AuditReport {
    let mut report = AuditReport::new();

    // 1. Verify localized README.md exists
    let readme_path = crate_dir.join("README.md");
    if !readme_path.is_file() {
        report.add(format!(
            "{}: Missing localized README.md at {}",
            crate_name,
            readme_path.display()
        ));
        return report;
    }

    // Read README.md content
    let readme_content = match fs::read_to_string(&readme_path) {
        Ok(c) => c,
        Err(e) => {
            report.add(format!(
                "{}: Failed to read README.md: {}",
                readme_path.display(),
                e
            ));
            return report;
        }
    };

    // 2. Reject prohibited placeholder strings across ALL crates
    report.extend(check_prohibited_placeholders(&readme_path, &readme_content));

    // 3. Enforce content thresholds and architectural headings (unless allowlisted)
    report.extend(check_content_thresholds(
        &readme_path,
        &readme_content,
        is_allowlisted,
    ));

    // 4. Verify crate root entrypoints include active doc attribute
    let lib_rs = crate_dir.join("src").join("lib.rs");
    let main_rs = crate_dir.join("src").join("main.rs");
    let mut checked_entrypoint = false;

    if lib_rs.is_file() {
        checked_entrypoint = true;
        if let Err(e) = check_entrypoint_attribute(&lib_rs) {
            report.add(e);
        }
    }

    if main_rs.is_file() {
        checked_entrypoint = true;
        if let Err(e) = check_entrypoint_attribute(&main_rs) {
            report.add(e);
        }
    }

    if !checked_entrypoint {
        report.add(format!(
            "{}: No root entrypoint found (neither src/lib.rs nor src/main.rs exists)",
            crate_name
        ));
    }

    report
}

/// Main entrypoint to audit all workspace crates.
pub fn run_doc_audit(args: &[String]) -> bool {
    println!("=== Automated Crate Documentation Verification Gate ===");

    let staging_mode = args
        .iter()
        .any(|a| a == "--staging" || a == "--allow-legacy");
    let strict_mode = args.iter().any(|a| a == "--strict");

    if staging_mode {
        println!("Running in staging mode (legacy allowlist enabled).");
    }
    if strict_mode {
        println!("Running in strict mode (all crates must satisfy full architecture headings).");
    }

    let (is_root, members) = oxidize_core::workspace::get_workspace_members();
    if members.is_empty() {
        eprintln!("[!] No workspace members discovered.");
        return false;
    }

    let base_dir = if !is_root && Path::new("../../Cargo.toml").exists() {
        Path::new("../../")
    } else {
        Path::new("")
    };

    let whitelist_file = base_dir.join("verification_whitelist.toml");
    let allowlist = if strict_mode {
        HashSet::new()
    } else {
        load_allowlist(&whitelist_file)
    };

    let mut report = AuditReport::new();

    for member in &members {
        let crate_dir = base_dir.join(member);
        let is_allowlisted = allowlist.contains(member) || staging_mode;

        let crate_report = audit_crate(member, &crate_dir, is_allowlisted);
        report.extend(crate_report.violations);
    }

    if report.is_success() {
        println!(
            "[OK] All {} workspace crates passed documentation verification.",
            members.len()
        );
        true
    } else {
        eprintln!(
            "\n[!] Documentation Verification Failed: {} violation(s) found:\n",
            report.violations.len()
        );
        for v in &report.violations {
            eprintln!("  - {}", v);
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn create_test_dir() -> PathBuf {
        let n = TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("test_doc_audit_{}_{}", std::process::id(), n));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_check_prohibited_placeholders_finds_stub() {
        let test_dir = create_test_dir();
        let readme = test_dir.join("README.md");
        let content = "Legacy crate.\nSome other text\n";
        fs::write(&readme, content).unwrap();

        let violations = check_prohibited_placeholders(&readme, content);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("README.md:1: Prohibited placeholder string found"));

        let _ = fs::remove_dir_all(&test_dir);
    }

    #[test]
    fn test_check_prohibited_placeholders_clean() {
        let test_dir = create_test_dir();
        let readme = test_dir.join("README.md");
        let content = "# Valid Crate\n\nDetailed documentation here.\n";
        fs::write(&readme, content).unwrap();

        let violations = check_prohibited_placeholders(&readme, content);
        assert!(violations.is_empty());

        let _ = fs::remove_dir_all(&test_dir);
    }

    #[test]
    fn test_check_content_thresholds_missing_headings() {
        let readme = Path::new("test/README.md");
        let content = "Line 1\nLine 2\nLine 3\nLine 4\n";
        let violations = check_content_thresholds(readme, content, false);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("below minimum threshold"))
        );
        assert!(violations.iter().any(|v| {
            v.contains("missing mandatory architectural heading '## 1. Overview & Architecture'")
        }));
        assert!(violations.iter().any(|v| {
            v.contains("missing mandatory architectural heading '## 2. Developer Instructions'")
        }));
    }

    #[test]
    fn test_check_content_thresholds_allowlisted_skips() {
        let readme = Path::new("test/README.md");
        let content = "Short stub\n";
        let violations = check_content_thresholds(readme, content, true);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_check_entrypoint_attribute_valid() {
        let test_dir = create_test_dir();
        let lib_rs = test_dir.join("lib.rs");
        fs::write(
            &lib_rs,
            "#![doc = include_str!(\"../README.md\")]\npub fn foo() {}\n",
        )
        .unwrap();

        assert!(check_entrypoint_attribute(&lib_rs).is_ok());
        let _ = fs::remove_dir_all(&test_dir);
    }

    #[test]
    fn test_check_entrypoint_attribute_missing() {
        let test_dir = create_test_dir();
        let lib_rs = test_dir.join("lib.rs");
        fs::write(&lib_rs, "pub fn foo() {}\n").unwrap();

        let err = check_entrypoint_attribute(&lib_rs).unwrap_err();
        assert!(err.contains("Missing required crate-level documentation attribute"));

        let _ = fs::remove_dir_all(&test_dir);
    }

    #[test]
    fn test_check_entrypoint_attribute_commented_out() {
        let test_dir = create_test_dir();
        let lib_rs = test_dir.join("lib.rs");
        fs::write(
            &lib_rs,
            "// #![doc = include_str!(\"../README.md\")]\npub fn foo() {}\n",
        )
        .unwrap();

        let err = check_entrypoint_attribute(&lib_rs).unwrap_err();
        assert!(err.contains("Missing required crate-level documentation attribute"));

        let _ = fs::remove_dir_all(&test_dir);
    }

    #[test]
    fn test_audit_crate_end_to_end_success() {
        let test_dir = create_test_dir();
        let readme_content = format!(
            "# Demo Crate\n\n## 1. Overview & Architecture\n\nIntro\n\n## 2. Developer Instructions\n\nInstructions\n{}",
            "\nMore content...".repeat(25)
        );
        fs::write(test_dir.join("README.md"), readme_content).unwrap();

        let src_dir = test_dir.join("src");
        fs::create_dir_all(&src_dir).unwrap();
        fs::write(
            src_dir.join("lib.rs"),
            "#![doc = include_str!(\"../README.md\")]\npub fn hello() {}\n",
        )
        .unwrap();

        let report = audit_crate("test_crate", &test_dir, false);
        assert!(report.is_success());

        let _ = fs::remove_dir_all(&test_dir);
    }
}
