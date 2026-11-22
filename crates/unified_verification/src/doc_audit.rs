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
use std::collections::HashSet;
use std::fs;
use std::path::Path;

pub mod facade_imports;

#[cfg(test)]
mod tests;

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

    let cfg = crate::whitelist::load_whitelist_or_default(None, &[whitelist_path]);
    if let Some(crates) = cfg.doc_verification.and_then(|d| d.allowlist) {
        allowlist.extend(crates);
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

fn resolve_doc_base_dir(is_root: bool) -> std::path::PathBuf {
    if is_root {
        return std::path::PathBuf::new();
    }
    oxidize_core::workspace::find_workspace_root().unwrap_or_else(|| {
        if Path::new("../../Cargo.toml").exists() {
            std::path::PathBuf::from("../..")
        } else {
            std::path::PathBuf::new()
        }
    })
}

/// Main entrypoint to audit all discovered workspace crates.
pub fn run_doc_audit(args: &[String]) -> bool {
    let (is_root, members) = oxidize_core::workspace::get_workspace_members();
    if members.is_empty() {
        println!("=== Automated Crate Documentation Verification Gate ===");
        eprintln!("[!] No workspace members discovered.");
        return false;
    }
    let base_dir = resolve_doc_base_dir(is_root);
    run_doc_audit_in_base(&base_dir, &members, args)
}

/// Audits an explicit slice of `members` relative to the active workspace root.
pub fn run_doc_audit_for_members(members: &[String], args: &[String]) -> bool {
    let (is_root, _) = oxidize_core::workspace::get_workspace_members();
    let base_dir = resolve_doc_base_dir(is_root);
    run_doc_audit_in_base(&base_dir, members, args)
}

fn run_doc_audit_in_base(base_dir: &Path, members: &[String], args: &[String]) -> bool {
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
    if members.is_empty() {
        eprintln!("[!] No workspace members discovered.");
        return false;
    }

    let candidate = base_dir.join(crate::whitelist::WHITELIST_FILE_NAME);
    let allowlist = if strict_mode {
        HashSet::new()
    } else {
        let whitelist_file =
            crate::whitelist::resolve_whitelist_path(&[candidate.as_path()]).unwrap_or(candidate);
        load_allowlist(&whitelist_file)
    };

    let mut report = AuditReport::new();
    for member in members {
        let crate_dir = base_dir.join(member);
        let is_allowlisted = allowlist.contains(member) || staging_mode;
        report.extend(audit_crate(member, &crate_dir, is_allowlisted).violations);
    }

    let workspace_root = if base_dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        base_dir
    };
    report.extend(facade_imports::audit_facade_import_hygiene(workspace_root));

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
