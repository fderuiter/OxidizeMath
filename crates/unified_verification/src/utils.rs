use regex::Regex;
use std::fs;
use std::path::Path;
use std::process::Command;
use walkdir::WalkDir;

pub fn check_file_lengths(members: &[String]) -> Vec<String> {
    let mut exceeding = Vec::new();

    for dir in members {
        if dir == "egui_plot" {
            continue;
        }
        if Path::new(dir).exists() {
            for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
                let path_str = entry.path().to_string_lossy().replace("\\", "/");
                if path_str.contains("/target/")
                    || path_str.contains("/egui_plot/")
                    || path_str.starts_with("egui_plot/")
                {
                    continue;
                }
                if entry.file_type().is_file()
                    && entry.path().extension().and_then(|s| s.to_str()) == Some("rs")
                    && let Ok(content) = fs::read_to_string(entry.path())
                {
                    let lines = content.lines().count();
                    if lines > 500 {
                        exceeding.push(format!(
                            "File length violation: {} ({} lines)",
                            entry.path().display(),
                            lines
                        ));
                    }
                }
            }
        }
    }
    exceeding
}

pub fn check_staged_duplicates() {
    let output = Command::new("git")
        .args(["diff", "--cached", "--no-color"])
        .output()
        .expect("Failed to run git diff --cached");

    if !output.status.success() {
        return;
    }

    let diff = String::from_utf8_lossy(&output.stdout);
    let mut current_file = String::new();
    let mut warnings = Vec::new();

    let trait_re = Regex::new(r"trait\s+\w*(Solver|Integrator)").unwrap();
    let proj_re =
        Regex::new(r"(fovy|aspect|projection|look_at|Perspective|matrix element)").unwrap();

    for line in diff.lines() {
        if let Some(stripped) = line.strip_prefix("+++ b/") {
            current_file = stripped.to_string();
        } else if line.starts_with('+') && !line.starts_with("+++") {
            let content = &line[1..];
            if trait_re.is_match(content) {
                warnings.push(format!("File {}: Possible duplicated mathematical trait detected: '{}'. Consider reusing existing utilities.", current_file, content.trim()));
            }
            if proj_re.is_match(content) {
                warnings.push(format!("File {}: Possible duplicated camera/projection logic detected: '{}'. Consider consolidating.", current_file, content.trim()));
            }
        }
    }

    if !warnings.is_empty() {
        println!("=== Code Duplication Warnings ===");
        for w in warnings {
            println!("WARN: {}", w);
        }
        println!(
            "Please verify you are not duplicating existing utilities in pure_math or math_commons."
        );
        println!("These warnings are non-blocking.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_file_lengths() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let members = vec!["crates/pure_math".to_string()];
        let violations = check_file_lengths(&members);
        // files should not exceed 500 lines limit in pure_math
        assert!(violations.is_empty(), "Violations: {:?}", violations);
    }

    #[test]
    fn test_check_staged_duplicates_run() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        // Should execute git diff without panic
        check_staged_duplicates();
    }
}
