use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use walkdir::WalkDir;

fn should_visit_entry(root: &Path, entry: &walkdir::DirEntry, excluded_dirs: &[&str]) -> bool {
    if entry.depth() == 0 {
        if root.is_relative() {
            let check_path = if entry.file_type().is_dir() {
                Some(root)
            } else {
                root.parent()
            };
            if let Some(p) = check_path {
                let norm = crate::whitelist::lossy_slash_path(p);
                let trimmed = norm.trim_start_matches("./");
                if trimmed
                    .split('/')
                    .filter(|seg| !seg.is_empty() && *seg != ".")
                    .any(|seg| excluded_dirs.contains(&seg))
                {
                    return false;
                }
            }
            return true;
        }
        if entry.file_type().is_dir()
            && let Some(name) = root.file_name()
        {
            return !excluded_dirs.contains(&name.to_string_lossy().as_ref());
        }
        return true;
    }
    if entry.file_type().is_dir() {
        return !excluded_dirs.contains(&entry.file_name().to_string_lossy().as_ref());
    }
    true
}

/// Collects sorted `.rs` file paths across `roots`, skipping any directories in `excluded_dirs`.
pub(crate) fn collect_rs_files_filtered<P: AsRef<Path>>(
    roots: &[P],
    excluded_dirs: &[&str],
) -> Vec<PathBuf> {
    let mut rs_files = Vec::new();
    for root_ref in roots {
        let root = root_ref.as_ref();
        if !root.exists() {
            continue;
        }
        for entry in WalkDir::new(root)
            .into_iter()
            .filter_entry(|e| should_visit_entry(root, e, excluded_dirs))
            .filter_map(Result::ok)
        {
            if entry.file_type().is_file()
                && entry.path().extension().and_then(|s| s.to_str()) == Some("rs")
            {
                rs_files.push(entry.path().to_path_buf());
            }
        }
    }
    rs_files.sort();
    rs_files
}

pub fn check_file_lengths(members: &[String]) -> Vec<String> {
    let mut exceeding = Vec::new();

    for path in collect_rs_files_filtered(members, &["target", "egui_plot"]) {
        if let Ok(content) = fs::read_to_string(&path) {
            let lines = content.lines().count();
            if lines > 500 {
                exceeding.push(format!(
                    "File length violation: {} ({} lines)",
                    path.display(),
                    lines
                ));
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
        if line.starts_with("+++ b/") {
            current_file = line["+++ b/".len()..].to_string();
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
pub(crate) mod test_utils {
    use std::fs;
    use std::ops::Deref;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEMP_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

    pub(crate) fn create_temp_test_dir(prefix: &str) -> PathBuf {
        let id = TEMP_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), id));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    pub(crate) struct TempTestDir(PathBuf);

    impl TempTestDir {
        pub(crate) fn new(prefix: &str) -> Self {
            Self(create_temp_test_dir(prefix))
        }
    }

    impl Deref for TempTestDir {
        type Target = Path;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl Drop for TempTestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    pub(crate) struct CurrentDirGuard(pub(crate) PathBuf);

    impl Drop for CurrentDirGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.0);
        }
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

    #[test]
    fn test_collect_rs_files_filtered_excludes_and_sorts() {
        let dir = test_utils::TempTestDir::new("test_rs_walker");
        for rel in [
            "src/z.rs",
            "src/a.rs",
            "src/notes.md",
            "target/gen.rs",
            ".git/hook.rs",
            "egui_plot/plot.rs",
        ] {
            let p = dir.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, "pub fn item() {}\n").unwrap();
        }

        let report_set = collect_rs_files_filtered(
            &[&*dir, &dir.join("missing")],
            &["target", ".git", "egui_plot"],
        );
        assert_eq!(report_set, vec![dir.join("src/a.rs"), dir.join("src/z.rs")]);

        let utils_set = collect_rs_files_filtered(&[&*dir], &["target", "egui_plot"]);
        assert_eq!(
            utils_set,
            vec![
                dir.join(".git/hook.rs"),
                dir.join("src/a.rs"),
                dir.join("src/z.rs")
            ]
        );
    }
}
