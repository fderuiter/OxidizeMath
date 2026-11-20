use std::fs;
use std::path::Path;
use toml::Value;

fn check_root_profile(root_path: &Path, auto_fix: bool) -> bool {
    if !root_path.exists() {
        return true;
    }

    let root_content = fs::read_to_string(root_path).unwrap_or_default();
    let root_parsed: Value =
        toml::from_str(&root_content).unwrap_or_else(|_| Value::Table(Default::default()));

    let root_has_overflow_checks = root_parsed
        .get("profile")
        .and_then(|p| p.get("release"))
        .and_then(|r| r.get("overflow-checks"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if root_has_overflow_checks {
        println!(
            "[+] Verified workspace root profile: overflow-checks = true is strictly enforced."
        );
        return true;
    }

    if auto_fix {
        println!(
            "[+] Auto-fixing: Adding [profile.release] overflow-checks = true to workspace root Cargo.toml"
        );
        let mut new_root_content = root_content;
        if !new_root_content.ends_with('\n') {
            new_root_content.push('\n');
        }
        new_root_content.push_str("\n[profile.release]\noverflow-checks = true\n");
        fs::write(root_path, new_root_content).unwrap();
        true
    } else {
        eprintln!(
            "[!] Profile mismatch: workspace root Cargo.toml is missing [profile.release] overflow-checks = true"
        );
        false
    }
}

fn check_member_manifest(member: &str, auto_fix: bool) -> bool {
    let path = Path::new(member).join("Cargo.toml");
    if !path.exists() {
        return true;
    }

    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read {}: {}", path.display(), e);
            return false;
        }
    };
    let parsed: Value = match toml::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to parse TOML for {}: {}", path.display(), e);
            return false;
        }
    };

    let mut passed = crate::policy_audit::lint_policy::check_manifest_lints(member, &path, &parsed);

    let has_profile_release = parsed
        .get("profile")
        .and_then(|p| p.get("release"))
        .is_some();

    if has_profile_release {
        if auto_fix {
            println!(
                "[+] Auto-fixing: Removing redundant [profile.release] from {}",
                path.display()
            );
            let new_content = remove_profile_release(&content);
            fs::write(&path, new_content).unwrap();
        } else {
            eprintln!(
                "[!] Profile mismatch in {}: redundant [profile.release] found. Configuration must be consolidated in workspace root.",
                path.display()
            );
            passed = false;
        }
    }

    passed
}

pub fn check_profiles(workspace_members: &[&str], auto_fix: bool) -> bool {
    let mut passed = check_root_profile(Path::new("Cargo.toml"), auto_fix);

    for member in workspace_members {
        if !check_member_manifest(member, auto_fix) {
            passed = false;
        }
    }

    passed
}

fn remove_profile_release(content: &str) -> String {
    let mut out = String::new();
    let mut in_profile_release = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[profile.release]") {
            in_profile_release = true;
            continue;
        }
        if in_profile_release && trimmed.starts_with('[') {
            in_profile_release = false;
        }
        if !in_profile_release {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.trim_end().to_string() + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    struct DirGuard {
        original: PathBuf,
        temp: PathBuf,
    }

    impl Drop for DirGuard {
        fn drop(&mut self) {
            let _ = env::set_current_dir(&self.original);
            let _ = fs::remove_dir_all(&self.temp);
        }
    }

    #[test]
    fn test_member_shadowing_failure() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let temp_dir = env::temp_dir().join("unified_verification_test");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let original_dir = env::current_dir().unwrap();
        env::set_current_dir(&temp_dir).unwrap();
        let _guard = DirGuard {
            original: original_dir,
            temp: temp_dir,
        };

        fs::write(
            "Cargo.toml",
            "[workspace]\nmembers = [\"member1\"]\n[profile.release]\noverflow-checks = true\n",
        )
        .unwrap();

        fs::create_dir_all("member1").unwrap();
        fs::write(
            "member1/Cargo.toml",
            "[package]\nname = \"member1\"\nversion = \"0.1.0\"\n\n[lints]\nworkspace = true\n\n[profile.release]\noverflow-checks = false\n",
        )
        .unwrap();

        let passed = check_profiles(&["member1"], false);
        assert!(
            !passed,
            "Linting should fail when member crate shadows [profile.release]"
        );
    }

    #[test]
    fn test_member_lints_inheritance_enforced() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let temp_dir = env::temp_dir().join("unified_verification_profile_lints_test");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let original_dir = env::current_dir().unwrap();
        env::set_current_dir(&temp_dir).unwrap();
        let _guard = DirGuard {
            original: original_dir,
            temp: temp_dir,
        };

        fs::write(
            "Cargo.toml",
            "[workspace]\nmembers = [\"member1\", \"egui_plot\"]\n[profile.release]\noverflow-checks = true\n",
        )
        .unwrap();

        fs::create_dir_all("member1").unwrap();
        fs::write(
            "member1/Cargo.toml",
            "[package]\nname = \"member1\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        assert!(
            !check_profiles(&["member1"], false),
            "check_profiles must reject member Cargo.toml missing [lints] workspace = true"
        );

        fs::write(
            "member1/Cargo.toml",
            "[package]\nname = \"member1\"\nversion = \"0.1.0\"\n\n[lints]\nworkspace = true\n",
        )
        .unwrap();
        assert!(check_profiles(&["member1"], false));

        fs::create_dir_all("egui_plot").unwrap();
        fs::write(
            "egui_plot/Cargo.toml",
            "[package]\nname = \"egui_plot\"\nversion = \"0.34.0\"\n\n[lints.rust]\nmissing_docs = \"warn\"\n[lints.clippy]\ntoo_many_lines = \"deny\"\n",
        )
        .unwrap();
        assert!(check_profiles(&["member1", "egui_plot"], false));
    }
}
