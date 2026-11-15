use std::fs;
use std::path::Path;
use toml::Value;

pub fn check_profiles(workspace_members: &[&str], auto_fix: bool) -> bool {
    let mut passed = true;

    // 1. Check workspace root
    let root_path = Path::new("Cargo.toml");
    if root_path.exists() {
        let root_content = fs::read_to_string(root_path).unwrap_or_default();
        let root_parsed: Value =
            toml::from_str(&root_content).unwrap_or_else(|_| Value::Table(Default::default()));

        let root_has_overflow_checks = root_parsed
            .get("profile")
            .and_then(|p| p.get("release"))
            .and_then(|r| r.get("overflow-checks"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if !root_has_overflow_checks {
            if auto_fix {
                println!(
                    "[+] Auto-fixing: Adding [profile.release] overflow-checks = true to workspace root Cargo.toml"
                );
                let mut new_root_content = root_content.clone();
                if !new_root_content.ends_with('\n') {
                    new_root_content.push('\n');
                }
                new_root_content.push_str("\n[profile.release]\noverflow-checks = true\n");
                fs::write(root_path, new_root_content).unwrap();
            } else {
                eprintln!(
                    "[!] Profile mismatch: workspace root Cargo.toml is missing [profile.release] overflow-checks = true"
                );
                passed = false;
            }
        } else {
            println!(
                "[+] Verified workspace root profile: overflow-checks = true is strictly enforced."
            );
        }
    }

    for member in workspace_members {
        let path = Path::new(member).join("Cargo.toml");
        if path.exists() {
            let content = fs::read_to_string(&path).unwrap();
            let parsed: Value = match toml::from_str(&content) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Failed to parse TOML for {}: {}", path.display(), e);
                    continue;
                }
            };

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

    #[test]
    fn test_member_shadowing_failure() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let temp_dir = env::temp_dir().join("unified_verification_test");
        fs::create_dir_all(&temp_dir).unwrap();

        // Save current directory and change to temp_dir
        let original_dir = env::current_dir().unwrap();
        env::set_current_dir(&temp_dir).unwrap();

        // Create root Cargo.toml
        fs::write(
            "Cargo.toml",
            "[workspace]\nmembers = [\"member1\"]\n[profile.release]\noverflow-checks = true\n",
        )
        .unwrap();

        // Create member Cargo.toml with shadowing profile
        fs::create_dir_all("member1").unwrap();
        fs::write(
            "member1/Cargo.toml",
            "[package]\nname = \"member1\"\nversion = \"0.1.0\"\n\n[profile.release]\noverflow-checks = false\n",
        )
        .unwrap();

        // Check profiles (auto_fix = false)
        let passed = check_profiles(&["member1"], false);

        // It should fail due to shadowing
        assert!(
            !passed,
            "Linting should fail when member crate shadows [profile.release]"
        );

        // Clean up
        env::set_current_dir(original_dir).unwrap();
        fs::remove_dir_all(&temp_dir).unwrap();
    }
}
