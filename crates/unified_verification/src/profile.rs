use crate::manifest_cache::ManifestCache;
use std::fs;
use std::path::Path;
use toml::Value;

fn check_root_profile(
    root_path: &Path,
    auto_fix: bool,
    manifest_cache: &mut ManifestCache,
) -> bool {
    let (root_content, root_has_overflow_checks) = match manifest_cache.get_or_load(root_path) {
        Ok(cached) => {
            let has_checks = cached
                .parsed
                .get("profile")
                .and_then(|p| p.get("release"))
                .and_then(|r| r.get("overflow-checks"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            (cached.content.clone(), has_checks)
        }
        Err(err) => {
            eprintln!("{err}");
            return false;
        }
    };

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
        manifest_cache.invalidate(root_path);
        true
    } else {
        eprintln!(
            "[!] Profile mismatch: workspace root Cargo.toml is missing [profile.release] overflow-checks = true"
        );
        false
    }
}

fn check_member_manifest(member: &str, auto_fix: bool, manifest_cache: &mut ManifestCache) -> bool {
    let path = crate::policy_audit::lint_policy::resolve_member_dir(member).join("Cargo.toml");
    let (content, parsed) = match manifest_cache.get_or_load(&path) {
        Ok(cached) => (cached.content.clone(), cached.parsed.clone()),
        Err(err) => {
            eprintln!("{err}");
            return false;
        }
    };

    let mut passed = manifest_cache.check_manifest_lints_once(member, &path, &parsed);

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
            manifest_cache.invalidate(&path);
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
    let mut manifest_cache = ManifestCache::new();
    check_profiles_with_cache(workspace_members, auto_fix, &mut manifest_cache)
}

pub(crate) fn check_profiles_with_cache(
    workspace_members: &[&str],
    auto_fix: bool,
    manifest_cache: &mut ManifestCache,
) -> bool {
    let root_manifest = crate::manifest_cache::resolve_workspace_path("Cargo.toml");
    let mut passed = check_root_profile(&root_manifest, auto_fix, manifest_cache);

    for member in workspace_members {
        if !check_member_manifest(member, auto_fix, manifest_cache) {
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
    use crate::utils::test_utils::{CurrentDirGuard, TempTestDir};
    use std::env;
    use std::fs;

    #[test]
    fn test_member_shadowing_failure() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let temp_dir = TempTestDir::new("unified_verification_profile_shadow");
        let _cwd_guard = CurrentDirGuard(env::current_dir().unwrap());
        env::set_current_dir(&*temp_dir).unwrap();

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
        let temp_dir = TempTestDir::new("unified_verification_profile_lints");
        let _cwd_guard = CurrentDirGuard(env::current_dir().unwrap());
        env::set_current_dir(&*temp_dir).unwrap();

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

    #[test]
    fn test_check_profiles_subdir_and_missing_manifest_fail_closed() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let temp_dir = TempTestDir::new("unified_verification_profile_subdir");
        let _cwd_guard = CurrentDirGuard(env::current_dir().unwrap());

        fs::write(
            temp_dir.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/m1\", \"crates/m2\"]\n[profile.release]\noverflow-checks = true\n",
        )
        .unwrap();
        fs::create_dir_all(temp_dir.join("crates/m1")).unwrap();
        fs::create_dir_all(temp_dir.join("crates/m2")).unwrap();
        fs::write(
            temp_dir.join("crates/m1/Cargo.toml"),
            "[package]\nname = \"m1\"\nversion = \"0.1.0\"\n\n[lints]\nworkspace = true\n",
        )
        .unwrap();
        fs::write(
            temp_dir.join("crates/m2/Cargo.toml"),
            "[package]\nname = \"m2\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();

        env::set_current_dir(&*temp_dir).unwrap();
        assert!(!check_profiles(&["crates/m1", "crates/m2"], false));
        assert!(!check_profiles(&["crates/m1", "crates/missing"], false));

        env::set_current_dir(temp_dir.join("crates/m1")).unwrap();
        assert!(!check_profiles(&["crates/m1", "crates/m2"], false));
        assert!(!check_profiles(&["crates/m1", "crates/missing"], false));

        fs::write(
            temp_dir.join("crates/m2/Cargo.toml"),
            "[package]\nname = \"m2\"\nversion = \"0.1.0\"\n\n[lints]\nworkspace = true\n",
        )
        .unwrap();
        assert!(check_profiles(&["crates/m1", "crates/m2"], false));
    }

    #[test]
    fn test_auto_fix_invalidates_cached_root_and_member_manifests() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let temp_dir = TempTestDir::new("unified_verification_profile_autofix");
        let _cwd_guard = CurrentDirGuard(env::current_dir().unwrap());
        env::set_current_dir(&*temp_dir).unwrap();

        fs::write("Cargo.toml", "[workspace]\nmembers = [\"crates/m1\"]\n").unwrap();
        fs::create_dir_all("crates/m1").unwrap();
        fs::write(
            "crates/m1/Cargo.toml",
            "[package]\nname = \"m1\"\nversion = \"0.1.0\"\n\n[lints]\nworkspace = true\n\n[profile.release]\noverflow-checks = false\n",
        )
        .unwrap();

        let mut cache = ManifestCache::new();
        assert!(cache.get_or_load(Path::new("Cargo.toml")).is_ok());
        assert!(cache.get_or_load(Path::new("crates/m1/Cargo.toml")).is_ok());

        assert!(check_profiles_with_cache(&["crates/m1"], true, &mut cache));
        assert!(!cache.contains(Path::new("Cargo.toml")));
        assert!(!cache.contains(Path::new("crates/m1/Cargo.toml")));
        assert!(check_profiles_with_cache(&["crates/m1"], false, &mut cache));
    }
}
