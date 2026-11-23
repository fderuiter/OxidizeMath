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

const FORBIDDEN_MEMBER_PROFILES: &[&str] = &["release", "dev", "test"];

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
    let mut updated_content: Option<String> = None;

    for profile_name in FORBIDDEN_MEMBER_PROFILES {
        let has_profile = parsed
            .get("profile")
            .and_then(|p| p.get(*profile_name))
            .is_some();
        if !has_profile {
            continue;
        }
        if auto_fix {
            println!(
                "[+] Auto-fixing: Removing redundant [profile.{profile_name}] from {}",
                path.display()
            );
            let base = updated_content.as_deref().unwrap_or(&content);
            updated_content = Some(remove_profile_section(base, profile_name));
        } else {
            eprintln!(
                "[!] Profile mismatch in {}: redundant [profile.{profile_name}] found. Configuration must be consolidated in workspace root.",
                path.display()
            );
            passed = false;
        }
    }

    if let Some(new_content) = updated_content {
        fs::write(&path, new_content).unwrap();
        manifest_cache.invalidate(&path);
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

fn is_matching_profile_header(trimmed: &str, profile_name: &str) -> bool {
    let Some(rest) = trimmed.strip_prefix("[profile.") else {
        return false;
    };
    let Some(after_name) = rest.strip_prefix(profile_name) else {
        return false;
    };
    after_name.starts_with(']') || after_name.starts_with('.')
}

fn remove_profile_section(content: &str, profile_name: &str) -> String {
    let mut out = String::new();
    let mut in_target_profile = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if is_matching_profile_header(trimmed, profile_name) {
            in_target_profile = true;
            continue;
        }
        if in_target_profile && trimmed.starts_with('[') {
            in_target_profile = false;
        }
        if !in_target_profile {
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

    #[test]
    fn test_member_dev_and_test_shadowing_rejected_and_autofixed() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let temp_dir = TempTestDir::new("unified_verification_profile_dev_test");
        let _cwd_guard = CurrentDirGuard(env::current_dir().unwrap());
        env::set_current_dir(&*temp_dir).unwrap();

        fs::write(
            "Cargo.toml",
            "[workspace]\nmembers = [\"crates/m_dev\", \"crates/m_test\"]\n[profile.release]\noverflow-checks = true\n",
        )
        .unwrap();
        fs::create_dir_all("crates/m_dev").unwrap();
        fs::create_dir_all("crates/m_test").unwrap();

        fs::write(
            "crates/m_dev/Cargo.toml",
            "[package]\nname = \"m_dev\"\nversion = \"0.1.0\"\n\n[lints]\nworkspace = true\n\n[profile.dev]\nopt-level = 1\n\n[profile.dev.package.\"*\"]\nopt-level = 3\n",
        )
        .unwrap();
        fs::write(
            "crates/m_test/Cargo.toml",
            "[package]\nname = \"m_test\"\nversion = \"0.1.0\"\n\n[lints]\nworkspace = true\n\n[profile.test]\nopt-level = 1\n",
        )
        .unwrap();

        assert!(
            !check_profiles(&["crates/m_dev"], false),
            "check_profiles must reject member [profile.dev] override"
        );
        assert!(
            !check_profiles(&["crates/m_test"], false),
            "check_profiles must reject member [profile.test] override"
        );

        let mut cache = ManifestCache::new();
        assert!(check_profiles_with_cache(
            &["crates/m_dev", "crates/m_test"],
            true,
            &mut cache
        ));
        assert!(check_profiles_with_cache(
            &["crates/m_dev", "crates/m_test"],
            false,
            &mut cache
        ));
    }

    #[test]
    fn test_workspace_root_dev_and_test_profiles_configured() {
        let root_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.toml");
        let raw = fs::read_to_string(&root_path).expect("read workspace Cargo.toml");
        let parsed: Value = toml::from_str(&raw).expect("parse workspace Cargo.toml");
        let profile = parsed.get("profile").expect("root [profile] table");

        for name in ["dev", "test"] {
            let section = profile
                .get(name)
                .unwrap_or_else(|| panic!("[profile.{name}]"));
            assert_eq!(
                section.get("opt-level").and_then(Value::as_integer),
                Some(0)
            );
            assert_eq!(
                section.get("debug").and_then(Value::as_str),
                Some("line-tables-only")
            );
            let star = section
                .get("package")
                .and_then(|p| p.get("*"))
                .unwrap_or_else(|| panic!("[profile.{name}.package.\"*\"]"));
            assert_eq!(star.get("opt-level").and_then(Value::as_integer), Some(2));
            assert_eq!(star.get("debug").and_then(Value::as_bool), Some(false));
        }

        let release = profile.get("release").expect("[profile.release]");
        assert_eq!(
            release.get("overflow-checks").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(release.get("lto").and_then(Value::as_bool), Some(true));
    }
}
