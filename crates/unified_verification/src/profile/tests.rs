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

fn write_synthetic_root(ver_line: &str) {
    fs::write(
        "Cargo.toml",
        format!(
            "[workspace]\nmembers = [\"math_explorer\", \"math_explorer_gui\", \"crates/pure_math\", \"crates/oxidize_core\", \"egui_plot\"]\n{ver_line}\n[patch.crates-io]\negui_plot = {{ path = \"egui_plot\" }}\n[profile.release]\noverflow-checks = true\n"
        ),
    )
    .unwrap();
}

fn write_synthetic_readme(badge_ver: &str, dep_ver: &str) {
    fs::write(
        "README.md",
        format!(
            "# Math Explorer\n![Version](https://img.shields.io/badge/version-{badge_ver}-blue.svg)\n```toml\n[dependencies]\nmath_explorer = {{ version = \"{dep_ver}\", features = [\"physics\"] }}\n```\n"
        ),
    )
    .unwrap();
}

fn seed_synthetic_version_workspace() {
    for dir in [
        "math_explorer",
        "math_explorer_gui",
        "crates/pure_math",
        "crates/oxidize_core",
        "egui_plot",
    ] {
        fs::create_dir_all(dir).unwrap();
    }
    write_synthetic_root("[workspace.package]\nversion = \"0.4.0\"");
    write_synthetic_readme("0.4.0", "0.4.0");
    fs::write(
        "math_explorer/Cargo.toml",
        "[package]\nname = \"math_explorer\"\nversion.workspace = true\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    fs::write(
        "math_explorer_gui/Cargo.toml",
        "[package]\nname = \"math_explorer_gui\"\nversion.workspace = true\n\n[package.metadata.packager]\nproduct-name = \"Math Explorer\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    fs::write(
        "crates/pure_math/Cargo.toml",
        "[package]\nname = \"pure_math\"\nversion.workspace = true\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    fs::write(
        "crates/oxidize_core/Cargo.toml",
        "[package]\nname = \"oxidize_core\"\nversion = \"0.1.0\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    fs::write(
        "egui_plot/Cargo.toml",
        "[package]\nname = \"egui_plot\"\nversion = \"0.34.1\"\n\n[lints.rust]\nmissing_docs = \"warn\"\n[lints.clippy]\ntoo_many_lines = \"deny\"\n",
    )
    .unwrap();
}

#[test]
fn test_workspace_version_inheritance_packager_override_and_readme_sync() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let temp_dir = TempTestDir::new("unified_verification_profile_versions");
    let _cwd_guard = CurrentDirGuard(env::current_dir().unwrap());
    env::set_current_dir(&*temp_dir).unwrap();
    seed_synthetic_version_workspace();

    let members = [
        "math_explorer",
        "math_explorer_gui",
        "crates/pure_math",
        "crates/oxidize_core",
        "egui_plot",
    ];
    assert!(check_profiles(&members, false));

    // Hardcoded release crate version must fail
    fs::write(
        "crates/pure_math/Cargo.toml",
        "[package]\nname = \"pure_math\"\nversion = \"0.4.0\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    assert!(!check_profiles(&members, false));
    fs::write(
        "crates/pure_math/Cargo.toml",
        "[package]\nname = \"pure_math\"\nversion.workspace = true\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();

    // Duplicate [package.metadata.packager].version must fail
    fs::write(
        "math_explorer_gui/Cargo.toml",
        "[package]\nname = \"math_explorer_gui\"\nversion.workspace = true\n\n[package.metadata.packager]\nversion = \"0.4.0\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    assert!(!check_profiles(&members, false));
    fs::write(
        "math_explorer_gui/Cargo.toml",
        "[package]\nname = \"math_explorer_gui\"\nversion.workspace = true\n\n[package.metadata.packager]\nproduct-name = \"Math Explorer\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();

    // README badge or dependency version drift, missing root version, or WiX bounds violations must fail
    for (badge, dep) in [("0.3.0", "0.4.0"), ("0.4.0", "0.3.0")] {
        write_synthetic_readme(badge, dep);
        assert!(!check_profiles(&members, false));
    }
    write_synthetic_readme("0.4.0", "0.4.0");
    for bad_root in [
        "",
        "[workspace.package]\nversion = \"0.4.0+build.1\"",
        "[workspace.package]\nversion = \"256.0.0\"",
    ] {
        write_synthetic_root(bad_root);
        assert!(!check_profiles(&members, false));
    }

    // Pre-release workspace version with shields.io '--' hyphen escaping must pass
    write_synthetic_root("[workspace.package]\nversion = \"0.5.0-rc.1\"");
    write_synthetic_readme("0.5.0--rc.1", "0.5.0-rc.1");
    assert!(check_profiles(&members, false));
}
