//! Integration tests verifying modular `xtask setup` CLI flags and `.githooks` ordering.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn local_hooks_path(dir: &Path) -> Option<String> {
    let out = Command::new("git")
        .current_dir(dir)
        .args(["config", "--local", "--get", "core.hooksPath"])
        .output()
        .expect("git config must execute");
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn create_isolated_setup_workspace(tag: &str) -> (PathBuf, PathBuf) {
    let base = env::temp_dir().join(format!("xtask_setup_cli_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    let repo = base.join("repo");
    let fake_bin = base.join("bin");
    fs::create_dir_all(repo.join(".githooks")).expect("create .githooks");
    fs::create_dir_all(&fake_bin).expect("create fake bin");
    fs::write(
        repo.join("Cargo.toml"),
        "[workspace]\nmembers = []\nresolver = \"2\"\n\n[workspace.dependencies]\nwasm-bindgen = \"0.2.130\"\n",
    )
    .expect("write Cargo.toml");
    fs::write(repo.join(".githooks/pre-commit"), "#!/bin/sh\nexit 0\n").expect("write pre-commit");

    let init = Command::new("git")
        .current_dir(&repo)
        .arg("init")
        .status()
        .expect("git init");
    assert!(init.success());

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let rustup_stub = fake_bin.join("rustup");
        fs::write(&rustup_stub, "#!/bin/sh\nexit 0\n").expect("write rustup stub");
        let mut perms = fs::metadata(&rustup_stub).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&rustup_stub, perms).unwrap();
    }

    (base, repo)
}

#[cfg(unix)]
fn isolated_path_without_browser_tools(fake_bin: &Path) -> String {
    let git_out = Command::new("which")
        .arg("git")
        .output()
        .expect("which git");
    let git_dir = Path::new(String::from_utf8_lossy(&git_out.stdout).trim())
        .parent()
        .expect("git parent dir")
        .to_path_buf();
    let link_git = fake_bin.join("git");
    if !link_git.exists() {
        std::os::unix::fs::symlink(git_dir.join("git"), &link_git).expect("symlink git");
    }
    let sh_link = fake_bin.join("sh");
    if !sh_link.exists() && Path::new("/bin/sh").exists() {
        let _ = std::os::unix::fs::symlink("/bin/sh", &sh_link);
    }
    fake_bin.to_string_lossy().into_owned()
}

#[test]
fn setup_hooks_only_and_check_only_succeed_without_browser_tools() {
    let xtask_bin = env!("CARGO_BIN_EXE_xtask");
    let (base, repo) = create_isolated_setup_workspace("hooks_and_check");

    let mut hooks_cmd = Command::new(xtask_bin);
    hooks_cmd.current_dir(&repo).args(["setup", "--hooks-only"]);
    #[cfg(unix)]
    hooks_cmd.env(
        "PATH",
        isolated_path_without_browser_tools(&base.join("bin")),
    );

    let hooks_out = hooks_cmd.output().expect("run xtask setup --hooks-only");
    assert!(
        hooks_out.status.success(),
        "setup --hooks-only failed: {}{}",
        String::from_utf8_lossy(&hooks_out.stdout),
        String::from_utf8_lossy(&hooks_out.stderr)
    );
    assert_eq!(local_hooks_path(&repo).as_deref(), Some(".githooks"));

    let _ = Command::new("git")
        .current_dir(&repo)
        .args(["config", "--local", "--unset", "core.hooksPath"])
        .status();

    let mut check_cmd = Command::new(xtask_bin);
    check_cmd.current_dir(&repo).args(["setup", "--check-only"]);
    #[cfg(unix)]
    check_cmd.env(
        "PATH",
        isolated_path_without_browser_tools(&base.join("bin")),
    );

    let check_out = check_cmd.output().expect("run xtask setup --check-only");
    assert!(
        check_out.status.success(),
        "setup --check-only failed: {}{}",
        String::from_utf8_lossy(&check_out.stdout),
        String::from_utf8_lossy(&check_out.stderr)
    );
    assert_eq!(local_hooks_path(&repo).as_deref(), Some(".githooks"));
    #[cfg(unix)]
    {
        let stdout = String::from_utf8_lossy(&check_out.stdout);
        assert!(stdout.contains("Warning: Optional WebAssembly/Playwright browser tools"));
    }

    let _ = fs::remove_dir_all(&base);
}

#[test]
#[cfg(unix)]
fn setup_web_configures_hooks_first_and_exits_one_with_install_commands() {
    let xtask_bin = env!("CARGO_BIN_EXE_xtask");
    let (base, repo) = create_isolated_setup_workspace("strict_web");
    let isolated_path = isolated_path_without_browser_tools(&base.join("bin"));

    let out = Command::new(xtask_bin)
        .current_dir(&repo)
        .env("PATH", &isolated_path)
        .args(["setup", "--web", "--check-only"])
        .output()
        .expect("run xtask setup --web");

    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        local_hooks_path(&repo).as_deref(),
        Some(".githooks"),
        "core.hooksPath must be configured before browser tool exit"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stdout.contains("wasm-bindgen-cli@0.2.130"));
    assert!(stdout.contains("trunk"));
    assert!(stdout.contains("-> Fix:"));
    assert!(stderr.contains("Setup failed (--web)"));

    let _ = fs::remove_dir_all(&base);
}
