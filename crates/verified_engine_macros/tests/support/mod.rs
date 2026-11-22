use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

static CASE_COUNTER: AtomicUsize = AtomicUsize::new(0);
static ARTIFACTS: OnceLock<CompilerArtifacts> = OnceLock::new();

struct CompilerArtifacts {
    rustc: PathBuf,
    proc_macro_dylib: PathBuf,
    scientific_metadata_rlib: PathBuf,
    verified_engine_stub_rlib: PathBuf,
    deps_dirs: Vec<PathBuf>,
}

pub struct RustcOutcome {
    pub success: bool,
    pub stderr: String,
    pub file_name: String,
}

impl RustcOutcome {
    pub fn assert_failed_without_panic(&self) {
        assert!(
            !self.success,
            "expected rustc to fail for `{}`, but compilation succeeded",
            self.file_name
        );
        for forbidden in [
            "panicked at",
            "proc-macro derive panicked",
            "custom attribute panicked",
            "proc macro panicked",
            "\"severity\"",
        ] {
            assert!(
                !self.stderr.contains(forbidden),
                "unexpected `{forbidden}` in rustc stderr:\n{}",
                self.stderr
            );
        }
    }

    pub fn assert_has_error(&self, expected: &str) {
        assert!(
            self.stderr.contains(expected),
            "expected stderr to contain `{expected}`, actual stderr:\n{}",
            self.stderr
        );
    }

    pub fn assert_has_span(&self, line: usize, col: usize) {
        let needle = format!("{}:{line}:{col}", self.file_name);
        assert!(
            self.stderr.contains(&needle),
            "expected span `{needle}` in rustc stderr:\n{}",
            self.stderr
        );
    }
}

fn rustc_bin() -> PathBuf {
    if let Some(rustc) = std::env::var_os("RUSTC") {
        return PathBuf::from(rustc);
    }
    if let Some(cargo) = std::env::var_os("CARGO") {
        let candidate =
            PathBuf::from(cargo).with_file_name(if cfg!(windows) { "rustc.exe" } else { "rustc" });
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("rustc")
}

fn detect_profile_layout() -> (Vec<PathBuf>, bool) {
    let exe = std::env::current_exe().expect("current test executable path");
    let parent = exe.parent().expect("executable parent dir");
    let (primary_profile, build_layout) =
        if parent.file_name().and_then(|s| s.to_str()) == Some("deps") {
            (parent.parent().expect("profile dir").to_path_buf(), false)
        } else if let Some(build_dir) = exe.ancestors().nth(4)
            && build_dir.file_name().and_then(|s| s.to_str()) == Some("build")
        {
            (build_dir.parent().expect("profile dir").to_path_buf(), true)
        } else {
            panic!("unexpected Cargo test executable path: {}", exe.display());
        };

    let mut profiles = vec![primary_profile.clone()];
    if let Some(profile_name) = primary_profile.file_name()
        && let Some(target_root) = primary_profile.parent().and_then(Path::parent)
    {
        let host_profile = target_root.join(profile_name);
        let marker = if build_layout { "build" } else { "deps" };
        if host_profile != primary_profile && host_profile.join(marker).is_dir() {
            profiles.push(host_profile);
        }
    }
    (profiles, build_layout)
}

fn crate_search_dirs(profiles: &[PathBuf], build_layout: bool, pkg_name: &str) -> Vec<PathBuf> {
    if !build_layout {
        return profiles
            .iter()
            .map(|p| p.join("deps"))
            .filter(|d| d.is_dir())
            .collect();
    }
    let mut dirs = Vec::new();
    for profile in profiles {
        let Ok(units) = fs::read_dir(profile.join("build").join(pkg_name)) else {
            continue;
        };
        for unit in units.flatten() {
            let out_dir = unit.path().join("out");
            if out_dir.is_dir() {
                dirs.push(out_dir);
            }
        }
    }
    dirs
}

fn is_proc_macro_ext(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|s| s.to_str()),
        Some("so" | "dylib" | "dll")
    )
}

fn find_newest_artifact<F>(dirs: &[PathBuf], pred: F, label: &str) -> PathBuf
where
    F: Fn(&Path, &str) -> bool,
{
    let mut matches = Vec::new();
    for dir in dirs {
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if pred(&path, name) {
                let mtime = entry.metadata().ok().and_then(|m| m.modified().ok());
                matches.push((mtime, path));
            }
        }
    }
    matches.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    matches
        .pop()
        .map(|(_, path)| path)
        .unwrap_or_else(|| panic!("could not locate compiled artifact `{label}` in {dirs:?}"))
}

fn make_temp_dir(tag: &str) -> PathBuf {
    let id = CASE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "oxidizemath_macro_diag_{}_{}_{}",
        std::process::id(),
        id,
        tag
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp test dir");
    dir
}

fn build_verified_engine_stub(rustc: &Path) -> PathBuf {
    let stub_dir = make_temp_dir("ve_stub");
    let stub_src = stub_dir.join("lib.rs");
    let stub_out = stub_dir.join("libverified_engine.rlib");
    let src = r#"
pub mod metrics {
    pub fn increment_calls() {}
    pub fn increment_arithmetic(_count: u64) {}
    pub fn increment_memory_loads(_count: u64) {}
}
"#;
    fs::write(&stub_src, src).expect("write verified_engine stub");
    let out = Command::new(rustc)
        .arg("--edition=2024")
        .arg("--crate-type=rlib")
        .arg("--crate-name=verified_engine")
        .arg(&stub_src)
        .arg("-o")
        .arg(&stub_out)
        .output()
        .expect("compile verified_engine stub");
    assert!(
        out.status.success(),
        "failed to compile verified_engine stub:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stub_out
}

fn artifacts() -> &'static CompilerArtifacts {
    ARTIFACTS.get_or_init(|| {
        let rustc = rustc_bin();
        let (profiles, build_layout) = detect_profile_layout();
        let macro_dirs = crate_search_dirs(&profiles, build_layout, "verified_engine_macros");
        let proc_macro_dylib = find_newest_artifact(
            &macro_dirs,
            |path, name| {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                (name.starts_with("libverified_engine_macros-")
                    || name.starts_with("verified_engine_macros-")
                    || stem == "libverified_engine_macros"
                    || stem == "verified_engine_macros")
                    && is_proc_macro_ext(path)
            },
            "verified_engine_macros dylib",
        );
        let sm_dirs = crate_search_dirs(&profiles, build_layout, "scientific_metadata");
        let scientific_metadata_rlib = find_newest_artifact(
            &sm_dirs,
            |path, name| {
                name.starts_with("libscientific_metadata-")
                    && path.extension().and_then(|s| s.to_str()) == Some("rlib")
            },
            "libscientific_metadata.rlib",
        );
        let verified_engine_stub_rlib = build_verified_engine_stub(&rustc);
        let mut deps_dirs = Vec::new();
        if let Some(p) = proc_macro_dylib.parent() {
            deps_dirs.push(p.to_path_buf());
        }
        if let Some(p) = scientific_metadata_rlib.parent()
            && !deps_dirs.iter().any(|d| d == p)
        {
            deps_dirs.push(p.to_path_buf());
        }
        CompilerArtifacts {
            rustc,
            proc_macro_dylib,
            scientific_metadata_rlib,
            verified_engine_stub_rlib,
            deps_dirs,
        }
    })
}

fn setup_workspace_files(
    case_name: &str,
    rel_manifest_dir: &str,
    tex_files: &[(&str, &str)],
    code: &str,
) -> (PathBuf, PathBuf, PathBuf, String) {
    let ws_root = make_temp_dir(case_name);
    fs::write(
        ws_root.join("Cargo.toml"),
        format!("[workspace]\nmembers = [\"{rel_manifest_dir}\"]\n"),
    )
    .expect("write workspace Cargo.toml");

    for (rel_path, contents) in tex_files {
        let full = ws_root.join(rel_path);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).expect("create tex parent dir");
        }
        fs::write(full, contents).expect("write tex fixture");
    }

    let manifest_dir = ws_root.join(rel_manifest_dir);
    let src_dir = manifest_dir.join("src");
    fs::create_dir_all(&src_dir).expect("create crate src dir");
    fs::write(
        manifest_dir.join("Cargo.toml"),
        format!("[package]\nname = \"{case_name}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n"),
    )
    .expect("write crate Cargo.toml");

    let file_name = format!("{case_name}.rs");
    let src_file = src_dir.join(&file_name);
    fs::write(&src_file, code).expect("write test snippet");
    (ws_root, manifest_dir, src_file, file_name)
}

pub fn compile_in_workspace(
    case_name: &str,
    rel_manifest_dir: &str,
    tex_files: &[(&str, &str)],
    code: &str,
) -> RustcOutcome {
    let art = artifacts();
    let (ws_root, manifest_dir, src_file, file_name) =
        setup_workspace_files(case_name, rel_manifest_dir, tex_files, code);
    let out_rlib = ws_root.join(format!("lib{case_name}.rlib"));
    let mut cmd = Command::new(&art.rustc);
    cmd.current_dir(&manifest_dir)
        .env("CARGO_MANIFEST_DIR", &manifest_dir)
        .args([
            "--edition=2024",
            "--crate-type=lib",
            "--crate-name",
            case_name,
        ])
        .args(["--color=never", "-D", "warnings"])
        .arg(format!(
            "--extern=verified_engine_macros={}",
            art.proc_macro_dylib.display()
        ))
        .arg(format!(
            "--extern=scientific_metadata={}",
            art.scientific_metadata_rlib.display()
        ))
        .arg(format!(
            "--extern=verified_engine={}",
            art.verified_engine_stub_rlib.display()
        ));
    for dir in &art.deps_dirs {
        cmd.arg("-L").arg(format!("dependency={}", dir.display()));
    }
    let output = cmd
        .arg(&src_file)
        .arg("-o")
        .arg(&out_rlib)
        .output()
        .expect("invoke rustc");

    let _ = fs::remove_dir_all(&ws_root);
    RustcOutcome {
        success: output.status.success(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        file_name,
    }
}

pub fn compile_snippet(case_name: &str, code: &str) -> RustcOutcome {
    compile_in_workspace(case_name, "crates/sample", &[], code)
}
