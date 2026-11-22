//! Integration tests for `verified_engine` macro re-exports and runtime complexity metrics.
use approx::assert_relative_eq;
use verified_engine::engine::VerifiedEngine;
use verified_engine::scientific_metadata::theory::TheoryDescribable;
use verified_engine::{Theory, embed_theory, verified};

trait StepTrait {
    fn advance(&mut self, delta: f64) -> f64;
}

struct Accumulator {
    total: f64,
}

impl Accumulator {
    #[verified]
    fn advance(&mut self, delta: f64) -> f64 {
        let scaled = delta * 2.0;
        self.total += scaled;
        self.total
    }
}

impl StepTrait for Accumulator {
    #[verified(allow_recursion = "delegates StepTrait::advance to inherent Accumulator::advance")]
    fn advance(&mut self, delta: f64) -> f64 {
        self.advance(delta)
    }
}

#[derive(Theory)]
#[theory(
    description = "Mixed integer and float parameter model",
    phonetic = "Mixed integer and float parameter model",
    citation = "Newton, I. (1687). Principia."
)]
struct MixedBoundsModel {
    #[theory(min = -10, max = 10, step = 1)]
    grid_offset: i32,
    #[theory(min = -2.5, max = 2.5, step = 0.5)]
    gain: f64,
    #[theory(min = 1, max = 64, step = 1)]
    iterations: usize,
}

trait RadarRangeModel {
    fn compute_beat_frequency(&self, b: f64, r: f64, c: f64, t: f64) -> f64;
}

struct FmcwRadar;

impl RadarRangeModel for FmcwRadar {
    #[embed_theory("papers/mmwave_radiotherapy_setup.tex", label = "eq:range")]
    #[verified]
    fn compute_beat_frequency(&self, b: f64, r: f64, c: f64, t: f64) -> f64 {
        (2.0 * b * r) / (c * t)
    }
}

#[embed_theory("papers/mmwave_radiotherapy_setup.tex", label = "eq:range")]
#[verified]
fn shared_range_helper_a(b: f64, r: f64, c: f64, t: f64) -> f64 {
    (2.0 * b * r) / (c * t)
}

#[embed_theory("papers/mmwave_radiotherapy_setup.tex", label = "eq:range")]
#[verified]
fn shared_range_helper_b(b: f64, r: f64, c: f64, t: f64) -> f64 {
    (2.0 * b * r) / (c * t)
}

#[test]
fn test_allow_recursion_records_runtime_complexity_metrics() {
    let mut acc = Accumulator { total: 1.0 };
    let (value, metrics) =
        VerifiedEngine::run_verified(|| <Accumulator as StepTrait>::advance(&mut acc, 3.0));
    assert_relative_eq!(value, 7.0);
    assert_eq!(metrics.function_calls, 2);
    assert_eq!(metrics.arithmetic_ops, 1);
}

#[test]
fn test_theory_supports_signed_integer_and_float_bounds() {
    let mut model = MixedBoundsModel {
        grid_offset: -3,
        gain: 1.5,
        iterations: 8,
    };
    let params = model.theory_parameters();
    assert_eq!(params.len(), 3);
    assert_relative_eq!(params["grid_offset"].min, -10.0);
    assert_relative_eq!(params["grid_offset"].max, 10.0);
    assert_relative_eq!(params["grid_offset"].step, 1.0);
    assert_relative_eq!(params["gain"].min, -2.5);
    assert_relative_eq!(params["gain"].max, 2.5);
    assert_relative_eq!(params["gain"].step, 0.5);

    assert_eq!(model.get_parameter("grid_offset"), Some(-3.0));
    model.set_parameter("grid_offset", 4.0);
    assert_eq!(model.grid_offset, 4);

    model.set_parameter("iterations", 16.0);
    assert_eq!(model.iterations, 16);
}

#[test]
fn test_embed_theory_in_trait_impl_and_shared_labels() {
    let radar = FmcwRadar;
    let f1 = radar.compute_beat_frequency(4.0, 3.0, 2.0, 1.0);
    let f2 = shared_range_helper_a(4.0, 3.0, 2.0, 1.0);
    let f3 = shared_range_helper_b(4.0, 3.0, 2.0, 1.0);
    assert_relative_eq!(f1, 12.0);
    assert_relative_eq!(f2, 12.0);
    assert_relative_eq!(f3, 12.0);
}

#[cfg(not(target_arch = "wasm32"))]
mod facade_isolation {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn rustc_bin() -> PathBuf {
        if let Some(rustc) = std::env::var_os("RUSTC") {
            return PathBuf::from(rustc);
        }
        if let Some(cargo) = std::env::var_os("CARGO") {
            let candidate = PathBuf::from(cargo).with_file_name(if cfg!(windows) {
                "rustc.exe"
            } else {
                "rustc"
            });
            if candidate.is_file() {
                return candidate;
            }
        }
        PathBuf::from("rustc")
    }

    fn detect_profiles() -> (Vec<PathBuf>, bool) {
        let exe = std::env::current_exe().expect("test exe");
        let parent = exe.parent().expect("exe parent");
        let (primary, build_layout) = if parent.file_name().and_then(|s| s.to_str()) == Some("deps")
        {
            (parent.parent().expect("profile dir").to_path_buf(), false)
        } else if let Some(build_dir) = exe.ancestors().nth(4)
            && build_dir.file_name().and_then(|s| s.to_str()) == Some("build")
        {
            (build_dir.parent().expect("profile dir").to_path_buf(), true)
        } else {
            panic!("unexpected test executable location: {}", exe.display());
        };
        let mut profiles = vec![primary.clone()];
        if let Some(name) = primary.file_name()
            && let Some(root) = primary.parent().and_then(Path::parent)
        {
            let host = root.join(name);
            let sub = if build_layout { "build" } else { "deps" };
            if host != primary && host.join(sub).is_dir() {
                profiles.push(host);
            }
        }
        (profiles, build_layout)
    }

    fn find_crate_rlibs(
        profiles: &[PathBuf],
        build_layout: bool,
        pkg_name: &str,
        prefix: &str,
    ) -> Vec<PathBuf> {
        let dirs: Vec<PathBuf> = if build_layout {
            profiles
                .iter()
                .filter_map(|p| fs::read_dir(p.join("build").join(pkg_name)).ok())
                .flatten()
                .flatten()
                .map(|e| e.path().join("out"))
                .filter(|d| d.is_dir())
                .collect()
        } else {
            profiles.iter().map(|p| p.join("deps")).collect()
        };
        let mut candidates = Vec::new();
        for dir in &dirs {
            let Ok(entries) = fs::read_dir(dir) else {
                continue;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.starts_with(prefix))
                    && p.extension().and_then(|s| s.to_str()) == Some("rlib")
                {
                    let mtime = e.metadata().ok().and_then(|m| m.modified().ok());
                    candidates.push((mtime, p));
                }
            }
        }
        candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
        assert!(!candidates.is_empty(), "compiled verified_engine rlib");
        candidates.into_iter().map(|(_, p)| p).collect()
    }

    fn link_artifact(src: &Path, dst: &Path) {
        if dst.exists() {
            return;
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(src, dst).expect("symlink dep artifact");
        #[cfg(not(unix))]
        fs::hard_link(src, dst)
            .or_else(|_| fs::copy(src, dst).map(|_| ()))
            .expect("stage dep artifact");
    }

    fn collect_profile_artifacts(profile: &Path) -> Vec<PathBuf> {
        let Ok(pkgs) = fs::read_dir(profile.join("build")) else {
            return Vec::new();
        };
        let mut found = Vec::new();
        for unit in pkgs
            .flatten()
            .filter_map(|pkg| fs::read_dir(pkg.path()).ok())
            .flatten()
            .flatten()
        {
            let Ok(files) = fs::read_dir(unit.path().join("out")) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if matches!(
                    path.extension().and_then(|s| s.to_str()),
                    Some("rlib" | "so" | "dylib" | "dll")
                ) {
                    let mtime = file.metadata().ok().and_then(|m| m.modified().ok());
                    found.push((mtime, path));
                }
            }
        }
        found.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
        found.into_iter().map(|(_, p)| p).collect()
    }

    fn stage_build_deps(profiles: &[PathBuf], tmp: &Path) -> PathBuf {
        let staged = if cfg!(unix) {
            tmp.join("deps")
        } else {
            profiles[0].join(format!("ve_facade_deps_{}", std::process::id()))
        };
        let _ = fs::remove_dir_all(&staged);
        fs::create_dir_all(&staged).expect("create staged deps dir");
        for profile in profiles {
            for path in collect_profile_artifacts(profile) {
                if let Some(name) = path.file_name() {
                    link_artifact(&path, &staged.join(name));
                }
            }
        }
        staged
    }

    fn run_downstream_rustc(
        manifest_dir: &Path,
        ve_rlib: &Path,
        dep_dirs: &[PathBuf],
        src: &Path,
        out_rlib: &Path,
    ) -> std::process::Output {
        let mut cmd = Command::new(rustc_bin());
        cmd.current_dir(manifest_dir)
            .env("CARGO_MANIFEST_DIR", manifest_dir)
            .args([
                "--edition=2024",
                "--crate-type=lib",
                "--crate-name=downstream_ve_only",
                "-D",
                "warnings",
            ])
            .arg(format!("--extern=verified_engine={}", ve_rlib.display()));
        for dir in dep_dirs {
            cmd.arg(format!("-Ldependency={}", dir.display()));
        }
        cmd.arg(src)
            .arg("-o")
            .arg(out_rlib)
            .output()
            .expect("run rustc")
    }

    const DOWNSTREAM_VE_SNIPPET: &str = r#"
use verified_engine::{Theory, embed_theory, verified};

#[derive(Theory)]
#[theory(description = "Downstream facade-only model", citation = "Test (2026)")]
pub struct ExternalModel {
    #[theory(min = -5, max = 5, step = 1)]
    pub count: i32,
}

#[embed_theory("papers/mmwave_radiotherapy_setup.tex", label = "eq:range")]
#[verified]
pub fn external_range(b: f64, r: f64, c: f64, t: f64) -> f64 {
    (2.0 * b * r) / (c * t)
}
"#;

    #[test]
    fn test_downstream_caller_with_only_verified_engine_extern() {
        let (profiles, build_layout) = detect_profiles();
        let ve_rlibs = find_crate_rlibs(
            &profiles,
            build_layout,
            "verified_engine",
            "libverified_engine-",
        );
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let tmp = std::env::temp_dir().join(format!("ve_facade_only_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("create tmp dir");
        let src = tmp.join("downstream.rs");
        let out_rlib = tmp.join("libdownstream.rlib");
        fs::write(&src, DOWNSTREAM_VE_SNIPPET).expect("write downstream.rs");

        let staged_deps = build_layout.then(|| stage_build_deps(&profiles, &tmp));
        let dep_dirs: Vec<PathBuf> = match staged_deps.as_ref() {
            Some(staged) => vec![staged.clone()],
            None => profiles.iter().map(|p| p.join("deps")).collect(),
        };

        let mut first_err = String::new();
        let mut succeeded = false;
        for ve_rlib in &ve_rlibs {
            let out = run_downstream_rustc(&manifest_dir, ve_rlib, &dep_dirs, &src, &out_rlib);
            if out.status.success() {
                succeeded = true;
                break;
            }
            if first_err.is_empty() {
                first_err = String::from_utf8_lossy(&out.stderr).into_owned();
            }
        }
        if let Some(staged) = staged_deps {
            let _ = fs::remove_dir_all(&staged);
        }
        let _ = fs::remove_dir_all(&tmp);
        assert!(
            succeeded,
            "verified_engine-only downstream crate failed to compile:\n{first_err}"
        );
    }
}
