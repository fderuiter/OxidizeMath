//! Integration tests verifying `math_explorer` macro and metadata facade re-exports.
use approx::assert_relative_eq;
use math_explorer::scientific_metadata::theory::TheoryDescribable;
use math_explorer::verified_engine::engine::VerifiedEngine;
use math_explorer::verified_engine::{Theory, embed_theory, verified};
use nalgebra::DVector;
use pure_math::pure_math::analysis::ode::traits::VectorOperations;

#[derive(Theory)]
#[theory(
    description = "Top-level math_explorer facade model",
    citation = "Newton, I. (1687). Principia."
)]
struct ExplorerFacadeModel {
    #[theory(min = -4, max = 4, step = 2)]
    mode_index: i32,
    #[theory(min = 0.0, max = 10.0, step = 0.25)]
    coupling: f64,
}

#[embed_theory("papers/mmwave_radiotherapy_setup.tex", label = "eq:range")]
#[verified]
fn top_level_beat_frequency(b: f64, r: f64, c: f64, t: f64) -> f64 {
    (2.0 * b * r) / (c * t)
}

#[test]
fn test_math_explorer_reexports_verified_theory_and_top_level_embed_theory() {
    let mut model = ExplorerFacadeModel {
        mode_index: -2,
        coupling: 1.25,
    };
    assert_eq!(
        model.theory_description(),
        "Top-level math_explorer facade model"
    );
    assert_eq!(model.get_parameter("mode_index"), Some(-2.0));
    model.set_parameter("mode_index", 2.0);
    assert_eq!(model.mode_index, 2);

    let (freq, metrics) =
        VerifiedEngine::run_verified(|| top_level_beat_frequency(4.0, 3.0, 2.0, 1.0));
    assert_relative_eq!(freq, 12.0);
    assert_eq!(metrics.function_calls, 1);
    assert!(metrics.arithmetic_ops >= 3);
}

#[test]
fn test_migrated_ode_allow_recursion_method_records_metrics() {
    let mut dst = DVector::from_vec(vec![0.0, 0.0, 0.0]);
    let src = DVector::from_vec(vec![1.5, -2.5, 4.0]);
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        <DVector<f64> as VectorOperations>::copy_from(&mut dst, &src);
    });
    assert_relative_eq!(dst[0], 1.5);
    assert_relative_eq!(dst[1], -2.5);
    assert_relative_eq!(dst[2], 4.0);
    assert_eq!(metrics.function_calls, 1);
}

#[cfg(feature = "physics")]
#[test]
fn test_migrated_physics_methods_record_verified_metrics() {
    use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
    use math_explorer::physics::fluid_dynamics::lattice_boltzmann::algorithms::LbmConfig;
    use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
        BgkCollision, LatticeBoltzmannD2Q9,
    };
    use oxidize_core::SimulationModel;
    use rand::SeedableRng;

    let mut lorenz = LorenzBuilder::new().build(LorenzState::new(1.0, 1.0, 1.0));
    let (res, lorenz_metrics) =
        VerifiedEngine::run_verified(|| <LorenzSystem as SimulationModel>::step(&mut lorenz));
    assert!(res.is_ok());
    assert!(lorenz_metrics.function_calls >= 1);

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let lbm_cfg = LbmConfig {
        width: 4,
        height: 4,
        tau: 0.8,
    };
    let mut lbm =
        <LatticeBoltzmannD2Q9<BgkCollision> as SimulationModel>::initialize(lbm_cfg, &mut rng)
            .unwrap();
    let (lbm_res, lbm_metrics) = VerifiedEngine::run_verified(|| {
        <LatticeBoltzmannD2Q9<BgkCollision> as SimulationModel>::step(&mut lbm)
    });
    assert!(lbm_res.is_ok());
    assert!(lbm_metrics.function_calls >= 2);
}

#[cfg(feature = "biology")]
#[test]
fn test_migrated_biology_allow_recursion_method_records_metrics() {
    use math_explorer::biology::morphogenesis::{DiffusionCoeff, TuringSystem};
    use math_explorer::math_kernel::types::{Dimension, StepSize};
    use pure_math::pure_math::analysis::ode::TimeStepper;

    let mut sys = TuringSystem::new(
        Dimension(4),
        DiffusionCoeff(0.16),
        DiffusionCoeff(0.08),
        StepSize(1.0),
    );
    let ((), bio_metrics) = VerifiedEngine::run_verified(|| {
        TimeStepper::step(&mut sys, 0.01);
    });
    assert!(bio_metrics.function_calls >= 2);
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
        assert!(!candidates.is_empty(), "compiled math_explorer rlib");
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
            profiles[0].join(format!("me_facade_deps_{}", std::process::id()))
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
        me_rlib: &Path,
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
                "--crate-name=downstream_me_only",
                "-D",
                "warnings",
            ])
            .arg(format!("--extern=math_explorer={}", me_rlib.display()));
        for dir in dep_dirs {
            cmd.arg(format!("-Ldependency={}", dir.display()));
        }
        cmd.arg(src)
            .arg("-o")
            .arg(out_rlib)
            .output()
            .expect("run rustc")
    }

    const FACADE_ONLY_SNIPPET: &str = r#"
use math_explorer::scientific_metadata::theory::TheoryDescribable;
use math_explorer::verified_engine::{Theory, embed_theory, verified};

#[derive(Theory)]
#[theory(description = "External math_explorer-only model", citation = "Test (2026)")]
pub struct FacadeOnlyModel {
    #[theory(min = 0, max = 100, step = 5)]
    pub steps: u32,
}

#[embed_theory("papers/mmwave_radiotherapy_setup.tex", label = "eq:range")]
#[verified]
pub fn facade_range(b: f64, r: f64, c: f64, t: f64) -> f64 {
    (2.0 * b * r) / (c * t)
}

pub fn check_model() -> Option<f64> {
    let m = FacadeOnlyModel { steps: 15 };
    m.get_parameter("steps")
}
"#;

    #[test]
    fn test_downstream_caller_with_only_math_explorer_extern() {
        let (profiles, build_layout) = detect_profiles();
        let me_rlibs = find_crate_rlibs(
            &profiles,
            build_layout,
            "math_explorer",
            "libmath_explorer-",
        );
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let tmp = std::env::temp_dir().join(format!("me_facade_only_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("create tmp dir");
        let src = tmp.join("external_caller.rs");
        let out_rlib = tmp.join("libexternal_caller.rlib");
        fs::write(&src, FACADE_ONLY_SNIPPET).expect("write external_caller.rs");

        let staged_deps = build_layout.then(|| stage_build_deps(&profiles, &tmp));
        let dep_dirs: Vec<PathBuf> = match staged_deps.as_ref() {
            Some(staged) => vec![staged.clone()],
            None => profiles.iter().map(|p| p.join("deps")).collect(),
        };

        let mut first_err = String::new();
        let mut succeeded = false;
        for me_rlib in &me_rlibs {
            let out = run_downstream_rustc(&manifest_dir, me_rlib, &dep_dirs, &src, &out_rlib);
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
            "math_explorer-only downstream crate failed to compile:\n{first_err}"
        );
    }
}
