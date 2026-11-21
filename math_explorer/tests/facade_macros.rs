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

    fn find_newest_rlib(deps_dir: &Path, prefix: &str) -> PathBuf {
        let mut candidates: Vec<_> = fs::read_dir(deps_dir)
            .expect("read deps dir")
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                let name = p.file_name()?.to_str()?;
                if name.starts_with(prefix) && p.extension()?.to_str()? == "rlib" {
                    Some((e.metadata().ok()?.modified().ok(), p))
                } else {
                    None
                }
            })
            .collect();
        candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        candidates.pop().expect("compiled math_explorer rlib").1
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
        let exe = std::env::current_exe().expect("test exe");
        let deps_dir = exe.parent().expect("deps dir");
        let me_rlib = find_newest_rlib(deps_dir, "libmath_explorer-");
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let tmp = std::env::temp_dir().join(format!("me_facade_only_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("create tmp dir");
        let src = tmp.join("external_caller.rs");
        let out_rlib = tmp.join("libexternal_caller.rlib");
        fs::write(&src, FACADE_ONLY_SNIPPET).expect("write external_caller.rs");

        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let mut cmd = Command::new(rustc);
        cmd.current_dir(&manifest_dir)
            .env("CARGO_MANIFEST_DIR", &manifest_dir)
            .args([
                "--edition=2024",
                "--crate-type=lib",
                "--crate-name=downstream_me_only",
                "-D",
                "warnings",
            ])
            .arg(format!("--extern=math_explorer={}", me_rlib.display()))
            .arg(format!("-Ldependency={}", deps_dir.display()));
        if let Some(prof) = deps_dir.parent()
            && let Some(name) = prof.file_name()
            && let Some(root) = prof.parent()
        {
            let host_deps = root.join(name).join("deps");
            if host_deps.is_dir() && host_deps != deps_dir {
                cmd.arg(format!("-Ldependency={}", host_deps.display()));
            }
        }
        let output = cmd
            .arg(&src)
            .arg("-o")
            .arg(&out_rlib)
            .output()
            .expect("run rustc");
        let _ = fs::remove_dir_all(&tmp);
        assert!(
            output.status.success(),
            "math_explorer-only downstream crate failed to compile:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
