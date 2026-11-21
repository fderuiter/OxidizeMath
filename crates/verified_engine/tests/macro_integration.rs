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
        candidates.pop().expect("compiled verified_engine rlib").1
    }

    #[test]
    fn test_downstream_caller_with_only_verified_engine_extern() {
        let exe = std::env::current_exe().expect("test exe");
        let deps_dir = exe.parent().expect("deps dir");
        let ve_rlib = find_newest_rlib(deps_dir, "libverified_engine-");
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let tmp = std::env::temp_dir().join(format!("ve_facade_only_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("create tmp dir");
        let src = tmp.join("downstream.rs");
        let out_rlib = tmp.join("libdownstream.rlib");
        let code = r#"
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
        fs::write(&src, code).expect("write downstream.rs");

        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let mut cmd = Command::new(rustc);
        cmd.current_dir(&manifest_dir)
            .env("CARGO_MANIFEST_DIR", &manifest_dir)
            .arg("--edition=2024")
            .arg("--crate-type=lib")
            .arg("--crate-name=downstream_ve_only")
            .arg("-D")
            .arg("warnings")
            .arg("--extern")
            .arg(format!("verified_engine={}", ve_rlib.display()))
            .arg("-L")
            .arg(format!("dependency={}", deps_dir.display()));
        if let Some(host_deps) = deps_dir.parent().and_then(|p| p.parent()).and_then(|t| {
            deps_dir
                .parent()?
                .file_name()
                .map(|prof| t.join(prof).join("deps"))
        }) && host_deps.is_dir()
            && host_deps != deps_dir
        {
            cmd.arg("-L")
                .arg(format!("dependency={}", host_deps.display()));
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
            "verified_engine-only downstream crate failed to compile:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
