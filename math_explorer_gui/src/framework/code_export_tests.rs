//! End-to-end unit and numerical parity tests for unified `.tex`, `.rs`, and `.py` code exports.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::presets::OdePreset;
use crate::tabs::analysis::complex_mapping::ComplexMappingTool;
#[cfg(not(target_arch = "wasm32"))]
use crate::tabs::analysis::complex_mapping::MappingFunction;
use crate::tabs::analysis::ode::OdeSolverTool;
#[cfg(not(target_arch = "wasm32"))]
use crate::tabs::analysis::riemann::FunctionPreset;
use crate::tabs::analysis::riemann::RiemannIntegrationTool;
use crate::tabs::battery_degradation::capacity_fade::CapacityFadeTool;
use crate::tabs::epidemiology::sir::SirTool;
use crate::tabs::medical::beam_profiling::BeamProfilingTool;
use eframe::egui;
use std::collections::HashMap;
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::process::Command;
use std::sync::Arc;

fn configured<T: Default>(f: impl FnOnce(&mut T)) -> T {
    let mut val = T::default();
    f(&mut val);
    val
}

#[test]
fn test_ode_custom_bookmark_calibrated_latex_rust_and_python_exact() {
    let tool = configured::<OdeSolverTool>(|t| {
        t.preset = OdePreset::Custom;
        t.math_input.set_expression("-k * y * sin(t)");
        t.param_k = 1.85;
        t.ic_y0 = 1.0;
        t.dt = 0.05;
        t.total_time = 10.0;
        t.recalculate();
    });

    let art = tool.export_code_artifact().expect("bookmark code artifact");
    assert!(art.validate().is_ok());
    assert!(art
        .calibrated_latex
        .contains(r"\frac{dy}{dt} &= -1.85 \cdot y \cdot \sin\left(t\right)"));
    assert!(art
        .calibrated_latex
        .contains(r"y(0) &= 1, \quad \Delta t = 0.05, \quad t \in [0, 10]"));
    assert!(art
        .rust_source
        .contains("pub fn rhs(t: f64, y: f64) -> f64 {\n    (-1.85_f64 * y) * t.sin()\n}"));
    assert!(art
        .rust_source
        .contains("pub fn rk4_step(t: f64, y: f64, dt: f64) -> f64"));
    assert!(art
        .python_source
        .contains("from scipy.integrate import solve_ivp"));
    assert!(art
        .python_source
        .contains("return [(-1.85 * y[0]) * np.sin(t)]"));
}

#[test]
fn test_all_six_flagship_tools_queue_file_and_clipboard_with_aria_status() {
    let temp_dir =
        std::env::temp_dir().join(format!("oxidize_code_export_6tools_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let tools: Vec<(Box<dyn InteractiveTool>, &str)> = vec![
        (Box::new(OdeSolverTool::default()), "ode_solvers_model"),
        (
            Box::new(ComplexMappingTool::default()),
            "complex_mapping_model",
        ),
        (
            Box::new(RiemannIntegrationTool::default()),
            "riemann_integration_model",
        ),
        (
            Box::new(BeamProfilingTool::default()),
            "beam_profiling_model",
        ),
        (Box::new(CapacityFadeTool::default()), "capacity_fade_model"),
        (Box::new(SirTool::default()), "sir_model_model"),
    ];

    for (tool, expected_stem) in tools {
        verify_tool_exports_in_framework(tool, expected_stem, &temp_dir);
    }
    let _ = std::fs::remove_dir_all(&temp_dir);
}

fn verify_tool_exports_in_framework(
    tool: Box<dyn InteractiveTool>,
    expected_stem: &str,
    temp_dir: &Path,
) {
    let mut fw = SimulationFramework::new("non_existent_domain_xyz");
    fw.active_tool = Some(tool);
    fw.export_queue.destination_dir = Some(temp_dir.to_path_buf());
    let clip_buf = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let clip_clone = Arc::clone(&clip_buf);
    fw.export_queue.custom_clipboard_writer = Some(Arc::new(move |s| {
        clip_clone.lock().unwrap().push(s.to_string());
        Ok(())
    }));

    let ctx = egui::Context::default();
    for fmt in [ExportFormat::Tex, ExportFormat::Rust, ExportFormat::Python] {
        crate::accessibility::clear_test_announcements();
        fw.queue_export(fmt);
        fw.export_queue.push_target(fmt, ExportTarget::Clipboard);
        let _ = ctx.run(egui::RawInput::default(), |ctx| fw.show(ctx, "code_q_test"));

        let expected_file = format!("{expected_stem}.{}", fmt.extension());
        let written = std::fs::read_to_string(temp_dir.join(&expected_file)).unwrap();
        assert!(!written.trim().is_empty());
        assert!(crate::accessibility::all_announcements()
            .iter()
            .any(|(msg, prio)| prio == "polite" && msg.contains("clipboard")));
    }
    assert_eq!(clip_buf.lock().unwrap().len(), 3);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_all_six_flagship_tools_compile_rustc_2024_and_execute_python() {
    let dir = std::env::temp_dir().join(format!("oxidize_rustc_py_6tools_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let (artifacts, ode_log) = build_flagship_artifacts();
    for (slug, art) in &artifacts {
        compile_and_run_rust_2024(&dir, slug, &art.rust_source);
        run_python_script(&dir, slug, &art.python_source);
    }
    verify_ode_rk4_parity(&dir, &ode_log);
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(not(target_arch = "wasm32"))]
fn build_flagship_artifacts() -> (
    Vec<(&'static str, crate::framework::CodeArtifact)>,
    OdeSolverTool,
) {
    let ode_custom = configured::<OdeSolverTool>(|t| {
        t.preset = OdePreset::Custom;
        t.param_k = 1.85;
        t.recalculate();
    });
    let ode_harm = configured::<OdeSolverTool>(|t| {
        t.preset = OdePreset::HarmonicOscillator;
        t.recalculate();
    });
    let ode_log = configured::<OdeSolverTool>(|t| {
        t.preset = OdePreset::LogisticGrowth;
        t.recalculate();
    });
    let cmap_custom = configured::<ComplexMappingTool>(|t| {
        t.function = MappingFunction::Custom;
    });
    let riemann_exp = configured::<RiemannIntegrationTool>(|t| {
        t.preset = FunctionPreset::EToX;
        t.recalculate();
    });

    let list = vec![
        (
            "ode_exp",
            OdeSolverTool::default().export_code_artifact().unwrap(),
        ),
        ("ode_custom", ode_custom.export_code_artifact().unwrap()),
        ("ode_harm", ode_harm.export_code_artifact().unwrap()),
        ("ode_log", ode_log.export_code_artifact().unwrap()),
        (
            "cmap_sq",
            ComplexMappingTool::default()
                .export_code_artifact()
                .unwrap(),
        ),
        ("cmap_custom", cmap_custom.export_code_artifact().unwrap()),
        (
            "riemann_sin",
            RiemannIntegrationTool::default()
                .export_code_artifact()
                .unwrap(),
        ),
        ("riemann_exp", riemann_exp.export_code_artifact().unwrap()),
        (
            "beam",
            BeamProfilingTool::default().export_code_artifact().unwrap(),
        ),
        (
            "battery",
            CapacityFadeTool::default().export_code_artifact().unwrap(),
        ),
        ("sir", SirTool::default().export_code_artifact().unwrap()),
    ];
    (list, ode_log)
}

#[cfg(not(target_arch = "wasm32"))]
fn compile_and_run_rust_2024(dir: &Path, slug: &str, rust_source: &str) {
    let rs_path = dir.join(format!("{slug}.rs"));
    let bin_path = dir.join(format!("{slug}_bin{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(&rs_path, rust_source).unwrap();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let compile_out = Command::new(rustc)
        .args(["--edition", "2024", "-D", "warnings"])
        .arg(&rs_path)
        .arg("-o")
        .arg(&bin_path)
        .output()
        .expect("invoke rustc");
    assert!(
        compile_out.status.success(),
        "rustc --edition 2024 failed for {slug}:\n{}",
        String::from_utf8_lossy(&compile_out.stderr)
    );
    let out = Command::new(&bin_path)
        .output()
        .expect("run compiled binary");
    assert!(out.status.success(), "compiled {slug} exited non-zero");
    assert!(!out.stdout.is_empty());
}

#[cfg(not(target_arch = "wasm32"))]
fn find_python_runner(probe_code: &str) -> Option<&'static str> {
    ["python3", "python"].into_iter().find(|cmd| {
        Command::new(cmd)
            .args(["-c", probe_code])
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn run_python_script(dir: &Path, slug: &str, python_source: &str) {
    let py_path = dir.join(format!("{slug}.py"));
    std::fs::write(&py_path, python_source).unwrap();
    if let Some(py_cmd) = find_python_runner("import numpy, scipy") {
        let out = Command::new(py_cmd)
            .arg(&py_path)
            .output()
            .expect("invoke scientific python");
        assert!(
            out.status.success(),
            "{py_cmd} failed for {slug}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!out.stdout.is_empty());
    } else if let Some(py_cmd) = find_python_runner("import ast") {
        let out = Command::new(py_cmd)
            .args([
                "-c",
                "import ast, pathlib, sys; ast.parse(pathlib.Path(sys.argv[1]).read_text('utf-8'))",
            ])
            .arg(&py_path)
            .output()
            .expect("validate python ast syntax");
        assert!(
            out.status.success(),
            "Python syntax check failed for {slug}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn read_last_csv_y(dir: &Path, stem: &str) -> f64 {
    let bin = dir.join(format!("{stem}{}", std::env::consts::EXE_SUFFIX));
    let stdout = String::from_utf8(Command::new(bin).output().unwrap().stdout).unwrap();
    stdout
        .lines()
        .last()
        .and_then(|l| l.split(',').nth(1))
        .unwrap()
        .parse()
        .unwrap()
}

#[cfg(not(target_arch = "wasm32"))]
fn verify_ode_rk4_parity(dir: &Path, ode_log: &OdeSolverTool) {
    let log_y = read_last_csv_y(dir, "ode_log_bin");
    let expected_log = *ode_log.y_series.last().unwrap();
    assert!((log_y - expected_log).abs() < 1e-8);

    let cust_y = read_last_csv_y(dir, "ode_custom_bin");
    let exact_y = (-1.85_f64 * (1.0 - 10.0_f64.cos())).exp();
    assert!(
        (cust_y - exact_y).abs() < 1e-6,
        "Standalone Rust RK4 final y {cust_y} diverged from exact {exact_y}"
    );
}

#[test]
fn test_nonfinite_parameters_and_unsupported_tools_are_guarded() {
    let bad_beam = configured::<BeamProfilingTool>(|t| {
        t.mu = f64::NAN;
    });
    assert!(!bad_beam.has_finite_parameters());
    assert!(bad_beam.export_code_artifact().is_none());

    let mut fw = SimulationFramework::new("non_existent_domain_xyz");
    fw.active_tool = Some(Box::new(bad_beam));
    crate::accessibility::clear_test_announcements();
    let ctx = egui::Context::default();
    fw.queue_export(ExportFormat::Rust);
    fw.dispatch_pending_exports_with_ctx(&ctx);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Error);
    assert!(fw.export_queue.last_exported_filename.is_none());
    assert!(crate::accessibility::all_announcements()
        .iter()
        .any(|(msg, prio)| prio == "assertive" && msg.contains("finite")));

    let bad_ode = configured::<OdeSolverTool>(|t| {
        t.preset = OdePreset::Custom;
        t.math_input.set_expression("1 +");
    });
    fw.active_tool = Some(Box::new(bad_ode));
    crate::accessibility::clear_test_announcements();
    fw.queue_export(ExportFormat::Python);
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        fw.show(ctx, "syntax_guard")
    });
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Error);
    assert!(fw.export_queue.last_exported_filename.is_none());
}

#[test]
fn test_code_export_modal_open_tab_switch_and_escape_close() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut fw = SimulationFramework::new("non_existent_domain_xyz");
    fw.active_tool = Some(Box::new(OdeSolverTool::default()));

    let screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1400.0, 800.0),
    ));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect,
            ..Default::default()
        },
        |ctx| {
            fw.show(ctx, "modal_code_test");
        },
    );
    assert!(!fw.show_code_export_modal);

    fw.show_code_export_modal = true;
    fw.code_export_tab = CodeExportTab::Rust;
    let out_open = ctx.run(
        egui::RawInput {
            screen_rect,
            ..Default::default()
        },
        |ctx| {
            fw.show(ctx, "modal_code_test");
        },
    );
    assert!(fw.show_code_export_modal);
    let ak = out_open.platform_output.accesskit_update.as_ref().unwrap();
    assert!(ak
        .nodes
        .iter()
        .any(|(_, n)| { n.label().is_some_and(|l| l.contains("Copy Rust")) }));

    let _ = ctx.run(
        egui::RawInput {
            screen_rect,
            events: vec![egui::Event::Key {
                key: egui::Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        },
        |ctx| fw.show(ctx, "modal_code_test"),
    );
    assert!(!fw.show_code_export_modal);
}

#[test]
fn test_math_input_with_params_inline_copy_buttons() {
    let mut bar = MathExpressionInputBar::new("y'", "-k * y * sin(t)", vec![]);
    let mut params = HashMap::new();
    params.insert("k".to_string(), 1.85);

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(800.0, 300.0),
    ));
    let out = ctx.run(
        egui::RawInput {
            screen_rect,
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                bar.ui_with_params(ui, &params);
            });
        },
    );
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let py_pos = ak
        .nodes
        .iter()
        .find_map(|(_, n)| {
            (n.role() == egui::accesskit::Role::Button && n.label() == Some("📋 Python"))
                .then(|| n.bounds().unwrap())
        })
        .map(|b| egui::pos2(((b.x0 + b.x1) * 0.5) as f32, ((b.y0 + b.y1) * 0.5) as f32))
        .expect("📋 Python button");

    crate::accessibility::clear_test_announcements();
    let mut copied_py = String::new();
    for pressed in [true, false] {
        let step_out = ctx.run(
            egui::RawInput {
                screen_rect,
                events: vec![
                    egui::Event::PointerMoved(py_pos),
                    egui::Event::PointerButton {
                        pos: py_pos,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    bar.ui_with_params(ui, &params);
                });
            },
        );
        for cmd in &step_out.platform_output.commands {
            if let egui::OutputCommand::CopyText(txt) = cmd {
                copied_py.clone_from(txt);
            }
        }
    }
    assert_eq!(copied_py, "(-1.85 * y) * np.sin(t)");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_custom_riemann_and_joukowsky_complex_mapping_compile_and_run() {
    let dir = std::env::temp_dir().join(format!("oxidize_custom_sandbox_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let riemann_custom = configured::<RiemannIntegrationTool>(|t| {
        t.preset = FunctionPreset::Custom;
        t.math_input.set_expression("clamp(sin(k * x), -0.5, 0.8)");
        t.math_input
            .set_params_map(&std::collections::BTreeMap::from([("k".into(), 3.0)]));
        t.recalculate();
    });
    let r_art = riemann_custom
        .export_code_artifact()
        .expect("custom riemann artifact");
    assert!(r_art.calibrated_latex.contains("3"));
    compile_and_run_rust_2024(&dir, "riemann_custom", &r_art.rust_source);
    run_python_script(&dir, "riemann_custom", &r_art.python_source);

    let joukowsky = configured::<ComplexMappingTool>(|t| {
        t.function = MappingFunction::Custom;
        t.math_input.set_expression("z + a^2 / z");
        t.math_input
            .set_params_map(&std::collections::BTreeMap::from([("a".into(), 0.85)]));
    });
    let j_art = joukowsky
        .export_code_artifact()
        .expect("joukowsky artifact");
    assert!(j_art.calibrated_latex.contains("0.85"));
    compile_and_run_rust_2024(&dir, "cmap_joukowsky", &j_art.rust_source);
    run_python_script(&dir, "cmap_joukowsky", &j_art.python_source);
    let _ = std::fs::remove_dir_all(&dir);
}
