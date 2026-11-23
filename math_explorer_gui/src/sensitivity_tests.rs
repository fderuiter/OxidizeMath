#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

pub(crate) fn collect_shape_texts(shapes: &[egui::epaint::Shape], out: &mut Vec<String>) {
    for shape in shapes {
        match shape {
            egui::epaint::Shape::Text(text_shape) => {
                out.push(text_shape.galley.text().to_string());
            }
            egui::epaint::Shape::Vec(nested) => collect_shape_texts(nested, out),
            _ => {}
        }
    }
}

#[test]
fn test_sensitivity_linear_function() {
    let mut engine = SensitivityEngine::new();
    let params = vec![("a".to_string(), 5.0), ("b".to_string(), 10.0)];

    // Y = 2*a + b -> dY/da = 2, dY/db = 1
    let eval_fn = |p: &HashMap<String, f64>| -> f64 {
        let a = p.get("a").copied().unwrap_or(0.0);
        let b = p.get("b").copied().unwrap_or(0.0);
        2.0 * a + b
    };

    let metrics = engine.compute(&params, eval_fn);
    assert_eq!(metrics.len(), 2);
    assert!((metrics[0].partial_derivative - 2.0).abs() < 1e-4);
    assert!(!metrics[0].is_high_sensitivity);
}

#[test]
fn test_high_sensitivity_warning() {
    let mut engine = SensitivityEngine::new();
    let params = vec![("x".to_string(), 3.0)];

    // Y = x^3 -> dY/dx = 3*x^2 = 27. Elasticity = 27 * 3 / 27 = 3.0 > 2.0
    let eval_fn = |p: &HashMap<String, f64>| -> f64 {
        let x = p.get("x").copied().unwrap_or(1.0);
        x.powi(3)
    };

    let metrics = engine.compute(&params, eval_fn);
    assert_eq!(metrics.len(), 1);
    assert!((metrics[0].elasticity - 3.0).abs() < 1e-2);
    assert!(metrics[0].is_high_sensitivity);
}

#[test]
fn test_bounded_evaluations_run_count() {
    let mut engine = SensitivityEngine::new();
    let params = vec![
        ("p1".to_string(), 1.0),
        ("p2".to_string(), 2.0),
        ("p3".to_string(), 3.0),
        ("p4".to_string(), 4.0),
    ];

    let mut call_count = 0;
    let eval_fn = |p: &HashMap<String, f64>| -> f64 {
        call_count += 1;
        p.values().sum()
    };

    engine.compute(&params, eval_fn);
    // 1 baseline evaluation + max 2 params * 2 perturbations = 5 evaluations max
    assert!(call_count <= 5);
    assert_eq!(call_count, 5);
}

#[test]
fn test_render_labeled_sensitivity_panel_ui_renders_custom_label() {
    let ctx = egui::Context::default();
    let mut engine = SensitivityEngine::new();
    let params = vec![("g_na".to_string(), 120.0)];
    engine.compute(&params, |p| p.get("g_na").copied().unwrap_or(0.0) * 0.25);

    let out = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            render_labeled_sensitivity_panel_ui(
                ui,
                &engine,
                "hh_test",
                "Peak Membrane Potential (mV)",
            );
        });
    });

    let raw_shapes: Vec<egui::epaint::Shape> = out.shapes.into_iter().map(|cs| cs.shape).collect();
    let mut texts = Vec::new();
    collect_shape_texts(&raw_shapes, &mut texts);

    assert!(
        texts
            .iter()
            .any(|t| t.contains("Baseline Peak Membrane Potential (mV): 30.0000")),
        "Expected labeled baseline metric in rendered UI texts, got: {:?}",
        texts
    );
}

#[test]
fn test_sensitivity_metric_card_uses_wcag_semantic_palette_across_themes() {
    let mut engine = SensitivityEngine::new();
    engine.metrics = vec![
        SensitivityMetric {
            param_name: "zero_p".to_string(),
            baseline_value: 1.0,
            partial_derivative: 0.0,
            elasticity: 0.0,
            is_high_sensitivity: false,
        },
        SensitivityMetric {
            param_name: "half_p".to_string(),
            baseline_value: 2.0,
            partial_derivative: 1.5,
            elasticity: 1.5,
            is_high_sensitivity: false,
        },
        SensitivityMetric {
            param_name: "high_p".to_string(),
            baseline_value: 3.0,
            partial_derivative: 27.0,
            elasticity: 3.0,
            is_high_sensitivity: true,
        },
    ];

    for mode in [
        crate::app::ThemeMode::Dark,
        crate::app::ThemeMode::Light,
        crate::app::ThemeMode::HighContrast,
    ] {
        let visuals = mode.visuals();
        let ctx = egui::Context::default();
        ctx.set_visuals(visuals.clone());
        let palette = mode.palette();
        let panel_bg = visuals.panel_fill;
        let bar_text_fg = visuals.selection.stroke.color;
        assert!(crate::app::contrast_ratio(palette.error, panel_bg) >= 4.5);
        assert!(crate::app::contrast_ratio(palette.warning, panel_bg) >= 4.5);
        assert!(crate::app::contrast_ratio(palette.success, panel_bg) >= 4.5);
        assert!(crate::app::contrast_ratio(visuals.extreme_bg_color, bar_text_fg) >= 4.5);
        assert!(crate::app::contrast_ratio(visuals.selection.bg_fill, bar_text_fg) >= 4.5);
        for m in &engine.metrics {
            let fill = crate::app::SemanticPalette::progress_bar_fill(&visuals, m.elasticity);
            assert!(crate::app::contrast_ratio(fill, bar_text_fg) >= 4.5);
        }

        let out = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_sensitivity_panel_ui(ui, &engine, "wcag_sens");
            });
        });
        let raw_shapes: Vec<egui::epaint::Shape> =
            out.shapes.into_iter().map(|cs| cs.shape).collect();
        let mut texts = Vec::new();
        collect_shape_texts(&raw_shapes, &mut texts);
        assert!(texts.iter().any(|t| t.contains("High Sensitivity")));
    }
}

#[test]
fn test_four_param_progressive_convergence_and_repaint_gating() {
    let mut engine = SensitivityEngine::new();
    let params = [("p1", 1.0), ("p2", 2.0), ("p3", 3.0), ("p4", 4.0)];
    let eval_sum = |p: &HashMap<String, f64>, calls: &mut usize| -> f64 {
        *calls += 1;
        p.get("p1").copied().unwrap_or(0.0)
            + 2.0 * p.get("p2").copied().unwrap_or(0.0)
            + 3.0 * p.get("p3").copied().unwrap_or(0.0)
            + 4.0 * p.get("p4").copied().unwrap_or(0.0)
    };

    let mut frame1_calls = 0;
    engine.compute(&params, |p| eval_sum(p, &mut frame1_calls));
    assert_eq!(frame1_calls, 5);
    assert!(!engine.is_converged());

    let ctx = egui::Context::default();
    let out1 = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            render_sensitivity_panel_ui(ui, &engine, "conv_test");
        });
    });
    assert!(out1.viewport_output[&egui::ViewportId::ROOT]
        .repaint_delay
        .is_zero());

    let mut frame2_calls = 0;
    engine.compute(&params, |p| eval_sum(p, &mut frame2_calls));
    assert_eq!(frame2_calls, 4);
    assert!(engine.is_converged());
    assert!((engine.metrics[2].partial_derivative - 3.0).abs() < 1e-4);
    assert!((engine.metrics[3].partial_derivative - 4.0).abs() < 1e-4);

    let mut frame3_calls = 0;
    engine.compute(&params, |p| eval_sum(p, &mut frame3_calls));
    assert_eq!(frame3_calls, 0);

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            render_sensitivity_panel_ui(ui, &engine, "conv_test");
        });
    });
    let out_idle = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            render_sensitivity_panel_ui(ui, &engine, "conv_test");
        });
    });
    assert!(!out_idle.viewport_output[&egui::ViewportId::ROOT]
        .repaint_delay
        .is_zero());
}

#[test]
fn test_nan_and_signed_zero_canonical_convergence() {
    assert_eq!(canonical_f64_bits(-0.0), canonical_f64_bits(0.0));
    assert_eq!(
        canonical_f64_bits(f64::NAN),
        canonical_f64_bits(f64::from_bits(0x7ff8_0000_0000_0001))
    );

    let mut engine = SensitivityEngine::new();
    let mut calls = 0;
    engine.compute(&[("y0", f64::NAN), ("k", -0.0)], |_| {
        calls += 1;
        f64::NAN
    });
    assert_eq!(calls, 5);
    assert!(engine.is_converged());

    calls = 0;
    engine.compute(&[("y0", -f64::NAN), ("k", 0.0)], |_| {
        calls += 1;
        f64::NAN
    });
    assert_eq!(calls, 0);
}

#[test]
fn test_context_invalidation_and_fair_batch_rotation() {
    let mut engine = SensitivityEngine::new();
    let params_a = [("p1", 1.0), ("p2", 2.0), ("p3", 3.0)];
    let mut calls = 0;
    engine.compute_with_context(&params_a, &1_u32, |p| {
        calls += 1;
        p.values().sum()
    });
    assert_eq!(calls, 5);
    assert_eq!(engine.batch_offset, 2);

    // Value edit preserves batch_offset so p3 updates immediately on continuous drag
    calls = 0;
    let params_b = [("p1", 1.5), ("p2", 2.0), ("p3", 3.0)];
    engine.compute_with_context(&params_b, &1_u32, |p| {
        calls += 1;
        p.values().sum()
    });
    assert_eq!(calls, 5);
    assert!((engine.metrics[2].partial_derivative - 1.0).abs() < 1e-4);

    calls = 0;
    engine.compute_with_context(&params_b, &1_u32, |p| {
        calls += 1;
        p.values().sum()
    });
    assert_eq!(calls, 2);
    assert!(engine.is_converged());

    // Context edit invalidates converged state while schema change resets batch_offset
    calls = 0;
    engine.compute_with_context(&params_b, &2_u32, |p| {
        calls += 1;
        2.0 * p.values().sum::<f64>()
    });
    assert_eq!(calls, 5);
    engine.compute_with_context(&[("only", 1.0)], &2_u32, |_| 1.0);
    assert_eq!(engine.batch_offset, 0);
}

#[test]
fn test_defer_for_gesture_retains_metrics_and_suppresses_repaint_until_release() {
    let mut engine = SensitivityEngine::new();
    let params_init = [("p1", 1.0), ("p2", 2.0), ("p3", 3.0), ("p4", 4.0)];
    let eval_weighted = |p: &HashMap<String, f64>, calls: &mut usize| -> f64 {
        *calls += 1;
        p.get("p1").copied().unwrap_or(0.0)
            + 2.0 * p.get("p2").copied().unwrap_or(0.0)
            + 3.0 * p.get("p3").copied().unwrap_or(0.0)
            + 4.0 * p.get("p4").copied().unwrap_or(0.0)
    };

    let mut calls = 0;
    engine.compute(&params_init, |p| eval_weighted(p, &mut calls));
    assert!(!engine.is_converged());
    let settled_p1 = engine.metrics[0].partial_derivative;

    // Deferring while unconverged retains metrics and suppresses request_repaint
    engine.defer_for_gesture();
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            render_sensitivity_panel_ui(ui, &engine, "defer_repaint_gate");
        });
    });
    let out_deferred = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            render_sensitivity_panel_ui(ui, &engine, "defer_repaint_gate");
        });
    });
    assert!(!out_deferred.viewport_output[&egui::ViewportId::ROOT]
        .repaint_delay
        .is_zero());
    assert!((engine.metrics[0].partial_derivative - settled_p1).abs() < 1e-6);

    // Release clears deferral, reuses baseline hint, and converges across 2 batches
    let params_released = [("p1", 2.0), ("p2", 2.0), ("p3", 3.0), ("p4", 4.0)];
    calls = 0;
    engine.compute_with_baseline(&params_released, &(), Some(31.0), |p| {
        eval_weighted(p, &mut calls)
    });
    assert_eq!(calls, 4);
    assert_eq!(engine.baseline_output, 31.0);
    calls = 0;
    engine.compute_with_baseline(&params_released, &(), Some(31.0), |p| {
        eval_weighted(p, &mut calls)
    });
    assert_eq!(calls, 4);
    assert!(engine.is_converged());
}

#[test]
fn test_compute_with_baseline_reuses_finite_hint_and_falls_back_on_nan() {
    let mut engine = SensitivityEngine::new();
    let params = [("a", 3.0), ("b", 4.0)];

    let mut calls = 0;
    engine.compute_with_baseline(&params, &1_u8, Some(10.0), |p| {
        calls += 1;
        p.get("a").copied().unwrap_or(0.0) * 2.0 + p.get("b").copied().unwrap_or(0.0)
    });
    assert_eq!(calls, 4, "Finite baseline hint must skip baseline eval_fn");
    assert_eq!(engine.baseline_output, 10.0);
    assert!(engine.is_converged());

    calls = 0;
    engine.compute_with_baseline(&params, &2_u8, Some(f64::NAN), |p| {
        calls += 1;
        p.values().sum()
    });
    assert_eq!(calls, 5, "NaN baseline hint must fall back to eval_fn");
    assert!((engine.baseline_output - 7.0).abs() < 1e-12);
}

#[test]
fn test_pane_scoped_sweep_isolation() {
    use crate::reflective_ui::slider::{
        begin_pane_sweep_scope, current_pass_active_sweep_count, merge_pane_sweep_flags,
        save_slider_state, AnimatedSlider, AnimatedSliderState, SweepMode,
    };

    let ctx = egui::Context::default();
    let (mut v0, mut v1) = (1.0, 2.0);
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.ctx()
                .data_mut(|d| d.insert_temp(egui::Id::new("CURRENT_RENDERING_PANE"), 0_usize));
            begin_pane_sweep_scope(ui.ctx(), 0);
            let id0 = crate::reflective_ui::slider::resolve_scoped_slider_id(
                ui,
                Some(egui::Id::new("pane_k")),
                "k",
            );
            save_slider_state(
                ui.ctx(),
                id0,
                AnimatedSliderState {
                    playing: true,
                    direction: 1.0,
                    sweep_mode: SweepMode::Oscillate,
                },
            );
            ui.add(AnimatedSlider::new(&mut v0, 0.0..=5.0).id_salt("pane_k"));
            assert_eq!(current_pass_active_sweep_count(ui.ctx()), 1);
            assert!(is_transient_parameter_gesture(ui.ctx(), true));
            merge_pane_sweep_flags(ui.ctx(), 0);

            ui.ctx()
                .data_mut(|d| d.insert_temp(egui::Id::new("CURRENT_RENDERING_PANE"), 1_usize));
            begin_pane_sweep_scope(ui.ctx(), 1);
            ui.add(AnimatedSlider::new(&mut v1, 0.0..=5.0).id_salt("pane_k"));
            assert_eq!(current_pass_active_sweep_count(ui.ctx()), 0);
            assert!(!is_transient_parameter_gesture(ui.ctx(), false));
            merge_pane_sweep_flags(ui.ctx(), 1);
        });
    });
}

#[test]
fn test_unified_sim_drag_deferral_and_release_convergence() {
    use crate::async_sim::unified::UnifiedSimTool;
    use crate::framework::InteractiveTool;
    use crate::tabs::neuroscience::hodgkin_huxley::HodgkinHuxleyUnified;

    let hh_ctx = egui::Context::default();
    let mut hh_tool = UnifiedSimTool::<HodgkinHuxleyUnified>::new();
    let base_in = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(960.0, 640.0),
        )),
        ..Default::default()
    };
    let _ = hh_ctx.run(base_in.clone(), |ctx| hh_tool.show(ctx));
    let _ = hh_ctx.run(base_in.clone(), |ctx| hh_tool.show(ctx));
    assert!(hh_tool.sensitivity_engine.is_converged());
    let settled_baseline = hh_tool.sensitivity_engine.baseline_output;

    hh_tool
        .params
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert("g_na".to_string(), 165.0);
    let mut drag_in = base_in.clone();
    drag_in.events.push(egui::Event::PointerButton {
        pos: egui::pos2(820.0, 520.0),
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: egui::Modifiers::NONE,
    });
    let _ = hh_ctx.run(drag_in, |ctx| hh_tool.show(ctx));
    assert_eq!(hh_tool.sensitivity_engine.baseline_output, settled_baseline);

    let mut release_in = base_in.clone();
    release_in.events.push(egui::Event::PointerButton {
        pos: egui::pos2(820.0, 520.0),
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    });
    let _ = hh_ctx.run(release_in, |ctx| hh_tool.show(ctx));
    let _ = hh_ctx.run(base_in, |ctx| hh_tool.show(ctx));
    assert!(hh_tool.sensitivity_engine.is_converged());
    assert!((hh_tool.sensitivity_engine.baseline_output - settled_baseline).abs() > 1e-3);
}
