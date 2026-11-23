use egui::{Pos2, Rect};

use crate::{
    HLine, Line, PlotBounds, PlotItem, VLine,
    poi::{CriticalPointKind, MAX_CRITICAL_POINTS, detect_critical_points},
    transform::PlotTransform,
};

fn make_transform(min: [f64; 2], max: [f64; 2]) -> PlotTransform {
    PlotTransform::new(
        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(800.0, 600.0)),
        PlotBounds::from_min_max(min, max),
        true,
    )
}

#[test]
fn harmonic_oscillator_dt_025_root_error_below_1e_3() {
    let dt = 0.25_f64;
    let pts: Vec<[f64; 2]> = (0..=40)
        .map(|i| {
            let t = i as f64 * dt;
            [t, t.cos()]
        })
        .collect();
    let items: Vec<Box<dyn PlotItem>> = vec![Box::new(Line::new("Position y(t)", pts))];
    let tf = make_transform([0.0, -1.2], [10.0, 1.2]);
    let pois = detect_critical_points(&items, &tf);
    let roots: Vec<f64> = pois
        .iter()
        .filter(|p| p.kind == CriticalPointKind::Root)
        .map(|p| p.point[0])
        .collect();
    assert_eq!(roots.len(), 3);
    for (idx, &r) in roots.iter().enumerate() {
        let exact = (idx as f64 + 0.5) * std::f64::consts::PI;
        assert!(
            (r - exact).abs() < 1e-3,
            "root {idx}: got {r}, exact {exact}, err {}",
            (r - exact).abs()
        );
    }
}

#[test]
fn capacity_fade_threshold_intersection_error_below_half_cycle() {
    let alpha = 850.0_f64;
    let beta = -1.35_f64;
    let dod = 0.45_f64;
    let n_70 = alpha * dod.powf(beta);
    let max_n = n_70 * 1.2;
    let pts: Vec<[f64; 2]> = (0..=100)
        .map(|i| {
            let n = (i as f64 / 100.0) * max_n;
            let cap = (1.0 - 0.3 * (n / n_70).powf(0.5)).max(0.0) * 100.0;
            [n, cap]
        })
        .collect();
    let items: Vec<Box<dyn PlotItem>> = vec![
        Box::new(Line::new("Remaining Capacity (%)", pts)),
        Box::new(HLine::new("EOL (70%)", 70.0)),
    ];
    let tf = make_transform([0.0, 50.0], [max_n, 102.0]);
    let pois = detect_critical_points(&items, &tf);
    let crossing = pois
        .iter()
        .find(|p| p.kind == CriticalPointKind::Intersection)
        .expect("Expected EOL threshold intersection");
    assert!((crossing.point[0] - n_70).abs() < 0.5);
    assert!((crossing.point[1] - 70.0).abs() < 1e-9);
}

#[test]
fn step_curve_detects_vertical_threshold_crossing() {
    let step_pts = vec![
        [0.0, 1.0],
        [4.0, 1.0],
        [4.0, 0.65],
        [9.5, 0.65],
        [9.5, 0.40],
        [15.0, 0.40],
    ];
    let items: Vec<Box<dyn PlotItem>> = vec![
        Box::new(Line::new("Kaplan-Meier S(t)", step_pts)),
        Box::new(HLine::new("Median S(t) = 0.50", 0.50)),
    ];
    let tf = make_transform([0.0, 0.0], [16.0, 1.05]);
    let pois = detect_critical_points(&items, &tf);
    let crossing = pois
        .iter()
        .find(|p| p.kind == CriticalPointKind::Intersection)
        .expect("Expected step crossing at t = 9.5");
    assert!((crossing.point[0] - 9.5).abs() < 1e-9);
    assert!((crossing.point[1] - 0.50).abs() < 1e-9);
}

#[test]
fn flat_series_and_nan_inputs_produce_no_spurious_pois() {
    let flat_zero = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
    let nan_series = vec![
        [0.0, f64::NAN],
        [1.0, 2.0],
        [2.0, f64::INFINITY],
        [3.0, -1.0],
        [f64::NEG_INFINITY, 0.5],
    ];
    let items: Vec<Box<dyn PlotItem>> = vec![
        Box::new(Line::new("FlatZero", flat_zero)),
        Box::new(Line::new("NaNSeries", nan_series)),
        Box::new(VLine::new("BadVLine", f64::NAN)),
    ];
    let tf = make_transform([-1.0, -5.0], [5.0, 5.0]);
    let pois = detect_critical_points(&items, &tf);
    assert!(
        pois.iter()
            .all(|p| p.point[0].is_finite() && p.point[1].is_finite())
    );
    assert!(
        pois.iter()
            .filter(|p| p.primary_series == "FlatZero")
            .all(|p| p.kind == CriticalPointKind::YIntercept)
    );
}

#[test]
fn critical_points_are_capped_at_256() {
    let pts: Vec<[f64; 2]> = (0..=1200)
        .map(|i| {
            let x = i as f64 * 0.5;
            [x, (x * 2.0).sin()]
        })
        .collect();
    let items: Vec<Box<dyn PlotItem>> = vec![Box::new(Line::new("DenseSine", pts))];
    let tf = make_transform([0.0, -2.0], [600.0, 2.0]);
    let pois = detect_critical_points(&items, &tf);
    assert_eq!(pois.len(), MAX_CRITICAL_POINTS);
}

#[test]
fn alt_arrow_keyboard_cycles_pois_and_emits_aria_live_summary() {
    let ctx = egui::Context::default();
    let plot_id = egui::Id::new("poi_kbd_test");
    let sine_pts: Vec<[f64; 2]> = (0..=40)
        .map(|i| {
            let t = i as f64 * 0.25;
            [t, t.cos()]
        })
        .collect();

    let mut focus_target = egui::Id::NULL;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let res = crate::Plot::new(plot_id).show(ui, |plot_ui| {
                plot_ui.line(Line::new("Oscillator", sine_pts.clone()));
            });
            focus_target = res.response.id;
        });
    });
    ctx.memory_mut(|m| m.request_focus(focus_target));

    let step_input = egui::RawInput {
        modifiers: egui::Modifiers::ALT,
        events: vec![egui::Event::Key {
            key: egui::Key::ArrowRight,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::ALT,
        }],
        ..Default::default()
    };
    let _ = ctx.run(step_input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let _ = crate::Plot::new(plot_id).show(ui, |plot_ui| {
                plot_ui.line(Line::new("Oscillator", sine_pts.clone()));
            });
        });
    });

    let live = ctx
        .data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")))
        .expect("Expected polite ARIA live summary");
    assert!(live.contains("Critical point 1 of"), "got: {live}");
    assert!(live.contains("Oscillator"), "got: {live}");
}

#[test]
fn clicking_poi_marker_pins_saved_callout_and_formats_table_exports() {
    let ctx = egui::Context::default();
    let plot_id = egui::Id::new("poi_pin_test");
    let pts = vec![[-2.0, -2.0], [2.0, 2.0]];
    let mut root_screen = Pos2::ZERO;

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let res = crate::Plot::new("poi_pin").id(plot_id).show(ui, |plot_ui| {
                plot_ui.line(Line::new("Ramp", pts.clone()));
            });
            root_screen = res
                .transform
                .position_from_point(&crate::PlotPoint::new(0.0, 0.0));
        });
    });

    let _ = ctx.run(
        egui::RawInput {
            events: vec![
                egui::Event::PointerMoved(root_screen),
                egui::Event::PointerButton {
                    pos: root_screen,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let _ = crate::Plot::new("poi_pin").id(plot_id).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Ramp", pts.clone()));
                });
            });
        },
    );
    let _ = ctx.run(
        egui::RawInput {
            events: vec![egui::Event::PointerButton {
                pos: root_screen,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let _ = crate::Plot::new("poi_pin").id(plot_id).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Ramp", pts.clone()));
                });
            });
        },
    );

    let mem = crate::PlotMemory::load(&ctx, plot_id).expect("Expected PlotMemory");
    assert_eq!(mem.callouts.len(), 1);
    assert!(mem.callouts[0].point[0].abs() < 1e-6);
    assert!(mem.callouts[0].point[1].abs() < 1e-6);

    let items: Vec<Box<dyn PlotItem>> = vec![Box::new(Line::new("Ramp", pts))];
    let pois = detect_critical_points(&items, &make_transform([-3.0, -3.0], [3.0, 3.0]));
    let tsv = crate::data_table::format::format_pois_tsv(&pois);
    let csv = crate::data_table::format::format_pois_csv(&pois);
    assert!(tsv.contains("Category\tSeries\tX\tY\nRoot\tRamp\t"));
    assert!(csv.contains("Category,Series,X,Y\nRoot,Ramp,"));
}

#[test]
fn fitted_curve_and_residual_series_appear_in_exported_datasets() {
    let ctx = egui::Context::default();
    let plot_id = egui::Id::new("fit_export_series_test");
    let pts: Vec<[f64; 2]> = (0..=20)
        .map(|i| {
            let x = i as f64 * 0.2;
            [x, 2.0 * (-0.5 * x).exp()]
        })
        .collect();

    let st = crate::AccessState {
        fit_model: Some(crate::FitType::Exponential),
        show_fit_overlay: true,
        show_fit_residuals: true,
        ..Default::default()
    };
    ctx.data_mut(|d| d.insert_temp(plot_id, st));

    crate::clear_last_plot_datasets();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let _ = crate::Plot::new("fit_exp").id(plot_id).show(ui, |plot_ui| {
                plot_ui.line(Line::new("Decay", pts.clone()));
            });
        });
    });

    let exported = crate::get_last_plot_datasets();
    assert!(
        exported
            .iter()
            .any(|(n, _)| n.starts_with("Fit (Exp): Decay"))
    );
    assert!(exported.iter().any(|(n, _)| n == "Decay Residuals"));
}

#[test]
fn plateau_200k_series_detects_extrema_in_linear_time_under_5ms() {
    let mut pts = Vec::with_capacity(200_000);
    pts.push([0.0, 0.0]);
    for i in 1..100_000 {
        pts.push([i as f64, 5.0]);
    }
    pts.push([100_000.0, 2.0]);
    for i in 100_001..199_999 {
        pts.push([i as f64, -3.0]);
    }
    pts.push([199_999.0, 1.0]);

    let items: Vec<Box<dyn PlotItem>> = vec![Box::new(Line::new("Plateau200k", pts))];
    let tf = make_transform([-10.0, -10.0], [200_010.0, 10.0]);

    // Warm up instruction cache once before timing
    let _ = detect_critical_points(&items, &tf);
    let start = std::time::Instant::now();
    let pois = detect_critical_points(&items, &tf);
    let elapsed = start.elapsed();

    let max_poi = pois
        .iter()
        .find(|p| p.kind == CriticalPointKind::LocalMax)
        .expect("expected plateau midpoint LocalMax");
    let min_poi = pois
        .iter()
        .find(|p| p.kind == CriticalPointKind::LocalMin)
        .expect("expected plateau midpoint LocalMin");
    assert!((max_poi.point[0] - 50_000.0).abs() <= 1.0);
    assert!((max_poi.point[1] - 5.0).abs() < 1e-9);
    assert!((min_poi.point[0] - 149_999.5).abs() <= 1.0);
    assert!((min_poi.point[1] + 3.0).abs() < 1e-9);
    assert!(
        elapsed < std::time::Duration::from_millis(5),
        "200k plateau detect_critical_points took {elapsed:?} (>= 5ms)"
    );
}

#[test]
fn pairwise_intersections_skip_disjoint_y_bounds_and_series_above_cap() {
    let tf = make_transform([0.0, -20.0], [20.0, 20.0]);
    let upper = Line::new("Upper", vec![[0.0, 10.0], [10.0, 12.0]]);
    let lower = Line::new("Lower", vec![[0.0, -12.0], [10.0, -10.0]]);
    let disjoint_items: Vec<Box<dyn PlotItem>> = vec![Box::new(upper), Box::new(lower)];
    let disjoint_pois = detect_critical_points(&disjoint_items, &tf);
    assert!(
        disjoint_pois
            .iter()
            .all(|p| p.kind != CriticalPointKind::Intersection)
    );

    // 17 crossing lines (> MAX_PAIRWISE_INTERSECTION_SERIES = 16) skip O(S^2) pairwise scan
    let mut many_items: Vec<Box<dyn PlotItem>> = Vec::new();
    for idx in 0..=crate::poi::MAX_PAIRWISE_INTERSECTION_SERIES {
        let slope = (idx as f64) * 0.2 - 1.6;
        many_items.push(Box::new(Line::new(
            format!("S{idx}"),
            vec![[0.0, -slope * 5.0], [10.0, slope * 5.0]],
        )));
    }
    let capped_pois = detect_critical_points(&many_items, &tf);
    assert!(
        capped_pois
            .iter()
            .all(|p| p.kind != CriticalPointKind::Intersection)
    );
}

#[test]
fn static_frames_reuse_cached_pois_and_defer_export_primitive_materialization() {
    let ctx = egui::Context::default();
    let plot_id = egui::Id::new("cached_poi_and_deferred_export_test");
    let pts = std::sync::Arc::new(vec![
        crate::PlotPoint::new(-2.0, -1.0),
        crate::PlotPoint::new(0.0, 2.0),
        crate::PlotPoint::new(2.0, -1.0),
    ]);

    crate::poi::reset_detect_critical_points_call_count();
    crate::export_primitives::reset_materialized_export_primitives_call_count();
    for _ in 0..4 {
        crate::clear_last_plot_datasets();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let _ = crate::Plot::new("cached_poi")
                    .id(plot_id)
                    .default_x_bounds(-3.0, 3.0)
                    .default_y_bounds(-3.0, 3.0)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new(
                            "ArcWave",
                            crate::PlotPoints::from_shared(std::sync::Arc::clone(&pts)),
                        ));
                    });
            });
        });
    }

    assert!(
        crate::poi::detect_critical_points_call_count() <= 2,
        "expected cached POIs across static frames, got {} calls",
        crate::poi::detect_critical_points_call_count()
    );
    assert_eq!(
        crate::export_primitives::materialized_export_primitives_call_count(),
        0
    );
    let prims = crate::get_last_plot_primitives();
    assert_eq!(prims.len(), 1);
    assert_eq!(
        crate::export_primitives::materialized_export_primitives_call_count(),
        1
    );
}

#[test]
fn chunked_plateau_preserves_y_intercept_and_nan_break_semantics() {
    // Negative-to-positive plateau crossing x = 0 must preserve YIntercept and LocalMax
    let mut cross_pts = vec![[-30.0, 1.0]];
    for x in -29..=29 {
        if x != 0 {
            cross_pts.push([x as f64, 4.0]);
        }
    }
    cross_pts.push([30.0, 1.0]);
    let tf = make_transform([-35.0, -5.0], [35.0, 10.0]);
    let pois = detect_critical_points(&[Box::new(Line::new("CrossPlateau", cross_pts))], &tf);
    assert!(
        pois.iter()
            .any(|p| p.kind == CriticalPointKind::YIntercept && (p.point[1] - 4.0).abs() < 1e-9)
    );
    assert!(
        pois.iter()
            .any(|p| p.kind == CriticalPointKind::LocalMax && p.point[0].abs() < 1e-9)
    );

    // Plateau broken by NaN must not emit a spurious extremum; terminal x = 0 emits YIntercept
    let mut nan_plateau: Vec<[f64; 2]> = (-25..=-1).map(|x| [x as f64, 3.0]).collect();
    nan_plateau.insert(0, [-26.0, 0.0]);
    nan_plateau[12] = [-14.0, f64::NAN];
    nan_plateau.push([0.0, 3.0]);
    let nan_pois = detect_critical_points(&[Box::new(Line::new("NaNPlateau", nan_plateau))], &tf);
    assert!(
        nan_pois
            .iter()
            .all(|p| p.kind != CriticalPointKind::LocalMax)
    );
    assert!(
        nan_pois
            .iter()
            .any(|p| p.kind == CriticalPointKind::YIntercept && (p.point[1] - 3.0).abs() < 1e-9)
    );
}

#[test]
fn flat_plateau_chunk_skip_preserves_y_intercept_and_nan_reset() {
    let tf = make_transform([-80.0, -10.0], [80.0, 10.0]);
    let mut pts = vec![[-75.0, 0.0]];
    for k in 0..35 {
        pts.push([-74.0 + k as f64, -4.0]);
    }
    pts.push([-39.0, 0.0]);
    for k in 0..37 {
        pts.push([-20.0 + k as f64, 4.0]);
    }
    pts.push([18.0, 0.0]);

    let items: Vec<Box<dyn PlotItem>> = vec![Box::new(Line::new("Plateau", pts))];
    let pois = detect_critical_points(&items, &tf);
    assert!(
        pois.iter()
            .any(|p| p.kind == CriticalPointKind::YIntercept && (p.point[1] - 4.0).abs() < 1e-9),
        "expected y-intercept on flat plateau crossing x = 0, got {pois:?}"
    );
    assert!(pois.iter().any(|p| p.kind == CriticalPointKind::LocalMin
        && (p.point[0] + 57.0).abs() < 1e-9
        && (p.point[1] + 4.0).abs() < 1e-9));
    assert!(pois.iter().any(|p| p.kind == CriticalPointKind::LocalMax
        && (p.point[0] + 2.0).abs() < 1e-9
        && (p.point[1] - 4.0).abs() < 1e-9));
}
