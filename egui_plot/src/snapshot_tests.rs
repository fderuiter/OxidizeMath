use std::collections::BTreeMap;

use egui::Id;

use crate::{
    FitType, ImportedDataset, Line, Plot, PlotBounds, PlotIdSource, PlotMemory,
    PlotOverlaySnapshot, SavedCallout,
    snapshot::{
        MAX_SNAPSHOT_TOTAL_POINTS, OrderedId, begin_plot_persistence_scope,
        end_plot_persistence_scope,
    },
};

#[test]
fn test_ordered_id_deterministic_sort() {
    let mut snap = PlotOverlaySnapshot::default();
    let ids = [
        Id::new("series_gamma"),
        Id::new("series_alpha"),
        Id::new("series_beta"),
    ];
    for id in ids {
        snap.hidden_items.insert(OrderedId(id));
    }
    let ordered_vals: Vec<u64> = snap.hidden_items.iter().map(|id| id.0.value()).collect();
    let mut sorted = ordered_vals.clone();
    sorted.sort_unstable();
    assert_eq!(ordered_vals, sorted);
}

#[test]
fn test_plot_id_source_keys_are_deterministic() {
    let (_, key_str, title_str) = "sir_plot".into_plot_id_key_and_title();
    assert_eq!(key_str, "sir_plot");
    assert_eq!(title_str.as_deref(), Some("SIR Plot"));

    let raw_id = Id::new("phase_portrait");
    let (_, key_id1, title_id1) = raw_id.into_plot_id_key_and_title();
    let (_, key_id2, _) = (&raw_id).into_plot_id_key_and_title();
    assert_eq!(key_id1, key_id2);
    assert!(key_id1.starts_with("id:"));
    assert!(title_id1.is_none());
}

#[test]
fn test_sanitize_clamps_fit_degree_and_rejects_non_finite() {
    let mut snap = PlotOverlaySnapshot {
        fit_degree: Some(99),
        grid_density: 250.0,
        ..Default::default()
    };
    snap.sanitize_and_validate().expect("valid after clamping");
    assert_eq!(snap.fit_model, Some(FitType::Quartic));
    assert_eq!(snap.fit_degree, None);
    assert!((snap.grid_density - 100.0).abs() < f32::EPSILON);

    let mut zero_deg = PlotOverlaySnapshot {
        fit_degree: Some(0),
        ..Default::default()
    };
    zero_deg.sanitize_and_validate().expect("clamps 0 to 1");
    assert_eq!(zero_deg.fit_model, Some(FitType::Linear));

    let mut quad_deg = PlotOverlaySnapshot {
        fit_degree: Some(2),
        ..Default::default()
    };
    quad_deg.sanitize_and_validate().expect("maps degree 2");
    assert_eq!(quad_deg.fit_model, Some(FitType::Quadratic));

    let mut bad_callout = PlotOverlaySnapshot::default();
    bad_callout
        .callouts
        .push(SavedCallout::new([f64::NAN, 1.0], "bad", [0.0, 0.0]));
    assert!(bad_callout.sanitize_and_validate().is_err());

    let mut bad_dataset = PlotOverlaySnapshot::default();
    bad_dataset.imported_datasets.push(ImportedDataset::new(
        "emp",
        vec![[0.0, 1.0], [1.0, f64::INFINITY]],
    ));
    assert!(bad_dataset.sanitize_and_validate().is_err());

    let mut bad_bounds = PlotOverlaySnapshot {
        bounds: Some(PlotBounds::from_min_max([10.0, 0.0], [2.0, 5.0])),
        auto_bounds: Some([false, false]),
        ..Default::default()
    };
    assert!(bad_bounds.sanitize_and_validate().is_err());
}

#[test]
fn test_dataset_points_downsample_within_visit_budget() {
    let mut snap = PlotOverlaySnapshot::default();
    let pts: Vec<[f64; 2]> = (0..2500).map(|i| [i as f64, (i as f64).sin()]).collect();
    snap.imported_datasets
        .push(ImportedDataset::new("large_csv", pts));
    snap.sanitize_and_validate()
        .expect("large dataset downsamples cleanly");
    assert!(snap.imported_datasets[0].points.len() <= MAX_SNAPSHOT_TOTAL_POINTS);
    assert_eq!(snap.imported_datasets[0].points.first(), Some(&[0.0, 0.0]));
    let last_pt = snap.imported_datasets[0].points.last().copied().unwrap();
    assert!((last_pt[0] - 2499.0).abs() < 1e-9);
}

#[cfg(feature = "serde")]
#[test]
fn test_snapshot_serde_roundtrip_and_compact_defaults() {
    let default_snap = PlotOverlaySnapshot::default();
    let compact_json = serde_json::to_string(&default_snap).expect("serialize default");
    assert_eq!(compact_json, "{}");

    let mut rich = PlotOverlaySnapshot {
        auto_bounds: Some([false, false]),
        bounds: Some(PlotBounds::from_min_max([1.0, 10.0], [100.0, 1000.0])),
        log_x: true,
        log_y: true,
        grid_density: 16.0,
        show_x_labels: false,
        clamp_grid: true,
        show_grid: [true, false],
        show_table: true,
        fit_model: Some(FitType::Logistic),
        fit_log_mode: true,
        show_fit_residuals: true,
        fit_dataset_idx: 2,
        custom_fit_formula: "a * x + b".to_owned(),
        custom_fit_seeds_text: "a=2, b=1".to_owned(),
        ..Default::default()
    };
    rich.hidden_items.insert(OrderedId(Id::new("hidden_b")));
    rich.hidden_items.insert(OrderedId(Id::new("hidden_a")));
    rich.imported_datasets.push(ImportedDataset::new(
        "Empirical",
        vec![[1.0, 12.0], [10.0, 120.0]],
    ));
    rich.callouts
        .push(SavedCallout::new([10.0, 120.0], "Peak", [24.0, -18.0]));

    let json = serde_json::to_string(&rich).expect("serialize rich snapshot");
    let mut restored: PlotOverlaySnapshot =
        serde_json::from_str(&json).expect("deserialize rich snapshot");
    restored
        .sanitize_and_validate()
        .expect("validate restored snapshot");
    assert_eq!(restored, rich);
}

#[test]
fn test_scoped_sync_preserves_manual_bounds_on_log_axes_and_skips_locked_3d_bounds() {
    let mut seed = BTreeMap::new();
    let manual_bounds = PlotBounds::from_min_max([1.0, 10.0], [100.0, 1000.0]);
    let snap = PlotOverlaySnapshot {
        auto_bounds: Some([false, false]),
        bounds: Some(manual_bounds),
        log_x: true,
        log_y: true,
        callouts: vec![SavedCallout::new([10.0, 100.0], "Decade", [20.0, -20.0])],
        fit_model: Some(FitType::PowerLaw),
        show_table: true,
        ..Default::default()
    };
    seed.insert("log_plot".to_owned(), snap);
    seed.insert(
        "locked_3d".to_owned(),
        PlotOverlaySnapshot {
            auto_bounds: Some([false, false]),
            bounds: Some(PlotBounds::from_min_max([500.0, 500.0], [600.0, 600.0])),
            ..Default::default()
        },
    );

    let ctx = egui::Context::default();
    let mut plot_2d_id = Id::NULL;
    let mut plot_3d_id = Id::NULL;

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        begin_plot_persistence_scope(ctx, 42, 1, &seed);
        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new("log_plot").height(180.0).show(ui, |plot_ui| {
                plot_2d_id = plot_ui.plot_id();
                plot_ui.line(Line::new(
                    "power",
                    vec![[0.1, 1.0], [10.0, 100.0], [1000.0, 10000.0]],
                ));
            });
            Plot::new("locked_3d")
                .height(180.0)
                .allow_2d_analysis(false)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_axis_zoom_drag(false)
                .allow_scroll(false)
                .allow_boxed_zoom(false)
                .default_x_bounds(-32.0, 32.0)
                .default_y_bounds(-32.0, 32.0)
                .show(ui, |plot_ui| {
                    plot_3d_id = plot_ui.plot_id();
                    plot_ui.set_plot_bounds_x(-32.0..=32.0);
                    plot_ui.set_plot_bounds_y(-32.0..=32.0);
                });
        });
        let after = end_plot_persistence_scope(ctx, 42);
        let synced_2d = after.get("log_plot").expect("2D plot snapshot retained");
        assert!(synced_2d.log_x && synced_2d.log_y);
        assert_eq!(synced_2d.auto_bounds, Some([false, false]));
        assert_eq!(synced_2d.bounds, Some(manual_bounds));
        assert_eq!(synced_2d.fit_model, Some(FitType::PowerLaw));
        assert!(synced_2d.show_table);
        assert!(
            !after.contains_key("locked_3d"),
            "locked 3D viewport must ignore injected manual 2D bounds and normalize to default"
        );
    });

    let mem_2d = PlotMemory::load(&ctx, plot_2d_id).expect("2D PlotMemory stored");
    assert!(!mem_2d.auto_bounds.x && !mem_2d.auto_bounds.y);
    assert!((mem_2d.bounds().min()[0] - 1.0).abs() < 1e-6);
    assert!((mem_2d.bounds().max()[1] - 1000.0).abs() < 1e-6);

    let mem_3d = PlotMemory::load(&ctx, plot_3d_id).expect("3D PlotMemory stored");
    assert!((mem_3d.bounds().min()[0] - -32.0).abs() < 1e-6);
    assert!((mem_3d.bounds().max()[0] - 32.0).abs() < 1e-6);
}
