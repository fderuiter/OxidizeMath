use super::*;

const ALL_THEMES: [ThemeMode; 3] = [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast];

#[test]
fn test_semantic_palette_text_tokens_meet_wcag_aa() {
    for mode in ALL_THEMES {
        let visuals = mode.visuals();
        let p = mode.palette();
        let bgs = [
            ("panel_fill", visuals.panel_fill),
            ("window_fill", visuals.window_fill),
            ("extreme_bg_color", visuals.extreme_bg_color),
        ];
        let text_tokens = [
            ("info", p.info),
            ("success", p.success),
            ("warning", p.warning),
            ("error", p.error),
            ("fatal", p.fatal),
            ("muted_text", p.muted_text),
        ];

        for (token_name, fg) in text_tokens {
            for (bg_name, bg) in bgs {
                let ratio = contrast_ratio(fg, bg);
                assert!(
                    ratio >= 4.5,
                    "{mode:?} text token '{token_name}' against {bg_name} achieved {ratio:.2}:1 (< 4.5:1)"
                );
            }
        }

        for (token_name, fg) in [
            ("error", p.error),
            ("warning", p.warning),
            ("muted_text", p.muted_text),
        ] {
            let ratio = contrast_ratio(fg, p.error_banner_bg);
            assert!(
                ratio >= 4.5,
                "{mode:?} banner text '{token_name}' against error_banner_bg achieved {ratio:.2}:1 (< 4.5:1)"
            );
        }
    }
}

#[test]
fn test_semantic_palette_series_axes_and_strokes_meet_wcag_contrast() {
    for mode in ALL_THEMES {
        let visuals = mode.visuals();
        let p = mode.palette();
        let bgs = [
            ("panel_fill", visuals.panel_fill),
            ("extreme_bg_color", visuals.extreme_bg_color),
        ];
        let graphical_tokens = [
            ("axis_x", p.axis_x),
            ("axis_y", p.axis_y),
            ("axis_z", p.axis_z),
            ("accent", p.accent),
            ("graph_edge", p.graph_edge),
            ("node_stroke", p.node_stroke),
            ("error_banner_stroke", p.error_banner_stroke),
            ("series[0]", p.series_color(0)),
            ("series[1]", p.series_color(1)),
            ("series[2]", p.series_color(2)),
            ("series[3]", p.series_color(3)),
            ("series[4]", p.series_color(4)),
            ("series[5]", p.series_color(5)),
        ];

        for (token_name, fg) in graphical_tokens {
            for (bg_name, bg) in bgs {
                let ratio = contrast_ratio(fg, bg);
                assert!(
                    ratio >= 3.0,
                    "{mode:?} graphical token '{token_name}' against {bg_name} achieved {ratio:.2}:1 (< 3.0:1)"
                );
            }
        }
    }
}

#[test]
fn test_contrasting_text_color_guarantees_wcag_aa_across_luminance_spectrum() {
    for step in 0..=255_u8 {
        for sample in [
            egui::Color32::from_gray(step),
            egui::Color32::from_rgb(step, 0, 255 - step),
            egui::Color32::from_rgb(0, step, step / 2),
        ] {
            let fg = contrasting_text_color(sample);
            let ratio = contrast_ratio(sample, fg);
            assert!(
                ratio >= 4.5,
                "contrasting_text_color({sample:?}) -> {fg:?} yielded {ratio:.2}:1 (< 4.5:1)"
            );
        }
    }
}

#[test]
fn test_high_contrast_selection_contrast_ratio() {
    let visuals = ThemeMode::HighContrast.visuals();
    let selection_bg = visuals.selection.bg_fill;
    assert_eq!(selection_bg, egui::Color32::from_rgb(0, 90, 158));

    // WCAG 2.2 AA requires >= 4.5:1 for normal text and strokes against background
    let active_fg = visuals.widgets.active.fg_stroke.color;
    let active_contrast = contrast_ratio(selection_bg, active_fg);
    assert!(
        active_contrast >= 4.5,
        "Selection fill contrast against active fg ({active_contrast:.2}:1) must exceed 4.5:1"
    );

    let noninteractive_fg = visuals.widgets.noninteractive.fg_stroke.color;
    let noninteractive_contrast = contrast_ratio(selection_bg, noninteractive_fg);
    assert!(
        noninteractive_contrast >= 4.5,
        "Selection fill contrast against noninteractive fg ({noninteractive_contrast:.2}:1) must exceed 4.5:1"
    );

    let inactive_fg = visuals.widgets.inactive.fg_stroke.color;
    let inactive_contrast = contrast_ratio(selection_bg, inactive_fg);
    assert!(
        inactive_contrast >= 4.5,
        "Selection fill contrast against inactive fg ({inactive_contrast:.2}:1) must exceed 4.5:1"
    );
}

#[test]
fn test_legacy_yellow_selection_fails_contrast_ratio() {
    let yellow = egui::Color32::YELLOW;
    let white = egui::Color32::WHITE;
    let ratio = contrast_ratio(yellow, white);
    // Demonstrates why legacy yellow fill violated WCAG contrast
    assert!(
        ratio < 4.5,
        "Legacy yellow on white ({ratio:.2}:1) should fail WCAG 4.5:1 requirement"
    );
}

#[cfg(feature = "ai")]
#[test]
fn test_q_table_and_tokenization_cell_labels_meet_wcag_aa() {
    use crate::framework::InteractiveTool;
    for mode in ALL_THEMES {
        let visuals = mode.visuals();
        let palette = mode.palette();
        for step in [0.0_f64, 0.25, 0.5, 0.75, 1.0] {
            let bg = crate::tabs::ai::q_table_inspector::q_value_cell_color(step, &palette);
            let fg = contrasting_text_color(bg);
            let ratio = contrast_ratio(bg, fg);
            assert!(
                ratio >= 4.5,
                "{mode:?} Q-table cell t={step} achieved {ratio:.2}:1 (< 4.5:1)"
            );
        }
        for val in [-1.0_f64, -0.5, 0.0, 0.5, 1.0] {
            let bg = crate::tabs::ai::tokenization::embedding_cell_color(val, &palette);
            let fg = contrasting_text_color(bg);
            let ratio = contrast_ratio(bg, fg);
            assert!(
                ratio >= 4.5,
                "{mode:?} Tokenization cell val={val} achieved {ratio:.2}:1 (< 4.5:1)"
            );
        }

        let ctx = egui::Context::default();
        ctx.set_visuals(visuals);
        let mut q_tool = crate::tabs::ai::q_table_inspector::QTableInspectorTool::default();
        let mut tok_tool = crate::tabs::ai::tokenization::TokenizationTool::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            q_tool.show(ctx);
            tok_tool.show(ctx);
        });
    }
}

fn collect_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

#[test]
fn test_no_unguarded_hardcoded_color32_regressions() {
    const BANNED: &[&str] = &[
        "Color32::YELLOW",
        "Color32::LIGHT_BLUE",
        "Color32::LIGHT_GREEN",
        "Color32::LIGHT_RED",
        "Color32::LIGHT_GRAY",
        "Color32::GOLD",
        "Color32::GREEN",
        "Color32::RED",
        "Color32::BLUE",
        "Color32::GRAY",
        "Color32::DARK_GRAY",
        "Color32::DARK_RED",
        "Color32::KHAKI",
        "Color32::BROWN",
    ];
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().expect("workspace root");
    let mut files = Vec::new();
    collect_rs_files(&manifest_dir.join("src"), &mut files);
    collect_rs_files(&workspace_root.join("egui_plot/src"), &mut files);

    let mut violations = Vec::new();
    for file in files {
        let rel = file.strip_prefix(workspace_root).unwrap_or(&file);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str.contains("test")
            || rel_str.ends_with("app/palette.rs")
            || rel_str.ends_with("app/theme.rs")
            || rel_str.ends_with("tabs/number_theory/prime_spiral.rs")
        {
            continue;
        }
        let content = std::fs::read_to_string(&file).expect("read source file");
        let prod_part = content.split("#[cfg(test)]").next().unwrap_or(&content);
        for (line_idx, line) in prod_part.lines().enumerate() {
            let code = line.split("//").next().unwrap_or(line);
            for banned in BANNED {
                if code.contains(banned) {
                    violations.push(format!("{rel_str}:{}: {banned}", line_idx + 1));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "Found unguarded hardcoded Color32 constants (use SemanticPalette instead):\n{}",
        violations.join("\n")
    );
}

#[test]
fn test_extract_explicit_theme_supports_raw_and_versioned_envelope() {
    use crate::app::{MathExplorerApp, WorkspaceSessionManager};
    let mut app = MathExplorerApp {
        theme: ThemeMode::Light,
        ..Default::default()
    };
    let raw = app.save_state_to_json();
    assert_eq!(
        extract_explicit_theme_from_json(&raw),
        Some(ThemeMode::Light)
    );
    assert_eq!(
        extract_explicit_theme_from_json(&format!("\u{FEFF}{raw}")),
        Some(ThemeMode::Light)
    );

    app.theme = ThemeMode::HighContrast;
    let env = WorkspaceSessionManager::default().create_envelope(app.get_app_state());
    assert_eq!(
        extract_explicit_theme_from_json(&env),
        Some(ThemeMode::HighContrast)
    );

    for invalid in [
        r#"{"schema_version":"2.0","app_state":{"theme":"Light"}}"#,
        "{ invalid",
        r#"{"selected_tab":0}"#,
    ] {
        assert_eq!(extract_explicit_theme_from_json(invalid), None);
    }
}

#[test]
fn test_creation_context_startup_theme_and_saved_override() {
    use crate::app::MathExplorerApp;
    use eframe::{App, Storage};
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let raw_in = egui::RawInput {
        system_theme: Some(egui::Theme::Light),
        ..Default::default()
    };
    let _ = ctx.run(raw_in, |_| {});

    let cold_app = MathExplorerApp::new(&eframe::CreationContext::_new_kittest(ctx.clone()));
    assert_eq!(cold_app.theme, ThemeMode::Light);
    assert_eq!(ctx.style().visuals, ThemeMode::Light.visuals());

    let mut storage = super::tests::TestStorage::default();
    let mut saved = MathExplorerApp {
        theme: ThemeMode::HighContrast,
        ..Default::default()
    };
    saved.save(&mut storage);
    let mut cc_warm = eframe::CreationContext::_new_kittest(ctx.clone());
    cc_warm.storage = Some(&storage);
    assert_eq!(
        MathExplorerApp::new(&cc_warm).theme,
        ThemeMode::HighContrast
    );
    assert_eq!(ctx.style().visuals, ThemeMode::HighContrast.visuals());

    storage.set_string(eframe::APP_KEY, "{ broken json".to_string());
    let mut cc_corrupt = eframe::CreationContext::_new_kittest(ctx.clone());
    cc_corrupt.storage = Some(&storage);
    assert_eq!(MathExplorerApp::new(&cc_corrupt).theme, ThemeMode::Light);
}
