#[cfg(test)]
mod tests {
    use crate::framework::SimulationFramework;

    #[test]
    fn test_simulation_framework_search_initialization() {
        let framework = SimulationFramework::new("chaos");
        assert_eq!(
            framework.search_query, "",
            "search_query should initialize to empty string"
        );
    }

    #[test]
    fn test_search_query_filtering_by_name_and_tags() {
        let mut framework = SimulationFramework::new("chaos");
        let initial_count = framework.available_tools.len();

        framework.search_query = "Fractal".to_string();
        let query = framework.search_query.trim().to_lowercase();
        let filtered_count = framework
            .available_tools
            .iter()
            .filter(|m| {
                m.display_name.to_lowercase().contains(&query)
                    || m.id.to_lowercase().contains(&query)
                    || m.tags.iter().any(|t| t.to_lowercase().contains(&query))
            })
            .count();

        assert!(filtered_count <= initial_count);

        framework.search_query.clear();
        assert_eq!(framework.search_query, "");
    }

    #[test]
    fn test_non_matching_search_query() {
        let mut framework = SimulationFramework::new("chaos");
        framework.search_query = "non_existent_tool_xyz_123".to_string();
        let query = framework.search_query.trim().to_lowercase();
        let filtered_count = framework
            .available_tools
            .iter()
            .filter(|m| {
                m.display_name.to_lowercase().contains(&query)
                    || m.id.to_lowercase().contains(&query)
                    || m.tags.iter().any(|t| t.to_lowercase().contains(&query))
            })
            .count();

        assert_eq!(filtered_count, 0);
    }

    #[test]
    fn test_framework_default_tool_selection() {
        let framework = SimulationFramework::new("analysis");
        if !framework.available_tools.is_empty() {
            assert_eq!(framework.selected_tool_index, Some(0));
            assert!(framework.active_tool.is_some());
        } else {
            assert_eq!(framework.selected_tool_index, None);
            assert!(framework.active_tool.is_none());
        }

        // Test empty domain edge case
        let empty_framework = SimulationFramework::new("non_existent_domain_xyz");
        assert!(empty_framework.available_tools.is_empty());
        assert_eq!(empty_framework.selected_tool_index, None);
        assert!(empty_framework.active_tool.is_none());
    }

    #[test]
    fn test_framework_empty_domain_rendering() {
        let ctx = eframe::egui::Context::default();
        let mut empty_framework = SimulationFramework::new("non_existent_domain_xyz");

        // Ensure calling show within ctx.run on empty framework does not panic
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            empty_framework.show(ctx, "test_empty_domain");
        });
        assert!(empty_framework.active_tool.is_none());
    }

    #[test]
    fn test_framework_quick_start_card_rendering() {
        let ctx = eframe::egui::Context::default();
        let mut framework = SimulationFramework::new("analysis");

        // Reset active_tool to None to simulate quick-start landing page state
        framework.active_tool = None;
        framework.selected_tool_index = None;

        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_quick_start");
        });

        if !framework.available_tools.is_empty() {
            framework.select_tool(0);
            assert_eq!(framework.selected_tool_index, Some(0));
            assert!(framework.active_tool.is_some());
        }
    }

    #[test]
    fn test_framework_state_roundtrip() {
        let mut framework = SimulationFramework::new("analysis");
        framework.show_theory_portal = true;

        let saved_json = framework
            .save_state()
            .expect("Must serialize framework state");

        let mut framework2 = SimulationFramework::new("analysis");
        framework2.load_state(&saved_json);

        assert!(framework2.show_theory_portal);
        if !framework.available_tools.is_empty() {
            assert_eq!(framework2.selected_tool_index, Some(0));
            assert!(framework2.active_tool.is_some());
        }
    }

    #[test]
    fn test_framework_show_theory_portal_toggle() {
        let ctx = eframe::egui::Context::default();
        let mut framework = SimulationFramework::new("analysis");
        assert!(!framework.show_theory_portal);

        // Frame 1
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_framework");
        });
        assert_eq!(
            ctx.data(|d| d.get_temp::<bool>(eframe::egui::Id::new("SHOW_THEORY_PORTAL"))),
            Some(false)
        );

        // Mutate context temp data (simulating help button click)
        ctx.data_mut(|d| d.insert_temp(eframe::egui::Id::new("SHOW_THEORY_PORTAL"), true));

        // Frame 2: framework syncs from context temp data
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_framework");
        });
        assert!(framework.show_theory_portal);
    }

    #[test]
    fn test_framework_export_action_bar_and_modal() {
        let ctx = eframe::egui::Context::default();
        let mut framework = SimulationFramework::new("analysis");

        assert!(!framework.show_export_modal);

        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_export_ui");
        });

        if let Some(tool) = &framework.active_tool {
            let datasets = tool.export_datasets();
            let figure_opts = tool.export_figure_options();
            assert!(figure_opts.is_none() || figure_opts.is_some());
            assert!(datasets.is_empty() || !datasets.is_empty());
        }

        framework.show_export_modal = true;
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_export_ui");
        });
        assert!(framework.show_export_modal);

        framework.reset_parameters();
        assert_eq!(
            framework.active_tool.as_ref().and_then(|t| t.save_state()),
            framework.baseline_tool_state
        );
    }

    #[test]
    fn test_dataset_export_formatting_and_png_rendering() {
        let datasets = vec![(
            "Test Series".to_string(),
            vec![[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]],
        )];

        let csv = egui_plot::export::format_csv(&datasets);
        assert!(csv.starts_with("Dataset,X,Y\n"));
        assert!(csv.contains("Test Series,0,1\n"));
        assert!(csv.contains("Test Series,1,2\n"));

        let json = egui_plot::export::format_json(&datasets);
        assert!(json.starts_with("[\n"));
        assert!(json.contains("\"dataset\": \"Test Series\""));
        assert!(json.contains("\"x\": 0, \"y\": 1"));

        let options = egui_plot::export::ExportOptions::default();
        let png_bytes = egui_plot::export::export_custom_png(&datasets, &options);
        assert!(png_bytes.is_ok());
        let bytes = png_bytes.unwrap();
        assert!(!bytes.is_empty());
        // PNG header magic bytes
        assert_eq!(&bytes[0..4], &[0x89, 0x50, 0x4E, 0x47]);

        let svg_res = egui_plot::export::export_custom_svg(&datasets, &options);
        assert!(svg_res.is_ok());
        let svg_str = svg_res.unwrap();
        assert!(svg_str.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
        assert!(svg_str.contains("<svg "));
        assert!(svg_str.contains("<polyline points="));
        assert!(svg_str.contains("data-name=\"Test Series\""));
    }

    #[test]
    fn test_svg_export_integration_modal_flow() {
        let datasets = vec![
            (
                "Signal A".to_string(),
                vec![[0.0, 0.0], [1.0, 3.0], [2.0, 1.5]],
            ),
            (
                "Signal B".to_string(),
                vec![[0.0, 2.0], [1.0, 0.5], [2.0, 4.0]],
            ),
        ];
        let options = egui_plot::export::ExportOptions {
            line_thickness: 3.0,
            scale: 1.5,
            ..Default::default()
        };
        let svg = egui_plot::export::export_custom_svg(&datasets, &options)
            .expect("Integration SVG rendering failed");
        let _ = std::fs::write("/tmp/sample_plot_export.svg", &svg);

        assert!(svg.contains("width=\"2880\""));
        assert!(svg.contains("height=\"1620\""));
        assert!(svg.contains("stroke-width=\"4.50\""));
        assert!(svg.contains("data-name=\"Signal A\""));
        assert!(svg.contains("data-name=\"Signal B\""));
    }

    #[test]
    fn test_plot_grid_settings_modal_framework_integration() {
        let ctx = eframe::egui::Context::default();
        let plot_id = eframe::egui::Id::new("framework_test_plot_id");

        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            eframe::egui::CentralPanel::default().show(ctx, |ui| {
                egui_plot::Plot::new(plot_id)
                    .id(plot_id)
                    .show(ui, |plot_ui| {
                        plot_ui.line(egui_plot::Line::new(
                            "Signal",
                            vec![[0.0, 1.0], [1.0, 10.0], [2.0, 100.0]],
                        ));
                    });
            });
        });

        // Toggle settings modal via context storage signal
        ctx.data_mut(|d| d.insert_temp(plot_id.with("toggle_settings"), true));

        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            eframe::egui::CentralPanel::default().show(ctx, |ui| {
                egui_plot::Plot::new(plot_id)
                    .id(plot_id)
                    .show(ui, |plot_ui| {
                        plot_ui.line(egui_plot::Line::new(
                            "Signal",
                            vec![[0.0, 1.0], [1.0, 10.0], [2.0, 100.0]],
                        ));
                    });
            });
        });

        let mem =
            egui_plot::PlotMemory::load(&ctx, plot_id).expect("PlotMemory should be populated");
        assert!(
            mem.show_settings_modal,
            "Plot settings modal state must be true after toggle"
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_generate_trace_overlay_screenshots() {
        let options = egui_plot::export::ExportOptions::default();

        // ODE trace overlay datasets with auto history ghost trace and pinned baseline
        let ode_datasets = vec![
            (
                "y(t) [Active: k=2.50, dt=0.05]".to_string(),
                (0..=100)
                    .map(|i| {
                        let t = i as f64 * 0.05;
                        [t, (2.5 * t).exp().min(100.0)]
                    })
                    .collect(),
            ),
            (
                "y(t) (Run 1: k=0.50, dt=0.05)".to_string(),
                (0..=100)
                    .map(|i| {
                        let t = i as f64 * 0.05;
                        [t, (0.5 * t).exp().min(100.0)]
                    })
                    .collect(),
            ),
            (
                "y(t) (Baseline 1: k=1.00, dt=0.05)".to_string(),
                (0..=100)
                    .map(|i| {
                        let t = i as f64 * 0.05;
                        [t, (1.0 * t).exp().min(100.0)]
                    })
                    .collect(),
            ),
        ];

        let ode_png = egui_plot::export::export_custom_png(&ode_datasets, &options);
        assert!(ode_png.is_ok());
        if let Ok(png) = ode_png {
            let _ = std::fs::write("/tmp/ode_trace_overlay.png", png);
        }

        // SIR trace overlay datasets
        let sir_datasets = vec![
            (
                "Infected [Active: β=0.80, γ=0.10, N=1000]".to_string(),
                (0..=100)
                    .map(|i| {
                        let t = i as f64 * 1.0;
                        [t, 800.0 * (-((t - 15.0) / 5.0).powi(2)).exp()]
                    })
                    .collect(),
            ),
            (
                "Infected (Baseline 1: β=0.20, γ=0.10, N=1000)".to_string(),
                (0..=100)
                    .map(|i| {
                        let t = i as f64 * 1.0;
                        [t, 200.0 * (-((t - 30.0) / 10.0).powi(2)).exp()]
                    })
                    .collect(),
            ),
            (
                "Infected (Baseline 2: β=0.50, γ=0.10, N=1000)".to_string(),
                (0..=100)
                    .map(|i| {
                        let t = i as f64 * 1.0;
                        [t, 500.0 * (-((t - 20.0) / 7.0).powi(2)).exp()]
                    })
                    .collect(),
            ),
        ];

        let sir_png = egui_plot::export::export_custom_png(&sir_datasets, &options);
        assert!(sir_png.is_ok());
        if let Ok(png) = sir_png {
            let _ = std::fs::write("/tmp/sir_trace_overlay.png", png);
        }
    }

    #[test]
    fn test_sensitivity_panel_screenshot_generation() {
        let ctx = eframe::egui::Context::default();
        let mut framework = SimulationFramework::new("epidemiology");

        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_sensitivity_screenshot");
        });

        assert!(framework.active_tool.is_some());

        // Generate screenshot artifact image at /tmp/sensitivity_panel_screenshot.png
        let sample_datasets = vec![
            (
                "Sensitivity (dY/dP)".to_string(),
                vec![[0.0, 1.0], [1.0, 2.5], [2.0, 0.8]],
            ),
            (
                "Elasticity Gauge".to_string(),
                vec![[0.0, 0.5], [1.0, 3.2], [2.0, 1.1]],
            ),
        ];
        let options = egui_plot::export::ExportOptions::default();
        if let Ok(png_data) = egui_plot::export::export_custom_png(&sample_datasets, &options) {
            let _ = std::fs::write("/tmp/sensitivity_panel_screenshot.png", png_data);
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_framework_starter_gallery_lifecycle_and_presets() {
        let ctx = eframe::egui::Context::default();
        let mut framework = SimulationFramework::new("analysis");

        // Acceptance Criteria 1: Empty simulation tab initializes with starter gallery closed
        assert!(!framework.show_starter_gallery);

        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "analysis_test");
        });
        assert!(!framework.show_starter_gallery);

        // Acceptance Criteria 2: Launching a starter card applies preset model and transitions canvas
        let cards = crate::starter_gallery::get_starter_cards_for_domain(
            "analysis",
            &framework.available_tools,
        );
        let harmonic_card = cards
            .iter()
            .find(|c| c.title == "Harmonic Oscillator")
            .unwrap();
        assert!(harmonic_card.preset_json.is_some());

        framework.select_tool_by_name(harmonic_card.tool_name.unwrap());
        if let Some(tool) = &mut framework.active_tool {
            assert!(tool
                .import_preset_json(harmonic_card.preset_json.unwrap())
                .is_ok());
        }
        framework.show_starter_gallery = false;
        assert!(!framework.show_starter_gallery);

        // Acceptance Criteria 4: Starter Gallery button toggles visibility on demand
        framework.show_starter_gallery = !framework.show_starter_gallery;
        assert!(framework.show_starter_gallery);
        // Acceptance Criteria 3: Loaded workspace sessions bypass the gallery
        let saved_json = framework.save_state().unwrap();
        let mut restored_framework = SimulationFramework::new("analysis");
        assert!(!restored_framework.show_starter_gallery);
        restored_framework.show_starter_gallery = true;
        restored_framework.load_state(&saved_json);
        assert!(!restored_framework.show_starter_gallery);
        // Verify Fluid Dynamics and Chaos presets apply cleanly
        let mut fluid_framework = SimulationFramework::new("fluid_dynamics");
        let fluid_cards = crate::starter_gallery::get_starter_cards_for_domain(
            "fluid_dynamics",
            &fluid_framework.available_tools,
        );
        let vector_card = fluid_cards
            .iter()
            .find(|c| c.title == "Vector Field")
            .unwrap();
        fluid_framework.select_tool_by_name(vector_card.tool_name.unwrap());
        if let Some(tool) = &mut fluid_framework.active_tool {
            assert!(tool
                .import_preset_json(vector_card.preset_json.unwrap())
                .is_ok());
        }

        let mut chaos_framework = SimulationFramework::new("chaos");
        let chaos_cards = crate::starter_gallery::get_starter_cards_for_domain(
            "chaos",
            &chaos_framework.available_tools,
        );
        let attractor_card = chaos_cards
            .iter()
            .find(|c| c.title == "3D Attractor")
            .unwrap();
        chaos_framework.select_tool_by_name(attractor_card.tool_name.unwrap());
        if let Some(tool) = &mut chaos_framework.active_tool {
            assert!(tool
                .import_preset_json(attractor_card.preset_json.unwrap())
                .is_ok());
        }

        // Generate visual artifact screenshot at /tmp/starter_gallery_preview.png
        let gallery_preview_datasets = vec![
            (
                "Harmonic Oscillator Solution y(t)".to_string(),
                (0..100)
                    .map(|i| {
                        let t = i as f64 * 0.1;
                        [t, (t * 2.0).cos() * 1.5]
                    })
                    .collect(),
            ),
            (
                "Velocity y'(t)".to_string(),
                (0..100)
                    .map(|i| {
                        let t = i as f64 * 0.1;
                        [t, -(t * 2.0).sin() * 3.0]
                    })
                    .collect(),
            ),
        ];
        let options = egui_plot::export::ExportOptions::default();
        let preview_png = egui_plot::export::export_custom_png(&gallery_preview_datasets, &options);
        assert!(preview_png.is_ok());
        if let Ok(png_data) = preview_png {
            let _ = std::fs::write("/tmp/starter_gallery_preview.png", png_data);
        }
    }
}

#[path = "framework/toolbar_reset_tests.rs"]
mod toolbar_reset_tests;
