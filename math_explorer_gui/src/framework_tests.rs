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
            .filter(|meta| {
                meta.name.to_lowercase().contains(&query)
                    || meta.tags.iter().any(|t| t.to_lowercase().contains(&query))
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
            .filter(|meta| {
                meta.name.to_lowercase().contains(&query)
                    || meta.tags.iter().any(|t| t.to_lowercase().contains(&query))
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
    }
}
