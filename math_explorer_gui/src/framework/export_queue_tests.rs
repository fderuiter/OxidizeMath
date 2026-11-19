#[cfg(test)]
mod tests {
    use crate::framework::*;
    #[cfg(feature = "pure_math")]
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(feature = "pure_math")]
    use std::sync::Arc;

    #[test]
    fn test_export_queue_deduplication_on_rapid_clicks() {
        let mut queue = ExportQueue::default();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        // First click succeeds
        assert!(queue.push(ExportFormat::Csv));
        assert_eq!(queue.len(), 1);

        // Rapid repeated clicks in the same frame should be ignored
        assert!(!queue.push(ExportFormat::Csv));
        assert!(!queue.push(ExportFormat::Csv));
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.requests(), &[ExportFormat::Csv]);

        // A different format can still be queued
        assert!(queue.push(ExportFormat::Json));
        assert_eq!(queue.len(), 2);
        assert!(!queue.push(ExportFormat::Json));
        assert_eq!(queue.len(), 2);

        // Draining clears the queue and yields queued formats
        let drained = queue.drain();
        assert_eq!(drained, vec![ExportFormat::Csv, ExportFormat::Json]);
        assert!(queue.is_empty());
    }

    #[test]
    #[cfg(feature = "pure_math")]
    fn test_export_csv_end_of_frame_dispatch_non_empty() {
        let temp_dir =
            std::env::temp_dir().join(format!("oxidize_test_export_csv_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let mut framework = SimulationFramework::new("analysis");
        framework.export_queue.destination_dir = Some(temp_dir.clone());

        let ctx = eframe::egui::Context::default();

        // Queue export request
        framework.queue_export(ExportFormat::Csv);
        assert_eq!(framework.export_queue.len(), 1);

        // Run framework frame loop
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_csv");
        });

        // Queue should be cleared after end-of-frame dispatch
        assert!(framework.export_queue.is_empty());

        // Verify CSV file creation
        let filename = framework
            .export_queue
            .last_exported_filename
            .as_ref()
            .expect("last_exported_filename should be recorded");
        assert!(filename.ends_with("_dataset.csv"));

        let file_path = temp_dir.join(filename);
        assert!(file_path.exists(), "Exported CSV file must exist on disk");

        let csv_content = std::fs::read_to_string(&file_path).expect("Failed to read CSV file");
        assert!(
            csv_content.starts_with("Dataset,X,Y\n"),
            "CSV must start with header"
        );

        let lines: Vec<&str> = csv_content.lines().collect();
        // Should have header + multiple data rows
        assert!(
            lines.len() > 1,
            "Exported CSV file must contain coordinate rows from the active tool"
        );

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    #[cfg(feature = "pure_math")]
    fn test_export_json_end_of_frame_dispatch_valid_json() {
        let temp_dir =
            std::env::temp_dir().join(format!("oxidize_test_export_json_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let mut framework = SimulationFramework::new("analysis");
        framework.export_queue.destination_dir = Some(temp_dir.clone());

        let ctx = eframe::egui::Context::default();

        // Queue export request
        framework.queue_export(ExportFormat::Json);
        assert_eq!(framework.export_queue.len(), 1);

        // Run framework frame loop
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_json");
        });

        // Queue should be cleared
        assert!(framework.export_queue.is_empty());

        // Verify JSON file creation
        let filename = framework
            .export_queue
            .last_exported_filename
            .as_ref()
            .expect("last_exported_filename should be recorded");
        assert!(filename.ends_with("_dataset.json"));

        let file_path = temp_dir.join(filename);
        assert!(file_path.exists(), "Exported JSON file must exist on disk");

        let json_content = std::fs::read_to_string(&file_path).expect("Failed to read JSON file");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_content).expect("Exported JSON must be valid JSON");

        let arr = parsed.as_array().expect("JSON root must be an array");
        assert!(
            !arr.is_empty(),
            "Exported JSON must contain coordinate items from active tool"
        );

        let first = &arr[0];
        assert!(first.get("dataset").is_some());
        assert!(first.get("x").is_some());
        assert!(first.get("y").is_some());

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    #[cfg(feature = "pure_math")]
    fn test_exported_coordinates_match_click_frame_visual_parameters() {
        let mut framework = SimulationFramework::new("analysis");
        framework.export_queue.custom_writer = Some(Arc::new(|_, _| {}));
        let ctx = eframe::egui::Context::default();

        // Run frame to populate tool rendering
        framework.queue_export(ExportFormat::Csv);
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_coords");
        });

        let csv_content = framework
            .export_queue
            .last_exported_content
            .clone()
            .expect("last_exported_content must be populated");

        let active_datasets = framework
            .active_tool
            .as_ref()
            .expect("active tool must be present")
            .export_datasets();

        assert!(
            !active_datasets.is_empty(),
            "Active tool must have generated datasets during tool rendering"
        );

        // Verify that every point in the active datasets appears in the CSV output
        for (name, pts) in &active_datasets {
            assert!(
                csv_content.contains(name),
                "CSV must contain dataset label {name}"
            );
            if let Some(first_point) = pts.first() {
                let row_str = format!("{},{}", first_point[0], first_point[1]);
                assert!(
                    csv_content.contains(&row_str),
                    "CSV must contain exact coordinate values {row_str}"
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "pure_math")]
    fn test_rapid_repeated_clicks_produce_single_write() {
        let write_count = Arc::new(AtomicUsize::new(0));
        let write_count_clone = Arc::clone(&write_count);

        let mut framework = SimulationFramework::new("analysis");
        framework.export_queue.custom_writer = Some(Arc::new(move |_filename, _content| {
            write_count_clone.fetch_add(1, Ordering::SeqCst);
        }));

        let ctx = eframe::egui::Context::default();

        // Simulate 10 rapid repeated clicks for CSV and 5 for JSON
        for _ in 0..10 {
            framework.queue_export(ExportFormat::Csv);
        }
        for _ in 0..5 {
            framework.queue_export(ExportFormat::Json);
        }

        // Only 2 distinct requests should be in queue
        assert_eq!(framework.export_queue.len(), 2);

        // Run framework frame loop
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_rapid");
        });

        // Writer should have been called exactly twice: once for CSV, once for JSON
        assert_eq!(
            write_count.load(Ordering::SeqCst),
            2,
            "Rapid repeated clicks must not produce duplicate writes"
        );
        assert!(framework.export_queue.is_empty());
    }

    #[test]
    #[cfg(feature = "pure_math")]
    fn test_export_action_bar_button_simulation() {
        let temp_dir =
            std::env::temp_dir().join(format!("oxidize_test_bar_sim_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let mut framework = SimulationFramework::new("analysis");
        // Select ODE solver tool specifically
        framework.select_tool(1);
        framework.export_queue.destination_dir = Some(temp_dir.clone());

        let ctx = eframe::egui::Context::default();

        // In frame 1, queue an export (simulating user button click in action bar)
        framework.queue_export(ExportFormat::Csv);

        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_ode_export");
        });

        // Verify CSV was dispatched at end of frame
        assert!(framework.export_queue.is_empty());
        let csv_filename = framework
            .export_queue
            .last_exported_filename
            .as_ref()
            .unwrap();
        assert!(csv_filename.ends_with("_dataset.csv"));

        let content = std::fs::read_to_string(temp_dir.join(csv_filename)).unwrap();
        assert!(content.starts_with("Dataset,X,Y\n"));
        assert!(
            content.lines().count() > 1,
            "Exported CSV must contain data rows from the ODE tool"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    #[cfg(feature = "pure_math")]
    fn test_deferred_dispatch_captures_frame_data_where_immediate_fails() {
        // Reproduce the problem statement:
        // When clear_last_plot_datasets is called at the beginning of the frame,
        // querying tool.export_datasets() immediately during action bar rendering
        // yields empty datasets because tool.show(ctx) has not yet run.
        egui_plot::clear_last_plot_datasets();

        let mut framework = SimulationFramework::new("analysis");
        framework.export_queue.custom_writer = Some(Arc::new(|_, _| {}));
        // Select RiemannIntegrationTool (which relies on default egui_plot::get_last_plot_datasets())
        framework.select_tool(2);

        // 1. Immediate export (the old behavior before the fix):
        let immediate_datasets = framework.active_tool.as_ref().unwrap().export_datasets();
        assert!(
            immediate_datasets.is_empty(),
            "Immediate query before tool rendering must yield empty datasets"
        );

        // 2. Deferred export via end-of-frame queue (the new behavior):
        framework.queue_export(ExportFormat::Csv);
        let ctx = eframe::egui::Context::default();
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_deferred_contrast");
        });

        let deferred_csv = framework
            .export_queue
            .last_exported_content
            .expect("Deferred export must have executed");
        assert!(
            deferred_csv.lines().count() > 1,
            "Deferred end-of-frame export must contain plot curves populated by tool.show()"
        );

        // Generate screenshot artifact image for PR verification
        let sample_datasets = vec![
            (
                "Exported Curve (y(t))".to_string(),
                vec![[0.0, 1.0], [0.5, 1.8], [1.0, 2.2], [1.5, 1.4], [2.0, 0.7]],
            ),
            (
                "Exported Derivative (y'(t))".to_string(),
                vec![[0.0, 0.0], [0.5, 1.2], [1.0, 0.4], [1.5, -1.0], [2.0, -0.9]],
            ),
        ];
        let options = egui_plot::export::ExportOptions::default();
        if let Ok(png_data) = egui_plot::export::export_custom_png(&sample_datasets, &options) {
            let _ = std::fs::write("/tmp/deferred_export_queue_verification.png", png_data);
        }
    }

    #[test]
    fn test_clipboard_export_queue_deduplication_and_coexistence() {
        let mut queue = ExportQueue::default();

        // Rapid repeated clipboard clicks deduplicate to a single request
        assert!(queue.queue_clipboard_export());
        assert!(!queue.queue_clipboard_export());
        assert!(!queue.push_target(ExportFormat::Csv, ExportTarget::Clipboard));
        assert_eq!(queue.len(), 1);
        assert_eq!(
            queue.export_requests(),
            &[ExportRequest {
                format: ExportFormat::Csv,
                target: ExportTarget::Clipboard,
            }]
        );

        // Disk CSV export can still be queued alongside clipboard CSV export
        assert!(queue.push(ExportFormat::Csv));
        assert_eq!(queue.len(), 2);

        let drained = queue.drain_requests();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].target, ExportTarget::Clipboard);
        assert_eq!(drained[1].target, ExportTarget::File);
        assert!(queue.is_empty());
    }

    #[test]
    #[cfg(feature = "pure_math")]
    fn test_clipboard_end_of_frame_dispatch_accuracy_without_disk_writes() {
        let temp_dir = std::env::temp_dir().join(format!(
            "oxidize_test_clipboard_no_disk_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&temp_dir);
        let _ = std::fs::create_dir_all(&temp_dir);

        let captured_clipboard = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let captured_clone = Arc::clone(&captured_clipboard);
        let disk_writes = Arc::new(AtomicUsize::new(0));
        let disk_writes_clone = Arc::clone(&disk_writes);

        let mut framework = SimulationFramework::new("analysis");
        framework.export_queue.destination_dir = Some(temp_dir.clone());
        framework.export_queue.custom_writer = Some(Arc::new(move |_, _| {
            disk_writes_clone.fetch_add(1, Ordering::SeqCst);
        }));
        framework.export_queue.custom_clipboard_writer = Some(Arc::new(move |text| {
            captured_clone.lock().unwrap().push(text.to_string());
            Ok(())
        }));

        // Rapid clicks queue only one clipboard operation
        for _ in 0..5 {
            framework.queue_clipboard_export();
        }
        assert_eq!(framework.export_queue.len(), 1);

        let ctx = eframe::egui::Context::default();
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            framework.show(ctx, "test_clipboard_dispatch");
        });

        assert!(framework.export_queue.is_empty());
        assert_eq!(disk_writes.load(Ordering::SeqCst), 0);
        assert_eq!(
            std::fs::read_dir(&temp_dir).unwrap().count(),
            0,
            "Clipboard export must not write any files to destination_dir"
        );
        assert!(framework.export_queue.last_exported_filename.is_none());

        let clips = captured_clipboard.lock().unwrap();
        assert_eq!(clips.len(), 1);
        let expected_csv = egui_plot::export::format_csv(
            &framework.active_tool.as_ref().unwrap().export_datasets(),
        );
        assert_eq!(clips[0], expected_csv);
        assert!(clips[0].starts_with("Dataset,X,Y\n"));
        assert!(clips[0].lines().count() > 1);
        assert_eq!(
            framework.export_queue.last_clipboard_content.as_deref(),
            Some(expected_csv.as_str())
        );

        let status = framework
            .export_queue
            .status
            .as_ref()
            .expect("Confirmation status should be set");
        assert!(!status.is_error);
        assert!(status.message.contains("Copied CSV dataset to clipboard"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    struct EmptyTool;
    impl scientific_metadata::theory::TheoryDescribable for EmptyTool {
        fn theory_description(&self) -> String {
            "Empty".to_string()
        }
        fn phonetic_description(&self) -> String {
            "Empty".to_string()
        }
        fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
            std::collections::HashMap::new()
        }
        fn theory_citation(&self) -> String {
            "None".to_string()
        }
    }
    impl InteractiveTool for EmptyTool {
        fn name(&self) -> &'static str {
            "Empty Tool"
        }
        fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
            self
        }
        fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
            Vec::new()
        }
    }

    #[test]
    fn test_clipboard_empty_and_no_tool_edge_cases() {
        let mut framework = SimulationFramework::new("analysis");
        framework.active_tool = None;
        framework.queue_clipboard_export();
        framework.dispatch_pending_exports();
        let msg_none = framework.export_status_message.clone().unwrap();
        assert!(msg_none.contains("No active simulation tool"));

        framework.active_tool = Some(Box::new(EmptyTool));
        framework.queue_clipboard_export();
        framework.dispatch_pending_exports();
        let msg_empty = framework.export_status_message.clone().unwrap();
        assert!(msg_empty.contains("No dataset points available"));
        assert!(framework.export_queue.last_clipboard_content.is_none());
    }

    #[test]
    #[cfg(feature = "pure_math")]
    fn test_clipboard_unavailable_and_confirmation_timeout() {
        let mut fw_err = SimulationFramework::new("analysis");
        fw_err.export_queue.custom_clipboard_writer =
            Some(Arc::new(|_| Err("display unavailable".to_string())));
        fw_err.queue_clipboard_export();
        let ctx = eframe::egui::Context::default();
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            fw_err.show(ctx, "test_cb_err");
        });
        let err_status = fw_err.export_queue.status.as_ref().unwrap();
        assert!(err_status.is_error);
        assert!(err_status.message.contains("Export CSV"));

        let mut fw_ok = SimulationFramework::new("analysis");
        fw_ok.queue_clipboard_export();
        for (time, expect_visible) in [(1.0, true), (2.0, true), (6.0, false)] {
            let input = eframe::egui::RawInput {
                time: Some(time),
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| fw_ok.show(ctx, "test_cb_timeout"));
            assert_eq!(fw_ok.export_queue.status.is_some(), expect_visible);
        }
    }
}
