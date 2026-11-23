#![cfg_attr(any(), verified(opt_out = "tests"))]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui;
use egui_plot::export::{AspectRatioOption, ExportOptions, decode_gif_animation_summary};
use scientific_metadata::theory::TheoryDescribable;

use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::framework::export_figure::encode_progressive_primitive_gif;
use crate::framework::*;

struct MockGifTheory;

impl TheoryDescribable for MockGifTheory {
    fn theory_description(&self) -> String {
        "Async GIF export verification theory".to_string()
    }
    fn phonetic_description(&self) -> String {
        "Async GIF export verification theory".to_string()
    }
    fn theory_citation(&self) -> String {
        "Test citation".to_string()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

struct StatefulRasterTool {
    theory: MockGifTheory,
    step_value: usize,
    step_calls: Arc<AtomicUsize>,
    restored_value: Arc<AtomicUsize>,
}

impl InteractiveTool for StatefulRasterTool {
    fn name(&self) -> &'static str {
        "Stateful Raster Exporter"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn show(&mut self, _ctx: &egui::Context) {}

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        ui.label(format!("Step: {}", self.step_value));
    }

    fn step_simulation(&mut self) {
        self.step_value += 1;
        self.step_calls.fetch_add(1, Ordering::SeqCst);
    }

    fn save_state(&self) -> Option<String> {
        Some(self.step_value.to_string())
    }

    fn load_state(&mut self, state_json: &str) {
        if let Ok(v) = state_json.trim().parse::<usize>() {
            self.step_value = v;
            self.restored_value.store(v, Ordering::SeqCst);
        }
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::raster_only()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        let base = ((self.step_value * 37) % 200) as u8 + 30;
        let pixels = vec![egui::Color32::from_rgb(base, 90, 220); 24 * 18];
        Some(egui::ColorImage::new([24, 18], pixels))
    }
}

fn screen_raw() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    }
}

fn pump_gif_until_settled(ctx: &egui::Context, fw: &mut SimulationFramework, domain: &str) {
    for _ in 0..250 {
        if fw.active_gif_progress().is_none() {
            break;
        }
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _ = ctx.run(screen_raw(), |c| fw.show(c, domain));
    }
}

#[test]
fn test_save_gif_restores_tool_immediately_and_preserves_sinks_and_announcements() {
    clear_test_announcements();
    let step_calls = Arc::new(AtomicUsize::new(0));
    let restored = Arc::new(AtomicUsize::new(0));
    let mut fw = SimulationFramework::new("__gif_job_sink_test__");
    fw.active_tool = Some(Box::new(StatefulRasterTool {
        theory: MockGifTheory,
        step_value: 42,
        step_calls: Arc::clone(&step_calls),
        restored_value: Arc::clone(&restored),
    }));
    fw.export_options = ExportOptions {
        width: 320,
        height: 180,
        aspect_ratio: AspectRatioOption::SixteenNine,
        ..Default::default()
    };
    fw.export_gif_frames = 6;
    fw.export_gif_fps = 20;

    let temp_dir = std::env::temp_dir().join(format!("oxidize_gif_sink_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).expect("create temp sink dir");
    fw.export_queue.destination_dir = Some(temp_dir.clone());

    let ctx = egui::Context::default();
    let _ = ctx.run(screen_raw(), |c| {
        fw.show_export_modal = true;
        fw.queue_export(ExportFormat::Gif);
        fw.show(c, "gif_sink_domain");
    });

    // Sampling stepped 5 times (frames 2..=6) and restored step_value = 42 BEFORE encoding completed
    assert_eq!(step_calls.load(Ordering::SeqCst), 5);
    assert_eq!(restored.load(Ordering::SeqCst), 42);
    assert_eq!(
        fw.active_tool
            .as_ref()
            .and_then(|t| t.save_state())
            .as_deref(),
        Some("42")
    );
    assert_eq!(fw.active_gif_progress(), Some((0, 6)));
    assert!(fw.export_queue.last_exported_bytes.is_none());

    pump_gif_until_settled(&ctx, &mut fw, "gif_sink_domain");
    assert!(fw.active_gif_progress().is_none());
    assert_eq!(
        fw.export_status_message.as_deref(),
        Some("GIF animation saved successfully.")
    );
    assert_decoded_sink_gif(&fw, &temp_dir);
}

fn assert_decoded_sink_gif(fw: &SimulationFramework, temp_dir: &std::path::Path) {
    let gif_bytes = fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("GIF bytes recorded");
    let written_path = temp_dir.join("stateful_raster_exporter_animation.gif");
    assert_eq!(
        std::fs::read(&written_path).expect("saved GIF file"),
        gif_bytes
    );
    let _ = std::fs::remove_dir_all(temp_dir);

    let summary = decode_gif_animation_summary(gif_bytes).expect("valid GIF animation");
    assert_eq!((summary.width, summary.height), (320, 180));
    assert_eq!(summary.frame_count, 6);
    assert!(summary.infinite_repeat);
    assert_eq!(summary.first_delay_ms, 50);
    assert!(
        all_announcements().iter().any(|(msg, _)| msg
            .contains("GIF animation saved successfully.")
            || msg.contains("GIF"))
    );
}

#[test]
fn test_custom_writer_duplicate_guard_and_user_cancel_aborts_cleanly() {
    let step_calls = Arc::new(AtomicUsize::new(0));
    let restored = Arc::new(AtomicUsize::new(0));
    let written: Arc<Mutex<Vec<(String, usize)>>> = Arc::new(Mutex::new(Vec::new()));
    let written_clone = Arc::clone(&written);

    let mut fw = SimulationFramework::new("__gif_cancel_test__");
    fw.active_tool = Some(Box::new(StatefulRasterTool {
        theory: MockGifTheory,
        step_value: 7,
        step_calls: Arc::clone(&step_calls),
        restored_value: Arc::clone(&restored),
    }));
    fw.export_options.width = 640;
    fw.export_options.height = 360;
    fw.export_gif_frames = 15;
    fw.export_queue.custom_writer = Some(Arc::new(move |name, bytes| {
        written_clone
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((name.to_string(), bytes.len()));
    }));

    let ctx = egui::Context::default();
    let _ = ctx.run(screen_raw(), |c| {
        fw.queue_export(ExportFormat::Gif);
        fw.show(c, "gif_cancel_domain");
    });
    assert!(fw.active_gif_progress().is_some());
    assert_eq!(step_calls.load(Ordering::SeqCst), 14);

    // Duplicate queue_export while active must be ignored
    fw.queue_export(ExportFormat::Gif);
    assert!(fw.export_queue.is_empty());
    assert_eq!(step_calls.load(Ordering::SeqCst), 14);

    // Cancel active GIF export
    assert!(fw.cancel_active_gif_export(true));
    assert!(fw.active_gif_progress().is_none());
    assert_eq!(
        fw.export_status_message.as_deref(),
        Some("GIF export canceled.")
    );
    assert!(fw.export_queue.last_exported_bytes.is_none());
    assert!(written.lock().unwrap_or_else(|e| e.into_inner()).is_empty());
}

#[test]
fn test_switching_tools_aborts_active_gif_job_and_prevents_stale_banners() {
    let mut fw = SimulationFramework::new("quantum");
    assert!(fw.available_tools.len() >= 2);
    fw.select_tool(0);
    fw.export_options.width = 640;
    fw.export_options.height = 360;
    fw.export_gif_frames = 16;

    let ctx = egui::Context::default();
    let _ = ctx.run(screen_raw(), |c| {
        fw.queue_export(ExportFormat::Gif);
        fw.show(c, "quantum_switch_test");
    });
    assert!(fw.active_gif_progress().is_some());

    // Switch to another tool before encoding completes
    fw.select_tool(1);
    assert!(fw.active_gif_progress().is_none());
    assert!(fw.export_status_message.is_none());

    for _ in 0..8 {
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(std::time::Duration::from_millis(3));
        let _ = ctx.run(screen_raw(), |c| fw.show(c, "quantum_switch_test"));
    }
    assert!(fw.active_gif_progress().is_none());
    assert!(fw.export_status_message.is_none());
    assert!(fw.export_queue.last_exported_bytes.is_none());
}

#[test]
fn test_streaming_gif_single_frame_live_heap_stays_bounded() {
    let datasets = vec![(
        "Trajectory".to_string(),
        (0..200)
            .map(|i| [f64::from(i) * 0.05, (f64::from(i) * 0.1).sin()])
            .collect::<Vec<_>>(),
    )];
    let sampled_20 = egui_plot::export::sample_progressive_frames(&datasets, 20);
    let opts = ExportOptions {
        width: 1280,
        height: 720,
        ..Default::default()
    };
    // Warm up one-time static font/layout caches before measuring steady-state live heap.
    let _ = egui_plot::export::export_custom_gif(&sampled_20[..1], &opts, 15);

    let single_frame_bytes = 1280 * 720 * 4;
    let buffered_twenty_frames = single_frame_bytes * 20;

    let (gif_res, peak_live_bytes) = crate::test_alloc::measure_peak_live_heap_bytes(|| {
        egui_plot::export::export_custom_gif(&sampled_20, &opts, 15)
    });
    let gif_bytes = gif_res.expect("streamed 720p GIF");
    assert!(gif_bytes.starts_with(b"GIF89a"));
    assert!(
        peak_live_bytes < buffered_twenty_frames / 4,
        "Streaming peak live heap ({peak_live_bytes} B) must stay O(1) and well below 20-frame buffer ({buffered_twenty_frames} B)"
    );
}

#[test]
fn test_encode_progressive_primitive_gif_streams_valid_animation() {
    let primitives = vec![
        egui_plot::ExportPrimitive::Bar {
            name: "B1".to_string(),
            min: [0.0, 0.0],
            max: [0.8, 2.0],
            fill: egui::Color32::from_rgb(60, 140, 240),
            stroke_color: egui::Color32::WHITE,
            stroke_width: 1.5,
        },
        egui_plot::ExportPrimitive::Bar {
            name: "B2".to_string(),
            min: [1.0, 0.0],
            max: [1.8, 3.5],
            fill: egui::Color32::from_rgb(240, 120, 60),
            stroke_color: egui::Color32::WHITE,
            stroke_width: 1.5,
        },
    ];
    let opts = ExportOptions {
        width: 240,
        height: 180,
        aspect_ratio: AspectRatioOption::FourThree,
        ..Default::default()
    };
    let gif_bytes =
        encode_progressive_primitive_gif(&primitives, &opts, 4, 20).expect("primitive GIF");
    let summary = decode_gif_animation_summary(&gif_bytes).expect("decode primitive GIF");
    assert_eq!((summary.width, summary.height), (240, 180));
    assert_eq!(summary.frame_count, 4);
    assert!(summary.infinite_repeat);
    assert_eq!(summary.first_delay_ms, 50);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn wasm_framework_encodes_one_gif_frame_per_repaint_and_cancels_cleanly() {
    let step_calls = Arc::new(AtomicUsize::new(0));
    let restored = Arc::new(AtomicUsize::new(0));
    let mut fw = SimulationFramework::new("__wasm_gif_job_test__");
    fw.active_tool = Some(Box::new(StatefulRasterTool {
        theory: MockGifTheory,
        step_value: 19,
        step_calls: Arc::clone(&step_calls),
        restored_value: Arc::clone(&restored),
    }));
    fw.export_options = ExportOptions {
        width: 160,
        height: 120,
        aspect_ratio: AspectRatioOption::FourThree,
        ..Default::default()
    };
    fw.export_gif_frames = 4;
    fw.export_gif_fps = 10;

    let ctx = egui::Context::default();
    let _ = ctx.run(screen_raw(), |c| {
        fw.show_export_modal = true;
        fw.queue_export(ExportFormat::Gif);
        fw.show(c, "wasm_gif_domain");
    });
    assert_eq!(restored.load(Ordering::SeqCst), 19);
    assert_eq!(fw.active_gif_progress(), Some((0, 4)));

    let _ = ctx.run(screen_raw(), |c| fw.show(c, "wasm_gif_domain"));
    assert_eq!(fw.active_gif_progress(), Some((1, 4)));
    let _ = ctx.run(screen_raw(), |c| fw.show(c, "wasm_gif_domain"));
    assert_eq!(fw.active_gif_progress(), Some((2, 4)));
    assert!(fw.cancel_active_gif_export(true));
    assert_eq!(fw.active_gif_progress(), None);
}
