use super::*;

#[derive(Default)]
pub(crate) struct FrameworkRevisionCache {
    pub(crate) framework_revision: u64,
    pub(crate) untracked_tool_revision: u64,
    pub(crate) active_tool_json:
        std::cell::RefCell<Option<(&'static str, u64, std::sync::Arc<str>)>>,
    pub(crate) last_snapshot_rev: Option<(&'static str, u64)>,
    pub(crate) last_plot_capture_hash: Option<(&'static str, u64)>,
    pub(crate) last_snapshot_plot_sig: Option<u64>,
}

impl SimulationFramework {
    /// Creates a new [`SimulationFramework`] populated with registered tools for `domain`.
    pub fn new(domain: &str) -> Self {
        let mut available_tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>
            .into_iter()
            .filter(|t| t.domain == domain)
            .collect();

        available_tools.sort_by_key(|t| (t.display_name, t.id));

        let mut framework = Self {
            available_tools,
            active_tool: None,
            selected_tool_index: None,
            input_mode: InputMode::Mouse,
            show_theory_portal: false,
            show_starter_gallery: false,
            saved_tool_states: std::collections::HashMap::new(),
            saved_plot_states: std::collections::BTreeMap::new(),
            plot_sync_generation: 1,
            search_query: String::new(),
            last_announced_count: None,
            last_announced_query: String::new(),
            show_export_modal: false,
            export_options: egui_plot::export::ExportOptions::default(),
            export_bg_dark: true,
            export_gif_fps: 15,
            export_gif_frames: 20,
            export_contour_levels: 8,
            export_texture_size: 1024,
            export_normal_strength: 2.0,
            export_status_message: None,
            export_status_severity: ExportStatusSeverity::Success,
            domain: domain.to_string(),
            preset_dialog: crate::presets::PresetDialogState::default(),
            export_queue: ExportQueue::default(),
            last_valid_snapshot: None,
            baseline_tool_state: None,
            current_valid_plot_datasets: Vec::new(),
            current_valid_plot_datasets_3d: Vec::new(),
            is_diverged: false,
            diagnostic_report: None,
            step_counter: 0,
            start_time: crate::web_io::PlatformInstant::now(),
            active_challenge: None,
            challenge_was_passed: false,
            instance_id: overlay_sync::next_framework_id(),
            sync_state: FrameworkSyncState::default(),
            rev_cache: FrameworkRevisionCache::default(),
        };

        if !framework.available_tools.is_empty() {
            framework.select_tool(0);
        }

        framework
    }
}
