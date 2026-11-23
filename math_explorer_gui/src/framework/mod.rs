pub mod atlas_host;
pub mod brush_codec;
pub mod brush_dispatch;
#[cfg(all(test, feature = "physics", feature = "biology"))]
mod brush_dispatch_tests;
pub mod callout_overlay;
pub mod challenge;
mod challenge_hud;
#[cfg(test)]
mod challenge_hud_tests;
pub mod clamp;
#[cfg(test)]
mod clamp_tests;
pub mod code_export;
pub mod code_export_modal;
#[cfg(all(
    test,
    feature = "pure_math",
    feature = "physics",
    feature = "applied",
    feature = "epidemiology"
))]
mod code_export_tests;
pub mod counterfactual;
pub mod decimate;
#[cfg(all(test, feature = "ai", feature = "physics", feature = "pure_math"))]
mod drag_bounds_and_rollback_tests;
mod export_3d;
mod export_audio;
mod export_dataset;
mod export_figure;
mod export_gif_job;
mod export_modal_host;
mod export_persistence;
mod export_queue;
#[cfg(test)]
pub(crate) use export_queue::deferred_export_tests;
#[cfg(test)]
mod export_queue_tests;
pub(crate) mod export_shader;
#[cfg(test)]
mod export_shader_tests;
pub mod export_types;
mod export_ui;
#[cfg(test)]
mod export_ui_tests;
pub mod fit_bridge;
mod history_ui;
pub mod math_input;
#[cfg(test)]
mod metadata_label_tests;
mod overlay_sync;
mod pane_scope;
#[cfg(test)]
mod perf_guard_tests;
mod persistence;
pub mod playback;
#[cfg(test)]
mod playback_tests;
#[cfg(test)]
mod playback_time_travel_tests;
pub mod plot_persistence;
#[cfg(test)]
mod plot_persistence_session_tests;
#[cfg(test)]
mod plot_persistence_tests;
pub mod recovery;
#[cfg(test)]
mod recovery_tests;
#[cfg(test)]
mod shared_points_perf_tests;
pub mod side_panel;
pub mod spatial_brush;
#[cfg(all(test, feature = "physics", feature = "biology"))]
mod spatial_brush_integration_tests;
#[cfg(test)]
mod spatial_brush_tests;
pub mod state_bounds;
#[cfg(test)]
mod state_bounds_tests;
pub mod state_guard;
#[cfg(test)]
mod state_guard_tests;
pub mod story_hud;
pub(crate) mod theory_links;
#[cfg(test)]
mod theory_links_tests;
mod theory_portal;
mod tool;
pub mod trace_overlay;
mod ui;
mod url_share;

pub use brush_codec::*;
pub use brush_dispatch::{PlotBrushController, PlotBrushFrame};
pub use callout_overlay::*;
pub use challenge::ChallengeEvaluation;
pub use clamp::*;
pub use code_export::*;
pub use code_export_modal::CodeExportTab;
pub use counterfactual::*;
pub use decimate::*;
pub use egui_plot::{ModalFrameContext, ModalLifecycle};
pub(crate) use export_queue::*;
pub use export_types::*;
pub use fit_bridge::ensure_custom_fit_bridge;
pub use history_ui::render_workspace_history_buttons;
pub use math_input::*;
#[cfg(any(feature = "pure_math", feature = "applied"))]
pub(crate) use overlay_sync::record_theory_portal_toggle;
pub use pane_scope::{
    is_button_activation_focused, is_pane_activation_shortcut_allowed, is_pane_shortcut_allowed,
    is_text_edit_focused, record_focused_widget_type_from_output, scoped_panel_id, InputMode,
    InteractionContext,
};
pub(crate) use persistence::framework_json_semantically_eq;
pub use playback::*;
pub use recovery::*;
pub use side_panel::{
    clear_pane_side_panel_records, pane_side_panel_records, PaneSidePanel, PaneSidePanelRecord,
    MAX_PANE_SIDE_PANEL_WIDTH_RATIO,
};
pub use spatial_brush::*;
pub use state_bounds::*;
pub use state_guard::*;
pub use story_hud::*;
pub use tool::InteractiveTool;
pub use trace_overlay::*;

use eframe::egui;

pub(super) use crate::accessibility::AccessibleHoverText;
pub(super) use overlay_sync::{
    FrameworkRevisionCache, FrameworkSyncState, UntrackedUrlDigestCache,
};

pub struct ToolMetadata {
    pub id: &'static str,
    pub display_name: &'static str,
    pub domain: &'static str,
    pub tags: &'static [&'static str],
    pub build: fn() -> Box<dyn InteractiveTool>,
}

inventory::collect!(ToolMetadata);
pub use crate::starter_gallery::{ChallengeMetadata, StarterPresetMetadata};

#[allow(missing_docs)]
pub struct SimulationFramework {
    pub available_tools: Vec<&'static ToolMetadata>,
    pub active_tool: Option<Box<dyn InteractiveTool>>,
    pub selected_tool_index: Option<usize>,
    pub input_mode: InputMode,
    pub show_theory_portal: bool,
    pub show_starter_gallery: bool,
    pub saved_tool_states: std::collections::HashMap<String, String>,
    pub saved_plot_states: plot_persistence::SavedPlotStatesMap,
    pub(super) plot_sync_generation: u64,
    pub search_query: String,
    pub(super) last_announced_count: Option<usize>,
    pub(super) last_announced_query: String,
    pub show_export_modal: bool,
    pub show_code_export_modal: bool,
    pub code_export_tab: CodeExportTab,
    pub export_options: egui_plot::export::ExportOptions,
    pub export_bg_dark: bool,
    pub export_gif_fps: u32,
    pub export_gif_frames: u32,
    pub export_status_message: Option<String>,
    pub export_status_severity: ExportStatusSeverity,
    pub domain: String,
    pub preset_dialog: crate::presets::PresetDialogState,
    pub(crate) export_queue: ExportQueue,
    gif_job: Option<export_gif_job::FrameworkGifJob>,
    /// Cached state snapshot before divergence for step-back rollback.
    pub last_valid_snapshot: Option<StateSnapshot>,
    /// Baseline parameters snapshot on tool load.
    pub baseline_tool_state: Option<String>,
    /// Last validated finite plot datasets.
    pub current_valid_plot_datasets: Vec<(String, egui_plot::SharedPoints)>,
    /// Last validated finite 3D world-coordinate plot datasets.
    pub current_valid_plot_datasets_3d: Vec<(String, Vec<[f64; 3]>)>,
    /// Whether divergence has been detected in the active tool.
    pub is_diverged: bool,
    /// Diagnostic report for divergence.
    pub diagnostic_report: Option<DiagnosticReport>,
    /// Simulation frame / step counter.
    pub step_counter: usize,
    /// Counterfactual shadow twin engine state synchronized with the active tool.
    pub branch_engine: CounterfactualBranchEngine,
    /// Start time of current active session.
    pub start_time: crate::web_io::PlatformInstant,
    /// Declarative guided story and AST challenge HUD state.
    pub challenge_hud: ChallengeHudState,
    /// Active inverse challenge contract displayed in the Challenge HUD, if any.
    pub active_challenge: Option<&'static ChallengeMetadata>,
    pub(super) challenge_was_passed: bool,
    pub(super) instance_id: u64,
    pub(super) sync_state: FrameworkSyncState,
    pub(super) rev_cache: FrameworkRevisionCache,
    pub atlas_host: atlas_host::FrameworkAtlasHost,
}

pub use crate::camera::{
    Camera3D, Plot3DViewport, ProjectionMode, Scene3DBuilder, CAMERA_3D_HELPER_TEXT,
    CAMERA_3D_MAX_PAN, CAMERA_3D_PERSPECTIVE_DISTANCE, CAMERA_3D_ZOOM_MAX, CAMERA_3D_ZOOM_MIN,
};
