pub mod data_bus;
pub mod dataflow_bus;
pub mod domain_bar;
pub(crate) mod global_shortcuts;
#[cfg(target_os = "macos")]
mod macos_menu;
#[cfg(all(test, not(target_os = "macos")))]
#[path = "macos_menu/edit.rs"]
pub(crate) mod macos_menu_edit;
#[cfg(target_os = "macos")]
mod macos_menu_view;
pub mod palette;
mod palette_dispatch;
mod pane_clone;
mod pane_ops;
mod pane_slots;
pub(crate) mod session;
mod share_link;
pub mod shared_json;
pub mod sim_bus;
mod state;
pub(crate) mod state_history;
pub mod theme;
pub(crate) mod ui_panels;
pub(crate) mod viewport;
mod visibility;
mod workspace_url;
mod workspace_url_pipeline;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_adaptive_chrome;
#[cfg(test)]
mod tests_cold_start_and_palette;
#[cfg(test)]
mod tests_command_palette;
#[cfg(all(
    test,
    feature = "biology",
    feature = "applied",
    feature = "epidemiology",
    feature = "physics",
    feature = "pure_math"
))]
mod tests_data_bus_coupling;
#[cfg(test)]
mod tests_dataflow_bus;
#[cfg(all(
    test,
    feature = "pure_math",
    feature = "epidemiology",
    feature = "biology",
    feature = "physics"
))]
mod tests_dataflow_pipelines;
#[cfg(test)]
mod tests_diagnostics;
#[cfg(test)]
mod tests_domain_bar;
#[cfg(test)]
mod tests_focus_routing;
#[cfg(test)]
mod tests_history_toolbar;
#[cfg(test)]
mod tests_multi_pane;
#[cfg(test)]
mod tests_multi_pane_a11y;
#[cfg(test)]
mod tests_multi_pane_plot_live;
#[cfg(test)]
mod tests_pane_clone_sync;
#[cfg(test)]
mod tests_pane_isolation;
#[cfg(test)]
mod tests_secondary_menus;
#[cfg(test)]
mod tests_session_lifecycle;
#[cfg(test)]
mod tests_shortcuts;
#[cfg(test)]
mod tests_sim_bus;
#[cfg(test)]
mod tests_split_view;
#[cfg(all(
    test,
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "climate",
    feature = "epidemiology",
    feature = "physics",
    feature = "pure_math"
))]
mod tests_tab_cache_perf;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;
#[cfg(all(
    test,
    target_arch = "wasm32",
    feature = "biology",
    feature = "applied",
    feature = "epidemiology",
    feature = "physics",
    feature = "pure_math"
))]
mod wasm_tests_data_bus;

#[allow(unused_imports)]
pub use data_bus::{EpidemicTraceSignal, SharedGraph, SharedNode, WorkspaceDataBus};
pub use dataflow_bus::{DataflowStreamKind, PaneDataflowBus, PanePipelineDag};
#[allow(unused_imports)]
pub use domain_bar::AcademicDomain;
#[allow(unused_imports)]
pub use session::{SessionLifecycleState, WorkspaceEnvelope, WorkspaceSessionManager};
pub use shared_json::SharedTabJson;
#[cfg(any(
    feature = "applied",
    feature = "biology",
    feature = "epidemiology",
    feature = "physics",
    feature = "pure_math"
))]
#[allow(unused_imports)]
pub use sim_bus::render_topology_source_bar;
#[cfg(feature = "biology")]
#[allow(unused_imports)]
pub use sim_bus::MAX_NEURAL_SHARED_NODES;
#[cfg(any(
    feature = "pure_math",
    all(test, any(feature = "biology", feature = "physics"))
))]
#[allow(unused_imports)]
pub use sim_bus::MAX_SHARED_GRAPH_EDGES;
#[cfg(any(
    feature = "pure_math",
    feature = "epidemiology",
    all(test, any(feature = "biology", feature = "physics"))
))]
#[allow(unused_imports)]
pub use sim_bus::MAX_SHARED_GRAPH_NODES;
#[allow(unused_imports)]
pub use sim_bus::{SharedGraphTopology, SharedPointCloud3D, SharedSimulationBus, TopologySource};
pub use state::{AppState, SplitLayoutMode};
#[allow(unused_imports)]
pub use theme::{
    contrast_ratio, contrasting_text_color, relative_luminance, SemanticPalette, ThemeMode,
};
pub use viewport::compare_sync::ComparisonSyncConfig;
pub use viewport::ViewportLayout;

use crate::command_palette::CommandPalette;
use crate::tabs::ExplorerTab;
use eframe::egui;
use federated_registry::TelemetryEvent;
#[cfg(test)]
use federated_registry::{global_registry, Severity};

pub struct MathExplorerApp {
    pub(super) tabs: Vec<Box<dyn ExplorerTab>>,
    pub selected_tab: usize,
    pub split_layout_mode: SplitLayoutMode,
    pub secondary_tab: usize,
    pub split_ratio: f32,
    diagnostic_events: Vec<TelemetryEvent>,
    pub show_diagnostics_panel: bool,
    pub(super) show_info: bool,
    pub(super) show_warnings: bool,
    pub(super) show_errors: bool,
    pub(super) show_help_menu: bool,
    pub show_starter_gallery: bool,
    pub theme: ThemeMode,
    pub(super) last_synced_theme: Option<ThemeMode>,
    pub command_palette: CommandPalette,
    pub session_manager: WorkspaceSessionManager,
    pub viewport_layout: ViewportLayout,
    pub pane_tabs: Vec<usize>,
    pub focused_pane: usize,
    pane_slots: pane_slots::SecondaryPaneSlots,
    pub is_dragging_divider: bool,
    /// Horizontal and vertical split ratios `(x, y)` for `ViewportLayout::QuadGrid`.
    pub quad_split_ratios: (f32, f32),
    pub(super) dragging_quad_axes: (bool, bool),
    tab_state_cache: state::TabStateCache,
    last_frame_had_input: bool,
    pending_focus_prune: bool,
    pub compare_sync: ComparisonSyncConfig,
    pub sim_bus: SharedSimulationBus,
    pub dataflow_bus: PaneDataflowBus,
    pub pipeline_dag: PanePipelineDag,
    pub(super) url_sync: crate::url_state::UrlSyncController,
    pub data_bus: data_bus::WorkspaceDataBus,
}

impl Default for MathExplorerApp {
    #[allow(clippy::vec_init_then_push)]
    fn default() -> Self {
        crate::accessibility::ensure_plot_live_bridge();
        crate::framework::ensure_custom_fit_bridge();
        let tabs: Vec<Box<dyn ExplorerTab>> = crate::tabs::instantiate_tabs();

        let mut app = Self {
            tabs,
            selected_tab: 0,
            split_layout_mode: SplitLayoutMode::Single,
            secondary_tab: 1,
            split_ratio: 0.5,
            diagnostic_events: Vec::new(),
            show_diagnostics_panel: false,
            show_info: true,
            show_warnings: true,
            show_errors: true,
            show_help_menu: false,
            show_starter_gallery: false,
            theme: ThemeMode::Dark,
            last_synced_theme: Some(ThemeMode::Dark),
            command_palette: CommandPalette::default(),
            session_manager: WorkspaceSessionManager::new(),
            viewport_layout: ViewportLayout::Single,
            pane_tabs: vec![0, 1, 2, 3],
            focused_pane: 0,
            pane_slots: pane_slots::SecondaryPaneSlots::default(),
            is_dragging_divider: false,
            quad_split_ratios: (0.5, 0.5),
            dragging_quad_axes: (false, false),
            tab_state_cache: state::TabStateCache::default(),
            last_frame_had_input: false,
            pending_focus_prune: false,
            compare_sync: ComparisonSyncConfig::default(),
            sim_bus: SharedSimulationBus::new(),
            dataflow_bus: PaneDataflowBus::default(),
            pipeline_dag: PanePipelineDag::default(),
            url_sync: crate::url_state::UrlSyncController::default(),
            data_bus: data_bus::WorkspaceDataBus::default(),
        };
        app.sync_viewport_visibility();
        app
    }
}

impl MathExplorerApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self::default();
        let restored = theme::init_startup_theme(&mut app, cc);
        let hydrated = app.hydrate_from_browser_url(&cc.egui_ctx, restored);
        if !restored && !hydrated && crate::starter_gallery::has_any_starter_presets() {
            app.show_starter_gallery = true;
        }

        #[cfg(target_os = "macos")]
        macos_menu::setup_macos_menu(&app);

        app
    }

    /// Extracts current application state into an `AppState` struct.
    pub fn get_app_state(&self) -> AppState {
        AppState::extract(self)
    }

    /// Restores application state from an `AppState` struct.
    pub fn apply_app_state(&mut self, app_state: AppState) {
        app_state.apply(self);
        for pane_idx in 1..self.viewport_layout.pane_count() {
            let _ = self.pane_tab_mut(pane_idx);
        }
        self.sync_viewport_visibility();
    }

    /// Serializes the entire application state (selected tab, panel toggles, individual tab states) to JSON.
    #[cfg(test)]
    pub fn save_state_to_json(&self) -> String {
        if self.session_manager.lifecycle_state == SessionLifecycleState::DragEditing {
            self.clear_active_pane_caches();
        }
        serde_json::to_string(&self.get_app_state()).unwrap_or_default()
    }

    /// Deserializes and restores the application state from JSON.
    #[cfg(test)]
    pub fn load_state_from_json(&mut self, json_str: &str) {
        let _ = self.try_load_state_from_json(json_str);
    }

    /// Selects an active tab by index, synchronizing primary pane and emitting accessibility announcements.
    pub fn select_tab(&mut self, ctx: &egui::Context, tab_idx: usize) {
        domain_bar::select_tab(self, ctx, tab_idx);
    }

    /// Assigns a simulation module to a specific viewport pane, synchronizing state and announcing the change.
    pub fn select_pane_tab(&mut self, ctx: &egui::Context, pane_idx: usize, tab_idx: usize) {
        domain_bar::select_pane_tab(self, ctx, pane_idx, tab_idx);
    }

    /// Restores the previous application state from the workspace undo history stack.
    pub fn undo(&mut self) {
        let mut sm = std::mem::take(&mut self.session_manager);
        sm.execute_undo(self);
        self.session_manager = sm;
    }

    /// Re-applies an undone application state from the workspace redo history stack.
    pub fn redo(&mut self) {
        let mut sm = std::mem::take(&mut self.session_manager);
        sm.execute_redo(self);
        self.session_manager = sm;
    }
    fn begin_update_frame(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        crate::reflective_ui::slider::reset_frame_sweep_flags(ctx);
        ctx.data_mut(|d| {
            d.insert_temp(
                egui::Id::new("CMD_REGISTRY"),
                egui_plot::commands::CommandRegistryData::default(),
            );
        });

        theme::sync_frame_theme(self, ctx, frame);
        crate::accessibility::ensure_plot_live_bridge();
        crate::accessibility::sync_context_live_messages(ctx, true);
        if ctx.input(|i| {
            i.events
                .iter()
                .any(|e| matches!(e, egui::Event::WindowFocused(false)))
        }) {
            crate::audio::notify_window_blurred();
        }

        self.flush_stale_pane_focus(ctx);
        let mut session_mgr = std::mem::take(&mut self.session_manager);
        session_mgr.register_and_check_shortcuts(ctx, self);
        session_mgr.poll(self, ctx);
        self.session_manager = session_mgr;
        theme::sync_frame_theme(self, ctx, frame);
    }

    fn render_top_chrome_and_viewports(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        #[cfg(target_os = "macos")]
        {
            macos_menu::sync_macos_edit_menu(self);
            macos_menu::handle_macos_menu_events(self, ctx);
            if self.session_manager.toast().is_some() {
                egui::TopBottomPanel::top("toast_panel").show(ctx, |ui| {
                    self.session_manager.render_toast(ui);
                });
            }
        }

        #[cfg(not(target_os = "macos"))]
        egui::TopBottomPanel::top("main_menu").show(ctx, |ui| {
            ui_panels::render_main_menu(self, ctx, ui);
        });

        egui::TopBottomPanel::top("domain_switcher_bar")
            .exact_height(34.0)
            .show(ctx, |ui| {
                domain_bar::render_domain_switcher_bar(self, ctx, ui);
            });

        let active_layout = self.resolve_active_viewport_layout();
        self.prune_inactive_pane_slots(active_layout.pane_count());
        domain_bar::poll_cross_tab_tool_handoffs(self, ctx);
        self.sync_viewport_visibility();
        self.sync_workspace_history_bridge(ctx);
        data_bus::sync_app_bus_before_viewport(self, ctx);

        if active_layout != ViewportLayout::Single {
            viewport::render_multi_pane_manager(self, ctx, frame);
            self.prune_incompatible_pipeline_links();
        } else {
            viewport::render_single_pane_viewport(self, ctx, frame);
        }
        data_bus::commit_app_bus_after_viewport(self, ctx);
        self.drain_workspace_history_bridge(ctx);
    }
}

impl eframe::App for MathExplorerApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.begin_update_frame(ctx, frame);
        self.register_and_handle_global_shortcuts(ctx);
        self.poll_and_render_diagnostics(ctx);
        self.render_top_chrome_and_viewports(ctx, frame);

        ui_panels::render_help_window(ctx, &mut self.show_help_menu);
        self.render_global_starter_gallery(ctx);
        self.apply_command_palette_selection(ctx);
        self.sync_viewport_visibility();
        crate::accessibility::sync_context_live_messages(ctx, false);

        self.last_frame_had_input =
            state::has_frame_input(ctx) || crate::reflective_ui::slider::has_sweep_mutation(ctx);
        let mut session_manager = std::mem::take(&mut self.session_manager);
        session_manager.track_frame_changes(ctx, self);
        self.session_manager = session_manager;
        #[cfg(target_os = "macos")]
        macos_menu::sync_macos_edit_menu(self);
        crate::framework::record_focused_widget_type_from_output(ctx);
        theme::sync_frame_theme(self, ctx, frame);
        self.tick_url_sync(ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        self.save_envelope_to_storage(storage);
    }

    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(5)
    }
    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}
