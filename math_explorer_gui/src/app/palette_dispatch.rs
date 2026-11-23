#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Unified command palette action execution and global starter gallery routing for [`MathExplorerApp`].

use super::{MathExplorerApp, ThemeMode, ViewportLayout};
use crate::command_palette::{PaletteAction, PaletteSelection};
use crate::framework::ExportFormat;
use crate::starter_gallery::{ChallengeCard, GallerySelection, StarterCard};
use eframe::egui;

impl MathExplorerApp {
    /// Opens the global cross-domain `StarterGallery` modal overlay.
    pub fn open_starter_gallery(&mut self, ctx: &egui::Context) {
        self.show_starter_gallery = true;
        crate::accessibility::announce_live_status(ctx, "Starter template gallery opened");
        ctx.request_repaint();
    }

    /// Renders the global workspace `StarterGallery` modal when opened via `View` or `CommandPalette`.
    pub(super) fn render_global_starter_gallery(&mut self, ctx: &egui::Context) {
        if !self.show_starter_gallery {
            crate::starter_gallery::filter_ui::clear_gallery_filter_state(
                ctx,
                egui::Id::new("global_workspace_starter_gallery_overlay"),
            );
            return;
        }
        let cards = crate::starter_gallery::get_all_starter_cards();
        let challenges = crate::starter_gallery::get_all_challenge_cards();
        let selected = crate::starter_gallery::show_starter_gallery_with_challenges(
            ctx,
            "global_workspace",
            &cards,
            &challenges,
            &mut self.show_starter_gallery,
        );
        if let Some(selection) = selected {
            self.show_starter_gallery = false;
            let pane = self.active_focused_pane();
            self.launch_gallery_selection_in_pane(ctx, pane, selection);
        }
    }

    /// Routes a root-hoisted per-pane `StarterGallery` selection, switching tabs if a cross-domain card or challenge was chosen.
    pub(crate) fn apply_deferred_starter_gallery_selection(
        &mut self,
        ctx: &egui::Context,
        pane_idx: usize,
        id_source: &str,
        is_open: bool,
        selected: Option<GallerySelection>,
    ) {
        let Some(selection) = selected else {
            if let Some(tab) = self.pane_tab_mut(pane_idx) {
                tab.apply_starter_gallery_selection(ctx, id_source, is_open, None);
            }
            return;
        };
        if let Some(tab) = self.pane_tab_mut(pane_idx) {
            tab.apply_starter_gallery_selection(ctx, id_source, false, None);
        }
        self.launch_gallery_selection_in_pane(ctx, pane_idx, selection);
    }

    fn launch_gallery_selection_in_pane(
        &mut self,
        ctx: &egui::Context,
        pane_idx: usize,
        selection: GallerySelection,
    ) {
        match selection {
            GallerySelection::Starter(card) => {
                self.launch_starter_card_in_pane(ctx, pane_idx, card)
            }
            GallerySelection::Challenge(ch) => {
                self.launch_challenge_card_in_pane(ctx, pane_idx, ch)
            }
        }
    }

    fn launch_challenge_card_in_pane(
        &mut self,
        ctx: &egui::Context,
        pane_idx: usize,
        ch: ChallengeCard,
    ) {
        if self.tabs.is_empty() {
            return;
        }
        let target_tab = crate::command_palette::resolve_domain_tab_index(&self.tabs, ch.domain);
        let cur_tab = if pane_idx == 0 {
            self.active_primary_tab_index()
        } else {
            self.pane_tabs.get(pane_idx).copied().unwrap_or(0)
        };
        if cur_tab != target_tab {
            self.select_pane_tab(ctx, pane_idx, target_tab);
        }
        if let Some(tab) = self.pane_tab_mut(pane_idx) {
            let slug = tab.url_slug();
            tab.apply_starter_gallery_selection(
                ctx,
                slug,
                false,
                Some(GallerySelection::Challenge(ch)),
            );
        }
        self.clear_active_pane_caches();
    }

    /// Switches `pane_idx` to the domain tab owning `card` and applies its tool and preset configuration.
    pub fn launch_starter_card_in_pane(
        &mut self,
        ctx: &egui::Context,
        pane_idx: usize,
        card: StarterCard,
    ) {
        if self.tabs.is_empty() {
            return;
        }
        let target_tab = crate::command_palette::resolve_domain_tab_index(&self.tabs, card.domain);
        let cur_tab = if pane_idx == 0 {
            self.active_primary_tab_index()
        } else {
            self.pane_tabs.get(pane_idx).copied().unwrap_or(0)
        };
        if cur_tab != target_tab {
            self.select_pane_tab(ctx, pane_idx, target_tab);
        }
        let card_title = card.title;
        let mut tab_label = "Simulation";
        if let Some(tab) = self.pane_tab_mut(pane_idx) {
            tab_label = tab.name();
            let slug = tab.url_slug();
            tab.apply_starter_gallery_outcome(ctx, slug, false, Some(card));
        }
        self.clear_active_pane_caches();
        let msg = format!("Launched starter preset {card_title} in {tab_label}");
        crate::accessibility::announce_live_status(ctx, msg);
    }

    /// Renders the global `CommandPalette` and executes any returned selection on `focused_pane`.
    pub(super) fn apply_command_palette_selection(&mut self, ctx: &egui::Context) {
        if let Some(selection) = self.command_palette.show(ctx, &self.tabs) {
            self.execute_palette_selection(ctx, selection);
        }
    }

    /// Executes a [`PaletteSelection`] (command action, starter preset, tool, or tab) on the workspace.
    pub fn execute_palette_selection(&mut self, ctx: &egui::Context, selection: PaletteSelection) {
        if let Some(action) = selection.action {
            self.execute_palette_action(ctx, action);
            return;
        }
        if let Some(preset) = selection.preset {
            let card = StarterCard {
                id: preset.id,
                title: preset.title,
                description: preset.description,
                category: preset.category,
                icon: preset.icon,
                domain: preset.domain,
                tool_name: preset.tool_name,
                preset_json: Some(preset.generate_preset()),
            };
            let pane = self.active_focused_pane();
            self.launch_starter_card_in_pane(ctx, pane, card);
            return;
        }
        if selection.tab_index >= self.tabs.len() {
            return;
        }
        self.select_tab(ctx, selection.tab_index);
        if let Some(tool_name) = &selection.tool_name {
            if let Some(tab) = self.focused_pane_tab_mut() {
                tab.select_tool_by_name(tool_name);
                let label = inventory::iter::<crate::framework::ToolMetadata>
                    .into_iter()
                    .find(|m| {
                        m.id.eq_ignore_ascii_case(tool_name)
                            || m.display_name.eq_ignore_ascii_case(tool_name)
                    })
                    .map_or(tool_name.as_str(), |m| m.display_name);
                let msg = format!("Switched to tool {} in tab {}", label, tab.name());
                crate::accessibility::announce_live_status(ctx, msg);
            }
        }
        self.clear_active_pane_caches();
    }

    fn execute_palette_action(&mut self, ctx: &egui::Context, action: PaletteAction) {
        if self.try_execute_workspace_action(ctx, &action)
            || self.try_execute_layout_action(ctx, &action)
            || self.try_execute_framework_action(ctx, &action)
        {
            ctx.request_repaint();
            return;
        }
        Self::execute_dispatched_widget_action(ctx, action);
    }

    fn try_execute_workspace_action(
        &mut self,
        ctx: &egui::Context,
        action: &PaletteAction,
    ) -> bool {
        use PaletteAction as A;
        match action {
            A::OpenWorkspaceSession => self.session_manager.trigger_open(),
            A::SaveWorkspaceSession => {
                self.clear_active_pane_caches();
                let env = self.session_manager.create_envelope(self.get_app_state());
                self.session_manager.trigger_save(env);
            }
            A::SaveWorkspaceSessionAs => {
                self.clear_active_pane_caches();
                let env = self.session_manager.create_envelope(self.get_app_state());
                self.session_manager.trigger_save_as(env);
            }
            A::CopyShareLink => self.copy_current_share_link(ctx),
            A::Undo => self.undo(),
            A::Redo => self.redo(),
            A::Quit => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            A::OpenStarterGallery => self.open_starter_gallery(ctx),
            A::ToggleTheoryPortal => {
                if let Some(fw) = self.focused_pane_tab_mut().and_then(|t| t.framework_mut()) {
                    fw.toggle_theory_portal_with_ctx(ctx);
                }
            }
            A::ToggleDiagnosticsPanel => self.show_diagnostics_panel = !self.show_diagnostics_panel,
            A::ToggleKeyboardShortcuts => self.show_help_menu = !self.show_help_menu,
            A::SetThemeDark => self.apply_palette_theme(ctx, ThemeMode::Dark),
            A::SetThemeLight => self.apply_palette_theme(ctx, ThemeMode::Light),
            A::SetThemeHighContrast => self.apply_palette_theme(ctx, ThemeMode::HighContrast),
            _ => return false,
        }
        true
    }

    fn apply_palette_theme(&mut self, ctx: &egui::Context, mode: ThemeMode) {
        self.theme = mode;
        self.theme.apply_to_context(ctx);
        crate::accessibility::announce_live_status(
            ctx,
            format!("Switched theme to {}", mode.name()),
        );
    }

    fn try_execute_layout_action(&mut self, ctx: &egui::Context, action: &PaletteAction) -> bool {
        use PaletteAction as A;
        match action {
            A::SetLayoutSingle => self.set_layout(ViewportLayout::Single),
            A::SetLayoutSplitHorizontal => self.set_layout(ViewportLayout::SplitHorizontal),
            A::SetLayoutSplitVertical => self.set_layout(ViewportLayout::SplitVertical),
            A::SetLayoutQuadGrid => self.set_layout(ViewportLayout::QuadGrid),
            A::DuplicateFocusedPane => {
                let _ = self.duplicate_focused_pane(Some(ctx));
            }
            A::SwapPanes => self.swap_panes(),
            A::RotatePanesClockwise => self.rotate_panes_clockwise(),
            A::RotatePanesCounterClockwise => self.rotate_panes_counter_clockwise(),
            A::FocusNextPane => super::viewport::focus::cycle_focused_pane(self, ctx, false),
            A::FocusPreviousPane => super::viewport::focus::cycle_focused_pane(self, ctx, true),
            A::FocusPane(idx) => super::viewport::focus::focus_pane_interactive(self, ctx, *idx),
            A::ToggleLinkPlotAxes => {
                self.set_link_plot_axes(Some(ctx), !self.compare_sync.link_axes);
            }
            A::ToggleLinkPlotCursors => {
                self.set_link_plot_cursors(Some(ctx), !self.compare_sync.link_cursors);
            }
            A::ToggleSyncPlayback => {
                self.set_sync_playback(Some(ctx), !self.compare_sync.sync_playback);
            }
            _ => return false,
        }
        true
    }

    fn try_execute_framework_action(
        &mut self,
        ctx: &egui::Context,
        action: &PaletteAction,
    ) -> bool {
        use PaletteAction as A;
        let is_fw_action = matches!(
            action,
            A::ExportCsv
                | A::CopyCsvClipboard
                | A::ExportJson
                | A::OpenExportImageModal
                | A::ExportWebGl2Shader
                | A::CopyWebGl2Shader
                | A::ExportShadertoyShader
                | A::CopyShadertoyShader
                | A::ImportPresetJson
                | A::ExportPresetJson
                | A::ResetParameters
        );
        if !is_fw_action {
            return false;
        }
        if let Some(fw) = self.focused_pane_tab_mut().and_then(|t| t.framework_mut()) {
            match action {
                A::ExportCsv => fw.queue_export(ExportFormat::Csv),
                A::CopyCsvClipboard => fw.queue_clipboard_export(),
                A::ExportJson => fw.queue_export(ExportFormat::Json),
                A::OpenExportImageModal => fw.trigger_palette_figure_export(),
                A::ExportWebGl2Shader => fw.queue_export(ExportFormat::GlslWebGl2),
                A::CopyWebGl2Shader => fw.queue_shader_clipboard_export(ExportFormat::GlslWebGl2),
                A::ExportShadertoyShader => fw.queue_export(ExportFormat::GlslShadertoy),
                A::CopyShadertoyShader => {
                    fw.queue_shader_clipboard_export(ExportFormat::GlslShadertoy);
                }
                A::ImportPresetJson => fw.preset_dialog.trigger_import(),
                A::ExportPresetJson => fw.trigger_palette_preset_export(),
                A::ResetParameters => {
                    fw.reset_parameters();
                    crate::accessibility::announce_live_status(
                        ctx,
                        "Parameters reset to baseline defaults",
                    );
                }
                _ => {}
            }
        }
        self.clear_active_pane_caches();
        true
    }

    fn execute_dispatched_widget_action(ctx: &egui::Context, action: PaletteAction) {
        use egui_plot::{
            commands::CommandRegistryData, dispatch_plot_ui_action, DispatchedPlotUiAction, FitType,
        };
        use PaletteAction as A;
        if let Some((context, name)) = dispatched_command_pair(&action) {
            CommandRegistryData::dispatch(ctx, name, context);
            return;
        }
        match action {
            A::DispatchedCommand { name, context } => {
                CommandRegistryData::dispatch(ctx, name, context);
            }
            A::PlotFitLinear => {
                dispatch_plot_ui_action(ctx, DispatchedPlotUiAction::SetFit(Some(FitType::Linear)))
            }
            A::PlotFitQuadratic => dispatch_plot_ui_action(
                ctx,
                DispatchedPlotUiAction::SetFit(Some(FitType::Quadratic)),
            ),
            A::PlotFitCubic => {
                dispatch_plot_ui_action(ctx, DispatchedPlotUiAction::SetFit(Some(FitType::Cubic)))
            }
            A::PlotFitExponential => dispatch_plot_ui_action(
                ctx,
                DispatchedPlotUiAction::SetFit(Some(FitType::Exponential)),
            ),
            A::PlotFitPowerLaw => dispatch_plot_ui_action(
                ctx,
                DispatchedPlotUiAction::SetFit(Some(FitType::PowerLaw)),
            ),
            A::PlotFitLogarithmic => dispatch_plot_ui_action(
                ctx,
                DispatchedPlotUiAction::SetFit(Some(FitType::Logarithmic)),
            ),
            A::PlotFitLogistic => dispatch_plot_ui_action(
                ctx,
                DispatchedPlotUiAction::SetFit(Some(FitType::Logistic)),
            ),
            A::PlotFitSinusoidal => dispatch_plot_ui_action(
                ctx,
                DispatchedPlotUiAction::SetFit(Some(FitType::Sinusoidal)),
            ),
            A::PlotClearFit => dispatch_plot_ui_action(ctx, DispatchedPlotUiAction::SetFit(None)),
            A::PlotToggleFitResiduals => {
                dispatch_plot_ui_action(ctx, DispatchedPlotUiAction::ToggleFitResiduals)
            }
            A::PlotToggleDataTable => {
                dispatch_plot_ui_action(ctx, DispatchedPlotUiAction::ToggleDataTable)
            }
            A::PlotOpenSettings => {
                dispatch_plot_ui_action(ctx, DispatchedPlotUiAction::OpenPlotSettings)
            }
            _ => {}
        }
    }
}

fn dispatched_command_pair(action: &PaletteAction) -> Option<(&'static str, &'static str)> {
    use PaletteAction as A;
    Some(match action {
        A::PlayPauseSimulation => ("Simulation Playback", "Play / Pause"),
        A::StepSimulationForward => ("Simulation Playback", "Step Forward"),
        A::ResetSimulationPlayback => ("Simulation Playback", "Reset Simulation"),
        A::CameraResetView => ("Camera 3D", "Reset View"),
        A::CameraToggleProjection => ("Camera 3D", "Toggle Projection"),
        A::CameraZoomIn => ("Camera 3D", "Zoom In"),
        A::CameraZoomOut => ("Camera 3D", "Zoom Out"),
        A::CameraPitchUp => ("Camera 3D", "Pitch Up"),
        A::CameraPitchDown => ("Camera 3D", "Pitch Down"),
        A::CameraYawLeft => ("Camera 3D", "Yaw Left"),
        A::CameraYawRight => ("Camera 3D", "Yaw Right"),
        A::CameraPanUp => ("Camera 3D", "Pan Up"),
        A::CameraPanDown => ("Camera 3D", "Pan Down"),
        A::CameraPanLeft => ("Camera 3D", "Pan Left"),
        A::CameraPanRight => ("Camera 3D", "Pan Right"),
        A::PlotZoomIn => ("Plot Viewport", "Zoom In"),
        A::PlotZoomOut => ("Plot Viewport", "Zoom Out"),
        A::PlotResetBounds => ("Plot Viewport", "Reset Bounds"),
        A::PlotPanLeft => ("Plot Viewport", "Pan Left"),
        A::PlotPanRight => ("Plot Viewport", "Pan Right"),
        A::PlotPanUp => ("Plot Viewport", "Pan Up"),
        A::PlotPanDown => ("Plot Viewport", "Pan Down"),
        A::PlotNextDataset => ("Plot Accessibility", "Next Dataset"),
        A::PlotPreviousDataset => ("Plot Accessibility", "Previous Dataset"),
        A::PlotNextPoint => ("Plot Accessibility", "Next Point"),
        A::PlotPreviousPoint => ("Plot Accessibility", "Previous Point"),
        A::PlotNextCriticalPoint => ("Plot Accessibility", "Next Critical Point"),
        A::PlotPreviousCriticalPoint => ("Plot Accessibility", "Previous Critical Point"),
        A::PlotIsolateSeries => ("Plot Legend", "Isolate Plot"),
        _ => return None,
    })
}
