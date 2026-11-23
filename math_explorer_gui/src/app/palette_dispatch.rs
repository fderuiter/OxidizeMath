#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Unified command palette action execution and preset routing for [`MathExplorerApp`].

use super::{MathExplorerApp, ThemeMode, ViewportLayout};
use crate::command_palette::{PaletteAction, PaletteSelection};
use crate::framework::{ExportFormat, QuickExportAction};
use eframe::egui;

impl MathExplorerApp {
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
            let card = preset.to_card();
            let _ = self.launch_starter_card_in_focused_pane(ctx, &card);
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
            A::OpenStarterGallery => self.open_explore_catalog(ctx),
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

    fn apply_palette_layout(&mut self, ctx: &egui::Context, layout: ViewportLayout) {
        self.set_layout(layout);
        crate::accessibility::announce_live_status(
            ctx,
            format!("Switched viewport layout to {}", layout.name()),
        );
    }

    fn try_execute_layout_action(&mut self, ctx: &egui::Context, action: &PaletteAction) -> bool {
        use PaletteAction as A;
        match action {
            A::SetLayoutSingle => self.apply_palette_layout(ctx, ViewportLayout::Single),
            A::SetLayoutSplitHorizontal => {
                self.apply_palette_layout(ctx, ViewportLayout::SplitHorizontal);
            }
            A::SetLayoutSplitVertical => {
                self.apply_palette_layout(ctx, ViewportLayout::SplitVertical);
            }
            A::SetLayoutQuadGrid => self.apply_palette_layout(ctx, ViewportLayout::QuadGrid),
            A::DuplicateFocusedPane => {
                if let Some(dst) = self.duplicate_focused_pane(Some(ctx)) {
                    let labels = self.viewport_layout.pane_labels();
                    let label = labels.get(dst).copied().unwrap_or("Pane");
                    crate::accessibility::announce_live_status(
                        ctx,
                        format!("Duplicated focused pane into {label}"),
                    );
                }
            }
            A::SwapPanes => {
                self.swap_panes();
                crate::accessibility::announce_live_status(
                    ctx,
                    "Swapped primary and secondary panes",
                );
            }
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

    fn apply_quick_export_action(&mut self, ctx: &egui::Context, action: QuickExportAction) {
        let outcome = match self.focused_pane_tab_mut() {
            Some(tab) => match tab.framework_mut() {
                Some(fw) => fw.trigger_quick_export(ctx, action),
                None => Err(format!(
                    "Tab '{}' does not support tool exports",
                    tab.name()
                )),
            },
            None => Err("No active simulation pane selected".to_string()),
        };
        match outcome {
            Ok(msg) => crate::accessibility::announce_live_status(ctx, msg),
            Err(err) => {
                self.session_manager.set_toast(err.clone(), true);
                crate::accessibility::announce_live_assertive_status(ctx, &err);
            }
        }
    }

    fn try_execute_framework_action(
        &mut self,
        ctx: &egui::Context,
        action: &PaletteAction,
    ) -> bool {
        use PaletteAction as A;
        let quick = match action {
            A::ExportCsv => Some(QuickExportAction::ExportCsv),
            A::CopyCsvClipboard => Some(QuickExportAction::CopyCsv),
            A::ExportJson => Some(QuickExportAction::ExportJson),
            A::OpenExportImageModal => Some(QuickExportAction::ExportPng),
            A::ExportPresetJson => Some(QuickExportAction::ExportPresetJson),
            _ => None,
        };
        if let Some(q) = quick {
            self.apply_quick_export_action(ctx, q);
            self.clear_active_pane_caches();
            return true;
        }

        let is_other_fw = matches!(
            action,
            A::ExportWebGl2Shader
                | A::CopyWebGl2Shader
                | A::ExportShadertoyShader
                | A::CopyShadertoyShader
                | A::ImportPresetJson
                | A::ResetParameters
        );
        if !is_other_fw {
            return false;
        }
        if let Some(fw) = self.focused_pane_tab_mut().and_then(|t| t.framework_mut()) {
            match action {
                A::ExportWebGl2Shader => fw.queue_export(ExportFormat::GlslWebGl2),
                A::CopyWebGl2Shader => fw.queue_shader_clipboard_export(ExportFormat::GlslWebGl2),
                A::ExportShadertoyShader => fw.queue_export(ExportFormat::GlslShadertoy),
                A::CopyShadertoyShader => {
                    fw.queue_shader_clipboard_export(ExportFormat::GlslShadertoy);
                }
                A::ImportPresetJson => fw.preset_dialog.trigger_import(),
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
        A::StepSimulationBack => ("Simulation Playback", "Step Back"),
        A::JumpPreviousKeyframe => ("Simulation Playback", "Previous Keyframe"),
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
