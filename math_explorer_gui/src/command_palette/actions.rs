#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Action catalog and active `CMD_REGISTRY` merging for the unified zero-prefix `CommandPalette`.

use super::layout_specs::LAYOUT_AND_EXPORT_SPECS;
use super::plot_specs::plot_and_camera_specs;
use super::scoring::fuzzy_match_score;
use eframe::egui;

/// Executable workspace, layout, theme, export, preset, playback, camera, or plot action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PaletteAction {
    OpenWorkspaceSession,
    SaveWorkspaceSession,
    SaveWorkspaceSessionAs,
    CopyShareLink,
    Undo,
    Redo,
    Quit,
    SetLayoutSingle,
    SetLayoutSplitHorizontal,
    SetLayoutSplitVertical,
    SetLayoutQuadGrid,
    DuplicateFocusedPane,
    SwapPanes,
    RotatePanesClockwise,
    RotatePanesCounterClockwise,
    FocusNextPane,
    FocusPreviousPane,
    FocusPane(usize),
    ToggleLinkPlotAxes,
    ToggleLinkPlotCursors,
    ToggleSyncPlayback,
    SetThemeDark,
    SetThemeLight,
    SetThemeHighContrast,
    OpenStarterGallery,
    ToggleTheoryPortal,
    ToggleDiagnosticsPanel,
    ToggleKeyboardShortcuts,
    ExportCsv,
    CopyCsvClipboard,
    ExportJson,
    OpenExportImageModal,
    ExportWebGl2Shader,
    CopyWebGl2Shader,
    ExportShadertoyShader,
    CopyShadertoyShader,
    ImportPresetJson,
    ExportPresetJson,
    ResetParameters,
    PlayPauseSimulation,
    StepSimulationForward,
    ResetSimulationPlayback,
    CameraResetView,
    CameraToggleProjection,
    CameraZoomIn,
    CameraZoomOut,
    CameraPitchUp,
    CameraPitchDown,
    CameraYawLeft,
    CameraYawRight,
    CameraPanUp,
    CameraPanDown,
    CameraPanLeft,
    CameraPanRight,
    PlotZoomIn,
    PlotZoomOut,
    PlotResetBounds,
    PlotPanLeft,
    PlotPanRight,
    PlotPanUp,
    PlotPanDown,
    PlotNextDataset,
    PlotPreviousDataset,
    PlotNextPoint,
    PlotPreviousPoint,
    PlotNextCriticalPoint,
    PlotPreviousCriticalPoint,
    PlotIsolateSeries,
    PlotFitLinear,
    PlotFitQuadratic,
    PlotFitCubic,
    PlotFitExponential,
    PlotFitPowerLaw,
    PlotFitLogarithmic,
    PlotFitLogistic,
    PlotFitSinusoidal,
    PlotClearFit,
    PlotToggleFitResiduals,
    PlotToggleDataTable,
    PlotOpenSettings,
    DispatchedCommand { name: String, context: String },
}

#[derive(Clone, Copy)]
pub(super) struct BuiltinCommandSpec {
    pub(super) action: fn() -> PaletteAction,
    pub(super) title: &'static str,
    pub(super) category: &'static str,
    pub(super) shortcut: Option<&'static str>,
    pub(super) description: &'static str,
    pub(super) aliases: &'static [&'static str],
    pub(super) registry_key: Option<(&'static str, &'static str)>,
}

pub(super) const fn cmd_key(mac: &'static str, other: &'static str) -> &'static str {
    if cfg!(target_os = "macos") {
        mac
    } else {
        other
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) const fn spec(
    action: fn() -> PaletteAction,
    title: &'static str,
    category: &'static str,
    shortcut: Option<&'static str>,
    description: &'static str,
    aliases: &'static [&'static str],
    registry_key: Option<(&'static str, &'static str)>,
) -> BuiltinCommandSpec {
    BuiltinCommandSpec {
        action,
        title,
        category,
        shortcut,
        description,
        aliases,
        registry_key,
    }
}

static WORKSPACE_AND_VIEW_SPECS: &[BuiltinCommandSpec] = &[
    spec(
        || PaletteAction::OpenWorkspaceSession,
        "Open Workspace Session...",
        "File",
        Some(cmd_key("Cmd+O", "Ctrl+O")),
        "Load a saved .mws or .json workspace session",
        &["open session", "load workspace", "mws"],
        Some(("File", "Open Workspace Session...")),
    ),
    spec(
        || PaletteAction::SaveWorkspaceSession,
        "Save Workspace Session",
        "File",
        Some(cmd_key("Cmd+S", "Ctrl+S")),
        "Save the current workspace session to disk",
        &["save session", "save workspace"],
        Some(("File", "Save Workspace Session")),
    ),
    spec(
        || PaletteAction::SaveWorkspaceSessionAs,
        "Save Workspace Session As...",
        "File",
        Some(cmd_key("Cmd+Shift+S", "Ctrl+Shift+S")),
        "Save the current workspace session to a new file",
        &["save as", "export workspace"],
        Some(("File", "Save Workspace Session As...")),
    ),
    spec(
        || PaletteAction::CopyShareLink,
        "Copy Share Link",
        "File",
        Some(cmd_key("Cmd+Shift+L", "Ctrl+Shift+L")),
        "Copy a shareable URL encoding the active simulation state",
        &["share url", "permalink", "copy link"],
        Some(("File", "Copy Share Link")),
    ),
    spec(
        || PaletteAction::Undo,
        "Undo",
        "Edit",
        Some(cmd_key("Cmd+Z", "Ctrl+Z")),
        "Undo the most recent workspace parameter or tab change",
        &["revert"],
        Some(("Edit", "Undo")),
    ),
    spec(
        || PaletteAction::Redo,
        "Redo",
        "Edit",
        Some(cmd_key("Cmd+Shift+Z", "Ctrl+Y")),
        "Re-apply the most recently undone workspace change",
        &["repeat"],
        Some(("Edit", "Redo")),
    ),
    spec(
        || PaletteAction::Quit,
        "Quit",
        "File",
        Some(cmd_key("Cmd+Q", "Ctrl+Q")),
        "Close the application window",
        &["exit", "close window"],
        Some(("Global", "Quit")),
    ),
    spec(
        || PaletteAction::OpenStarterGallery,
        "Open Starter Gallery",
        "View",
        None,
        "Browse and filter pre-configured starter templates across all domains",
        &["starter gallery", "starter templates", "presets gallery"],
        None,
    ),
    spec(
        || PaletteAction::ToggleTheoryPortal,
        "Toggle Theory Context Portal",
        "View",
        None,
        "Show or hide mathematical equations, derivations, and citations",
        &["theory portal", "equations", "citations"],
        None,
    ),
    spec(
        || PaletteAction::ToggleDiagnosticsPanel,
        "Toggle Issues & Diagnostics",
        "View",
        Some(cmd_key("Cmd+Shift+D", "Ctrl+Shift+D")),
        "Toggle visibility of the Issues & Diagnostics telemetry panel",
        &["diagnostics", "issues panel", "logs"],
        Some(("Global", "Toggle Issues & Diagnostics")),
    ),
    spec(
        || PaletteAction::ToggleKeyboardShortcuts,
        "Keyboard Shortcuts",
        "Help",
        Some(cmd_key("Cmd+/", "Ctrl+/")),
        "Open or close the keyboard shortcuts reference modal",
        &["help", "cheatsheet", "hotkeys"],
        Some(("Global", "Keyboard Shortcuts")),
    ),
    spec(
        || PaletteAction::SetThemeDark,
        "Theme: Dark",
        "Appearance",
        None,
        "Switch application color palette to Dark theme",
        &["dark", "dark mode", "dark theme"],
        None,
    ),
    spec(
        || PaletteAction::SetThemeLight,
        "Theme: Light",
        "Appearance",
        None,
        "Switch application color palette to Light theme",
        &["light", "light mode", "light theme"],
        None,
    ),
    spec(
        || PaletteAction::SetThemeHighContrast,
        "Theme: High Contrast",
        "Appearance",
        None,
        "Switch application color palette to WCAG AAA High Contrast theme",
        &["high contrast", "contrast theme"],
        None,
    ),
];

pub(super) fn all_builtin_commands() -> impl Iterator<Item = &'static BuiltinCommandSpec> {
    WORKSPACE_AND_VIEW_SPECS
        .iter()
        .chain(LAYOUT_AND_EXPORT_SPECS.iter())
        .chain(plot_and_camera_specs())
}

pub(super) fn score_builtin_command(spec: &BuiltinCommandSpec, query: &str) -> Option<usize> {
    if query.trim().is_empty() {
        return Some(usize::MAX);
    }
    let mut best = fuzzy_match_score(spec.title, query);
    for &alias in spec.aliases {
        if let Some(s) = fuzzy_match_score(alias, query) {
            best = Some(best.map_or(s, |cur| cur.min(s)));
        }
    }
    if let Some(sc) = spec.shortcut {
        if let Some(s) = fuzzy_match_score(sc, query) {
            let weighted = s.saturating_add(4);
            best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
        }
    }
    for extra in [spec.category, spec.description] {
        if let Some(s) = fuzzy_match_score(extra, query) {
            let weighted = s.saturating_add(20);
            best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
        }
    }
    best.map(|s| if s == 0 { 0 } else { s.saturating_add(3) })
}

pub(super) fn extra_registry_commands(
    ctx: &egui::Context,
) -> Vec<egui_plot::commands::CommandMetadata> {
    let reg = ctx.data(|d| {
        d.get_temp::<egui_plot::commands::CommandRegistryData>(egui::Id::new("CMD_REGISTRY"))
            .unwrap_or_default()
    });
    let mut out: Vec<egui_plot::commands::CommandMetadata> = Vec::new();
    for cmd in reg.commands {
        if cmd.name == "Command Palette" {
            continue;
        }
        let covered = all_builtin_commands().any(|b| {
            b.registry_key
                .is_some_and(|(cat, name)| cat == cmd.context && name == cmd.name)
        });
        let dup = out
            .iter()
            .any(|existing| existing.context == cmd.context && existing.name == cmd.name);
        if !covered && !dup {
            out.push(cmd);
        }
    }
    out
}
