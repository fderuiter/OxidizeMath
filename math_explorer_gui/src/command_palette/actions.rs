use super::scoring::fuzzy_match_score;

/// Workspace-level layout, theme, catalog, and export/share actions indexed in `CommandPalette`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkspaceAction {
    /// Open the global Starter Template Gallery across all domains.
    ExploreCatalog,
    /// Switch to Single viewport layout.
    LayoutSingle,
    /// Switch to Split Horizontal (left/right) viewport layout.
    LayoutSplitHorizontal,
    /// Switch to Split Vertical (top/bottom) viewport layout.
    LayoutSplitVertical,
    /// Switch to 2x2 Quad Grid viewport layout.
    LayoutQuadGrid,
    /// Duplicate the currently focused pane into an adjacent split pane.
    DuplicateFocusedPane,
    /// Swap primary and secondary viewport panes.
    SwapPanes,
    /// Activate Dark theme mode.
    ThemeDark,
    /// Activate Light theme mode.
    ThemeLight,
    /// Activate High Contrast theme mode.
    ThemeHighContrast,
    /// Queue CSV dataset export for the active tool in the focused pane.
    ExportCsv,
    /// Copy active dataset points as CSV to the clipboard.
    CopyCsv,
    /// Queue JSON dataset export for the active tool in the focused pane.
    ExportJson,
    /// Open high-resolution PNG/SVG/GIF figure export dialog for the active tool.
    ExportPng,
    /// Export the active tool's parameter preset JSON envelope.
    ExportPresetJson,
    /// Copy a shareable `#tab=...` simulation URL to the clipboard.
    CopyShareLink,
}

impl WorkspaceAction {
    /// Ordered catalog of all workspace actions available in `Ctrl+K` / `Cmd+K`.
    pub const ALL: &'static [WorkspaceAction] = &[
        Self::ExploreCatalog,
        Self::LayoutSingle,
        Self::LayoutSplitHorizontal,
        Self::LayoutSplitVertical,
        Self::LayoutQuadGrid,
        Self::DuplicateFocusedPane,
        Self::SwapPanes,
        Self::ThemeDark,
        Self::ThemeLight,
        Self::ThemeHighContrast,
        Self::ExportCsv,
        Self::CopyCsv,
        Self::ExportJson,
        Self::ExportPng,
        Self::ExportPresetJson,
        Self::CopyShareLink,
    ];

    /// User-facing action title displayed in `CommandPalette`.
    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Self::ExploreCatalog => "Explore Catalog (Starter Template Gallery)",
            Self::LayoutSingle => "Viewport Layout: Single",
            Self::LayoutSplitHorizontal => "Viewport Layout: Split Horizontal",
            Self::LayoutSplitVertical => "Viewport Layout: Split Vertical",
            Self::LayoutQuadGrid => "Viewport Layout: 2x2 Quad Grid",
            Self::DuplicateFocusedPane => "Viewport Layout: Duplicate Focused Pane",
            Self::SwapPanes => "Viewport Layout: Swap Panes",
            Self::ThemeDark => "Theme: Dark Mode",
            Self::ThemeLight => "Theme: Light Mode",
            Self::ThemeHighContrast => "Theme: High Contrast Mode",
            Self::ExportCsv => "Export: Save Dataset CSV",
            Self::CopyCsv => "Export: Copy Dataset CSV to Clipboard",
            Self::ExportJson => "Export: Save Dataset JSON",
            Self::ExportPng => "Export: High-Res PNG / SVG / GIF Figure...",
            Self::ExportPresetJson => "Export: Save Parameter Preset JSON",
            Self::CopyShareLink => "Share: Copy Shareable Simulation Link",
        }
    }

    /// Leading icon glyph for the action row in `CommandPalette`.
    #[must_use]
    pub const fn icon(self) -> &'static str {
        match self {
            Self::ExploreCatalog => "🧭",
            Self::LayoutSingle
            | Self::LayoutSplitHorizontal
            | Self::LayoutSplitVertical
            | Self::LayoutQuadGrid
            | Self::DuplicateFocusedPane
            | Self::SwapPanes => "🪟",
            Self::ThemeDark | Self::ThemeLight | Self::ThemeHighContrast => "🎨",
            Self::ExportCsv
            | Self::CopyCsv
            | Self::ExportJson
            | Self::ExportPng
            | Self::ExportPresetJson => "📊",
            Self::CopyShareLink => "🔗",
        }
    }

    /// Curated search keywords for matching workspace actions in `CommandPalette`.
    #[must_use]
    pub const fn keywords(self) -> &'static [&'static str] {
        match self {
            Self::ExploreCatalog => &[
                "explore catalog",
                "starter gallery",
                "starter templates",
                "browse presets",
            ],
            Self::LayoutSingle => &["single layout", "single pane", "one pane"],
            Self::LayoutSplitHorizontal => {
                &["split horizontal", "side by side", "two panes", "dual pane"]
            }
            Self::LayoutSplitVertical => &["split vertical", "stacked panes", "top bottom"],
            Self::LayoutQuadGrid => &["quad grid", "2x2 layout", "four panes"],
            Self::DuplicateFocusedPane => &["duplicate pane", "clone pane"],
            Self::SwapPanes => &["swap panes", "switch panes"],
            Self::ThemeDark => &["dark theme", "dark mode", "appearance dark"],
            Self::ThemeLight => &["light theme", "light mode", "appearance light"],
            Self::ThemeHighContrast => &["high contrast", "contrast theme", "accessible theme"],
            Self::ExportCsv => &["export csv", "download csv", "save csv"],
            Self::CopyCsv => &["copy csv", "clipboard csv"],
            Self::ExportJson => &["export json", "download json", "save json"],
            Self::ExportPng => &[
                "export png",
                "export svg",
                "export gif",
                "export figure",
                "save image",
            ],
            Self::ExportPresetJson => &["export preset", "save preset json"],
            Self::CopyShareLink => &["copy share link", "share url", "copy link"],
        }
    }
}

/// Scores a [`WorkspaceAction`] against `query`, returning `None` when unmatched.
pub(crate) fn score_workspace_action(action: WorkspaceAction, query: &str) -> Option<usize> {
    let q_trim = query.trim();
    if q_trim.is_empty() {
        return Some(0);
    }

    let mut best: Option<usize> = None;
    let mut update = |cand: Option<usize>| {
        if let Some(c) = cand {
            best = Some(best.map_or(c, |cur| cur.min(c)));
        }
    };

    update(fuzzy_match_score(action.title(), q_trim).map(|s| if s == 0 { 0 } else { s + 10 }));
    for kw in action.keywords() {
        update(fuzzy_match_score(kw, q_trim).map(|s| if s == 0 { 0 } else { s + 8 }));
    }
    best
}
