pub use super::palette::SemanticPalette;
use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum ThemeMode {
    #[default]
    Dark,
    Light,
    HighContrast,
}

impl ThemeMode {
    pub fn name(self) -> &'static str {
        match self {
            ThemeMode::Dark => "Dark",
            ThemeMode::Light => "Light",
            ThemeMode::HighContrast => "High Contrast",
        }
    }

    /// Returns the `html[data-theme]` attribute value for this [`ThemeMode`].
    #[inline]
    #[must_use]
    pub const fn data_theme(self) -> &'static str {
        match self {
            ThemeMode::Dark => "dark",
            ThemeMode::Light => "light",
            ThemeMode::HighContrast => "high-contrast",
        }
    }

    /// Returns the CSS and `<meta name="color-scheme">` token (`"dark"` or `"light"`).
    #[inline]
    #[must_use]
    pub const fn color_scheme(self) -> &'static str {
        match self {
            ThemeMode::Dark | ThemeMode::HighContrast => "dark",
            ThemeMode::Light => "light",
        }
    }

    /// Returns the `<meta name="theme-color">` hex token (`"#121214"`, `"#f8f8f8"`, or `"#000000"`).
    #[inline]
    #[must_use]
    pub const fn theme_color_hex(self) -> &'static str {
        match self {
            ThemeMode::Dark => "#121214",
            ThemeMode::Light => "#f8f8f8",
            ThemeMode::HighContrast => "#000000",
        }
    }

    /// Returns the underlying [`egui::Theme`] slot backing this [`ThemeMode`].
    #[inline]
    #[must_use]
    pub const fn egui_theme(self) -> egui::Theme {
        match self {
            ThemeMode::Dark | ThemeMode::HighContrast => egui::Theme::Dark,
            ThemeMode::Light => egui::Theme::Light,
        }
    }

    /// Locks [`egui::Context::set_theme`] and applies [`Self::visuals`] so OS `system_theme`
    /// events cannot overwrite active canvas visuals on Frame 1 or runtime switches.
    pub fn apply_to_context(self, ctx: &egui::Context) {
        let base_theme = self.egui_theme();
        ctx.set_theme(base_theme);
        ctx.set_visuals_of(base_theme, self.visuals());
    }

    /// Parses a serialized or DOM theme token into [`ThemeMode`].
    #[must_use]
    pub fn from_token(raw: &str) -> Option<Self> {
        let trimmed = raw.trim();
        if trimmed.eq_ignore_ascii_case("dark") {
            Some(ThemeMode::Dark)
        } else if trimmed.eq_ignore_ascii_case("light") {
            Some(ThemeMode::Light)
        } else if trimmed.eq_ignore_ascii_case("highcontrast")
            || trimmed.eq_ignore_ascii_case("high-contrast")
            || trimmed.eq_ignore_ascii_case("high_contrast")
        {
            Some(ThemeMode::HighContrast)
        } else {
            None
        }
    }

    /// Infers the active [`ThemeMode`] directly from [`egui::Visuals`].
    ///
    /// Sub-panes in multi-pane split view clone the root [`egui::Style`], so resolving
    /// [`ThemeMode`] from `ui.visuals()` keeps every pane synchronized with zero state plumbing.
    #[inline]
    pub fn from_visuals(visuals: &egui::Visuals) -> Self {
        if visuals.dark_mode
            && visuals.extreme_bg_color == egui::Color32::BLACK
            && visuals.selection.bg_fill == egui::Color32::from_rgb(0, 90, 158)
        {
            ThemeMode::HighContrast
        } else if visuals.dark_mode {
            ThemeMode::Dark
        } else {
            ThemeMode::Light
        }
    }

    /// Returns the stack-allocated [`SemanticPalette`] for this [`ThemeMode`].
    #[inline]
    pub const fn palette(self) -> SemanticPalette {
        SemanticPalette::for_theme(self)
    }

    pub fn visuals(self) -> egui::Visuals {
        match self {
            ThemeMode::Dark => egui::Visuals::dark(),
            ThemeMode::Light => egui::Visuals::light(),
            ThemeMode::HighContrast => {
                let mut visuals = egui::Visuals::dark();
                visuals.extreme_bg_color = egui::Color32::BLACK;
                visuals.panel_fill = egui::Color32::BLACK;
                visuals.window_fill = egui::Color32::BLACK;
                visuals.widgets.noninteractive.bg_fill = egui::Color32::BLACK;
                visuals.widgets.noninteractive.bg_stroke =
                    egui::Stroke::new(1.0_f32, super::palette::HIGH_CONTRAST_PALETTE.graph_edge);
                visuals.window_stroke = visuals.widgets.noninteractive.bg_stroke;
                visuals.widgets.noninteractive.fg_stroke =
                    egui::Stroke::new(1.0_f32, egui::Color32::WHITE);
                visuals.widgets.inactive.fg_stroke =
                    egui::Stroke::new(1.0_f32, egui::Color32::WHITE);
                visuals.widgets.hovered.fg_stroke =
                    egui::Stroke::new(2.0_f32, egui::Color32::YELLOW);
                visuals.widgets.active.fg_stroke = egui::Stroke::new(2.0_f32, egui::Color32::WHITE);
                visuals.selection.bg_fill = egui::Color32::from_rgb(0, 90, 158);
                visuals.selection.stroke = egui::Stroke::new(2.0_f32, egui::Color32::WHITE);
                visuals
            }
        }
    }
}

/// Resolves cold-boot [`ThemeMode`] precedence from media flags and `egui` system theme.
#[inline]
#[must_use]
pub const fn resolve_startup_theme(
    forced_colors: bool,
    prefers_more_contrast: bool,
    prefers_light: bool,
    system_theme: Option<egui::Theme>,
) -> ThemeMode {
    if forced_colors || prefers_more_contrast {
        ThemeMode::HighContrast
    } else if prefers_light || matches!(system_theme, Some(egui::Theme::Light)) {
        ThemeMode::Light
    } else {
        ThemeMode::Dark
    }
}

#[cfg(target_arch = "wasm32")]
fn media_query_matches(window: &web_sys::Window, query: &str) -> bool {
    window
        .match_media(query)
        .ok()
        .flatten()
        .is_some_and(|mql| mql.matches())
}

#[cfg(target_arch = "wasm32")]
fn query_browser_media_flags() -> (bool, bool, bool) {
    if let Some(window) = web_sys::window() {
        (
            media_query_matches(&window, "(forced-colors: active)"),
            media_query_matches(&window, "(prefers-contrast: more)"),
            media_query_matches(&window, "(prefers-color-scheme: light)"),
        )
    } else {
        (false, false, false)
    }
}

#[cfg(not(target_arch = "wasm32"))]
const fn query_browser_media_flags() -> (bool, bool, bool) {
    (false, false, false)
}

/// Detects the startup [`ThemeMode`] from browser media queries (`wasm32`) and `egui` system theme.
#[must_use]
pub fn detect_startup_theme(cc: &eframe::CreationContext<'_>) -> ThemeMode {
    let (forced_colors, prefers_more_contrast, prefers_light) = query_browser_media_flags();
    resolve_startup_theme(
        forced_colors,
        prefers_more_contrast,
        prefers_light,
        cc.egui_ctx.system_theme(),
    )
}

/// Computes `(data-theme, inline color-scheme, meta color-scheme, meta theme-color)` tokens
/// while preserving system `Canvas` metadata under `(forced-colors: active)`.
#[inline]
#[must_use]
pub(crate) const fn resolve_document_theme_metadata(
    theme: ThemeMode,
    forced_colors: bool,
    prefers_light: bool,
) -> (
    &'static str,
    Option<&'static str>,
    &'static str,
    &'static str,
) {
    let data_theme = theme.data_theme();
    if forced_colors {
        let theme_color = if prefers_light {
            ThemeMode::Light.theme_color_hex()
        } else {
            ThemeMode::HighContrast.theme_color_hex()
        };
        (data_theme, None, "dark light", theme_color)
    } else {
        let scheme = theme.color_scheme();
        (data_theme, Some(scheme), scheme, theme.theme_color_hex())
    }
}

/// Synchronizes `html[data-theme]`, `color-scheme`, and `<meta name="theme-color">` on `wasm32`.
#[cfg(target_arch = "wasm32")]
pub fn sync_document_theme(theme: ThemeMode) {
    use wasm_bindgen::JsCast;

    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let (forced_colors, _prefers_more_contrast, prefers_light) = query_browser_media_flags();
    let (data_theme, inline_scheme, meta_scheme, theme_color) =
        resolve_document_theme_metadata(theme, forced_colors, prefers_light);

    if let Some(root) = document.document_element() {
        let _ = root.set_attribute("data-theme", data_theme);
        if let Ok(html_root) = root.dyn_into::<web_sys::HtmlElement>() {
            match inline_scheme {
                Some(scheme) => {
                    let _ = html_root.style().set_property("color-scheme", scheme);
                }
                None => {
                    let _ = html_root.style().remove_property("color-scheme");
                }
            }
        }
    }

    if let Ok(Some(scheme_meta)) = document.query_selector("meta[name=\"color-scheme\"]") {
        let _ = scheme_meta.set_attribute("content", meta_scheme);
    }

    if let Ok(nodes) = document.query_selector_all("meta[name=\"theme-color\"]") {
        let count = nodes.length();
        if count > 0 {
            for idx in 0..count {
                if let Some(el) = nodes
                    .item(idx)
                    .and_then(|n| n.dyn_into::<web_sys::Element>().ok())
                {
                    let _ = el.set_attribute("content", theme_color);
                }
            }
        } else if let (Ok(Some(head)), Ok(meta)) = (
            document.query_selector("head"),
            document.create_element("meta"),
        ) {
            let _ = meta.set_attribute("name", "theme-color");
            let _ = meta.set_attribute("content", theme_color);
            let _ = head.append_child(&meta);
        }
    }
}

/// Host no-op stub for [`sync_document_theme`] outside `wasm32`.
#[cfg(not(target_arch = "wasm32"))]
pub fn sync_document_theme(theme: ThemeMode) {
    let _ = resolve_document_theme_metadata(theme, false, false);
}

/// Extracts an explicit [`ThemeMode`] from valid raw `AppState` or versioned `WorkspaceEnvelope` JSON.
pub(crate) fn extract_explicit_theme_from_json(json_str: &str) -> Option<ThemeMode> {
    let cleaned = egui_plot::export::strip_utf8_bom(json_str);
    let val: serde_json::Value = serde_json::from_str(cleaned).ok()?;
    let theme_str = if val.get("schema_version").is_some() {
        val.get("app_state")?.get("theme")?.as_str()?
    } else {
        val.get("theme")?.as_str()?
    };
    let explicit = ThemeMode::from_token(theme_str)?;
    super::WorkspaceSessionManager::default()
        .parse_envelope(cleaned)
        .ok()
        .map(|_| explicit)
}

/// Initializes startup theme from browser/OS detection, restores `eframe::APP_KEY`, and syncs DOM.
pub(super) fn init_startup_theme(
    app: &mut super::MathExplorerApp,
    cc: &eframe::CreationContext<'_>,
) -> bool {
    let detected = detect_startup_theme(cc);
    app.theme = detected;
    let mut restored_from_storage = false;
    if let Some(storage) = cc.storage {
        if let Some(state_str) = storage.get_string(eframe::APP_KEY) {
            let explicit_theme = extract_explicit_theme_from_json(&state_str);
            restored_from_storage = app.try_load_state_from_json(&state_str);
            if explicit_theme.is_none() {
                app.theme = detected;
            }
        }
    }
    app.theme.apply_to_context(&cc.egui_ctx);
    sync_document_theme(app.theme);
    app.last_synced_theme = Some(app.theme);
    restored_from_storage
}

/// Synchronizes `egui` visuals, DOM theme metadata, and immediate `eframe::Storage` persistence.
pub(crate) fn sync_theme_with_storage(
    app: &mut super::MathExplorerApp,
    ctx: &egui::Context,
    storage: Option<&mut (dyn eframe::Storage + '_)>,
) {
    let target_visuals = app.theme.visuals();
    let target_egui_theme = app.theme.egui_theme();
    if ctx.theme() != target_egui_theme || ctx.style().visuals != target_visuals {
        app.theme.apply_to_context(ctx);
    }
    if app.last_synced_theme != Some(app.theme) {
        sync_document_theme(app.theme);
        app.last_synced_theme = Some(app.theme);
        if let Some(storage) = storage {
            use eframe::App as _;
            app.save(storage);
            storage.flush();
        }
        ctx.request_repaint();
    }
}

/// Per-frame theme synchronization hook invoked by `<MathExplorerApp as eframe::App>::update`.
pub(super) fn sync_frame_theme(
    app: &mut super::MathExplorerApp,
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
) {
    sync_theme_with_storage(app, ctx, frame.storage_mut());
}

impl std::fmt::Display for ThemeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[inline]
fn channel_to_linear(channel: u8) -> f64 {
    let c = f64::from(channel) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Computes WCAG 2.2 relative luminance in `[0.0, 1.0]` for an sRGB [`egui::Color32`].
#[inline]
pub fn relative_luminance(color: egui::Color32) -> f64 {
    0.2126 * channel_to_linear(color.r())
        + 0.7152 * channel_to_linear(color.g())
        + 0.0722 * channel_to_linear(color.b())
}

/// Computes the WCAG 2.2 contrast ratio `(L_lighter + 0.05) / (L_darker + 0.05)` in `[1.0, 21.0]`.
#[inline]
pub fn contrast_ratio(c1: egui::Color32, c2: egui::Color32) -> f64 {
    let l1 = relative_luminance(c1);
    let l2 = relative_luminance(c2);
    let lighter = l1.max(l2);
    let darker = l1.min(l2);
    (lighter + 0.05) / (darker + 0.05)
}

/// Selects [`egui::Color32::BLACK`] or [`egui::Color32::WHITE`] to maximize WCAG 2.2 contrast
/// against `bg` (guaranteeing at least `4.58:1` contrast for any opaque sRGB background).
#[inline]
pub fn contrasting_text_color(bg: egui::Color32) -> egui::Color32 {
    let black_ratio = contrast_ratio(bg, egui::Color32::BLACK);
    let white_ratio = contrast_ratio(bg, egui::Color32::WHITE);
    if black_ratio >= white_ratio {
        egui::Color32::BLACK
    } else {
        egui::Color32::WHITE
    }
}

#[cfg(test)]
#[path = "theme_palette_tests.rs"]
mod palette_tests;

#[cfg(test)]
#[path = "theme_tests.rs"]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "theme_wasm_tests.rs"]
mod wasm_tests;
