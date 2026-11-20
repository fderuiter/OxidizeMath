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
                visuals.widgets.noninteractive.fg_stroke =
                    egui::Stroke::new(1.0_f32, egui::Color32::WHITE);
                visuals.widgets.inactive.fg_stroke =
                    egui::Stroke::new(1.0_f32, egui::Color32::WHITE);
                visuals.widgets.hovered.fg_stroke =
                    egui::Stroke::new(2.0_f32, egui::Color32::YELLOW);
                visuals.widgets.active.fg_stroke = egui::Stroke::new(2.0_f32, egui::Color32::WHITE);
                visuals.selection.bg_fill = egui::Color32::from_rgb(0, 90, 158);
                visuals.selection.stroke = egui::Stroke::new(1.0_f32, egui::Color32::BLACK);
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
// theory_verification!
fn media_query_matches(window: &web_sys::Window, query: &str) -> bool {
    window
        .match_media(query)
        .ok()
        .flatten()
        .is_some_and(|mql| mql.matches())
}

#[cfg(target_arch = "wasm32")]
// theory_verification!
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

/// Synchronizes `html[data-theme]`, `color-scheme`, and `<meta name="theme-color">` on `wasm32`.
#[cfg(target_arch = "wasm32")]
// theory_verification!
pub fn sync_document_theme(theme: ThemeMode) {
    use wasm_bindgen::JsCast;

    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let data_theme = theme.data_theme();
    let color_scheme = theme.color_scheme();
    let theme_color = theme.theme_color_hex();

    if let Some(root) = document.document_element() {
        let _ = root.set_attribute("data-theme", data_theme);
        if let Ok(html_root) = root.dyn_into::<web_sys::HtmlElement>() {
            let _ = html_root.style().set_property("color-scheme", color_scheme);
        }
    }

    if let Ok(Some(scheme_meta)) = document.query_selector("meta[name=\"color-scheme\"]") {
        let _ = scheme_meta.set_attribute("content", color_scheme);
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
    let _ = (
        theme.data_theme(),
        theme.color_scheme(),
        theme.theme_color_hex(),
    );
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
) {
    let detected = detect_startup_theme(cc);
    app.theme = detected;
    if let Some(storage) = cc.storage {
        if let Some(state_str) = storage.get_string(eframe::APP_KEY) {
            let explicit_theme = extract_explicit_theme_from_json(&state_str);
            app.load_state_from_json(&state_str);
            if explicit_theme.is_none() {
                app.theme = detected;
            }
        }
    }
    app.theme.apply_to_context(&cc.egui_ctx);
    sync_document_theme(app.theme);
    app.last_synced_theme = Some(app.theme);
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
mod tests {
    use super::*;
    use crate::app::MathExplorerApp;
    use eframe::{App, Storage};
    use std::collections::HashMap;

    #[derive(Default)]
    pub(super) struct TestStorage {
        entries: HashMap<String, String>,
        pub(super) flushes: usize,
    }

    impl Storage for TestStorage {
        fn get_string(&self, key: &str) -> Option<String> {
            self.entries.get(key).cloned()
        }
        fn set_string(&mut self, key: &str, value: String) {
            self.entries.insert(key.to_string(), value);
        }
        fn flush(&mut self) {
            self.flushes += 1;
        }
    }

    #[test]
    fn test_theme_mode_roundtrips_through_visuals_and_dom_tokens() {
        let cases = [
            (ThemeMode::Dark, "dark", "dark", "#121214"),
            (ThemeMode::Light, "light", "light", "#f8f8f8"),
            (ThemeMode::HighContrast, "high-contrast", "dark", "#000000"),
        ];
        for (mode, data_tok, scheme, hex) in cases {
            let visuals = mode.visuals();
            assert_eq!(ThemeMode::from_visuals(&visuals), mode);
            assert_eq!(SemanticPalette::from_visuals(&visuals), mode.palette());
            assert_eq!(mode.data_theme(), data_tok);
            assert_eq!(mode.color_scheme(), scheme);
            assert_eq!(mode.theme_color_hex(), hex);
            assert_eq!(ThemeMode::from_token(data_tok), Some(mode));
            assert_eq!(ThemeMode::from_token(&format!("{mode:?}")), Some(mode));
        }
        assert_eq!(ThemeMode::from_token("unknown"), None);
    }

    #[test]
    fn test_startup_media_query_precedence() {
        use egui::Theme::{Dark, Light};
        use ThemeMode::{Dark as TD, HighContrast as THC, Light as TL};
        for (fc, mc, pl, sys, expected) in [
            (true, false, true, Some(Light), THC),
            (false, true, true, Some(Light), THC),
            (false, false, true, None, TL),
            (false, false, false, Some(Light), TL),
            (false, false, false, Some(Dark), TD),
            (false, false, false, None, TD),
        ] {
            assert_eq!(resolve_startup_theme(fc, mc, pl, sys), expected);
        }
    }

    #[test]
    fn test_immediate_storage_writes_on_switch_undo_redo_and_session_open() {
        let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
        let ctx = egui::Context::default();
        let mut app = MathExplorerApp::default();
        let mut frame = eframe::Frame::_new_kittest();
        let mut storage = TestStorage::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| app.update(ctx, &mut frame));
        let saved_theme = |s: &TestStorage| {
            extract_explicit_theme_from_json(&s.get_string(eframe::APP_KEY).unwrap())
        };

        app.theme = ThemeMode::Light;
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            let mut mgr = std::mem::take(&mut app.session_manager);
            mgr.track_frame_changes(ctx, &app);
            app.session_manager = mgr;
            sync_theme_with_storage(&mut app, ctx, Some(&mut storage));
        });
        assert_eq!(
            (saved_theme(&storage), storage.flushes),
            (Some(ThemeMode::Light), 1)
        );

        app.undo();
        sync_theme_with_storage(&mut app, &ctx, Some(&mut storage));
        assert_eq!(saved_theme(&storage), Some(ThemeMode::Dark));

        app.redo();
        sync_theme_with_storage(&mut app, &ctx, Some(&mut storage));
        assert_eq!(saved_theme(&storage), Some(ThemeMode::Light));

        let (tx, rx) = std::sync::mpsc::channel();
        app.session_manager.open_rx = Some(rx);
        app.session_manager.lifecycle_state = crate::app::SessionLifecycleState::AwaitingFileDialog;
        let mut hc_state = app.get_app_state();
        hc_state.theme = ThemeMode::HighContrast;
        tx.send(Ok((
            "hc.mws".into(),
            app.session_manager.create_envelope(hc_state),
        )))
        .unwrap();
        let mut mgr = std::mem::take(&mut app.session_manager);
        mgr.poll(&mut app, &ctx);
        app.session_manager = mgr;
        sync_theme_with_storage(&mut app, &ctx, Some(&mut storage));
        assert_eq!(saved_theme(&storage), Some(ThemeMode::HighContrast));
        assert_eq!(ctx.style().visuals, ThemeMode::HighContrast.visuals());
    }
}
