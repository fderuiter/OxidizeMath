//! `sessionStorage` startup boot sentinel and automatic safe-mode workspace quarantine.
//!
//! Brackets synchronous `localStorage` and share-URL hydration through Frame 1 with a
//! `sessionStorage` sentinel. If a reload begins while the previous boot's sentinel is still
//! active, the coordinator quarantines saved state for `.mws` backup export, clears crashing
//! share URL parameters via [`crate::url_state::replace_browser_url_fragment`], preserves the
//! user's [`ThemeMode`], and surfaces a dismissible safe-mode notice banner.

use super::{theme, MathExplorerApp, SemanticPalette, ThemeMode};
use eframe::egui;
use std::collections::HashMap;
#[cfg(not(all(test, not(target_arch = "wasm32"))))]
use std::sync::mpsc::channel;

/// `sessionStorage` key written before startup hydration and cleared after Frame 1 completes.
pub const BOOT_SENTINEL_SESSION_KEY: &str = "math_explorer_boot_sentinel";
/// `localStorage` key holding the most recently quarantined `.mws` / JSON workspace payload.
pub const QUARANTINED_WORKSPACE_STORAGE_KEY: &str = "math_explorer_quarantined_workspace";
/// `localStorage` key preserving an explicit theme across workspace quarantine or manual reset.
pub const PRESERVED_THEME_STORAGE_KEY: &str = "math_explorer_preserved_theme";
/// Default filename used when exporting a quarantined workspace backup.
pub const QUARANTINED_BACKUP_FILENAME: &str = "math-explorer-quarantined-backup.mws";
const BOOT_SENTINEL_ACTIVE_VALUE: &str = "hydrating";

thread_local! {
    static LIVE_ARMED_SENTINELS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Runtime state tracking Frame 1 sentinel completion and the dismissible safe-mode banner.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct BootSentinelState {
    /// `true` while startup hydration or Frame 1 has not yet disarmed the boot sentinel.
    pub sentinel_armed: bool,
    /// `true` when the workspace entered safe mode after quarantining a crashing startup session.
    pub safe_mode_banner_open: bool,
    /// Serialized `.mws` / JSON backup payload available for one-click download from safe mode.
    pub quarantined_backup_json: String,
}

impl Clone for BootSentinelState {
    fn clone(&self) -> Self {
        if self.sentinel_armed {
            LIVE_ARMED_SENTINELS.with(|c| c.set(c.get().saturating_add(1)));
        }
        Self {
            sentinel_armed: self.sentinel_armed,
            safe_mode_banner_open: self.safe_mode_banner_open,
            quarantined_backup_json: self.quarantined_backup_json.clone(),
        }
    }
}

impl Drop for BootSentinelState {
    fn drop(&mut self) {
        if self.sentinel_armed {
            self.sentinel_armed = false;
            let remaining = LIVE_ARMED_SENTINELS.with(|c| {
                let next = c.get().saturating_sub(1);
                c.set(next);
                next
            });
            if remaining == 0 && !std::thread::panicking() {
                remove_session_storage_item(BOOT_SENTINEL_SESSION_KEY);
            }
        }
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct MockBootStorage {
    pub session: HashMap<String, String>,
    pub local: HashMap<String, String>,
    pub dom_theme: Option<String>,
    pub last_downloaded_backup: Option<(String, String)>,
}

#[cfg(test)]
thread_local! {
    static MOCK_BOOT_STORAGE: std::cell::RefCell<Option<MockBootStorage>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_mock_boot_storage(mock: Option<MockBootStorage>) {
    if mock.is_some() {
        LIVE_ARMED_SENTINELS.with(|c| c.set(0));
    }
    MOCK_BOOT_STORAGE.with(|slot| *slot.borrow_mut() = mock);
}

#[cfg(test)]
pub(crate) fn snapshot_mock_boot_storage() -> Option<MockBootStorage> {
    MOCK_BOOT_STORAGE.with(|slot| slot.borrow().clone())
}

#[cfg(test)]
fn mutate_mock_map(is_session: bool, f: impl FnOnce(&mut HashMap<String, String>)) -> bool {
    MOCK_BOOT_STORAGE.with(|slot| {
        let mut guard = slot.borrow_mut();
        let Some(mock) = guard.as_mut() else {
            return false;
        };
        f(if is_session {
            &mut mock.session
        } else {
            &mut mock.local
        });
        true
    })
}

fn read_storage_item(key: &str, is_session: bool) -> Option<String> {
    #[cfg(test)]
    if let Some(res) = MOCK_BOOT_STORAGE.with(|slot| {
        slot.borrow().as_ref().map(|m| {
            let map = if is_session { &m.session } else { &m.local };
            map.get(key).cloned()
        })
    }) {
        return res;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let storage = web_sys::window().and_then(|w| {
            if is_session {
                w.session_storage().ok().flatten()
            } else {
                w.local_storage().ok().flatten()
            }
        });
        storage.and_then(|s| s.get_item(key).ok().flatten())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (key, is_session);
        None
    }
}

fn write_storage_item(key: &str, value: &str, is_session: bool) {
    #[cfg(test)]
    if mutate_mock_map(is_session, |m| {
        m.insert(key.to_string(), value.to_string());
    }) {
        return;
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(storage) = web_sys::window().and_then(|w| {
        if is_session {
            w.session_storage().ok().flatten()
        } else {
            w.local_storage().ok().flatten()
        }
    }) {
        let _ = storage.set_item(key, value);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = (key, value, is_session);
}

fn remove_storage_item(key: &str, is_session: bool) {
    #[cfg(test)]
    if mutate_mock_map(is_session, |m| {
        m.remove(key);
    }) {
        return;
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(storage) = web_sys::window().and_then(|w| {
        if is_session {
            w.session_storage().ok().flatten()
        } else {
            w.local_storage().ok().flatten()
        }
    }) {
        let _ = storage.remove_item(key);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = (key, is_session);
}

fn read_session_storage_item(key: &str) -> Option<String> {
    read_storage_item(key, true)
}

fn write_session_storage_item(key: &str, value: &str) {
    write_storage_item(key, value, true);
}

fn remove_session_storage_item(key: &str) {
    remove_storage_item(key, true);
}

fn read_local_storage_item(key: &str) -> Option<String> {
    read_storage_item(key, false)
}

fn write_local_storage_item(key: &str, value: &str) {
    write_storage_item(key, value, false);
}

fn remove_local_storage_item(key: &str) {
    remove_storage_item(key, false);
}

fn read_dom_theme_fallback() -> Option<ThemeMode> {
    #[cfg(test)]
    if let Some(dom_tok) =
        MOCK_BOOT_STORAGE.with(|slot| slot.borrow().as_ref().and_then(|m| m.dom_theme.clone()))
    {
        return ThemeMode::from_token(&dom_tok);
    }
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.document_element())
            .and_then(|el| el.get_attribute("data-theme"))
            .and_then(|raw| ThemeMode::from_token(&raw))
    }
    #[cfg(not(target_arch = "wasm32"))]
    None
}

/// Extracts an explicit [`ThemeMode`] from raw JSON without deserializing full workspace tab state.
#[must_use]
pub fn extract_theme_token_lenient(raw_json: &str) -> Option<ThemeMode> {
    let cleaned = egui_plot::export::strip_utf8_bom(raw_json);
    let val: serde_json::Value = serde_json::from_str(cleaned).ok()?;
    let token = val
        .get("app_state")
        .and_then(|s| s.get("theme"))
        .and_then(serde_json::Value::as_str)
        .or_else(|| val.get("theme").and_then(serde_json::Value::as_str))?;
    ThemeMode::from_token(token)
}

#[cfg(target_arch = "wasm32")]
fn sync_js_quarantined_envelope(backup_json: &str) {
    use wasm_bindgen::{JsCast, JsValue};
    let Some(win) = web_sys::window() else {
        return;
    };
    let Ok(boot) = js_sys::Reflect::get(win.as_ref(), &JsValue::from_str("__mathExplorerBoot"))
    else {
        return;
    };
    if boot.is_null() || boot.is_undefined() {
        return;
    }
    if let Ok(set_fn) = js_sys::Reflect::get(&boot, &JsValue::from_str("setSavedWorkspaceEnvelope"))
    {
        if let Ok(func) = set_fn.dyn_into::<js_sys::Function>() {
            let _ = func.call3(
                &boot,
                &JsValue::from_str(backup_json),
                &JsValue::from_str(QUARANTINED_BACKUP_FILENAME),
                &JsValue::FALSE,
            );
        }
    }
}

fn is_crashed_boot_sentinel_present() -> bool {
    read_session_storage_item(BOOT_SENTINEL_SESSION_KEY).is_some_and(|val| !val.trim().is_empty())
        && LIVE_ARMED_SENTINELS.with(std::cell::Cell::get) == 0
}

fn arm_sentinel_for_app(app: &mut MathExplorerApp) {
    write_session_storage_item(BOOT_SENTINEL_SESSION_KEY, BOOT_SENTINEL_ACTIVE_VALUE);
    if !app.boot_sentinel.sentinel_armed {
        app.boot_sentinel.sentinel_armed = true;
        LIVE_ARMED_SENTINELS.with(|c| c.set(c.get().saturating_add(1)));
    }
}

fn read_raw_saved_app_payload(cc: &eframe::CreationContext<'_>) -> Option<String> {
    if let Some(raw) = read_local_storage_item(eframe::APP_KEY) {
        if !raw.trim().is_empty() {
            return Some(raw);
        }
    }
    cc.storage
        .and_then(|s| s.get_string(eframe::APP_KEY))
        .filter(|raw| !raw.trim().is_empty())
}

fn quarantine_crashing_startup_state(app: &mut MathExplorerApp, cc: &eframe::CreationContext<'_>) {
    let raw_saved = read_raw_saved_app_payload(cc);
    let preserved_theme = raw_saved
        .as_deref()
        .and_then(extract_theme_token_lenient)
        .or_else(|| {
            read_local_storage_item(PRESERVED_THEME_STORAGE_KEY)
                .and_then(|tok| ThemeMode::from_token(&tok))
        })
        .or_else(read_dom_theme_fallback)
        .unwrap_or_else(|| theme::detect_startup_theme(cc));

    remove_local_storage_item(eframe::APP_KEY);
    write_local_storage_item(PRESERVED_THEME_STORAGE_KEY, preserved_theme.data_theme());
    crate::url_state::install_browser_history_listeners(&cc.egui_ctx);
    let _ = crate::url_state::replace_browser_url_fragment("");

    app.theme = preserved_theme;
    app.theme.apply_to_context(&cc.egui_ctx);
    theme::sync_document_theme(app.theme);
    app.last_synced_theme = Some(app.theme);
    app.show_starter_gallery = false;
    app.url_sync
        .suppress_initial_overwrite(app.current_url_signature());

    let backup_json = raw_saved
        .or_else(|| {
            read_local_storage_item(QUARANTINED_WORKSPACE_STORAGE_KEY)
                .filter(|s| !s.trim().is_empty())
        })
        .unwrap_or_else(|| {
            app.session_manager
                .create_compact_envelope(app.get_app_state())
        });
    write_local_storage_item(QUARANTINED_WORKSPACE_STORAGE_KEY, &backup_json);
    #[cfg(target_arch = "wasm32")]
    sync_js_quarantined_envelope(&backup_json);

    app.boot_sentinel = BootSentinelState {
        sentinel_armed: false,
        safe_mode_banner_open: true,
        quarantined_backup_json: backup_json,
    };
    arm_sentinel_for_app(app);

    let notice =
        "Previous startup crashed; entered safe mode and quarantined saved workspace state.";
    crate::telemetry::emit_telemetry(
        "BootSentinel",
        federated_registry::Severity::Warning,
        notice.to_string(),
        HashMap::new(),
    );
    crate::accessibility::announce_live_status(&cc.egui_ctx, notice);
}

/// Coordinates startup storage and share-URL hydration inside a `sessionStorage` sentinel bracket.
pub(super) fn hydrate_with_boot_sentinel(
    app: &mut MathExplorerApp,
    cc: &eframe::CreationContext<'_>,
) {
    if is_crashed_boot_sentinel_present() {
        quarantine_crashing_startup_state(app, cc);
        return;
    }

    arm_sentinel_for_app(app);

    let restored = theme::init_startup_theme(app, cc);
    if restored {
        remove_local_storage_item(PRESERVED_THEME_STORAGE_KEY);
    } else if let Some(preserved) = read_local_storage_item(PRESERVED_THEME_STORAGE_KEY)
        .and_then(|tok| ThemeMode::from_token(&tok))
    {
        remove_local_storage_item(PRESERVED_THEME_STORAGE_KEY);
        app.theme = preserved;
        app.theme.apply_to_context(&cc.egui_ctx);
        theme::sync_document_theme(app.theme);
        app.last_synced_theme = Some(app.theme);
    }

    let hydrated = app.hydrate_from_browser_url(&cc.egui_ctx, restored);
    if !restored && !hydrated && crate::starter_gallery::has_any_starter_presets() {
        app.show_starter_gallery = true;
    }
}

/// Clears the active `sessionStorage` boot sentinel once Frame 1 finishes rendering cleanly.
pub(crate) fn complete_first_frame(app: &mut MathExplorerApp) {
    if app.boot_sentinel.sentinel_armed {
        app.boot_sentinel.sentinel_armed = false;
        let remaining = LIVE_ARMED_SENTINELS.with(|c| {
            let next = c.get().saturating_sub(1);
            c.set(next);
            next
        });
        if remaining == 0 {
            remove_session_storage_item(BOOT_SENTINEL_SESSION_KEY);
        }
    }
}

/// Triggers export of the quarantined `.mws` backup file from safe mode.
pub fn trigger_quarantined_backup_download(app: &mut MathExplorerApp, ctx: &egui::Context) -> bool {
    let payload = app.boot_sentinel.quarantined_backup_json.trim().to_string();
    if payload.is_empty() {
        return false;
    }
    #[cfg(test)]
    MOCK_BOOT_STORAGE.with(|slot| {
        if let Some(mock) = slot.borrow_mut().as_mut() {
            mock.last_downloaded_backup =
                Some((QUARANTINED_BACKUP_FILENAME.to_string(), payload.clone()));
        }
    });

    #[cfg(target_arch = "wasm32")]
    {
        let (tx, _rx) = channel();
        crate::web_io::spawn_save_direct_file(
            tx,
            crate::web_io::DirectSaveConfig {
                fallback_filename: QUARANTINED_BACKUP_FILENAME,
                write_error_prefix: "Failed to write quarantined workspace backup",
            },
            std::path::PathBuf::from(QUARANTINED_BACKUP_FILENAME),
            payload,
        );
    }
    #[cfg(all(not(target_arch = "wasm32"), not(test)))]
    {
        let (tx, _rx) = channel();
        crate::web_io::spawn_save_as_file_dialog(
            tx,
            crate::web_io::SaveAsDialogConfig {
                filter_name: "Workspace Session (*.mws, *.json)",
                extensions: &["mws", "json"],
                fallback_filename: QUARANTINED_BACKUP_FILENAME,
                write_error_prefix: "Failed to write quarantined workspace backup",
            },
            QUARANTINED_BACKUP_FILENAME,
            payload,
            |path, _default_name| path,
        );
    }

    let msg = format!("Exported quarantined workspace backup {QUARANTINED_BACKUP_FILENAME}.");
    app.session_manager.set_toast(&msg, false);
    crate::accessibility::announce_live_status(ctx, msg);
    true
}

/// Dismisses the safe-mode quarantine notice banner.
pub fn dismiss_safe_mode_banner(app: &mut MathExplorerApp, ctx: &egui::Context) {
    app.boot_sentinel.safe_mode_banner_open = false;
    crate::accessibility::announce_live_status(ctx, "Safe-mode quarantine notice dismissed.");
}

/// Renders the dismissible safe-mode quarantine notice banner when automatic quarantine occurred.
pub(super) fn render_safe_mode_banner(app: &mut MathExplorerApp, ctx: &egui::Context) {
    if !app.boot_sentinel.safe_mode_banner_open {
        return;
    }
    let mut download_clicked = false;
    let mut dismiss_clicked = false;
    let has_backup = !app.boot_sentinel.quarantined_backup_json.trim().is_empty();

    egui::TopBottomPanel::top("quarantine_safe_mode_banner").show(ctx, |ui| {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new(
                    "Safe mode: Quarantined workspace state after a startup crash to restore a clean session.",
                )
                .strong()
                .color(palette.warning),
            );
            if has_backup
                && ui
                    .button("Download Quarantined Backup (.mws)")
                    .on_hover_text("Export the quarantined workspace JSON before continuing")
                    .clicked()
            {
                download_clicked = true;
            }
            if ui
                .button("Dismiss")
                .on_hover_text("Dismiss safe-mode quarantine notice")
                .clicked()
            {
                dismiss_clicked = true;
            }
        });
    });

    if download_clicked {
        let _ = trigger_quarantined_backup_download(app, ctx);
    }
    if dismiss_clicked {
        dismiss_safe_mode_banner(app, ctx);
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "boot_sentinel_wasm_tests.rs"]
mod wasm_tests;
