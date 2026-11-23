#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::pane_slots::MAX_SECONDARY_PANES;
use super::state::{AppState, SplitLayoutMode};
use super::theme::ThemeMode;
use super::viewport::ViewportLayout;
use super::{MathExplorerApp, SharedTabJson};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HistoryFingerprint {
    selected_tab: usize,
    secondary_tab: usize,
    split_mode_discriminant: u8,
    viewport_discriminant: u8,
    compare_sync_bits: u8,
    split_ratio_bits: u32,
    quad_split_ratio_bits: (u32, u32),
    pane_tabs: [usize; 4],
    primary_history_hash: u64,
    secondary_history_hash: u64,
}

/// Zero-allocation O(1) state fingerprint separating semantic history from transient UI chrome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AppStateFingerprint {
    history: HistoryFingerprint,
    chrome_hash: u64,
    pub(crate) all_revisions_tracked: bool,
}

impl AppStateFingerprint {
    pub(crate) fn same_history_as(&self, other: &Self) -> bool {
        self.history == other.history
    }

    pub(crate) fn capture(app: &MathExplorerApp) -> Self {
        let mut primary_hist = std::collections::hash_map::DefaultHasher::new();
        let mut chrome_hasher = std::collections::hash_map::DefaultHasher::new();
        let mut all_tracked = true;
        let active_primary = app.active_primary_tab_index();
        for (idx, tab) in app.tabs.iter().enumerate() {
            let full_rev = tab.state_revision();
            let hist_rev = tab.history_revision();
            if idx == active_primary && full_rev == 0 {
                all_tracked = false;
            }
            idx.hash(&mut primary_hist);
            hist_rev.hash(&mut primary_hist);
            idx.hash(&mut chrome_hasher);
            full_rev.hash(&mut chrome_hasher);
        }
        let (sec_full, sec_hist, sec_tracked) = app.secondary_slots_revision_hashes();
        all_tracked &= sec_tracked;
        sec_full.hash(&mut chrome_hasher);
        hash_app_chrome_fields(app, &mut chrome_hasher);

        Self {
            history: build_history_fingerprint(app, primary_hist.finish(), sec_hist),
            chrome_hash: chrome_hasher.finish(),
            all_revisions_tracked: all_tracked,
        }
    }
}

fn hash_app_chrome_fields(app: &MathExplorerApp, hasher: &mut impl Hasher) {
    app.show_info.hash(hasher);
    app.show_warnings.hash(hasher);
    app.show_errors.hash(hasher);
    app.show_diagnostics_panel.hash(hasher);
    let theme_disc: u8 = match app.theme {
        ThemeMode::Dark => 0,
        ThemeMode::Light => 1,
        ThemeMode::HighContrast => 2,
    };
    theme_disc.hash(hasher);
    app.active_focused_pane().hash(hasher);
}

fn build_history_fingerprint(
    app: &MathExplorerApp,
    primary_history_hash: u64,
    secondary_history_hash: u64,
) -> HistoryFingerprint {
    let mut pane_tabs = [0_usize; 4];
    for (i, slot) in pane_tabs.iter_mut().enumerate() {
        *slot = app.pane_tabs.get(i).copied().unwrap_or(i);
    }
    pane_tabs[0] = app.selected_tab;
    if app.viewport_layout != ViewportLayout::QuadGrid {
        pane_tabs[1] = app.secondary_tab;
    }
    HistoryFingerprint {
        selected_tab: app.selected_tab,
        secondary_tab: app.secondary_tab,
        split_mode_discriminant: match app.split_layout_mode {
            SplitLayoutMode::Single => 0,
            SplitLayoutMode::Horizontal => 1,
            SplitLayoutMode::Vertical => 2,
        },
        viewport_discriminant: match app.viewport_layout {
            ViewportLayout::Single => 0,
            ViewportLayout::SplitHorizontal => 1,
            ViewportLayout::SplitVertical => 2,
            ViewportLayout::QuadGrid => 3,
        },
        compare_sync_bits: u8::from(app.compare_sync.link_axes)
            | (u8::from(app.compare_sync.link_cursors) << 1)
            | (u8::from(app.compare_sync.sync_playback) << 2),
        split_ratio_bits: app.split_ratio.clamp(0.20, 0.80).to_bits(),
        quad_split_ratio_bits: (
            app.quad_split_ratios.0.clamp(0.20, 0.80).to_bits(),
            app.quad_split_ratios.1.clamp(0.20, 0.80).to_bits(),
        ),
        pane_tabs,
        primary_history_hash,
        secondary_history_hash,
    }
}

fn tab_maps_semantically_eq(
    a: &HashMap<String, SharedTabJson>,
    b: &HashMap<String, SharedTabJson>,
) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().all(|(k, va)| {
        b.get(k)
            .is_some_and(|vb| va == vb || crate::framework::framework_json_semantically_eq(va, vb))
    })
}

fn pane_tab_maps_semantically_eq(
    a: &[HashMap<String, SharedTabJson>],
    b: &[HashMap<String, SharedTabJson>],
) -> bool {
    let empty = HashMap::new();
    let max_len = a.len().max(b.len());
    (0..max_len).all(|idx| {
        let ma = a.get(idx).unwrap_or(&empty);
        let mb = b.get(idx).unwrap_or(&empty);
        tab_maps_semantically_eq(ma, mb)
    })
}

struct LiveChromeSnapshot {
    show_info: bool,
    show_warnings: bool,
    show_errors: bool,
    show_diagnostics_panel: bool,
    theme: ThemeMode,
    focused_pane: usize,
    primary_overlays: HashMap<String, (bool, bool)>,
    secondary_overlays: [Option<(String, bool, bool)>; MAX_SECONDARY_PANES],
}

impl LiveChromeSnapshot {
    fn capture(app: &MathExplorerApp) -> Self {
        let mut primary_overlays = HashMap::with_capacity(app.tabs.len());
        for tab in &app.tabs {
            if let Some(fw) = tab.framework() {
                primary_overlays.insert(
                    tab.name().to_string(),
                    (fw.show_theory_portal, fw.show_starter_gallery),
                );
            }
        }
        let secondary_overlays = std::array::from_fn(|idx| {
            let tab = app.pane_tab(idx + 1)?;
            let fw = tab.framework()?;
            Some((
                tab.name().to_string(),
                fw.show_theory_portal,
                fw.show_starter_gallery,
            ))
        });
        Self {
            show_info: app.show_info,
            show_warnings: app.show_warnings,
            show_errors: app.show_errors,
            show_diagnostics_panel: app.show_diagnostics_panel,
            theme: app.theme,
            focused_pane: app.active_focused_pane(),
            primary_overlays,
            secondary_overlays,
        }
    }

    fn restore(self, app: &mut MathExplorerApp) {
        app.show_info = self.show_info;
        app.show_warnings = self.show_warnings;
        app.show_errors = self.show_errors;
        app.show_diagnostics_panel = self.show_diagnostics_panel;
        app.theme = self.theme;
        if app.viewport_layout == ViewportLayout::Single {
            app.focused_pane = 0;
        } else {
            let max_pane = app.viewport_layout.pane_count().saturating_sub(1);
            app.focused_pane = self.focused_pane.min(max_pane);
        }
        for tab in &mut app.tabs {
            if let Some(&(portal, gallery)) = self.primary_overlays.get(tab.name()) {
                if let Some(fw) = tab.framework_mut() {
                    fw.restore_overlay_chrome(portal, gallery);
                }
            }
        }
        for (idx, entry) in self.secondary_overlays.into_iter().enumerate() {
            if let Some((tab_name, portal, gallery)) = entry {
                app.restore_secondary_slot_overlay_chrome(idx + 1, &tab_name, portal, gallery);
            }
        }
        app.tab_state_cache.clear();
        app.clear_active_pane_caches();
    }
}

impl AppState {
    /// Compares two `AppState` snapshots while ignoring transient UI chrome
    /// (`focused_pane`, `theme`, `show_diagnostics_panel`, diagnostic filter flags,
    /// `show_theory_portal`, and `show_starter_gallery`).
    pub fn semantically_eq(&self, other: &Self) -> bool {
        self.selected_tab == other.selected_tab
            && self.selected_tab_name == other.selected_tab_name
            && self.split_layout_mode == other.split_layout_mode
            && self.secondary_tab == other.secondary_tab
            && self.secondary_tab_name == other.secondary_tab_name
            && self.split_ratio.to_bits() == other.split_ratio.to_bits()
            && self.quad_split_ratios.0.to_bits() == other.quad_split_ratios.0.to_bits()
            && self.quad_split_ratios.1.to_bits() == other.quad_split_ratios.1.to_bits()
            && self.viewport_layout == other.viewport_layout
            && self.compare_sync == other.compare_sync
            && self.pane_tabs == other.pane_tabs
            && tab_maps_semantically_eq(&self.tab_states, &other.tab_states)
            && pane_tab_maps_semantically_eq(&self.pane_tab_states, &other.pane_tab_states)
    }

    /// Restores historical workspace state during `Undo` / `Redo` while preserving live UI chrome.
    pub fn apply_preserving_chrome(self, app: &mut MathExplorerApp) {
        let chrome = LiveChromeSnapshot::capture(app);
        self.apply(app);
        chrome.restore(app);
    }
}
