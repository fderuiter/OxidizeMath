//! 300 ms debounced address-bar `replaceState` controller with zero idle allocations.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::web_io::PlatformInstant;
use std::time::Duration;

/// Debounce window (`300 ms`) applied before flushing live parameter edits to `location.hash`.
pub const URL_SYNC_DEBOUNCE_MS: u64 = 300;

/// Per-pane O(1) revision token included inside [`UrlStateSignature`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PaneUrlSignature {
    /// Assigned domain tab index for this pane (`0` when pane is inactive).
    pub tab_idx: usize,
    /// Optional active tool identifier within `SimulationFramework` tabs.
    pub active_tool_id: Option<&'static str>,
    /// Deterministic parameter revision counter for this pane's active tab/tool.
    pub state_revision: u64,
}

/// Lightweight O(1) signature of the workspace's shareable single-pane or multi-pane state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UrlStateSignature {
    /// Active focused/primary tab index.
    pub selected_tab: usize,
    /// Optional active tool identifier in the focused pane.
    pub active_tool_id: Option<&'static str>,
    /// Deterministic parameter revision counter for the focused pane.
    pub state_revision: u64,
    /// Active viewport layout (`Single`, `SplitHorizontal`, `SplitVertical`, or `QuadGrid`).
    pub viewport_layout: crate::app::ViewportLayout,
    /// Canonical `f32::to_bits()` of the 2-pane split ratio (`0.5` when not in 2-pane split).
    pub split_ratio_bits: u32,
    /// Canonical `f32::to_bits()` of `(x, y)` `QuadGrid` split ratios (`(0.5, 0.5)` otherwise).
    pub quad_split_ratio_bits: (u32, u32),
    /// Focused pane index (`0..viewport_layout.pane_count()`).
    pub focused_pane: usize,
    /// Multi-pane comparison sync toggles (`Default` when in `ViewportLayout::Single`).
    pub compare_sync: crate::app::ComparisonSyncConfig,
    /// Active per-pane signatures for panes `0..viewport_layout.pane_count()`.
    pub pane_revisions: [PaneUrlSignature; 4],
}

impl Default for UrlStateSignature {
    fn default() -> Self {
        let half = 0.5_f32.to_bits();
        Self {
            selected_tab: 0,
            active_tool_id: None,
            state_revision: 0,
            viewport_layout: crate::app::ViewportLayout::Single,
            split_ratio_bits: half,
            quad_split_ratio_bits: (half, half),
            focused_pane: 0,
            compare_sync: crate::app::ComparisonSyncConfig::default(),
            pane_revisions: [PaneUrlSignature::default(); 4],
        }
    }
}

/// Tracks changes to [`UrlStateSignature`] and flushes encoded `#tab=...` or `#ws=...` fragments
/// after [`URL_SYNC_DEBOUNCE_MS`] of idle pointer/parameter activity.
#[derive(Default)]
pub struct UrlSyncController {
    last_observed: Option<UrlStateSignature>,
    last_flushed: Option<UrlStateSignature>,
    clock_origin: Option<PlatformInstant>,
    dirty_since_ms: Option<u64>,
    last_synced_fragment: Option<String>,
    suppress_until_user_change: bool,
}

impl UrlSyncController {
    /// Marks the current signature as already synchronized (e.g., right after startup hydration).
    pub fn mark_synced(&mut self, sig: UrlStateSignature, fragment: Option<String>) {
        self.last_observed = Some(sig);
        self.last_flushed = Some(sig);
        self.dirty_since_ms = None;
        self.suppress_until_user_change = false;
        if let Some(frag) = fragment {
            self.last_synced_fragment = Some(frag);
        }
    }

    /// Suppresses automatic `replaceState` flushing until the user changes tab, tool, layout, or parameters.
    pub fn suppress_initial_overwrite(&mut self, sig: UrlStateSignature) {
        self.last_observed = Some(sig);
        self.last_flushed = Some(sig);
        self.dirty_since_ms = None;
        self.suppress_until_user_change = true;
    }

    /// Returns the most recently flushed URL fragment (`""` if cleared after budget overflow).
    #[must_use]
    pub fn last_synced_fragment(&self) -> Option<&str> {
        self.last_synced_fragment.as_deref()
    }

    /// Observes the current frame's [`UrlStateSignature`] at [`PlatformInstant`] `now`.
    pub fn poll_with_now<F>(
        &mut self,
        sig: UrlStateSignature,
        pointer_down: bool,
        now: PlatformInstant,
        debounce: Duration,
        encode_fragment: F,
    ) -> Option<Duration>
    where
        F: FnOnce() -> Result<String, String>,
    {
        let origin = *self.clock_origin.get_or_insert(now);
        let now_ms = u64::try_from(now.duration_since(origin).as_millis()).unwrap_or(u64::MAX);
        let debounce_ms = u64::try_from(debounce.as_millis()).unwrap_or(u64::MAX);
        self.poll_at_millis(sig, pointer_down, now_ms, debounce_ms, encode_fragment)
    }

    /// Observes the current frame's [`UrlStateSignature`] at monotonic `now_ms`, deferring while
    /// `pointer_down` is true and invoking `encode_fragment` + browser `replaceState` once
    /// `debounce_ms` has elapsed.
    pub fn poll_at_millis<F>(
        &mut self,
        sig: UrlStateSignature,
        pointer_down: bool,
        now_ms: u64,
        debounce_ms: u64,
        encode_fragment: F,
    ) -> Option<Duration>
    where
        F: FnOnce() -> Result<String, String>,
    {
        if self.last_observed != Some(sig) {
            let is_initial = self.last_observed.is_none();
            self.last_observed = Some(sig);
            if is_initial {
                self.last_flushed = Some(sig);
                return None;
            }
            self.suppress_until_user_change = false;
            if self.last_flushed == Some(sig) {
                self.dirty_since_ms = None;
                return None;
            }
            self.dirty_since_ms = Some(now_ms);
        }

        let dirty_at_ms = self.dirty_since_ms?;
        if self.suppress_until_user_change {
            self.dirty_since_ms = None;
            return None;
        }
        if pointer_down {
            self.dirty_since_ms = Some(now_ms);
            return Some(Duration::from_millis(debounce_ms));
        }

        let elapsed_ms = now_ms.saturating_sub(dirty_at_ms);
        if elapsed_ms < debounce_ms {
            return Some(Duration::from_millis(debounce_ms - elapsed_ms));
        }

        self.dirty_since_ms = None;
        self.last_flushed = Some(sig);
        self.flush_encoded_or_clear(encode_fragment());
        None
    }

    fn flush_encoded_or_clear(&mut self, encoded: Result<String, String>) {
        match encoded {
            Ok(fragment) => {
                if self.last_synced_fragment.as_deref() != Some(fragment.as_str()) {
                    let _ = super::replace_browser_url_fragment(&fragment);
                    self.last_synced_fragment = Some(fragment);
                }
            }
            Err(_) => {
                if self.last_synced_fragment.as_deref() != Some("") {
                    let _ = super::replace_browser_url_fragment("");
                    self.last_synced_fragment = Some(String::new());
                }
            }
        }
    }
}

// theory_verification!
