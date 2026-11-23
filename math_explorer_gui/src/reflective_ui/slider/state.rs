#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;

/// Animation repeat behavior for [`super::AnimatedSlider`], matching GeoGebra slider repeat modes.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum SweepMode {
    /// Ping-pong sweep reversing direction at `min` and `max` (`⇔`).
    #[default]
    Oscillate,
    /// Forward sawtooth sweep from `min` to `max`, wrapping back to `min` (`⇒`).
    Increasing,
    /// Reverse sawtooth sweep from `max` to `min`, wrapping back to `max` (`⇐`).
    Decreasing,
    /// Single forward sweep stopping playback upon reaching `max` (`⇒|`).
    Once,
}

impl SweepMode {
    /// Ordered list of all supported sweep modes for UI selection.
    pub const ALL: [Self; 4] = [
        Self::Oscillate,
        Self::Increasing,
        Self::Decreasing,
        Self::Once,
    ];

    /// Human-readable display label for UI controls and screen-reader announcements.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Oscillate => "Oscillate",
            Self::Increasing => "Increasing",
            Self::Decreasing => "Decreasing",
            Self::Once => "Once",
        }
    }
}

/// Persistent animated slider playback state per scoped widget ID.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AnimatedSliderState {
    /// Whether parameter animation is currently playing.
    pub playing: bool,
    /// Sweep direction (`+1.0` for forward, `-1.0` for reverse).
    pub direction: f64,
    /// Active repeat mode for parameter sweeps.
    #[serde(default)]
    pub sweep_mode: SweepMode,
}

impl Default for AnimatedSliderState {
    fn default() -> Self {
        Self {
            playing: false,
            direction: 1.0,
            sweep_mode: SweepMode::Oscillate,
        }
    }
}

/// Persisted user overrides for soft `[min, max]` bounds, `step`, and `speed` per scoped slider ID.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AnimatedSliderSettings {
    /// Custom soft lower bound override.
    pub min: Option<f64>,
    /// Custom soft upper bound override.
    pub max: Option<f64>,
    /// Custom step size override (`0.0` = continuous).
    pub step: Option<f64>,
    /// Custom sweep speed override (units per second).
    pub speed: Option<f64>,
}

impl AnimatedSliderSettings {
    /// Returns `true` when at least one soft bound or step/speed setting has been customized.
    #[must_use]
    pub const fn has_overrides(&self) -> bool {
        self.min.is_some() || self.max.is_some() || self.step.is_some() || self.speed.is_some()
    }
}

/// Resolves the pane- and owner-scoped `egui::Id` used by [`super::AnimatedSlider`].
#[must_use]
pub fn resolve_scoped_slider_id(ui: &egui::Ui, id_salt: Option<egui::Id>, label: &str) -> egui::Id {
    match id_salt {
        Some(raw_id) => {
            let scope_key = ui.ctx().data(|d| {
                let pane = d.get_temp::<usize>(egui::Id::new("CURRENT_RENDERING_PANE"));
                let owner = d.get_temp::<u64>(egui::Id::new("SCOPED_ANIMATED_SLIDER_OWNER"));
                (pane, owner)
            });
            if scope_key.0.is_some() || scope_key.1.is_some() {
                ui.make_persistent_id((raw_id, scope_key.0, scope_key.1))
            } else {
                raw_id
            }
        }
        None => ui.make_persistent_id(label),
    }
}

/// Loads persisted [`AnimatedSliderState`] for `id`, initializing with `default_mode` if absent.
#[must_use]
pub fn load_slider_state(
    ctx: &egui::Context,
    id: egui::Id,
    default_mode: SweepMode,
) -> AnimatedSliderState {
    ctx.data_mut(|d| d.get_persisted::<AnimatedSliderState>(id))
        .unwrap_or(AnimatedSliderState {
            playing: false,
            direction: if matches!(default_mode, SweepMode::Decreasing) {
                -1.0
            } else {
                1.0
            },
            sweep_mode: default_mode,
        })
}

/// Persists [`AnimatedSliderState`] for `id` in `egui` memory.
pub fn save_slider_state(ctx: &egui::Context, id: egui::Id, state: AnimatedSliderState) {
    ctx.data_mut(|d| d.insert_persisted(id, state));
}

/// Loads persisted [`AnimatedSliderSettings`] for `id`.
#[must_use]
pub fn load_slider_settings(ctx: &egui::Context, id: egui::Id) -> AnimatedSliderSettings {
    ctx.data_mut(|d| d.get_persisted::<AnimatedSliderSettings>(id.with("slider_settings")))
        .unwrap_or_default()
}

/// Persists [`AnimatedSliderSettings`] for `id` in `egui` memory.
pub fn save_slider_settings(ctx: &egui::Context, id: egui::Id, settings: AnimatedSliderSettings) {
    ctx.data_mut(|d| d.insert_persisted(id.with("slider_settings"), settings));
}

/// Per-frame animated slider sweep counters stored in `egui::Context` temporary memory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SweepFrameFlags {
    /// Cumulative egui pass number when these flags were recorded.
    pub pass_nr: Option<u64>,
    /// Number of `AnimatedSlider` widgets actively playing during the frame.
    pub active_sweep_count: usize,
    /// Number of `AnimatedSlider` widgets that mutated bound parameters during the frame.
    pub mutation_count: u64,
}

const ROOT_SWEEP_FLAGS_KEY: &str = "ANIMATED_SLIDER_SWEEP_FLAGS";
const LIVE_SWEEP_FLAGS_KEY: &str = "ANIMATED_SLIDER_LIVE_SWEEP_FLAGS";
const PANE_SWEEP_FLAGS_KEY: &str = "ANIMATED_SLIDER_PANE_SWEEP_FLAGS";

/// Resets root and live frame sweep flags at the start of an application frame.
pub fn reset_frame_sweep_flags(ctx: &egui::Context) {
    let flags = SweepFrameFlags {
        pass_nr: Some(ctx.cumulative_pass_nr()),
        active_sweep_count: 0,
        mutation_count: 0,
    };
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(ROOT_SWEEP_FLAGS_KEY), flags);
        d.insert_temp(egui::Id::new(LIVE_SWEEP_FLAGS_KEY), flags);
    });
}

/// Starts a per-pane sweep accumulation scope prior to rendering `pane_idx`.
pub fn begin_pane_sweep_scope(ctx: &egui::Context, pane_idx: usize) {
    let flags = SweepFrameFlags {
        pass_nr: Some(ctx.cumulative_pass_nr()),
        active_sweep_count: 0,
        mutation_count: 0,
    };
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new((PANE_SWEEP_FLAGS_KEY, pane_idx)), flags);
    });
}

/// Merges per-pane sweep flags recorded while rendering `pane_idx` into the root `egui::Context`.
pub fn merge_pane_sweep_flags(ctx: &egui::Context, pane_idx: usize) {
    let pass_nr = ctx.cumulative_pass_nr();
    ctx.data_mut(|d| {
        let Some(pane_flags) =
            d.remove_temp::<SweepFrameFlags>(egui::Id::new((PANE_SWEEP_FLAGS_KEY, pane_idx)))
        else {
            return;
        };
        let root_id = egui::Id::new(ROOT_SWEEP_FLAGS_KEY);
        let mut root = d.get_temp::<SweepFrameFlags>(root_id).unwrap_or_default();
        if root.pass_nr != Some(pass_nr) {
            root = SweepFrameFlags {
                pass_nr: Some(pass_nr),
                active_sweep_count: 0,
                mutation_count: 0,
            };
        }
        root.active_sweep_count = root
            .active_sweep_count
            .saturating_add(pane_flags.active_sweep_count);
        root.mutation_count = root
            .mutation_count
            .saturating_add(pane_flags.mutation_count);
        d.insert_temp(root_id, root);
    });
}

fn read_valid_flags(ctx: &egui::Context, id: egui::Id) -> SweepFrameFlags {
    let current_pass = ctx.cumulative_pass_nr();
    ctx.data(|d| d.get_temp::<SweepFrameFlags>(id))
        .filter(|f| {
            f.pass_nr
                .is_some_and(|p| current_pass <= p.saturating_add(1))
        })
        .unwrap_or_default()
}

/// Returns the merged count of actively playing `AnimatedSlider` sweeps in the root context.
#[must_use]
pub fn active_sweep_count(ctx: &egui::Context) -> usize {
    read_valid_flags(ctx, egui::Id::new(ROOT_SWEEP_FLAGS_KEY)).active_sweep_count
}

/// Returns `true` when at least one `AnimatedSlider` sweep is active on the current frame.
#[must_use]
pub fn has_active_sweeps(ctx: &egui::Context) -> bool {
    active_sweep_count(ctx) > 0 || live_active_sweep_count(ctx) > 0
}

/// Returns the merged count of `AnimatedSlider` sweep mutations in the root context.
#[must_use]
pub fn frame_sweep_mutation_count(ctx: &egui::Context) -> u64 {
    read_valid_flags(ctx, egui::Id::new(ROOT_SWEEP_FLAGS_KEY)).mutation_count
}

/// Returns `true` when at least one `AnimatedSlider` sweep mutated a bound value this frame.
#[must_use]
pub fn has_sweep_mutation(ctx: &egui::Context) -> bool {
    frame_sweep_mutation_count(ctx) > 0
}

/// Returns the live intra-pass count of actively playing `AnimatedSlider` sweeps.
#[must_use]
pub fn live_active_sweep_count(ctx: &egui::Context) -> usize {
    read_valid_flags(ctx, egui::Id::new(LIVE_SWEEP_FLAGS_KEY)).active_sweep_count
}

/// Returns the live intra-pass count of `AnimatedSlider` parameter mutations.
#[must_use]
pub fn live_sweep_mutation_count(ctx: &egui::Context) -> u64 {
    read_valid_flags(ctx, egui::Id::new(LIVE_SWEEP_FLAGS_KEY)).mutation_count
}

pub(super) fn record_slider_frame(ctx: &egui::Context, playing: bool, mutated: bool) {
    let pass_nr = ctx.cumulative_pass_nr();
    let add_active = usize::from(playing);
    let add_mut = u64::from(mutated);
    ctx.data_mut(|d| {
        let live_id = egui::Id::new(LIVE_SWEEP_FLAGS_KEY);
        let mut live = d.get_temp::<SweepFrameFlags>(live_id).unwrap_or_default();
        if live.pass_nr != Some(pass_nr) {
            live = SweepFrameFlags {
                pass_nr: Some(pass_nr),
                active_sweep_count: 0,
                mutation_count: 0,
            };
        }
        live.active_sweep_count = live.active_sweep_count.saturating_add(add_active);
        live.mutation_count = live.mutation_count.saturating_add(add_mut);
        d.insert_temp(live_id, live);

        let target_id = d
            .get_temp::<usize>(egui::Id::new("CURRENT_RENDERING_PANE"))
            .map(|p| egui::Id::new((PANE_SWEEP_FLAGS_KEY, p)))
            .filter(|pane_id| d.get_temp::<SweepFrameFlags>(*pane_id).is_some())
            .unwrap_or_else(|| egui::Id::new(ROOT_SWEEP_FLAGS_KEY));
        let mut flags = d.get_temp::<SweepFrameFlags>(target_id).unwrap_or_default();
        if flags.pass_nr != Some(pass_nr) {
            flags = SweepFrameFlags {
                pass_nr: Some(pass_nr),
                active_sweep_count: 0,
                mutation_count: 0,
            };
        }
        flags.active_sweep_count = flags.active_sweep_count.saturating_add(add_active);
        flags.mutation_count = flags.mutation_count.saturating_add(add_mut);
        d.insert_temp(target_id, flags);
    });
}
