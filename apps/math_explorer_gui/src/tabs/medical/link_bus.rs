//! Zero-allocation thread-local 4D `GatingTelemetry` bus coupling `RadarGatingTool` and `DoseCalculationTool`.

use math_explorer::physics::medical::radar_gating::BeamState;
use std::cell::Cell;

/// Fixed capacity of the thread-local `GatingTelemetry` snapshot ring.
pub const GATING_RING_CAPACITY: usize = 64;

/// Stack-allocated 4D respiratory gating telemetry snapshot exchanged across medical panes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatingTelemetry {
    /// Monotonic sequence number assigned when published (`0` when uninitialized).
    pub seq: u64,
    /// Radar slow-time simulation clock in seconds.
    pub sim_time_s: f64,
    /// Ground-truth thoracic displacement (including cough transients) in millimeters.
    pub true_disp_mm: f64,
    /// Unwrapped + surface-smoothed radar chest displacement in millimeters.
    pub measured_disp_mm: f64,
    /// Latency-compensated Kalman lookahead prediction in millimeters.
    pub predicted_disp_mm: f64,
    /// Kalman-estimated chest-wall velocity in millimeters per second.
    pub velocity_mm_s: f64,
    /// Cumulative `BeamActive` duty cycle in `[0.0, 100.0]` percent.
    pub duty_cycle_pct: f64,
    /// Current verified Schmitt-trigger `BeamState`.
    pub beam_state: BeamState,
    /// Verified Pearson correlation coefficient `r` across the active trace window.
    pub pearson_r: f64,
    /// Verified root-mean-square tracking error in millimeters.
    pub rmse_mm: f64,
    /// Verified SGRT slant-range spatial accuracy percentage error.
    pub spatial_err_pct: f64,
    /// True when this snapshot advances one 4D dosimetry step; false for non-step state changes.
    pub step_event: bool,
}

impl GatingTelemetry {
    /// Zeroed `Standby` snapshot suitable for `const` initialization.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            seq: 0,
            sim_time_s: 0.0,
            true_disp_mm: 0.0,
            measured_disp_mm: 0.0,
            predicted_disp_mm: 0.0,
            velocity_mm_s: 0.0,
            duty_cycle_pct: 0.0,
            beam_state: BeamState::Standby,
            pearson_r: 1.0,
            rmse_mm: 0.0,
            spatial_err_pct: 0.0,
            step_event: false,
        }
    }
}

impl Default for GatingTelemetry {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Clone, Copy)]
struct LinkBusRing {
    slots: [GatingTelemetry; GATING_RING_CAPACITY],
    write_seq: u64,
    latest: GatingTelemetry,
}

impl LinkBusRing {
    const fn new() -> Self {
        Self {
            slots: [GatingTelemetry::empty(); GATING_RING_CAPACITY],
            write_seq: 0,
            latest: GatingTelemetry::empty(),
        }
    }
}

thread_local! {
    static MEDICAL_LINK_BUS: Cell<LinkBusRing> = const { Cell::new(LinkBusRing::new()) };
}

/// Publishes a `GatingTelemetry` snapshot into the thread-local ring with zero heap allocations.
pub fn publish_gating_telemetry(mut snapshot: GatingTelemetry) -> GatingTelemetry {
    MEDICAL_LINK_BUS.with(|cell| {
        let mut bus = cell.get();
        bus.write_seq = bus.write_seq.saturating_add(1);
        snapshot.seq = bus.write_seq;
        bus.slots[((snapshot.seq - 1) as usize) % GATING_RING_CAPACITY] = snapshot;
        bus.latest = snapshot;
        cell.set(bus);
    });
    snapshot
}

/// Returns the current write sequence number without consuming ring entries.
#[must_use]
pub fn current_write_seq() -> u64 {
    MEDICAL_LINK_BUS.with(|cell| cell.get().write_seq)
}

/// Returns the most recently published `GatingTelemetry` snapshot, if any.
#[cfg(test)]
#[must_use]
pub fn latest_gating_telemetry() -> Option<GatingTelemetry> {
    MEDICAL_LINK_BUS.with(|cell| {
        let bus = cell.get();
        (bus.write_seq > 0).then_some(bus.latest)
    })
}

/// Drains unconsumed `GatingTelemetry` entries newer than `*cursor_seq` into `out` with zero heap allocations.
pub fn drain_gating_telemetry_into(cursor_seq: &mut u64, out: &mut [GatingTelemetry]) -> usize {
    if out.is_empty() {
        return 0;
    }
    MEDICAL_LINK_BUS.with(|cell| {
        let bus = cell.get();
        if bus.write_seq <= *cursor_seq {
            return 0;
        }
        let earliest = bus.write_seq.saturating_sub(GATING_RING_CAPACITY as u64) + 1;
        let start = (*cursor_seq + 1).max(earliest);
        let count = ((bus.write_seq - start + 1) as usize).min(out.len());
        let first_seq = bus.write_seq - (count as u64) + 1;
        for (i, dst) in out.iter_mut().take(count).enumerate() {
            *dst = bus.slots[((first_seq + i as u64 - 1) as usize) % GATING_RING_CAPACITY];
        }
        *cursor_seq = bus.write_seq;
        count
    })
}

/// Resets the thread-local medical link bus to its empty initial state.
#[cfg(test)]
pub fn clear_link_bus() {
    MEDICAL_LINK_BUS.with(|cell| cell.set(LinkBusRing::new()));
}
// [cite:mmwave_radiotherapy_setup]
