#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use math_explorer::pure_math::statistics::tda::Point3D;
use std::sync::{Mutex, OnceLock};

/// Payload transferring up to `MAX_VIETORIS_RIPS_POINTS` 3D chaotic landmarks and
/// Lorenz system parameters from `AttractorPlotter` / `AttractorRunner` to `VietorisRipsTool`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChaosLandmarkHandoff {
    /// Subsampled 3D landmark coordinates in state space.
    pub points: Vec<Point3D>,
    /// Prandtl parameter `σ`.
    pub sigma: f64,
    /// Rayleigh parameter `ρ`.
    pub rho: f64,
    /// Geometric parameter `β`.
    pub beta: f64,
    /// Integration time step `dt`.
    pub dt: f64,
}

struct LiveStreamSlot {
    seq: u64,
    latest: Option<ChaosLandmarkHandoff>,
}

fn handoff_slot() -> &'static Mutex<Option<ChaosLandmarkHandoff>> {
    static SLOT: OnceLock<Mutex<Option<ChaosLandmarkHandoff>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

fn live_stream_slot() -> &'static Mutex<LiveStreamSlot> {
    static STREAM: OnceLock<Mutex<LiveStreamSlot>> = OnceLock::new();
    STREAM.get_or_init(|| {
        Mutex::new(LiveStreamSlot {
            seq: 0,
            latest: None,
        })
    })
}

/// Queues a one-click 3D landmark handoff for `VietorisRipsTool`.
#[cfg(feature = "physics")]
pub fn submit_chaos_landmark_handoff(handoff: ChaosLandmarkHandoff) {
    let mut guard = handoff_slot().lock().unwrap_or_else(|e| e.into_inner());
    *guard = Some(handoff);
}

/// Drains any pending one-click 3D landmark handoff queued by `AttractorPlotter`.
pub fn take_chaos_landmark_handoff() -> Option<ChaosLandmarkHandoff> {
    let mut guard = handoff_slot().lock().unwrap_or_else(|e| e.into_inner());
    guard.take()
}

/// Publishes the latest live 3D landmarks from `AttractorRunner` for split-pane / live streaming.
#[cfg(feature = "physics")]
pub fn publish_live_attractor_landmarks(handoff: ChaosLandmarkHandoff) {
    let mut guard = live_stream_slot().lock().unwrap_or_else(|e| e.into_inner());
    guard.seq = guard.seq.wrapping_add(1).max(1);
    guard.latest = Some(handoff);
}

/// Reads a newer live landmark frame if `seq > last_seen_seq`.
pub fn poll_live_attractor_landmarks(last_seen_seq: u64) -> Option<(u64, ChaosLandmarkHandoff)> {
    let guard = live_stream_slot().lock().unwrap_or_else(|e| e.into_inner());
    if guard.seq > last_seen_seq {
        guard.latest.clone().map(|h| (guard.seq, h))
    } else {
        None
    }
}

const OPEN_VR_REQUEST_ID: &str = "OPEN_VIETORIS_RIPS_HANDOFF";

/// Signals `MathExplorerApp` to focus the `Geometry & Topology` tab and select `VietorisRipsTool`.
#[cfg(feature = "physics")]
pub fn request_open_vietoris_rips(ctx: &egui::Context) {
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(OPEN_VR_REQUEST_ID), true);
    });
    ctx.request_repaint();
}

/// Consumes any pending `VietorisRipsTool` cross-tab navigation request.
pub fn take_open_vietoris_rips_request(ctx: &egui::Context) -> bool {
    ctx.data_mut(|d| {
        d.remove_temp::<bool>(egui::Id::new(OPEN_VR_REQUEST_ID))
            .unwrap_or(false)
    })
}
