//! Cross-tab launch handoff from `FluidDynamicsTab` to `AdvectiveMorphogenesisTool` on a `100 x 50` grid.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use std::sync::{Mutex, OnceLock};

const OPEN_ADVECTIVE_MORPHO_ID: &str = "OPEN_ADVECTIVE_MORPHOGENESIS_HANDOFF";

/// Deterministic `100 x 50` fluid-to-morphogenesis handoff payload.
#[derive(Clone, Debug, PartialEq)]
pub struct AdvectiveMorphogenesisHandoff {
    /// Kinematic viscosity `nu` transferred from `FluidDynamicsTab`.
    pub viscosity: f64,
    /// Optional `1,250`-character hex obstacle mask transferred from `LatticeBoltzmannTool`.
    pub mask_hex: Option<String>,
}

impl Default for AdvectiveMorphogenesisHandoff {
    fn default() -> Self {
        Self {
            viscosity: 0.020,
            mask_hex: None,
        }
    }
}

fn handoff_slot() -> &'static Mutex<Option<AdvectiveMorphogenesisHandoff>> {
    static SLOT: OnceLock<Mutex<Option<AdvectiveMorphogenesisHandoff>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Queues a cross-tab launch into `MorphogenesisTab` -> `AdvectiveMorphogenesisTool`.
pub fn request_open_advective_morphogenesis(
    ctx: &egui::Context,
    handoff: AdvectiveMorphogenesisHandoff,
) {
    let mut guard = handoff_slot().lock().unwrap_or_else(|e| e.into_inner());
    *guard = Some(handoff);
    drop(guard);
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(OPEN_ADVECTIVE_MORPHO_ID), true);
    });
    ctx.request_repaint();
}

/// Consumes any pending `AdvectiveMorphogenesisTool` cross-tab navigation trigger.
pub fn take_open_advective_morphogenesis_request(ctx: &egui::Context) -> bool {
    ctx.data_mut(|d| {
        d.remove_temp::<bool>(egui::Id::new(OPEN_ADVECTIVE_MORPHO_ID))
            .unwrap_or(false)
    })
}

/// Drains any pending [`AdvectiveMorphogenesisHandoff`] payload queued by `FluidDynamicsTab`.
pub fn take_advective_morphogenesis_handoff() -> Option<AdvectiveMorphogenesisHandoff> {
    let mut guard = handoff_slot().lock().unwrap_or_else(|e| e.into_inner());
    guard.take()
}
