#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::atlas::ParameterAtlasController;
use crate::framework::{CalloutManager, CalloutSpec, ExportCapabilities, InteractiveTool};
use eframe::egui;
use math_explorer::applied::game_theory::equilibrium::{
    braess_avg_physical_latency, braess_optimal_pigou_toll, check_braess_wardrop_fixed_point,
    smith_route_step, solve_braess_wheatstone_flows, BraessWheatstoneParams,
};
use math_explorer::applied::game_theory::mean_field::{
    Density, FixedPointSolver, MFGConfigBuilder, MFGSolver, Position,
};
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

/// Serializable bounded state for `BraessNetworkParadoxTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BraessState {
    /// Total commuter traffic demand `D` from source `S` to sink `T`.
    pub demand: f64,
    /// Linear congestion slope `alpha` on links `S->A` and `B->T`.
    pub congestion_slope: f64,
    /// Constant bypass travel time `beta` on links `A->T` and `S->B`.
    pub bypass_latency: f64,
    /// Physical base travel cost `c_short` on shortcut `A->B`.
    pub shortcut_cost: f64,
    /// Pigouvian congestion toll `tau` charged on shortcut `A->B`.
    pub shortcut_toll: f64,
    /// Whether central shortcut `A->B` is open to traffic.
    pub shortcut_open: bool,
    /// Whether to display coupled HJB / Fokker-Planck continuum crowd curves.
    pub continuum_mfg_mode: bool,
    /// Viscosity `nu` for the continuum Mean-Field Game PDEs.
    pub mfg_viscosity: f64,
    /// Spatial grid resolution `Nx` for `FixedPointSolver`.
    pub mfg_grid_points: usize,
}

impl Default for BraessState {
    fn default() -> Self {
        Self {
            demand: 1.0,
            congestion_slope: 1.0,
            bypass_latency: 1.0,
            shortcut_cost: 0.0,
            shortcut_toll: 0.0,
            shortcut_open: true,
            continuum_mfg_mode: false,
            mfg_viscosity: 0.08,
            mfg_grid_points: 33,
        }
    }
}

impl BraessState {
    pub(super) fn wheatstone_params(&self) -> BraessWheatstoneParams {
        BraessWheatstoneParams {
            demand: self.demand,
            congestion_slope: self.congestion_slope,
            bypass_latency: self.bypass_latency,
            shortcut_cost: self.shortcut_cost,
            shortcut_toll: self.shortcut_toll,
            shortcut_open: self.shortcut_open,
        }
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        let ok_net = self.demand.is_finite()
            && (0.2..=2.0).contains(&self.demand)
            && self.congestion_slope.is_finite()
            && (0.2..=3.0).contains(&self.congestion_slope)
            && self.bypass_latency.is_finite()
            && (0.2..=3.0).contains(&self.bypass_latency);
        let ok_short = self.shortcut_cost.is_finite()
            && (0.0..=1.5).contains(&self.shortcut_cost)
            && self.shortcut_toll.is_finite()
            && (0.0..=1.5).contains(&self.shortcut_toll);
        let ok_mfg = self.mfg_viscosity.is_finite()
            && (0.02..=0.25).contains(&self.mfg_viscosity)
            && (17..=65).contains(&self.mfg_grid_points);
        if !ok_net || !ok_short || !ok_mfg {
            return Err("Braess state parameter out of bounds".into());
        }
        Ok(())
    }
}

/// Equilibrium, social optimum, and Nash verification metrics on the Wheatstone network.
#[derive(Clone, Copy, Debug, Default)]
pub struct BraessMetrics {
    /// Selfish Wardrop route flows `[upper, lower, zig_zag]`.
    pub wardrop_flow: [f64; 3],
    /// Social-optimum Pigou route flows `[upper, lower, zig_zag]`.
    pub pigou_flow: [f64; 3],
    /// Average commuter physical travel time at Wardrop equilibrium.
    pub wardrop_latency: f64,
    /// Average commuter travel time when shortcut `A->B` is closed.
    pub closed_latency: f64,
    /// Minimum possible average physical travel time (Pigou optimum).
    pub pigou_latency: f64,
    /// Optimal Pigouvian toll `tau*` restoring social efficiency.
    pub optimal_toll: f64,
    /// Price of Anarchy `wardrop_latency / pigou_latency`.
    pub price_of_anarchy: f64,
    /// Verified via `BoxSet`, `BestResponseCorrespondence`, and `is_fixed_point`.
    pub nash_fixed_point: bool,
    /// Effective CFL diffusion ratio `nu * dt / dx^2` (`< 0.25`).
    pub mfg_cfl_ratio: f64,
}

/// Interactive Braess's Paradox and Mean-Field Crowd Congestion tool.
pub struct BraessNetworkParadoxTool {
    pub(super) state: BraessState,
    pub(super) metrics: BraessMetrics,
    pub(super) revision: u64,
    pub(super) dynamic_flow: [f64; 3],
    pub(super) curve_wardrop: Vec<[f64; 2]>,
    pub(super) curve_closed: Vec<[f64; 2]>,
    pub(super) curve_pigou: Vec<[f64; 2]>,
    pub(super) mfg_u0: Vec<[f64; 2]>,
    pub(super) mfg_m0: Vec<[f64; 2]>,
    pub(super) mfg_m_mid: Vec<[f64; 2]>,
    pub(super) mfg_m_final: Vec<[f64; 2]>,
    /// Progressive 2D parameter regime atlas controller.
    pub atlas: ParameterAtlasController,
    /// Plot callout overlay manager synchronized with stories, challenges, and atlas teleports.
    pub callouts: CalloutManager,
}

pub(super) fn avg_physical_latency(flow: [f64; 3], s: &BraessState) -> f64 {
    braess_avg_physical_latency(flow, &s.wheatstone_params())
}

pub(super) fn solve_flows_for_demand(s: &BraessState, demand: f64) -> ([f64; 3], [f64; 3], f64) {
    solve_braess_wheatstone_flows(&s.wheatstone_params(), demand)
}

/// Checks whether `candidate` is a Nash / Wardrop fixed point using `BoxSet` and `BestResponseCorrespondence`.
#[cfg(test)]
pub fn check_braess_fixed_point(state: &BraessState, candidate: [f64; 3]) -> bool {
    check_braess_wardrop_fixed_point(&state.wheatstone_params(), candidate)
}

impl Default for BraessNetworkParadoxTool {
    fn default() -> Self {
        let state = BraessState::default();
        let mut tool = Self {
            metrics: BraessMetrics::default(),
            revision: 1,
            dynamic_flow: [0.5, 0.5, 0.0],
            curve_wardrop: Vec::new(),
            curve_closed: Vec::new(),
            curve_pigou: Vec::new(),
            mfg_u0: Vec::new(),
            mfg_m0: Vec::new(),
            mfg_m_mid: Vec::new(),
            mfg_m_final: Vec::new(),
            atlas: ParameterAtlasController::new(vec![
                ("demand", "shortcut_cost"),
                ("demand", "shortcut_toll"),
            ]),
            callouts: CalloutManager::new(),
            state,
        };
        tool.recompute_all();
        tool
    }
}

impl BraessNetworkParadoxTool {
    /// Returns the latest Wheatstone network and Mean-Field Game metrics.
    pub fn metrics(&self) -> BraessMetrics {
        self.metrics
    }

    #[cfg(test)]
    pub(super) fn state_ref(&self) -> &BraessState {
        &self.state
    }

    #[cfg(test)]
    pub(super) fn apply_optimal_toll_and_mfg(&mut self, viscosity: f64, grid_points: usize) {
        self.state.shortcut_toll = self.metrics.optimal_toll;
        self.state.continuum_mfg_mode = true;
        self.state.mfg_viscosity = viscosity;
        self.state.mfg_grid_points = grid_points;
        self.recompute_all();
    }

    #[cfg(test)]
    pub(super) fn has_finite_mfg_profiles(&self) -> bool {
        self.mfg_u0.iter().all(|p| p[1].is_finite())
            && self
                .mfg_m_final
                .iter()
                .all(|p| p[1].is_finite() && p[1] >= 0.0)
    }

    /// Advances zero-allocation Smith route-switching flow toward the Wardrop equilibrium.
    pub fn step_braess_zero_alloc(&mut self) {
        self.dynamic_flow = smith_route_step(self.dynamic_flow, &self.state.wheatstone_params());
    }

    pub(super) fn recompute_all(&mut self) {
        let demand = self.state.demand;
        let (wardrop, pigou, closed_lat, wardrop_lat, pigou_lat, opt_toll, nash_ok) = {
            let s = &self.state;
            let params = s.wheatstone_params();
            let (w, p, cl) = solve_braess_wheatstone_flows(&params, demand);
            let wl = braess_avg_physical_latency(w, &params);
            let pl = braess_avg_physical_latency(p, &params);
            let toll = braess_optimal_pigou_toll(&params, demand, p[2]);
            (
                w,
                p,
                cl,
                wl,
                pl,
                toll,
                check_braess_wardrop_fixed_point(&params, w),
            )
        };
        let cfl = self.recompute_mfg_profiles();
        self.recompute_demand_curves();
        self.dynamic_flow = [0.5 * demand, 0.5 * demand, 0.0];
        for _ in 0..8 {
            self.step_braess_zero_alloc();
        }
        self.metrics = BraessMetrics {
            wardrop_flow: wardrop,
            pigou_flow: pigou,
            wardrop_latency: wardrop_lat,
            closed_latency: closed_lat,
            pigou_latency: pigou_lat,
            optimal_toll: opt_toll,
            price_of_anarchy: wardrop_lat / pigou_lat.max(1e-9),
            nash_fixed_point: nash_ok,
            mfg_cfl_ratio: cfl,
        };
        self.revision = self.revision.wrapping_add(1);
    }

    fn recompute_demand_curves(&mut self) {
        self.curve_wardrop.clear();
        self.curve_closed.clear();
        self.curve_pigou.clear();
        let samples = 41usize;
        for idx in 0..samples {
            let d = 0.2 + 1.8 * (idx as f64) / ((samples - 1) as f64);
            let mut probe = self.state.clone();
            probe.demand = d;
            let (w, p, closed) = solve_flows_for_demand(&probe, d);
            self.curve_wardrop
                .push([d, avg_physical_latency(w, &probe)]);
            self.curve_closed.push([d, closed]);
            self.curve_pigou.push([d, avg_physical_latency(p, &probe)]);
        }
    }

    fn recompute_mfg_profiles(&mut self) -> f64 {
        let nx = self.state.mfg_grid_points.clamp(17, 65);
        let nu = self.state.mfg_viscosity.clamp(0.02, 0.25);
        let dx = 2.0 / ((nx - 1) as f64);
        let nt = (((nu * 1.0) / (0.20 * dx * dx)).ceil() as usize + 2).max(40);
        let cfg = MFGConfigBuilder::new()
            .viscosity(nu)
            .time_horizon(1.0)
            .grid_points(NonZeroUsize::new(nx).unwrap())
            .time_steps(NonZeroUsize::new(nt).unwrap())
            .space_bounds(-1.0, 1.0)
            .unwrap()
            .build()
            .unwrap();

        let alpha = self.state.congestion_slope * self.state.demand;
        let beta = self.state.bypass_latency;
        let well = if self.state.shortcut_open {
            -0.35 * (1.0 - self.state.shortcut_toll).max(-1.0)
        } else {
            0.55
        };
        let running = |p: Position, d: Density| alpha * d.0 + well * (-p.0 * p.0 / 0.08).exp();
        let terminal = |p: Position, d: Density| 0.4 * beta * p.0 * p.0 + 0.2 * alpha * d.0;
        let bimodal =
            |x: f64| (-((x + 0.42).powi(2)) / 0.06).exp() + (-((x - 0.42).powi(2)) / 0.06).exp();
        let mass = (0..nx)
            .map(|i| bimodal(cfg.space_min + (i as f64) * cfg.dx))
            .sum::<f64>()
            * cfg.dx;
        let init = |p: Position| bimodal(p.0) / mass.max(1e-9);

        let (u, m) = FixedPointSolver::new(10).solve(&cfg, &running, &terminal, &init);
        for buf in [
            &mut self.mfg_u0,
            &mut self.mfg_m0,
            &mut self.mfg_m_mid,
            &mut self.mfg_m_final,
        ] {
            buf.clear();
        }
        for i in 0..nx {
            let x = cfg.space_min + (i as f64) * cfg.dx;
            self.mfg_u0.push([x, u[(i, 0)]]);
            self.mfg_m0.push([x, m[(i, 0)]]);
            self.mfg_m_mid.push([x, m[(i, nt / 2)]]);
            self.mfg_m_final.push([x, m[(i, nt)]]);
        }
        cfg.viscosity * cfg.dt / (cfg.dx * cfg.dx)
    }
}

impl InteractiveTool for BraessNetworkParadoxTool {
    fn name(&self) -> &'static str {
        "Braess's Paradox"
    }
    fn domain(&self) -> &'static str {
        "paradox_lab"
    }
    fn set_suspended(&mut self, suspended: bool) {
        self.atlas.set_suspended(suspended);
    }
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.render_braess_inside(parent_ui);
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn has_finite_parameters(&self) -> bool {
        self.state.validate().is_ok()
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }
    fn try_load_state(&mut self, json: &str) -> Result<(), String> {
        self.state_bounds().validate_json(json)?;
        let parsed: BraessState =
            serde_json::from_str(json).map_err(|e| format!("Invalid Braess state: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.recompute_all();
        Ok(())
    }
    fn clamp_parameters(&mut self) -> bool {
        self.state.demand = self.state.demand.clamp(0.2, 2.0);
        self.state.congestion_slope = self.state.congestion_slope.clamp(0.2, 3.0);
        self.state.bypass_latency = self.state.bypass_latency.clamp(0.2, 3.0);
        self.state.shortcut_cost = self.state.shortcut_cost.clamp(0.0, 1.5);
        self.state.shortcut_toll = self.state.shortcut_toll.clamp(0.0, 1.5);
        self.state.mfg_viscosity = self.state.mfg_viscosity.clamp(0.02, 0.25);
        self.state.mfg_grid_points = self.state.mfg_grid_points.clamp(17, 65);
        self.recompute_all();
        true
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        if self.state.continuum_mfg_mode {
            vec![
                ("HJB Value u(x, 0)".into(), self.mfg_u0.clone()),
                ("Initial Density m(x, 0)".into(), self.mfg_m0.clone()),
                ("Midpoint Density m(x, T/2)".into(), self.mfg_m_mid.clone()),
                ("Terminal Density m(x, T)".into(), self.mfg_m_final.clone()),
            ]
        } else {
            vec![
                ("Wardrop Selfish Latency".into(), self.curve_wardrop.clone()),
                ("Closed-Shortcut Latency".into(), self.curve_closed.clone()),
                ("Pigou Social Optimum".into(), self.curve_pigou.clone()),
            ]
        }
    }
    fn step_simulation(&mut self) {
        self.state.demand = (self.state.demand + 0.05).min(2.0);
        self.recompute_all();
    }
    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("wardrop_latency", self.metrics.wardrop_latency),
            ("pigou_latency", self.metrics.pigou_latency),
            ("closed_latency", self.metrics.closed_latency),
            ("price_of_anarchy", self.metrics.price_of_anarchy),
        ]
    }
    fn sync_callouts(&mut self, specs: &[CalloutSpec]) {
        self.callouts.sync_specs(specs);
    }
}

// [cite:paradox_lab]
