//! Unified Braess Wardrop Frank-Wolfe and Smith route-switching congestion solvers.
//!
//! Combines general directed-graph Frank-Wolfe Wardrop routing from [`pure_math::pure_math::graph_theory::wardrop`]
//! with analytical Wardrop/Pigou solvers, zero-allocation Smith route-switching dynamics, and
//! [`BestResponseCorrespondence`] fixed-point checks on the canonical 4-node Wheatstone network.

use super::{BestResponseCorrespondence, BoxSet, ConvexSet, is_fixed_point};
use nalgebra::DVector;
pub use pure_math::pure_math::graph_theory::wardrop::{
    AffineLatency, FrankWolfeScratch, WardropError, WardropSummary, solve_wardrop_in_place,
};

/// Parameters of the 4-node Wheatstone congestion routing network (`S -> {A, B} -> T` with `A -> B`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BraessWheatstoneParams {
    /// Total commuter traffic demand $D \ge 0$ from source $S$ to sink $T$.
    pub demand: f64,
    /// Linear congestion slope $\alpha > 0$ on bottleneck links $S \to A$ and $B \to T$.
    pub congestion_slope: f64,
    /// Constant bypass latency $\beta \ge 0$ on links $A \to T$ and $S \to B$.
    pub bypass_latency: f64,
    /// Base physical travel cost $c_{AB} \ge 0$ on shortcut $A \to B$.
    pub shortcut_cost: f64,
    /// Pigouvian congestion toll $\tau \ge 0$ levied on shortcut $A \to B$.
    pub shortcut_toll: f64,
    /// Whether directed central shortcut $A \to B$ is open to traffic.
    pub shortcut_open: bool,
}

impl Default for BraessWheatstoneParams {
    fn default() -> Self {
        Self {
            demand: 1.0,
            congestion_slope: 1.0,
            bypass_latency: 1.0,
            shortcut_cost: 0.0,
            shortcut_toll: 0.0,
            shortcut_open: true,
        }
    }
}

/// Evaluates `[upper, lower, zig_zag]` route costs on the Wheatstone network.
#[must_use]
#[verified_engine::verified]
pub fn braess_route_costs(
    flow: [f64; 3],
    params: &BraessWheatstoneParams,
    include_toll: bool,
) -> [f64; 3] {
    let (f_sa, f_bt) = (flow[0] + flow[2], flow[1] + flow[2]);
    let toll = if include_toll {
        params.shortcut_toll
    } else {
        0.0
    };
    let costs = [
        params.congestion_slope * f_sa + params.bypass_latency,
        params.bypass_latency + params.congestion_slope * f_bt,
        params.congestion_slope * (f_sa + f_bt) + params.shortcut_cost + toll,
    ];
    debug_assert!(costs.iter().all(|c| !c.is_nan()));
    costs
}

/// Computes average physical commuter travel time (excluding Pigouvian toll transfers).
#[must_use]
#[verified_engine::verified]
pub fn braess_avg_physical_latency(flow: [f64; 3], params: &BraessWheatstoneParams) -> f64 {
    let c = braess_route_costs(flow, params, false);
    let latency = (flow[0] * c[0] + flow[1] * c[1] + flow[2] * c[2]) / params.demand.max(1e-9);
    debug_assert!(!latency.is_nan());
    latency
}

/// Computes `(wardrop_flow, pigou_flow, closed_shortcut_latency)` for a given `demand`.
///
/// # Examples
///
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::game_theory::equilibrium::{
///     BraessWheatstoneParams, braess_avg_physical_latency, solve_braess_wheatstone_flows,
/// };
///
/// let params = BraessWheatstoneParams::default();
/// let (wardrop, pigou, closed_lat) = solve_braess_wheatstone_flows(&params, 1.0);
/// assert!((closed_lat - 1.50).abs() < 1e-12);
/// assert!((braess_avg_physical_latency(wardrop, &params) - 2.00).abs() < 1e-12);
/// assert!((braess_avg_physical_latency(pigou, &params) - 1.50).abs() < 1e-12);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn solve_braess_wheatstone_flows(
    params: &BraessWheatstoneParams,
    demand: f64,
) -> ([f64; 3], [f64; 3], f64) {
    let slope = params.congestion_slope.max(1e-12);
    let closed_lat = 0.5 * slope * demand + params.bypass_latency;
    if !params.shortcut_open {
        let half = [0.5 * demand, 0.5 * demand, 0.0];
        return (half, half, closed_lat);
    }
    let x2_w = ((2.0 * (params.bypass_latency - params.shortcut_cost - params.shortcut_toll)
        / slope)
        - demand)
        .clamp(0.0, demand);
    let wardrop = [0.5 * (demand - x2_w), 0.5 * (demand - x2_w), x2_w];
    let x2_p =
        (((params.bypass_latency - params.shortcut_cost) / slope) - demand).clamp(0.0, demand);
    let pigou = [0.5 * (demand - x2_p), 0.5 * (demand - x2_p), x2_p];
    debug_assert!(wardrop[2] >= 0.0 && pigou[2] >= 0.0);
    (wardrop, pigou, closed_lat)
}

/// Computes the marginal-externality Pigouvian toll $\tau^*$ restoring social optimality.
#[must_use]
#[verified_engine::verified]
pub fn braess_optimal_pigou_toll(
    params: &BraessWheatstoneParams,
    demand: f64,
    pigou_zigzag_flow: f64,
) -> f64 {
    if params.shortcut_open {
        (params.bypass_latency
            - params.shortcut_cost
            - 0.5 * params.congestion_slope * (demand + pigou_zigzag_flow))
            .max(0.0)
    } else {
        0.0
    }
}

/// Advances stack-allocated Smith proportional pairwise route-switching dynamics by one step.
#[must_use]
#[verified_engine::verified]
pub fn smith_route_step(flow: [f64; 3], params: &BraessWheatstoneParams) -> [f64; 3] {
    let k = if params.shortcut_open { 3 } else { 2 };
    let costs = braess_route_costs(flow, params, true);
    let denom = (params.congestion_slope * params.demand + params.bypass_latency + 1.0).max(1e-9);
    let eta = 0.25 / denom;
    let mut next = [0.0; 3];
    for i in 0..k {
        let mut delta = 0.0;
        for j in 0..k {
            delta +=
                flow[j] * (costs[j] - costs[i]).max(0.0) - flow[i] * (costs[i] - costs[j]).max(0.0);
        }
        next[i] = (flow[i] + eta * delta).max(0.0);
    }
    let sum: f64 = next[..k].iter().sum();
    if sum > 0.0 {
        for val in &mut next[..k] {
            *val *= params.demand / sum;
        }
    }
    debug_assert!(next.iter().all(|v| v.is_finite() && *v >= 0.0));
    next
}

/// Checks whether `candidate` is a Wardrop/Nash fixed point using [`BoxSet`] and [`BestResponseCorrespondence`].
#[must_use]
#[verified_engine::verified]
pub fn check_braess_wardrop_fixed_point(
    params: &BraessWheatstoneParams,
    candidate: [f64; 3],
) -> bool {
    if !params.demand.is_finite() || params.demand < 0.0 {
        return false;
    }
    let k = if params.shortcut_open { 3 } else { 2 };
    let Ok(strategy_box) = BoxSet::try_new(vec![0.0; k], vec![params.demand; k]) else {
        return false;
    };
    let point = DVector::from_column_slice(&candidate[..k]);
    if !strategy_box.is_convex() || !strategy_box.contains(&point) {
        return false;
    }
    let captured = *params;
    let correspondence = BestResponseCorrespondence {
        mapping: Box::new(move |x: &DVector<f64>| {
            let mut arr = [0.0; 3];
            for i in 0..k.min(x.len()) {
                arr[i] = x[i];
            }
            let stepped = smith_route_step(arr, &captured);
            DVector::from_column_slice(&stepped[..k])
        }),
        tolerance: 1e-5,
    };
    is_fixed_point(&correspondence, &point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    #[verified_engine::verified]
    fn test_wheatstone_analytic_matches_frank_wolfe_and_smith_zero_alloc() {
        let params = BraessWheatstoneParams::default();
        let (wardrop, pigou, closed_lat) = solve_braess_wheatstone_flows(&params, 1.0);
        assert!(check_braess_wardrop_fixed_point(&params, wardrop));
        assert!(!check_braess_wardrop_fixed_point(&params, pigou));

        let edges = [
            (0, 1, 0.0),
            (1, 3, 1.0),
            (0, 2, 1.0),
            (2, 3, 0.0),
            (1, 2, 0.0),
        ];
        let latencies = [
            AffineLatency::clamped(1.0, 0.0),
            AffineLatency::clamped(0.0, 1.0),
            AffineLatency::clamped(0.0, 1.0),
            AffineLatency::clamped(1.0, 0.0),
            AffineLatency::clamped(0.0, 0.0),
        ];
        let mut scratch = FrankWolfeScratch::new();
        scratch.prepare_topology(0..4, &edges, &latencies);
        let mut fw_flows = [0.0; 5];
        let mut fw_lats = [0.0; 5];

        let strict_params = BraessWheatstoneParams {
            bypass_latency: 1.25,
            ..params
        };
        let _guard = crate::VERIFIED_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ((fw_open, smith_deg, smith_strict), _) = VerifiedEngine::run_verified(|| {
            let summary =
                solve_wardrop_in_place(&mut scratch, 0, 3, 1.0, None, &mut fw_flows, &mut fw_lats)
                    .unwrap();
            let mut dyn_deg = [0.5, 0.5, 0.0];
            let mut dyn_strict = [0.5, 0.5, 0.0];
            for _ in 0..900 {
                dyn_deg = smith_route_step(dyn_deg, &params);
                dyn_strict = smith_route_step(dyn_strict, &strict_params);
            }
            (summary.commuter_latency, dyn_deg, dyn_strict)
        });
        assert!((closed_lat - 1.50).abs() < 1e-9);
        assert!((fw_open - braess_avg_physical_latency(wardrop, &params)).abs() < 1e-6);
        assert!(smith_deg[2] > 0.85);
        assert!((smith_strict[2] - 1.0).abs() < 1e-6);
    }
}
