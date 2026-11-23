#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:modular_polynomials_review]
// [cite:hurwitz_kronecker]
//! Bidirectional cross-tab state handoff transferring `(p, a, b, P, Q, k)` across
//! `WeierstrassPlaygroundTool`, `ModularInvariantsTool`, and `EllipticCurvePlayground`.

use super::chord_tangent_tool::{CurveFieldMode, WeierstrassPlaygroundTool, snap_prime_i64};
use super::modular_invariants_tool::{ModularInvariantsTool, snap_prime_u64};
use crate::accessibility::AccessibleHoverText;
use crate::tabs::number_theory::elliptic_curve_playground::EllipticCurvePlayground;
use eframe::egui;
use math_explorer::pure_math::elliptic_curves::{AffinePointFp, CurvePoint};

const HANDOFF_ELLIPTIC_KEY: &str = "HANDOFF_ELLIPTIC_TORUS_PAYLOAD";

/// Destination elliptic curve tool for a one-click cross-tab handoff.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EllipticHandoffTarget {
    /// `EllipticCurvesTab` -> `WeierstrassPlaygroundTool` (Finite Torus $\mathbb{F}_p$).
    #[default]
    WeierstrassPlayground,
    /// `EllipticCurvesTab` -> `ModularInvariantsTool` (Hasse-Weil / Hurwitz / Breuer).
    ModularInvariants,
    /// `NumberTheoryTab` -> `EllipticCurvePlayground` (Torus & Hasse Arch).
    NumberTheoryTorus,
}

impl EllipticHandoffTarget {
    /// Target `ExplorerTab::name()` hosting this tool.
    #[must_use]
    pub const fn tab_name(self) -> &'static str {
        match self {
            Self::WeierstrassPlayground | Self::ModularInvariants => "Elliptic Curves",
            Self::NumberTheoryTorus => "Number Theory",
        }
    }

    /// Canonical `ToolMetadata::id` to activate on the target tab.
    #[must_use]
    pub const fn tool_id(self) -> &'static str {
        match self {
            Self::WeierstrassPlayground => "WeierstrassPlaygroundTool",
            Self::ModularInvariants => "ModularInvariantsTool",
            Self::NumberTheoryTorus => "EllipticCurvePlayground",
        }
    }
}

/// Complete `(p, a, b, P, Q, k)` algebraic state transferred across elliptic curve tools.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EllipticTorusHandoff {
    /// Destination tool to focus and hydrate.
    pub target: EllipticHandoffTarget,
    /// Prime field modulus $p \in [5, 997]$.
    pub p: i64,
    /// Short-Weierstrass linear coefficient $a$.
    pub a: i64,
    /// Short-Weierstrass constant coefficient $b$.
    pub b: i64,
    /// Primary curve point $P$.
    pub point_p: AffinePointFp,
    /// Secondary curve point $Q$.
    pub point_q: AffinePointFp,
    /// Scalar multiplier $k \ge 1$.
    pub scalar_k: i64,
}

/// Converts a `CurvePoint<i64>` into `AffinePointFp`.
#[must_use]
pub const fn curve_point_to_affine_fp(pt: CurvePoint<i64>) -> AffinePointFp {
    match pt {
        CurvePoint::Infinity => AffinePointFp::Infinity,
        CurvePoint::Affine { x, y } => AffinePointFp::Finite { x, y },
    }
}

/// Queues a pending cross-tab elliptic torus handoff on `ctx`.
pub fn request_elliptic_torus_handoff(ctx: &egui::Context, payload: EllipticTorusHandoff) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new(HANDOFF_ELLIPTIC_KEY), payload));
    ctx.request_repaint();
}

/// Consumes and clears any pending cross-tab elliptic torus handoff on `ctx`.
#[must_use]
pub fn take_elliptic_torus_handoff(ctx: &egui::Context) -> Option<EllipticTorusHandoff> {
    ctx.data_mut(|d| d.remove_temp::<EllipticTorusHandoff>(egui::Id::new(HANDOFF_ELLIPTIC_KEY)))
}

/// Applies `payload` to [`WeierstrassPlaygroundTool`], aligning `P` and `Q` lattice indices.
pub fn apply_to_weierstrass_playground(
    tool: &mut WeierstrassPlaygroundTool,
    payload: EllipticTorusHandoff,
) {
    tool.state.field_mode = CurveFieldMode::FiniteField;
    tool.state.prime_p = snap_prime_i64(payload.p);
    tool.state.mod_a = payload.a.clamp(-997, 997);
    tool.state.mod_b = payload.b.clamp(-997, 997);
    tool.state.real_a = (payload.a as f64).clamp(-20.0, 20.0);
    tool.state.real_b = (payload.b as f64).clamp(-20.0, 20.0);
    tool.state.scalar_k = payload.scalar_k.clamp(1, 1024);
    tool.rebuild_lattice();
    if let AffinePointFp::Finite { x, y } = payload.point_p
        && let Some(idx) = tool.lattice_points.iter().position(|pt| {
            pt.coords()
                == Some((
                    x.rem_euclid(tool.state.prime_p),
                    y.rem_euclid(tool.state.prime_p),
                ))
        })
    {
        tool.state.selected_p_idx = idx;
    }
    if let AffinePointFp::Finite { x, y } = payload.point_q
        && let Some(idx) = tool.lattice_points.iter().position(|pt| {
            pt.coords()
                == Some((
                    x.rem_euclid(tool.state.prime_p),
                    y.rem_euclid(tool.state.prime_p),
                ))
        })
    {
        tool.state.selected_q_idx = idx;
    }
    tool.bump_revision();
}

/// Applies `payload` to [`ModularInvariantsTool`], focusing `inspect_prime = p`.
pub fn apply_to_modular_invariants(
    tool: &mut ModularInvariantsTool,
    payload: EllipticTorusHandoff,
) {
    tool.state.coeff_a = payload.a.clamp(-50, 50);
    tool.state.coeff_b = payload.b.clamp(-50, 50);
    let snapped = snap_prime_u64(payload.p.max(5) as u64);
    tool.state.inspect_prime = snapped;
    tool.state.max_prime = tool.state.max_prime.max(snapped).clamp(11, 997);
    tool.recompute();
}

/// Applies `payload` to [`EllipticCurvePlayground`], preserving `(p, a, b, P, Q, k)`.
pub fn apply_to_number_theory_torus(
    tool: &mut EllipticCurvePlayground,
    payload: EllipticTorusHandoff,
) {
    tool.p = payload.p.clamp(5, 997);
    tool.a = payload.a.clamp(-997, 997);
    tool.b = payload.b.clamp(-997, 997);
    tool.point_p = payload.point_p;
    tool.point_q = payload.point_q;
    tool.scalar_k = payload.scalar_k.clamp(1, 2048);
    tool.hasse_p_max = tool.hasse_p_max.max(tool.p);
    tool.recompute_all();
}

/// Renders compact one-click cross-tab handoff buttons excluding `current`.
pub fn render_handoff_buttons(
    ui: &mut egui::Ui,
    current: EllipticHandoffTarget,
    base: EllipticTorusHandoff,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Handoff:");
        for (target, label, tip) in [
            (
                EllipticHandoffTarget::WeierstrassPlayground,
                "↗ Weierstrass",
                "Transfer (p, a, b, P, Q, k) to Elliptic Curves -> Weierstrass Curve Playground",
            ),
            (
                EllipticHandoffTarget::ModularInvariants,
                "↗ Modular",
                "Transfer (p, a, b) to Elliptic Curves -> Modular Forms & Class Invariants",
            ),
            (
                EllipticHandoffTarget::NumberTheoryTorus,
                "↗ NT Torus",
                "Transfer (p, a, b, P, Q, k) to Number Theory -> Elliptic Curve Playground",
            ),
        ] {
            if target == current {
                continue;
            }
            if ui.small_button(label).accessible_hover_text(tip).clicked() {
                let mut req = base;
                req.target = target;
                request_elliptic_torus_handoff(ui.ctx(), req);
            }
        }
    });
}
