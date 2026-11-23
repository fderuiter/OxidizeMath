//! Algebraic, constant-folding, and trigonometric rewrite rules for [`crate::egraph::EGraph`].
// [cite:engine_symbolic]

use alloc::vec::Vec;
use num_traits::Float;

use crate::egraph::{EGraph, MAX_EGRAPH_NODES};
use crate::sym::{SymFunc1, SymFunc2, SymOp};

#[derive(Clone, Copy)]
enum RewriteAction {
    MergeExisting(u32, u32),
    AddAndMerge(u32, SymOp),
}

#[verified_engine::verified]
fn class_const(egraph: &EGraph, cid: u32) -> Option<f64> {
    let r = egraph.find(cid) as usize;
    for node in &egraph.classes[r] {
        if let Some(v) = node.as_f64() {
            return Some(v);
        }
    }
    None
}

#[verified_engine::verified]
fn is_approx_int(v: f64, target: f64) -> bool {
    v == target
}

#[verified_engine::verified]
pub(crate) fn collect_rewrites_step(egraph: &mut EGraph) -> bool {
    let mut actions: Vec<RewriteAction> = Vec::new();
    let n_classes = egraph.classes.len();

    let mut cid = 0usize;
    while cid < n_classes {
        if egraph.parents[cid] as usize == cid {
            let snapshot = egraph.classes[cid].clone();
            for node in &snapshot {
                collect_node_rewrites(egraph, cid as u32, node, &mut actions);
            }
        }
        cid += 1;
    }

    let mut changed = false;
    for act in actions {
        match act {
            RewriteAction::MergeExisting(a, b) => {
                if egraph.union(a, b) {
                    changed = true;
                }
            }
            RewriteAction::AddAndMerge(target, op) => {
                if egraph.node_count() >= MAX_EGRAPH_NODES {
                    break;
                }
                if let Some(new_id) = egraph.add(op)
                    && egraph.union(target, new_id)
                {
                    changed = true;
                }
            }
        }
    }
    changed
}

#[verified_engine::verified]
fn collect_node_rewrites(
    egraph: &EGraph,
    cid: u32,
    node: &SymOp,
    actions: &mut Vec<RewriteAction>,
) {
    if let Some(folded) = try_constant_fold(egraph, node)
        && folded.is_finite()
    {
        actions.push(RewriteAction::AddAndMerge(cid, SymOp::num(folded)));
        return;
    }

    match *node {
        SymOp::Neg(a) => rewrite_neg(egraph, cid, a, actions),
        SymOp::Add([a, b]) => rewrite_add(egraph, cid, a, b, actions),
        SymOp::Sub([a, b]) => rewrite_sub(egraph, cid, a, b, actions),
        SymOp::Mul([a, b]) => rewrite_mul(egraph, cid, a, b, actions),
        SymOp::Div([a, b]) => rewrite_div(egraph, cid, a, b, actions),
        SymOp::Pow([a, b]) => rewrite_pow(egraph, cid, a, b, actions),
        SymOp::Func1(f, a) => rewrite_func1(egraph, cid, f, a, actions),
        SymOp::Func2(_, [a, b]) if egraph.find(a) == egraph.find(b) => {
            actions.push(RewriteAction::MergeExisting(cid, a));
        }
        _ => {}
    }
}

#[verified_engine::verified]
fn try_constant_fold(egraph: &EGraph, node: &SymOp) -> Option<f64> {
    match *node {
        SymOp::Const(_) | SymOp::Symbol(_) => None,
        SymOp::Neg(a) => Some(-class_const(egraph, a)?),
        SymOp::Add([a, b]) => Some(class_const(egraph, a)? + class_const(egraph, b)?),
        SymOp::Sub([a, b]) => Some(class_const(egraph, a)? - class_const(egraph, b)?),
        SymOp::Mul([a, b]) => Some(class_const(egraph, a)? * class_const(egraph, b)?),
        SymOp::Div([a, b]) => {
            let rhs = class_const(egraph, b)?;
            (rhs != 0.0).then(|| class_const(egraph, a).unwrap_or(f64::NAN) / rhs)
        }
        SymOp::Pow([a, b]) => Some(Float::powf(
            class_const(egraph, a)?,
            class_const(egraph, b)?,
        )),
        SymOp::Func1(f, a) => Some(fold_func1(f, class_const(egraph, a)?)),
        SymOp::Func2(f, [a, b]) => {
            let (u, v) = (class_const(egraph, a)?, class_const(egraph, b)?);
            Some(match f {
                SymFunc2::Min => u.min(v),
                SymFunc2::Max => u.max(v),
                SymFunc2::Atan2 => Float::atan2(u, v),
            })
        }
        SymOp::Clamp([x, lo, hi]) => {
            let (xv, lov, hiv) = (
                class_const(egraph, x)?,
                class_const(egraph, lo)?,
                class_const(egraph, hi)?,
            );
            (lov <= hiv).then(|| xv.max(lov).min(hiv))
        }
    }
}

#[verified_engine::verified]
fn fold_func1(f: SymFunc1, x: f64) -> f64 {
    match f {
        SymFunc1::Sin => Float::sin(x),
        SymFunc1::Cos => Float::cos(x),
        SymFunc1::Tan => Float::tan(x),
        SymFunc1::Exp => Float::exp(x),
        SymFunc1::Ln => Float::ln(x),
        SymFunc1::Log10 => Float::log10(x),
        SymFunc1::Sqrt => Float::sqrt(x),
        SymFunc1::Abs => Float::abs(x),
        SymFunc1::Sinh => Float::sinh(x),
        SymFunc1::Cosh => Float::cosh(x),
        SymFunc1::Tanh => Float::tanh(x),
        SymFunc1::Asin => Float::asin(x),
        SymFunc1::Acos => Float::acos(x),
        SymFunc1::Atan => Float::atan(x),
        SymFunc1::Step => {
            if x >= 0.0 {
                1.0
            } else {
                0.0
            }
        }
        SymFunc1::Rect => {
            if Float::abs(x) <= 0.5 {
                1.0
            } else {
                0.0
            }
        }
    }
}

#[verified_engine::verified]
fn rewrite_neg(egraph: &EGraph, cid: u32, a: u32, actions: &mut Vec<RewriteAction>) {
    let ra = egraph.find(a) as usize;
    for sub in &egraph.classes[ra] {
        if let SymOp::Neg(inner) = *sub {
            actions.push(RewriteAction::MergeExisting(cid, inner));
        }
    }
}

#[verified_engine::verified]
fn rewrite_add(egraph: &EGraph, cid: u32, a: u32, b: u32, actions: &mut Vec<RewriteAction>) {
    let ca = class_const(egraph, a);
    let cb = class_const(egraph, b);
    if ca.is_some_and(|v| is_approx_int(v, 0.0)) {
        actions.push(RewriteAction::MergeExisting(cid, b));
        return;
    }
    if cb.is_some_and(|v| is_approx_int(v, 0.0)) {
        actions.push(RewriteAction::MergeExisting(cid, a));
        return;
    }
    let ra = egraph.find(a);
    let rb = egraph.find(b);
    for sub in &egraph.classes[rb as usize] {
        if let SymOp::Neg(inner) = *sub {
            if egraph.find(inner) == ra {
                actions.push(RewriteAction::AddAndMerge(cid, SymOp::num(0.0)));
            } else {
                actions.push(RewriteAction::AddAndMerge(cid, SymOp::Sub([ra, inner])));
            }
        }
    }
    for sub in &egraph.classes[ra as usize] {
        if let SymOp::Neg(inner) = *sub {
            if egraph.find(inner) == rb {
                actions.push(RewriteAction::AddAndMerge(cid, SymOp::num(0.0)));
            } else {
                actions.push(RewriteAction::AddAndMerge(cid, SymOp::Sub([rb, inner])));
            }
        }
    }
    if check_pythagorean_trig(egraph, ra, rb) {
        actions.push(RewriteAction::AddAndMerge(cid, SymOp::num(1.0)));
    }
}

#[verified_engine::verified]
fn check_pythagorean_trig(egraph: &EGraph, ra: u32, rb: u32) -> bool {
    let sin_arg_a = extract_trig_sq_arg(egraph, ra, SymFunc1::Sin);
    let cos_arg_b = extract_trig_sq_arg(egraph, rb, SymFunc1::Cos);
    if let (Some(u), Some(v)) = (sin_arg_a, cos_arg_b)
        && u == v
    {
        return true;
    }
    let cos_arg_a = extract_trig_sq_arg(egraph, ra, SymFunc1::Cos);
    let sin_arg_b = extract_trig_sq_arg(egraph, rb, SymFunc1::Sin);
    matches!((cos_arg_a, sin_arg_b), (Some(u), Some(v)) if u == v)
}

#[verified_engine::verified]
fn extract_trig_sq_arg(egraph: &EGraph, cid: u32, trig: SymFunc1) -> Option<u32> {
    let r = egraph.find(cid) as usize;
    for node in &egraph.classes[r] {
        match *node {
            SymOp::Pow([base, exp]) if class_const(egraph, exp) == Some(2.0) => {
                let rb = egraph.find(base) as usize;
                for bnode in &egraph.classes[rb] {
                    if let SymOp::Func1(f, arg) = *bnode
                        && f == trig
                    {
                        return Some(egraph.find(arg));
                    }
                }
            }
            SymOp::Mul([u, v]) if egraph.find(u) == egraph.find(v) => {
                let ru = egraph.find(u) as usize;
                for unode in &egraph.classes[ru] {
                    if let SymOp::Func1(f, arg) = *unode
                        && f == trig
                    {
                        return Some(egraph.find(arg));
                    }
                }
            }
            _ => {}
        }
    }
    None
}

#[verified_engine::verified]
fn rewrite_sub(egraph: &EGraph, cid: u32, a: u32, b: u32, actions: &mut Vec<RewriteAction>) {
    if egraph.find(a) == egraph.find(b) {
        actions.push(RewriteAction::AddAndMerge(cid, SymOp::num(0.0)));
        return;
    }
    if class_const(egraph, b).is_some_and(|v| is_approx_int(v, 0.0)) {
        actions.push(RewriteAction::MergeExisting(cid, a));
        return;
    }
    if class_const(egraph, a).is_some_and(|v| is_approx_int(v, 0.0)) {
        actions.push(RewriteAction::AddAndMerge(cid, SymOp::Neg(b)));
    }
}

#[verified_engine::verified]
fn rewrite_mul(egraph: &EGraph, cid: u32, a: u32, b: u32, actions: &mut Vec<RewriteAction>) {
    let ca = class_const(egraph, a);
    let cb = class_const(egraph, b);
    if ca.is_some_and(|v| is_approx_int(v, 0.0)) || cb.is_some_and(|v| is_approx_int(v, 0.0)) {
        actions.push(RewriteAction::AddAndMerge(cid, SymOp::num(0.0)));
        return;
    }
    if ca.is_some_and(|v| is_approx_int(v, 1.0)) {
        actions.push(RewriteAction::MergeExisting(cid, b));
        return;
    }
    if cb.is_some_and(|v| is_approx_int(v, 1.0)) {
        actions.push(RewriteAction::MergeExisting(cid, a));
        return;
    }
    if ca.is_some_and(|v| is_approx_int(v, -1.0)) {
        actions.push(RewriteAction::AddAndMerge(cid, SymOp::Neg(b)));
        return;
    }
    if cb.is_some_and(|v| is_approx_int(v, -1.0)) {
        actions.push(RewriteAction::AddAndMerge(cid, SymOp::Neg(a)));
    }
}

#[verified_engine::verified]
fn rewrite_div(egraph: &EGraph, cid: u32, a: u32, b: u32, actions: &mut Vec<RewriteAction>) {
    let ca = class_const(egraph, a);
    let cb = class_const(egraph, b);
    if cb.is_some_and(|v| is_approx_int(v, 1.0)) {
        actions.push(RewriteAction::MergeExisting(cid, a));
        return;
    }
    if ca.is_some_and(|v| is_approx_int(v, 0.0)) && cb.is_none_or(|v| v != 0.0) {
        actions.push(RewriteAction::AddAndMerge(cid, SymOp::num(0.0)));
    }
}

#[verified_engine::verified]
fn rewrite_pow(egraph: &EGraph, cid: u32, a: u32, b: u32, actions: &mut Vec<RewriteAction>) {
    let ca = class_const(egraph, a);
    let cb = class_const(egraph, b);
    if cb.is_some_and(|v| is_approx_int(v, 0.0)) {
        actions.push(RewriteAction::AddAndMerge(cid, SymOp::num(1.0)));
    } else if cb.is_some_and(|v| is_approx_int(v, 1.0)) {
        actions.push(RewriteAction::MergeExisting(cid, a));
    } else if ca.is_some_and(|v| is_approx_int(v, 1.0)) {
        actions.push(RewriteAction::AddAndMerge(cid, SymOp::num(1.0)));
    }
}

#[verified_engine::verified]
fn rewrite_func1(egraph: &EGraph, cid: u32, f: SymFunc1, a: u32, actions: &mut Vec<RewriteAction>) {
    if f == SymFunc1::Ln {
        let ra = egraph.find(a) as usize;
        for sub in &egraph.classes[ra] {
            if let SymOp::Func1(SymFunc1::Exp, inner) = *sub {
                actions.push(RewriteAction::MergeExisting(cid, inner));
            }
        }
    }
}
