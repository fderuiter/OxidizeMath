//! Iterative symbolic differentiation (`Expr::differentiate`) and e-graph simplification (`Expr::simplify`).
// [cite:math_commons]

use engine_symbolic::{EGraph, MAX_REWRITE_PASSES, SymExpr, SymFunc1, SymFunc2, SymOp};

use super::expr::{BinaryOp, Expr, Func, UnaryOp, is_universal_constant};

impl Expr {
    /// Simplifies this expression via bounded equality saturation (`<= 2048` e-nodes, `<= 16` passes)
    /// and minimum-cost iterative AST extraction.
    #[must_use]
    #[verified_engine::verified]
    pub fn simplify(&self) -> Self {
        let mut egraph = EGraph::new();
        let Some(root_id) = lower_expr_iterative(self, &mut egraph) else {
            return self.clone();
        };
        let _ = egraph.saturate(MAX_REWRITE_PASSES);
        let dag = egraph.extract(root_id);
        lift_sym_expr(&dag)
    }

    /// Computes the simplified symbolic derivative $\frac{\partial}{\partial \text{var}}$ of this expression,
    /// applying exact Clarke subgradients on piecewise primitives (`clamp`, `max`, `min`, `step`, `rect`, `abs`).
    #[must_use]
    #[verified_engine::verified]
    pub fn differentiate(&self, var: &str) -> Self {
        let raw = differentiate_raw_iterative(self, var);
        raw.simplify()
    }
}

#[verified_engine::verified]
fn map_func_to_sym1(func: Func) -> Option<SymFunc1> {
    match func {
        Func::Sin => Some(SymFunc1::Sin),
        Func::Cos => Some(SymFunc1::Cos),
        Func::Tan => Some(SymFunc1::Tan),
        Func::Exp => Some(SymFunc1::Exp),
        Func::Ln => Some(SymFunc1::Ln),
        Func::Log10 => Some(SymFunc1::Log10),
        Func::Sqrt => Some(SymFunc1::Sqrt),
        Func::Abs => Some(SymFunc1::Abs),
        Func::Sinh => Some(SymFunc1::Sinh),
        Func::Cosh => Some(SymFunc1::Cosh),
        Func::Tanh => Some(SymFunc1::Tanh),
        Func::Asin => Some(SymFunc1::Asin),
        Func::Acos => Some(SymFunc1::Acos),
        Func::Atan => Some(SymFunc1::Atan),
        Func::Step => Some(SymFunc1::Step),
        Func::Rect => Some(SymFunc1::Rect),
        Func::Min | Func::Max | Func::Atan2 | Func::Clamp => None,
    }
}

#[verified_engine::verified]
fn map_sym1_to_func(sym: SymFunc1) -> Func {
    match sym {
        SymFunc1::Sin => Func::Sin,
        SymFunc1::Cos => Func::Cos,
        SymFunc1::Tan => Func::Tan,
        SymFunc1::Exp => Func::Exp,
        SymFunc1::Ln => Func::Ln,
        SymFunc1::Log10 => Func::Log10,
        SymFunc1::Sqrt => Func::Sqrt,
        SymFunc1::Abs => Func::Abs,
        SymFunc1::Sinh => Func::Sinh,
        SymFunc1::Cosh => Func::Cosh,
        SymFunc1::Tanh => Func::Tanh,
        SymFunc1::Asin => Func::Asin,
        SymFunc1::Acos => Func::Acos,
        SymFunc1::Atan => Func::Atan,
        SymFunc1::Step => Func::Step,
        SymFunc1::Rect => Func::Rect,
    }
}

#[verified_engine::verified]
fn lower_expr_iterative(root: &Expr, egraph: &mut EGraph) -> Option<u32> {
    let mut stack: Vec<(&Expr, bool)> = vec![(root, false)];
    let mut out_ids: Vec<u32> = Vec::new();

    while let Some((node, visited)) = stack.pop() {
        if !visited {
            stack.push((node, true));
            match node {
                Expr::Number(_) | Expr::Variable(_) => {}
                Expr::Unary { expr, .. } => stack.push((expr, false)),
                Expr::Binary { left, right, .. } => {
                    stack.push((right, false));
                    stack.push((left, false));
                }
                Expr::Function { args, .. } => {
                    for arg in args.iter().rev() {
                        stack.push((arg, false));
                    }
                }
            }
        } else {
            let id = lower_postorder_node(node, egraph, &mut out_ids)?;
            out_ids.push(id);
        }
    }
    out_ids.pop()
}

#[verified_engine::verified]
fn lower_postorder_node(node: &Expr, egraph: &mut EGraph, out_ids: &mut Vec<u32>) -> Option<u32> {
    match node {
        Expr::Number(v) => egraph.add(SymOp::num(*v)),
        Expr::Variable(name) => egraph.add_symbol(name),
        Expr::Unary { op, .. } => {
            let a = out_ids.pop()?;
            match op {
                UnaryOp::Pos => Some(a),
                UnaryOp::Neg => egraph.add(SymOp::Neg(a)),
            }
        }
        Expr::Binary { op, .. } => {
            let r = out_ids.pop()?;
            let l = out_ids.pop()?;
            let sym_op = match op {
                BinaryOp::Add => SymOp::Add([l, r]),
                BinaryOp::Sub => SymOp::Sub([l, r]),
                BinaryOp::Mul => SymOp::Mul([l, r]),
                BinaryOp::Div => SymOp::Div([l, r]),
                BinaryOp::Pow => SymOp::Pow([l, r]),
            };
            egraph.add(sym_op)
        }
        Expr::Function { func, args } => {
            if args.len() != func.arity() {
                return None;
            }
            if let Some(f1) = map_func_to_sym1(*func) {
                let a = out_ids.pop()?;
                return egraph.add(SymOp::Func1(f1, a));
            }
            match func {
                Func::Min | Func::Max | Func::Atan2 => {
                    let b = out_ids.pop()?;
                    let a = out_ids.pop()?;
                    let f2 = match func {
                        Func::Min => SymFunc2::Min,
                        Func::Max => SymFunc2::Max,
                        _ => SymFunc2::Atan2,
                    };
                    egraph.add(SymOp::Func2(f2, [a, b]))
                }
                Func::Clamp => {
                    let hi = out_ids.pop()?;
                    let lo = out_ids.pop()?;
                    let x = out_ids.pop()?;
                    egraph.add(SymOp::Clamp([x, lo, hi]))
                }
                _ => None,
            }
        }
    }
}

#[verified_engine::verified]
fn lift_sym_expr(dag: &SymExpr) -> Expr {
    if dag.nodes.is_empty() {
        return Expr::Number(0.0);
    }
    let mut built: Vec<Expr> = Vec::with_capacity(dag.nodes.len());
    for op in &dag.nodes {
        built.push(lift_sym_node(*op, &dag.symbols, &built));
    }
    built
        .get(dag.root as usize)
        .cloned()
        .unwrap_or(Expr::Number(0.0))
}

#[verified_engine::verified]
fn lift_sym_node(op: SymOp, symbols: &[String], built: &[Expr]) -> Expr {
    let at = |idx: u32| built[idx as usize].clone();
    match op {
        SymOp::Const(bits) => Expr::Number(f64::from_bits(bits)),
        SymOp::Symbol(sym_id) => Expr::Variable(
            symbols
                .get(sym_id as usize)
                .cloned()
                .unwrap_or_else(|| "x".to_string()),
        ),
        SymOp::Neg(a) => Expr::unary(UnaryOp::Neg, at(a)),
        SymOp::Add([a, b]) => Expr::binary(BinaryOp::Add, at(a), at(b)),
        SymOp::Sub([a, b]) => Expr::binary(BinaryOp::Sub, at(a), at(b)),
        SymOp::Mul([a, b]) => Expr::binary(BinaryOp::Mul, at(a), at(b)),
        SymOp::Div([a, b]) => Expr::binary(BinaryOp::Div, at(a), at(b)),
        SymOp::Pow([a, b]) => Expr::binary(BinaryOp::Pow, at(a), at(b)),
        SymOp::Func1(f1, a) => Expr::function(map_sym1_to_func(f1), vec![at(a)]),
        SymOp::Func2(f2, [a, b]) => {
            let func = match f2 {
                SymFunc2::Min => Func::Min,
                SymFunc2::Max => Func::Max,
                SymFunc2::Atan2 => Func::Atan2,
            };
            Expr::function(func, vec![at(a), at(b)])
        }
        SymOp::Clamp([x, lo, hi]) => Expr::function(Func::Clamp, vec![at(x), at(lo), at(hi)]),
    }
}

#[inline]
#[verified_engine::verified]
fn is_zero_expr(e: &Expr) -> bool {
    matches!(e, Expr::Number(v) if *v == 0.0)
}

#[verified_engine::verified]
fn differentiate_raw_iterative(root: &Expr, var: &str) -> Expr {
    let mut stack: Vec<(&Expr, bool)> = vec![(root, false)];
    let mut derivs: Vec<Expr> = Vec::new();

    while let Some((node, visited)) = stack.pop() {
        if !visited {
            stack.push((node, true));
            match node {
                Expr::Number(_) | Expr::Variable(_) => {}
                Expr::Unary { expr, .. } => stack.push((expr, false)),
                Expr::Binary { left, right, .. } => {
                    stack.push((right, false));
                    stack.push((left, false));
                }
                Expr::Function { args, .. } => {
                    for arg in args.iter().rev() {
                        stack.push((arg, false));
                    }
                }
            }
        } else {
            let d = diff_postorder_node(node, var, &mut derivs);
            derivs.push(d);
        }
    }
    derivs.pop().unwrap_or(Expr::Number(0.0))
}

#[verified_engine::verified]
fn diff_postorder_node(node: &Expr, var: &str, derivs: &mut Vec<Expr>) -> Expr {
    match node {
        Expr::Number(_) => Expr::Number(0.0),
        Expr::Variable(name) => {
            if !is_universal_constant(name) && name == var {
                Expr::Number(1.0)
            } else {
                Expr::Number(0.0)
            }
        }
        Expr::Unary { op, .. } => {
            let du = derivs.pop().unwrap_or(Expr::Number(0.0));
            match op {
                UnaryOp::Pos => du,
                UnaryOp::Neg => Expr::unary(UnaryOp::Neg, du),
            }
        }
        Expr::Binary { op, left, right } => {
            let dv = derivs.pop().unwrap_or(Expr::Number(0.0));
            let du = derivs.pop().unwrap_or(Expr::Number(0.0));
            diff_binary(*op, left, right, du, dv)
        }
        Expr::Function { func, args } => {
            let n = args.len();
            let split = derivs.len().saturating_sub(n);
            let d_args: Vec<Expr> = derivs.drain(split..).collect();
            diff_function(*func, args, d_args)
        }
    }
}

#[verified_engine::verified]
fn diff_binary(op: BinaryOp, u: &Expr, v: &Expr, du: Expr, dv: Expr) -> Expr {
    match op {
        BinaryOp::Add => Expr::binary(BinaryOp::Add, du, dv),
        BinaryOp::Sub => Expr::binary(BinaryOp::Sub, du, dv),
        BinaryOp::Mul => Expr::binary(
            BinaryOp::Add,
            Expr::binary(BinaryOp::Mul, du, v.clone()),
            Expr::binary(BinaryOp::Mul, u.clone(), dv),
        ),
        BinaryOp::Div => {
            let num = Expr::binary(
                BinaryOp::Sub,
                Expr::binary(BinaryOp::Mul, du, v.clone()),
                Expr::binary(BinaryOp::Mul, u.clone(), dv),
            );
            let den = Expr::binary(BinaryOp::Pow, v.clone(), Expr::Number(2.0));
            Expr::binary(BinaryOp::Div, num, den)
        }
        BinaryOp::Pow => {
            if is_zero_expr(&du) && is_zero_expr(&dv) {
                return Expr::Number(0.0);
            }
            if is_zero_expr(&dv) {
                let exp_m1 = Expr::binary(BinaryOp::Sub, v.clone(), Expr::Number(1.0));
                let u_pow = Expr::binary(BinaryOp::Pow, u.clone(), exp_m1);
                return Expr::binary(
                    BinaryOp::Mul,
                    Expr::binary(BinaryOp::Mul, v.clone(), u_pow),
                    du,
                );
            }
            let self_pow = Expr::binary(BinaryOp::Pow, u.clone(), v.clone());
            let ln_u = Expr::function(Func::Ln, vec![u.clone()]);
            let t1 = Expr::binary(BinaryOp::Mul, dv, ln_u);
            let t2 = Expr::binary(
                BinaryOp::Div,
                Expr::binary(BinaryOp::Mul, v.clone(), du),
                u.clone(),
            );
            Expr::binary(BinaryOp::Mul, self_pow, Expr::binary(BinaryOp::Add, t1, t2))
        }
    }
}

#[verified_engine::verified]
fn clarke_gt_indicator(a: &Expr, b: &Expr) -> Expr {
    let a_minus_b = Expr::binary(BinaryOp::Sub, a.clone(), b.clone());
    let b_minus_a = Expr::binary(BinaryOp::Sub, b.clone(), a.clone());
    let s1 = Expr::function(Func::Step, vec![a_minus_b]);
    let s2 = Expr::binary(
        BinaryOp::Sub,
        Expr::Number(1.0),
        Expr::function(Func::Step, vec![b_minus_a]),
    );
    Expr::binary(
        BinaryOp::Mul,
        Expr::Number(0.5),
        Expr::binary(BinaryOp::Add, s1, s2),
    )
}

#[verified_engine::verified]
fn diff_max_clarke(a: &Expr, b: &Expr, da: Expr, db: Expr) -> Expr {
    let w = clarke_gt_indicator(a, b);
    let one_minus_w = Expr::binary(BinaryOp::Sub, Expr::Number(1.0), w.clone());
    Expr::binary(
        BinaryOp::Add,
        Expr::binary(BinaryOp::Mul, w, da),
        Expr::binary(BinaryOp::Mul, one_minus_w, db),
    )
}

#[verified_engine::verified]
fn diff_function(func: Func, args: &[Expr], mut d_args: Vec<Expr>) -> Expr {
    if args.len() != func.arity() || d_args.iter().all(is_zero_expr) {
        return Expr::Number(0.0);
    }
    match func {
        Func::Step | Func::Rect => Expr::Number(0.0),
        Func::Min => {
            let db = d_args.pop().unwrap_or(Expr::Number(0.0));
            let da = d_args.pop().unwrap_or(Expr::Number(0.0));
            diff_max_clarke(&args[1], &args[0], da, db)
        }
        Func::Max => {
            let db = d_args.pop().unwrap_or(Expr::Number(0.0));
            let da = d_args.pop().unwrap_or(Expr::Number(0.0));
            diff_max_clarke(&args[0], &args[1], da, db)
        }
        Func::Clamp => {
            let d_hi = d_args.pop().unwrap_or(Expr::Number(0.0));
            let d_lo = d_args.pop().unwrap_or(Expr::Number(0.0));
            let dx = d_args.pop().unwrap_or(Expr::Number(0.0));
            let max_x_lo = Expr::function(Func::Max, vec![args[0].clone(), args[1].clone()]);
            let d_max = diff_max_clarke(&args[0], &args[1], dx, d_lo);
            diff_max_clarke(&args[2], &max_x_lo, d_max, d_hi)
        }
        Func::Atan2 => {
            let dx = d_args.pop().unwrap_or(Expr::Number(0.0));
            let dy = d_args.pop().unwrap_or(Expr::Number(0.0));
            let (y, x) = (&args[0], &args[1]);
            let num = Expr::binary(
                BinaryOp::Sub,
                Expr::binary(BinaryOp::Mul, x.clone(), dy),
                Expr::binary(BinaryOp::Mul, y.clone(), dx),
            );
            let den = Expr::binary(
                BinaryOp::Add,
                Expr::binary(BinaryOp::Pow, x.clone(), Expr::Number(2.0)),
                Expr::binary(BinaryOp::Pow, y.clone(), Expr::Number(2.0)),
            );
            Expr::binary(BinaryOp::Div, num, den)
        }
        _ => diff_unary_function(func, &args[0], d_args.pop().unwrap_or(Expr::Number(0.0))),
    }
}

#[verified_engine::verified]
fn sq_expr(e: Expr) -> Expr {
    Expr::binary(BinaryOp::Pow, e, Expr::Number(2.0))
}

#[verified_engine::verified]
fn recip_expr(den: Expr) -> Expr {
    Expr::binary(BinaryOp::Div, Expr::Number(1.0), den)
}

#[verified_engine::verified]
fn diff_unary_function(func: Func, u: &Expr, du: Expr) -> Expr {
    let uc = u.clone();
    let local = match func {
        Func::Sin => Expr::function(Func::Cos, vec![uc]),
        Func::Cos => Expr::unary(UnaryOp::Neg, Expr::function(Func::Sin, vec![uc])),
        Func::Tan => recip_expr(sq_expr(Expr::function(Func::Cos, vec![uc]))),
        Func::Exp => Expr::function(Func::Exp, vec![uc]),
        Func::Ln => recip_expr(uc),
        Func::Log10 => recip_expr(Expr::binary(
            BinaryOp::Mul,
            uc,
            Expr::Number(std::f64::consts::LN_10),
        )),
        Func::Sqrt => recip_expr(Expr::binary(
            BinaryOp::Mul,
            Expr::Number(2.0),
            Expr::function(Func::Sqrt, vec![uc]),
        )),
        Func::Abs => Expr::binary(
            BinaryOp::Sub,
            Expr::function(Func::Step, vec![uc.clone()]),
            Expr::function(Func::Step, vec![Expr::unary(UnaryOp::Neg, uc)]),
        ),
        Func::Sinh => Expr::function(Func::Cosh, vec![uc]),
        Func::Cosh => Expr::function(Func::Sinh, vec![uc]),
        Func::Tanh => Expr::binary(
            BinaryOp::Sub,
            Expr::Number(1.0),
            sq_expr(Expr::function(Func::Tanh, vec![uc])),
        ),
        Func::Asin | Func::Acos | Func::Atan => diff_inverse_trig(func, uc),
        _ => Expr::Number(0.0),
    };
    Expr::binary(BinaryOp::Mul, local, du)
}

#[verified_engine::verified]
fn diff_inverse_trig(func: Func, uc: Expr) -> Expr {
    match func {
        Func::Asin | Func::Acos => {
            let rad = Expr::binary(BinaryOp::Sub, Expr::Number(1.0), sq_expr(uc));
            let inv = recip_expr(Expr::function(Func::Sqrt, vec![rad]));
            if func == Func::Acos {
                Expr::unary(UnaryOp::Neg, inv)
            } else {
                inv
            }
        }
        Func::Atan => recip_expr(Expr::binary(BinaryOp::Add, Expr::Number(1.0), sq_expr(uc))),
        _ => Expr::Number(0.0),
    }
}
