//! Forward-mode (`Dual`, `Jet2`) and `SolverWorkspace`-backed reverse-mode (`eval_grad_into`)
//! automatic differentiation on [`CompiledExpr`].
// [cite:math_commons]

use engine_symbolic::{
    Dual, DualComplex64, Jet2, MAX_WORKSPACE_STACK, MAX_WORKSPACE_TAPE, SolverWorkspace,
};

use super::compiled::{CompiledExpr, Instruction};
use super::eval::{EvalScalar, eval_binop, eval_func, eval_unary};
use super::expr::{BinaryOp, Func, UnaryOp};

impl CompiledExpr {
    /// Evaluates a 1D real formula (`slot 0 = x`, `slots 1.. = params`) over [`Dual<f64>`].
    #[must_use]
    pub fn eval_1d_dual(&self, x: Dual<f64>, params: &[f64]) -> Dual<f64> {
        self.exec_bytecode(|slot| match slot {
            0 => x,
            idx => params
                .get(idx - 1)
                .copied()
                .map_or(Dual::NAN, Dual::constant),
        })
    }

    /// Evaluates the bytecode over an arbitrary slice of [`Dual<f64>`] variables.
    #[must_use]
    pub fn eval_slice_dual(&self, vars: &[Dual<f64>]) -> Dual<f64> {
        self.exec_bytecode(|slot| vars.get(slot).copied().unwrap_or(Dual::NAN))
    }

    /// Evaluates primal value and partial derivative with respect to `active_slot` over `&[f64]`.
    #[must_use]
    pub fn eval_dir_dual(&self, vars: &[f64], active_slot: usize) -> Dual<f64> {
        self.exec_bytecode(|slot| match vars.get(slot) {
            Some(&v) => Dual::new(v, if slot == active_slot { 1.0 } else { 0.0 }),
            None => Dual::NAN,
        })
    }

    /// Evaluates a complex map `w = f(z)` over [`DualComplex64`] with zero allocations.
    #[must_use]
    pub fn eval_complex_z_dual(&self, z: DualComplex64) -> DualComplex64 {
        self.eval_complex_z_params_dual(z, &[])
    }

    /// Evaluates `w = f(z; params)` over [`DualComplex64`] (`slot 0 = z`, `slots 1.. = real params`).
    #[must_use]
    pub fn eval_complex_z_params_dual(&self, z: DualComplex64, params: &[f64]) -> DualComplex64 {
        self.exec_bytecode(|slot| match slot {
            0 => z,
            idx => params
                .get(idx - 1)
                .copied()
                .map_or(DualComplex64::NAN, DualComplex64::from_real),
        })
    }

    /// Evaluates a 2D real formula (`slot 0 = u`, `slot 1 = v`, `slots 2.. = params`) over [`Jet2<2>`].
    #[must_use]
    pub fn eval_2d_jet2(&self, u: Jet2<2>, v: Jet2<2>, params: &[f64]) -> Jet2<2> {
        self.exec_bytecode(|slot| match slot {
            0 => u,
            1 => v,
            idx => params
                .get(idx - 2)
                .copied()
                .map_or(Jet2::NAN, Jet2::constant),
        })
    }

    /// Evaluates a 3D real formula (`slots 0..3 = (x, y, z)`, `slots 3.. = params`) over [`Jet2<3>`].
    #[must_use]
    pub fn eval_3d_jet2(&self, x: Jet2<3>, y: Jet2<3>, z: Jet2<3>, params: &[f64]) -> Jet2<3> {
        self.exec_bytecode(|slot| match slot {
            0 => x,
            1 => y,
            2 => z,
            idx => params
                .get(idx - 3)
                .copied()
                .map_or(Jet2::NAN, Jet2::constant),
        })
    }

    /// Evaluates the bytecode over an arbitrary slice of [`Jet2<N>`] variables.
    #[must_use]
    pub fn eval_slice_jet2<const N: usize>(&self, vars: &[Jet2<N>]) -> Jet2<N> {
        self.exec_bytecode(|slot| vars.get(slot).copied().unwrap_or(Jet2::NAN))
    }

    /// Executes 1 forward tape pass and 1 reverse adjoint sweep on `workspace`, writing
    /// $\nabla_{\text{vars}} f$ into `grad_out` with zero heap allocations.
    pub fn eval_grad_into(
        &self,
        vars: &[f64],
        workspace: &mut SolverWorkspace,
        grad_out: &mut [f64],
    ) -> f64 {
        for g in grad_out.iter_mut() {
            *g = 0.0;
        }
        let insts = self.instructions();
        let n = insts.len();
        if n == 0 || n > MAX_WORKSPACE_TAPE {
            fill_nan(grad_out);
            return f64::NAN;
        }

        let Some(root_idx) = record_forward_tape(insts, vars, workspace) else {
            fill_nan(grad_out);
            return f64::NAN;
        };
        let primal = workspace.values[root_idx];
        if primal.is_nan() {
            fill_nan(grad_out);
            return f64::NAN;
        }

        workspace.clear_adjoints(n);
        workspace.adjoints[root_idx] = 1.0;

        let mut pc = n;
        while pc > 0 {
            pc -= 1;
            let bar = workspace.adjoints[pc];
            if bar == 0.0 {
                continue;
            }
            propagate_reverse_step(insts[pc], pc, bar, workspace, grad_out);
        }
        primal
    }
}

#[inline]
fn fill_nan(slice: &mut [f64]) {
    for v in slice.iter_mut() {
        *v = f64::NAN;
    }
}

fn record_forward_tape(
    insts: &[Instruction],
    vars: &[f64],
    ws: &mut SolverWorkspace,
) -> Option<usize> {
    let mut sp = 0usize;
    let mut pc = 0usize;
    while pc < insts.len() {
        match insts[pc] {
            Instruction::PushConst(c) => {
                if sp >= MAX_WORKSPACE_STACK {
                    return None;
                }
                ws.values[pc] = c;
                ws.stack[sp] = pc as u16;
                sp += 1;
            }
            Instruction::PushImagUnit => return None,
            Instruction::LoadVar(slot) => {
                if sp >= MAX_WORKSPACE_STACK {
                    return None;
                }
                ws.values[pc] = vars.get(usize::from(slot)).copied().unwrap_or(f64::NAN);
                ws.stack[sp] = pc as u16;
                sp += 1;
            }
            Instruction::Unary(op) => {
                if sp < 1 {
                    return None;
                }
                let a = ws.stack[sp - 1];
                ws.arg0[pc] = a;
                ws.values[pc] = eval_unary(op, ws.values[usize::from(a)]);
                ws.stack[sp - 1] = pc as u16;
            }
            Instruction::Binary(op) => {
                if sp < 2 {
                    return None;
                }
                let (b, a) = (ws.stack[sp - 1], ws.stack[sp - 2]);
                ws.arg0[pc] = a;
                ws.arg1[pc] = b;
                ws.values[pc] =
                    eval_binop(op, ws.values[usize::from(a)], ws.values[usize::from(b)]);
                ws.stack[sp - 2] = pc as u16;
                sp -= 1;
            }
            Instruction::Call(func) => {
                sp = record_call_tape(func, pc, sp, ws)?;
            }
        }
        pc += 1;
    }
    (sp == 1).then(|| usize::from(ws.stack[0]))
}

fn record_call_tape(func: Func, pc: usize, sp: usize, ws: &mut SolverWorkspace) -> Option<usize> {
    let arity = func.arity();
    if sp < arity {
        return None;
    }
    let base = sp - arity;
    let a = ws.stack[base];
    ws.arg0[pc] = a;
    let mut args_val = [ws.values[usize::from(a)], 0.0, 0.0];
    if arity >= 2 {
        let b = ws.stack[base + 1];
        ws.arg1[pc] = b;
        args_val[1] = ws.values[usize::from(b)];
    }
    if arity == 3 {
        let c = ws.stack[base + 2];
        ws.arg2[pc] = c;
        args_val[2] = ws.values[usize::from(c)];
    }
    ws.values[pc] = eval_func(func, &args_val[..arity]);
    ws.stack[base] = pc as u16;
    Some(base + 1)
}

fn propagate_reverse_step(
    inst: Instruction,
    pc: usize,
    bar: f64,
    ws: &mut SolverWorkspace,
    grad_out: &mut [f64],
) {
    match inst {
        Instruction::PushConst(_) | Instruction::PushImagUnit => {}
        Instruction::LoadVar(slot) => {
            let idx = usize::from(slot);
            if idx < grad_out.len() {
                grad_out[idx] += bar;
            }
        }
        Instruction::Unary(op) => {
            let i0 = usize::from(ws.arg0[pc]);
            match op {
                UnaryOp::Pos => ws.adjoints[i0] += bar,
                UnaryOp::Neg => ws.adjoints[i0] -= bar,
            }
        }
        Instruction::Binary(op) => propagate_binary(op, pc, bar, ws),
        Instruction::Call(func) => propagate_call(func, pc, bar, ws),
    }
}

fn propagate_binary(op: BinaryOp, pc: usize, bar: f64, ws: &mut SolverWorkspace) {
    let i0 = usize::from(ws.arg0[pc]);
    let i1 = usize::from(ws.arg1[pc]);
    let a = ws.values[i0];
    let b = ws.values[i1];
    let out = ws.values[pc];

    match op {
        BinaryOp::Add => {
            ws.adjoints[i0] += bar;
            ws.adjoints[i1] += bar;
        }
        BinaryOp::Sub => {
            ws.adjoints[i0] += bar;
            ws.adjoints[i1] -= bar;
        }
        BinaryOp::Mul => {
            ws.adjoints[i0] += bar * b;
            ws.adjoints[i1] += bar * a;
        }
        BinaryOp::Div => {
            let inv_b = 1.0 / b;
            ws.adjoints[i0] += bar * inv_b;
            ws.adjoints[i1] -= bar * out * inv_b;
        }
        BinaryOp::Pow => {
            let da = Dual::new(a, 1.0).powf(Dual::constant(b)).eps;
            let db = Dual::constant(a).powf(Dual::new(b, 1.0)).eps;
            ws.adjoints[i0] += bar * da;
            ws.adjoints[i1] += bar * db;
        }
    }
}

fn propagate_call(func: Func, pc: usize, bar: f64, ws: &mut SolverWorkspace) {
    let i0 = usize::from(ws.arg0[pc]);
    let a = ws.values[i0];
    let out = ws.values[pc];

    match func {
        Func::Sin => ws.adjoints[i0] += bar * a.cos(),
        Func::Cos => ws.adjoints[i0] -= bar * a.sin(),
        Func::Tan => {
            let c = a.cos();
            ws.adjoints[i0] += bar / (c * c);
        }
        Func::Exp => ws.adjoints[i0] += bar * out,
        Func::Ln => ws.adjoints[i0] += bar / a,
        Func::Log10 => ws.adjoints[i0] += bar / (a * std::f64::consts::LN_10),
        Func::Sqrt => {
            if out != 0.0 {
                ws.adjoints[i0] += 0.5 * bar / out;
            }
        }
        Func::Abs => {
            if a > 0.0 {
                ws.adjoints[i0] += bar;
            } else if a < 0.0 {
                ws.adjoints[i0] -= bar;
            }
        }
        Func::Sinh => ws.adjoints[i0] += bar * a.cosh(),
        Func::Cosh => ws.adjoints[i0] += bar * a.sinh(),
        Func::Tanh => ws.adjoints[i0] += bar * (1.0 - out * out),
        Func::Asin => ws.adjoints[i0] += bar / (1.0 - a * a).sqrt(),
        Func::Acos => ws.adjoints[i0] -= bar / (1.0 - a * a).sqrt(),
        Func::Atan => ws.adjoints[i0] += bar / (1.0 + a * a),
        Func::Step | Func::Rect => {}
        Func::Min | Func::Max | Func::Atan2 | Func::Clamp => {
            propagate_multivariate_call(func, pc, bar, ws);
        }
    }
}

fn propagate_multivariate_call(func: Func, pc: usize, bar: f64, ws: &mut SolverWorkspace) {
    let i0 = usize::from(ws.arg0[pc]);
    let i1 = usize::from(ws.arg1[pc]);
    let a = ws.values[i0];
    let b = ws.values[i1];

    match func {
        Func::Min => {
            if a < b {
                ws.adjoints[i0] += bar;
            } else if b < a {
                ws.adjoints[i1] += bar;
            } else {
                ws.adjoints[i0] += 0.5 * bar;
                ws.adjoints[i1] += 0.5 * bar;
            }
        }
        Func::Max => {
            if a > b {
                ws.adjoints[i0] += bar;
            } else if b > a {
                ws.adjoints[i1] += bar;
            } else {
                ws.adjoints[i0] += 0.5 * bar;
                ws.adjoints[i1] += 0.5 * bar;
            }
        }
        Func::Atan2 => {
            let r2 = a * a + b * b;
            if r2 != 0.0 {
                ws.adjoints[i0] += bar * b / r2;
                ws.adjoints[i1] -= bar * a / r2;
            }
        }
        Func::Clamp => {
            let i2 = usize::from(ws.arg2[pc]);
            let hi = ws.values[i2];
            let dx = Dual::variable(a)
                .clamp(Dual::constant(b), Dual::constant(hi))
                .eps;
            let dlo = Dual::constant(a)
                .clamp(Dual::variable(b), Dual::constant(hi))
                .eps;
            let dhi = Dual::constant(a)
                .clamp(Dual::constant(b), Dual::variable(hi))
                .eps;
            ws.adjoints[i0] += bar * dx;
            ws.adjoints[i1] += bar * dlo;
            ws.adjoints[i2] += bar * dhi;
        }
        _ => {}
    }
}
