//! Generic [`math_scalars::Scalar`] bytecode evaluator for zero-allocation AD and enclosure propagation.
// [cite:math_commons]

use math_scalars::Scalar;

use super::compiled::{CompiledExpr, Instruction};
use super::eval::MAX_EVAL_DEPTH;
use super::expr::{BinaryOp, Func, UnaryOp};

impl CompiledExpr {
    /// Evaluates the compiled postfix bytecode over an arbitrary slice of [`Scalar`] values.
    #[must_use]
    pub fn eval_slice_scalar<S: Scalar>(&self, vars: &[S]) -> S {
        let nan = S::from_f64(f64::NAN);
        self.exec_scalar(|slot| vars.get(slot).copied().unwrap_or(nan))
    }

    /// Evaluates a 1D formula (`slot 0 = x`, `slots 1.. = params`) over any [`Scalar`] `S`.
    #[must_use]
    pub fn eval_1d_scalar<S: Scalar>(&self, x: S, params: &[f64]) -> S {
        let nan = S::from_f64(f64::NAN);
        self.exec_scalar(|slot| match slot {
            0 => x,
            idx => params.get(idx - 1).copied().map_or(nan, S::from_f64),
        })
    }

    /// Evaluates a 2D formula (`slot 0 = x`, `slot 1 = y`, `slots 2.. = params`) over any [`Scalar`] `S`.
    #[must_use]
    pub fn eval_2d_scalar<S: Scalar>(&self, x: S, y: S, params: &[f64]) -> S {
        let nan = S::from_f64(f64::NAN);
        self.exec_scalar(|slot| match slot {
            0 => x,
            1 => y,
            idx => params.get(idx - 2).copied().map_or(nan, S::from_f64),
        })
    }

    /// Evaluates a 3D formula (`slot 0..3 = x, y, z`, `slots 3.. = params`) over any [`Scalar`] `S`.
    #[must_use]
    pub fn eval_3d_scalar<S: Scalar>(&self, x: S, y: S, z: S, params: &[f64]) -> S {
        let nan = S::from_f64(f64::NAN);
        self.exec_scalar(|slot| match slot {
            0 => x,
            1 => y,
            2 => z,
            idx => params.get(idx - 3).copied().map_or(nan, S::from_f64),
        })
    }

    #[inline]
    fn exec_scalar<S: Scalar, F: Fn(usize) -> S>(&self, load_var: F) -> S {
        let nan = S::from_f64(f64::NAN);
        let mut stack = [S::ZERO; MAX_EVAL_DEPTH];
        let mut sp = 0usize;
        for &inst in self.instructions() {
            if !step_scalar_inst(inst, &mut stack, &mut sp, &load_var) {
                return nan;
            }
        }
        if sp == 1 { stack[0] } else { nan }
    }
}

#[inline]
fn step_scalar_inst<S: Scalar, F: Fn(usize) -> S>(
    inst: Instruction,
    stack: &mut [S; MAX_EVAL_DEPTH],
    sp: &mut usize,
    load_var: &F,
) -> bool {
    match inst {
        Instruction::PushConst(c) => push_s(stack, sp, S::from_f64(c)),
        Instruction::PushImagUnit => push_s(stack, sp, S::from_f64(f64::NAN)),
        Instruction::LoadVar(slot) => push_s(stack, sp, load_var(usize::from(slot))),
        Instruction::Unary(op) => {
            if *sp < 1 {
                return false;
            }
            if op == UnaryOp::Neg {
                stack[*sp - 1] = -stack[*sp - 1];
            }
            true
        }
        Instruction::Binary(op) => {
            if *sp < 2 {
                return false;
            }
            let r = stack[*sp - 1];
            let l = stack[*sp - 2];
            stack[*sp - 2] = eval_scalar_binop(op, l, r);
            *sp -= 1;
            true
        }
        Instruction::Call(func) => {
            let arity = func.arity();
            if *sp < arity {
                return false;
            }
            let base = *sp - arity;
            stack[base] = eval_scalar_func(func, &stack[base..*sp]);
            *sp = base + 1;
            true
        }
    }
}

#[inline]
fn push_s<S: Copy>(stack: &mut [S; MAX_EVAL_DEPTH], sp: &mut usize, val: S) -> bool {
    if *sp >= MAX_EVAL_DEPTH {
        return false;
    }
    stack[*sp] = val;
    *sp += 1;
    true
}

#[inline]
fn eval_scalar_binop<S: Scalar>(op: BinaryOp, l: S, r: S) -> S {
    match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => l / r,
        BinaryOp::Pow => l.powf(r),
    }
}

fn eval_scalar_func<S: Scalar>(func: Func, args: &[S]) -> S {
    match func {
        Func::Sin => args[0].sin(),
        Func::Cos => args[0].cos(),
        Func::Tan => args[0].tan(),
        Func::Exp => args[0].exp(),
        Func::Ln => args[0].ln(),
        Func::Log10 => args[0].log10(),
        Func::Sqrt => args[0].sqrt(),
        Func::Abs => args[0].abs(),
        Func::Sinh => args[0].sinh(),
        Func::Cosh => args[0].cosh(),
        Func::Tanh => args[0].tanh(),
        Func::Asin => args[0].asin(),
        Func::Acos => args[0].acos(),
        Func::Atan => args[0].atan(),
        Func::Atan2 => args[0].atan2(args[1]),
        Func::Step => {
            if args[0].real_part() < 0.0 {
                S::ZERO
            } else {
                S::ONE
            }
        }
        Func::Rect => {
            if args[0].real_part().abs() <= 0.5 {
                S::ONE
            } else {
                S::ZERO
            }
        }
        Func::Min => {
            if args[0].real_part() <= args[1].real_part() {
                args[0]
            } else {
                args[1]
            }
        }
        Func::Max => {
            if args[0].real_part() >= args[1].real_part() {
                args[0]
            } else {
                args[1]
            }
        }
        Func::Clamp => {
            let (x, lo, hi) = (args[0], args[1], args[2]);
            if lo.real_part() > hi.real_part() {
                S::from_f64(f64::NAN)
            } else if x.real_part() < lo.real_part() {
                lo
            } else if x.real_part() > hi.real_part() {
                hi
            } else {
                x
            }
        }
    }
}
