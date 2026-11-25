//! Generic [`math_scalars::Scalar`] postfix bytecode evaluator for [`CompiledExpr`].
//!
//! Enables zero-allocation automatic differentiation (`Dual`, `DualVec`, `HyperDual`) and
//! outward-rounded interval evaluation (`Interval`, `Ball`) over compiled symbolic expressions.
// [cite:math_commons]

use super::compiled::{CompiledExpr, Instruction};
use super::eval::MAX_EVAL_DEPTH;
use super::expr::{BinaryOp, Func, UnaryOp};
use math_scalars::Scalar;

impl CompiledExpr {
    /// Evaluates compiled bytecode over an arbitrary slice of generic [`Scalar`] values (`None` on invalid/unsupported ops).
    #[must_use]
    pub fn eval_slice_scalar<S: Scalar>(&self, vars: &[S]) -> Option<S> {
        self.exec_scalar_bytecode(|slot| vars.get(slot).copied())
    }

    /// Evaluates a 1D formula (`slot 0 = x`, `slots 1.. = real params`) over generic [`Scalar`].
    #[must_use]
    pub fn eval_1d_scalar<S: Scalar>(&self, x: S, params: &[f64]) -> Option<S> {
        self.exec_scalar_bytecode(|slot| match slot {
            0 => Some(x),
            idx => params.get(idx - 1).copied().map(S::from_f64),
        })
    }

    /// Evaluates a 2D formula (`slot 0 = u`, `slot 1 = v`, `slots 2.. = real params`) over generic [`Scalar`].
    #[must_use]
    pub fn eval_2d_scalar<S: Scalar>(&self, u: S, v: S, params: &[f64]) -> Option<S> {
        self.exec_scalar_bytecode(|slot| match slot {
            0 => Some(u),
            1 => Some(v),
            idx => params.get(idx - 2).copied().map(S::from_f64),
        })
    }

    /// Evaluates a 3D formula (`slot 0 = x`, `slot 1 = y`, `slot 2 = z`, `slots 3.. = real params`) over generic [`Scalar`].
    #[must_use]
    pub fn eval_3d_scalar<S: Scalar>(&self, x: S, y: S, z: S, params: &[f64]) -> Option<S> {
        self.exec_scalar_bytecode(|slot| match slot {
            0 => Some(x),
            1 => Some(y),
            2 => Some(z),
            idx => params.get(idx - 3).copied().map(S::from_f64),
        })
    }

    #[inline]
    fn exec_scalar_bytecode<S: Scalar, F: Fn(usize) -> Option<S>>(&self, load_var: F) -> Option<S> {
        let mut stack = [S::ZERO; MAX_EVAL_DEPTH];
        let mut sp = 0usize;
        for &inst in self.instructions() {
            exec_scalar_inst(inst, &mut stack, &mut sp, &load_var)?;
        }
        (sp == 1).then(|| stack[0])
    }
}

#[inline]
fn exec_scalar_inst<S: Scalar, F: Fn(usize) -> Option<S>>(
    inst: Instruction,
    stack: &mut [S; MAX_EVAL_DEPTH],
    sp: &mut usize,
    load_var: &F,
) -> Option<()> {
    match inst {
        Instruction::PushConst(c) => push_scalar(stack, sp, S::from_f64(c)),
        Instruction::PushImagUnit => None,
        Instruction::LoadVar(slot) => push_scalar(stack, sp, load_var(usize::from(slot))?),
        Instruction::Unary(UnaryOp::Pos) => (*sp >= 1).then_some(()),
        Instruction::Unary(UnaryOp::Neg) => {
            if *sp < 1 {
                return None;
            }
            stack[*sp - 1] = -stack[*sp - 1];
            Some(())
        }
        Instruction::Binary(op) => {
            if *sp < 2 {
                return None;
            }
            let r = stack[*sp - 1];
            let l = stack[*sp - 2];
            stack[*sp - 2] = eval_scalar_binop(op, l, r);
            *sp -= 1;
            Some(())
        }
        Instruction::Call(func) => {
            let arity = func.arity();
            if *sp < arity {
                return None;
            }
            let base = *sp - arity;
            stack[base] = eval_scalar_func(func, &stack[base..*sp])?;
            *sp = base + 1;
            Some(())
        }
    }
}

#[inline]
fn push_scalar<S: Copy>(stack: &mut [S; MAX_EVAL_DEPTH], sp: &mut usize, val: S) -> Option<()> {
    if *sp >= MAX_EVAL_DEPTH {
        return None;
    }
    stack[*sp] = val;
    *sp += 1;
    Some(())
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

#[inline]
fn eval_scalar_func<S: Scalar>(func: Func, args: &[S]) -> Option<S> {
    let val = match func {
        Func::Sin => args[0].sin(),
        Func::Cos => args[0].cos(),
        Func::Tan => args[0].tan(),
        Func::Exp => args[0].exp(),
        Func::Ln => args[0].ln(),
        Func::Log10 => args[0].ln() / S::from_f64(core::f64::consts::LN_10),
        Func::Sqrt => args[0].sqrt(),
        Func::Abs => args[0].abs(),
        Func::Sinh => args[0].sinh(),
        Func::Cosh => args[0].cosh(),
        Func::Tanh => args[0].tanh(),
        Func::Asin => args[0].asin(),
        Func::Acos => args[0].acos(),
        Func::Atan => args[0].atan(),
        Func::Atan2 => args[0].atan2(args[1]),
        Func::Step | Func::Rect | Func::Min | Func::Max | Func::Clamp => return None,
    };
    Some(val)
}
