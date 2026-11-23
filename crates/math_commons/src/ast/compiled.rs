//! Flat postfix stack bytecode compiler and zero-allocation iterative evaluator.
// [cite:math_commons]

use super::eval::{
    EvalError, EvalScalar, MAX_EVAL_DEPTH, eval_binop, eval_func, eval_unary,
    lookup_universal_real_const,
};
use super::expr::{BinaryOp, Expr, Func, UnaryOp};
use num_complex::Complex64;

/// Single postfix bytecode instruction executed by [`CompiledExpr`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Instruction {
    /// Push a literal real constant onto the evaluation stack.
    PushConst(f64),
    /// Push the imaginary unit $i$ (`0 + 1i` in complex evaluation, `NaN` in real evaluation).
    PushImagUnit,
    /// Load the variable value stored at the indexed variable slot.
    LoadVar(u16),
    /// Apply a unary operator to the top stack operand.
    Unary(UnaryOp),
    /// Apply a binary operator to the top two stack operands.
    Binary(BinaryOp),
    /// Invoke a built-in function on the top `func.arity()` stack operands.
    Call(Func),
}

/// Zero-allocation postfix stack bytecode compiled from an [`Expr`].
///
/// Variables are resolved once at compile time into contiguous `u16` slot indices:
/// `0..axis_count` store explicit coordinate axes (matched case-insensitively), followed by
/// discovered free parameters in left-to-right source order.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledExpr {
    instructions: Vec<Instruction>,
    var_names: Vec<String>,
    axis_count: usize,
}

enum CompileFrame<'a> {
    Visit { expr: &'a Expr, depth: usize },
    Emit(Instruction),
}

impl Expr {
    /// Compiles this AST into flat postfix bytecode with auto-discovered variable slots.
    pub fn compile(&self) -> Result<CompiledExpr, EvalError> {
        CompiledExpr::compile(self)
    }
}

impl CompiledExpr {
    /// Compiles an [`Expr`] with all non-constant identifiers assigned to indexed variable slots.
    pub fn compile(expr: &Expr) -> Result<Self, EvalError> {
        Self::compile_internal(expr, &[], &[], false)
    }

    /// Compiles an [`Expr`] with `axes` bound to leading slots `0..axes.len()` and remaining
    /// free parameters assigned to subsequent slots in source order.
    pub fn compile_with_vars(expr: &Expr, axes: &[&str]) -> Result<Self, EvalError> {
        Self::compile_internal(expr, axes, &[], false)
    }

    /// Compiles an [`Expr`] with `axes` at `0..axes.len()` and `shared_params` pre-bound to
    /// `axes.len()..axes.len() + shared_params.len()` so coupled formulas share parameter slots.
    pub fn compile_with_vars_and_params(
        expr: &Expr,
        axes: &[&str],
        shared_params: &[&str],
    ) -> Result<Self, EvalError> {
        Self::compile_internal(expr, axes, shared_params, false)
    }

    /// Compiles a complex-valued [`Expr`] with `axes` bound to leading slots `0..axes.len()`,
    /// mapping `i` / `I` to [`Instruction::PushImagUnit`] unless bound as an explicit axis.
    pub fn compile_complex(expr: &Expr, axes: &[&str]) -> Result<Self, EvalError> {
        Self::compile_internal(expr, axes, &[], true)
    }

    fn compile_internal(
        root: &Expr,
        axes: &[&str],
        shared_params: &[&str],
        complex_mode: bool,
    ) -> Result<Self, EvalError> {
        let mut var_names: Vec<String> = axes.iter().map(|&s| s.to_string()).collect();
        let axis_count = var_names.len();
        for &param in shared_params {
            if !var_names[..axis_count]
                .iter()
                .any(|ax| ax.eq_ignore_ascii_case(param))
                && !var_names[axis_count..].iter().any(|p| p == param)
            {
                var_names.push(param.to_string());
            }
        }
        let mut instructions = Vec::new();
        let mut work = vec![CompileFrame::Visit {
            expr: root,
            depth: 0,
        }];

        while let Some(frame) = work.pop() {
            match frame {
                CompileFrame::Emit(inst) => instructions.push(inst),
                CompileFrame::Visit { expr, depth } => {
                    compile_node(
                        expr,
                        depth,
                        axis_count,
                        complex_mode,
                        &mut var_names,
                        &mut instructions,
                        &mut work,
                    )?;
                }
            }
        }

        verify_stack_depth(&instructions)?;
        Ok(Self {
            instructions,
            var_names,
            axis_count,
        })
    }

    /// Returns the compiled postfix instruction slice.
    #[must_use]
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// Returns all indexed variable names (`axes` followed by discovered free parameters).
    #[must_use]
    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }

    /// Returns the number of leading coordinate axis slots.
    #[must_use]
    pub const fn axis_count(&self) -> usize {
        self.axis_count
    }

    /// Returns the discovered free parameter names (slots `axis_count..`).
    #[must_use]
    pub fn param_names(&self) -> &[String] {
        &self.var_names[self.axis_count..]
    }

    /// Looks up the slot index assigned to `name`.
    #[must_use]
    pub fn var_slot(&self, name: &str) -> Option<usize> {
        self.var_names.iter().position(|v| v == name).or_else(|| {
            self.var_names
                .iter()
                .position(|v| v.eq_ignore_ascii_case(name))
        })
    }

    /// Evaluates the bytecode over an arbitrary slice of real variable values.
    #[must_use]
    pub fn eval_slice(&self, vars: &[f64]) -> f64 {
        self.exec_bytecode(|slot| vars.get(slot).copied().unwrap_or(f64::NAN))
    }

    /// Evaluates a 1D real formula (`slot 0 = x`, `slots 1.. = params`) with zero allocations.
    #[must_use]
    pub fn eval_1d(&self, x: f64, params: &[f64]) -> f64 {
        self.exec_bytecode(|slot| match slot {
            0 => x,
            idx => params.get(idx - 1).copied().unwrap_or(f64::NAN),
        })
    }

    /// Evaluates a 2D real formula (`slot 0 = x`, `slot 1 = y`, `slots 2.. = params`).
    #[must_use]
    pub fn eval_2d(&self, x: f64, y: f64, params: &[f64]) -> f64 {
        self.exec_bytecode(|slot| match slot {
            0 => x,
            1 => y,
            idx => params.get(idx - 2).copied().unwrap_or(f64::NAN),
        })
    }

    /// Evaluates a 3D real formula (`slot 0 = x`, `slot 1 = y`, `slot 2 = z`, `slots 3.. = params`).
    #[must_use]
    pub fn eval_3d(&self, x: f64, y: f64, z: f64, params: &[f64]) -> f64 {
        self.exec_bytecode(|slot| match slot {
            0 => x,
            1 => y,
            2 => z,
            idx => params.get(idx - 3).copied().unwrap_or(f64::NAN),
        })
    }

    /// Evaluates the bytecode over an arbitrary slice of complex variable values.
    #[must_use]
    pub fn eval_complex_slice(&self, vars: &[Complex64]) -> Complex64 {
        self.exec_bytecode(|slot| vars.get(slot).copied().unwrap_or(Complex64::NAN))
    }

    /// Evaluates a complex conformal map `w = f(z)` with no free parameters.
    #[must_use]
    pub fn eval_complex_z(&self, z: Complex64) -> Complex64 {
        self.eval_complex_z_params(z, &[])
    }

    /// Evaluates a parameterized complex map `w = f(z; params)` (`slot 0 = z`, `slots 1.. = real params`).
    #[must_use]
    pub fn eval_complex_z_params(&self, z: Complex64, params: &[f64]) -> Complex64 {
        self.exec_bytecode(|slot| match slot {
            0 => z,
            idx => params
                .get(idx - 1)
                .copied()
                .map_or(Complex64::NAN, Complex64::from_real),
        })
    }

    #[inline]
    fn exec_bytecode<T: EvalScalar, F: Fn(usize) -> T>(&self, load_var: F) -> T {
        let mut stack = [T::NAN; MAX_EVAL_DEPTH];
        let mut sp = 0usize;
        for &inst in &self.instructions {
            if !exec_instruction(inst, &mut stack, &mut sp, &load_var) {
                return T::NAN;
            }
        }
        if sp == 1 { stack[0] } else { T::NAN }
    }
}

fn compile_node<'a>(
    expr: &'a Expr,
    depth: usize,
    axis_count: usize,
    complex_mode: bool,
    var_names: &mut Vec<String>,
    instructions: &mut Vec<Instruction>,
    work: &mut Vec<CompileFrame<'a>>,
) -> Result<(), EvalError> {
    if depth > MAX_EVAL_DEPTH {
        return Err(EvalError::RecursionLimitExceeded(MAX_EVAL_DEPTH));
    }
    match expr {
        Expr::Number(n) => instructions.push(Instruction::PushConst(*n)),
        Expr::Variable(name) => {
            let inst = resolve_variable_instruction(name, axis_count, complex_mode, var_names)?;
            instructions.push(inst);
        }
        Expr::Unary { op, expr: inner } => {
            work.push(CompileFrame::Emit(Instruction::Unary(*op)));
            work.push(CompileFrame::Visit {
                expr: inner,
                depth: depth + 1,
            });
        }
        Expr::Binary { op, left, right } => {
            work.push(CompileFrame::Emit(Instruction::Binary(*op)));
            work.push(CompileFrame::Visit {
                expr: right,
                depth: depth + 1,
            });
            work.push(CompileFrame::Visit {
                expr: left,
                depth: depth + 1,
            });
        }
        Expr::Function { func, args } => compile_function_call(*func, args, depth, work)?,
    }
    Ok(())
}

fn compile_function_call<'a>(
    func: Func,
    args: &'a [Expr],
    depth: usize,
    work: &mut Vec<CompileFrame<'a>>,
) -> Result<(), EvalError> {
    if matches!(func, Func::Min | Func::Max) {
        if args.len() < 2 {
            return Err(EvalError::WrongArgumentCount {
                func,
                expected: 2,
                found: args.len(),
            });
        }
        for arg in args[1..].iter().rev() {
            work.push(CompileFrame::Emit(Instruction::Call(func)));
            work.push(CompileFrame::Visit {
                expr: arg,
                depth: depth + 1,
            });
        }
        work.push(CompileFrame::Visit {
            expr: &args[0],
            depth: depth + 1,
        });
        return Ok(());
    }
    let expected = func.arity();
    if args.len() != expected {
        return Err(EvalError::WrongArgumentCount {
            func,
            expected,
            found: args.len(),
        });
    }
    work.push(CompileFrame::Emit(Instruction::Call(func)));
    for arg in args.iter().rev() {
        work.push(CompileFrame::Visit {
            expr: arg,
            depth: depth + 1,
        });
    }
    Ok(())
}

fn resolve_variable_instruction(
    name: &str,
    axis_count: usize,
    complex_mode: bool,
    var_names: &mut Vec<String>,
) -> Result<Instruction, EvalError> {
    if let Some(c) = lookup_universal_real_const(name) {
        return Ok(Instruction::PushConst(c));
    }
    if let Some(axis_idx) = var_names[..axis_count]
        .iter()
        .position(|ax| ax.eq_ignore_ascii_case(name))
    {
        let slot = u16::try_from(axis_idx)
            .map_err(|_| EvalError::DomainError("Too many variable slots".into()))?;
        return Ok(Instruction::LoadVar(slot));
    }
    if complex_mode && name.eq_ignore_ascii_case("i") {
        return Ok(Instruction::PushImagUnit);
    }
    let slot_idx = if let Some(rel) = var_names[axis_count..].iter().position(|v| v == name) {
        axis_count + rel
    } else {
        let idx = var_names.len();
        var_names.push(name.to_string());
        idx
    };
    let slot = u16::try_from(slot_idx)
        .map_err(|_| EvalError::DomainError("Too many variable slots".into()))?;
    Ok(Instruction::LoadVar(slot))
}

fn verify_stack_depth(instructions: &[Instruction]) -> Result<(), EvalError> {
    let mut sp = 0usize;
    for &inst in instructions {
        match inst {
            Instruction::PushConst(_) | Instruction::PushImagUnit | Instruction::LoadVar(_) => {
                sp += 1;
                if sp > MAX_EVAL_DEPTH {
                    return Err(EvalError::RecursionLimitExceeded(MAX_EVAL_DEPTH));
                }
            }
            Instruction::Unary(_) => {
                if sp < 1 {
                    return Err(EvalError::DomainError("Stack underflow".into()));
                }
            }
            Instruction::Binary(_) => {
                if sp < 2 {
                    return Err(EvalError::DomainError("Stack underflow".into()));
                }
                sp -= 1;
            }
            Instruction::Call(func) => {
                let arity = func.arity();
                if sp < arity {
                    return Err(EvalError::DomainError("Stack underflow".into()));
                }
                sp = sp + 1 - arity;
            }
        }
    }
    Ok(())
}

#[inline]
fn exec_instruction<T: EvalScalar, F: Fn(usize) -> T>(
    inst: Instruction,
    stack: &mut [T; MAX_EVAL_DEPTH],
    sp: &mut usize,
    load_var: &F,
) -> bool {
    match inst {
        Instruction::PushConst(c) => push_val(stack, sp, T::from_real(c)),
        Instruction::PushImagUnit => push_val(stack, sp, T::imag_unit()),
        Instruction::LoadVar(slot) => push_val(stack, sp, load_var(usize::from(slot))),
        Instruction::Unary(op) => {
            if *sp < 1 {
                return false;
            }
            stack[*sp - 1] = eval_unary(op, stack[*sp - 1]);
            true
        }
        Instruction::Binary(op) => {
            if *sp < 2 {
                return false;
            }
            let r = stack[*sp - 1];
            let l = stack[*sp - 2];
            stack[*sp - 2] = eval_binop(op, l, r);
            *sp -= 1;
            true
        }
        Instruction::Call(func) => {
            let arity = func.arity();
            if *sp < arity {
                return false;
            }
            let base = *sp - arity;
            stack[base] = eval_func(func, &stack[base..*sp]);
            *sp = base + 1;
            true
        }
    }
}

#[inline]
fn push_val<T: Copy>(stack: &mut [T; MAX_EVAL_DEPTH], sp: &mut usize, val: T) -> bool {
    if *sp >= MAX_EVAL_DEPTH {
        return false;
    }
    stack[*sp] = val;
    *sp += 1;
    true
}
