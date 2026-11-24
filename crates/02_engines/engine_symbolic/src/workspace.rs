//! Fixed-capacity zero-allocation scratch workspace for reverse-mode bytecode differentiation.
// [cite:engine_symbolic]

/// Maximum number of bytecode instructions recorded in a [`SolverWorkspace`] reverse tape.
pub const MAX_WORKSPACE_TAPE: usize = 512;

/// Maximum operand stack depth during forward/reverse bytecode evaluation.
pub const MAX_WORKSPACE_STACK: usize = 128;

/// Zero-allocation inline tape and operand stack workspace for reverse-mode autodiff.
///
/// Pre-allocates fixed-size inline arrays so `CompiledExpr::eval_grad_into` executes
/// both forward primal evaluation and reverse adjoint accumulation without heap allocations
/// under `VerifiedAllocator`.
#[derive(Debug, Clone)]
pub struct SolverWorkspace {
    /// Primal value recorded for each executed tape slot.
    pub values: [f64; MAX_WORKSPACE_TAPE],
    /// Reverse-mode adjoint $\bar{v}_i = \partial y / \partial v_i$ for each tape slot.
    pub adjoints: [f64; MAX_WORKSPACE_TAPE],
    /// First child tape index for each recorded node.
    pub arg0: [u16; MAX_WORKSPACE_TAPE],
    /// Second child tape index for each recorded node.
    pub arg1: [u16; MAX_WORKSPACE_TAPE],
    /// Third child tape index for ternary operations (`Clamp`).
    pub arg2: [u16; MAX_WORKSPACE_TAPE],
    ///Scratch stack of tape node indices during forward evaluation.
    pub stack: [u16; MAX_WORKSPACE_STACK],
}

impl Default for SolverWorkspace {
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn default() -> Self {
        Self::new()
    }
}

impl SolverWorkspace {
    /// Constructs a zeroed inline [`SolverWorkspace`] on the stack or inside a solver struct.
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub const fn new() -> Self {
        Self {
            values: [0.0; MAX_WORKSPACE_TAPE],
            adjoints: [0.0; MAX_WORKSPACE_TAPE],
            arg0: [0; MAX_WORKSPACE_TAPE],
            arg1: [0; MAX_WORKSPACE_TAPE],
            arg2: [0; MAX_WORKSPACE_TAPE],
            stack: [0; MAX_WORKSPACE_STACK],
        }
    }

    /// Clears `adjoints[..tape_len]` to `0.0` prior to a reverse sweep.
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub const fn clear_adjoints(&mut self, tape_len: usize) {
        let cap = if tape_len < MAX_WORKSPACE_TAPE {
            tape_len
        } else {
            MAX_WORKSPACE_TAPE
        };
        let mut idx = 0;
        while idx < cap {
            self.adjoints[idx] = 0.0;
            idx += 1;
        }
    }
}
