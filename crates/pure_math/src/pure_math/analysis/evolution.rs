//! Unified Numerical Evolution Engine
//!
//! Provides a single trait structure merging AI optimizers and Physics solvers.
//! Enforces fixed loop bounds and centralized RNG injection.

#![cfg_attr(any(), verified(opt_out = "infrastructure"))]

use rand::RngCore;
use thiserror::Error;

/// Enumeration `EvolutionError` classifying EvolutionError variants.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum EvolutionError {
    /// Variant `ConversionError` (ConversionError).
    #[error("Failed to convert numerical value")]
    ConversionError,
    /// Variant `BoundsError` (BoundsError).
    #[error("Loop bound exceeded")]
    BoundsError,
    /// Variant `InvalidState` (InvalidState).
    #[error("Invalid state")]
    InvalidState,
    /// Variant `OptimizationError` (OptimizationError).
    #[error("Optimization error")]
    OptimizationError,
}

/// The unified numerical evolution engine trait.
/// Merges the AI optimizer and Physics solver hierarchies.
pub trait EvolutionEngine<State, AuxState> {
    /// Explicit hook for fused streaming and collision steps, or gradient updates.
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut State,
        aux: &mut AuxState,
        rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError>;

    /// Evolve the state over a fixed number of iterations.
    /// Strictly enforces fixed loop bounds via const generics to comply with NASA Power of 10.
    fn evolve<R: RngCore + ?Sized, const ITERS: usize>(
        &mut self,
        state: &mut State,
        aux: &mut AuxState,
        rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        for _ in 0..ITERS {
            self.step(state, aux, rng, dt)?;
        }
        Ok(())
    }
}

/// Trait `DoubleBufferedState` defining DoubleBufferedState operations.
pub trait DoubleBufferedState {
    /// Computes `swap_buffers` (swap buffers).
    fn swap_buffers(&mut self);
}

/// Trait `DoubleBufferedEvolutionEngine` defining DoubleBufferedEvolutionEngine operations.
pub trait DoubleBufferedEvolutionEngine<State: DoubleBufferedState, AuxState>:
    EvolutionEngine<State, AuxState>
{
    /// Native double buffered step: delegates to step and then swaps buffers.
    fn step_buffered<R: RngCore + ?Sized>(
        &mut self,
        state: &mut State,
        aux: &mut AuxState,
        rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        self.step(state, aux, rng, dt)?;
        state.swap_buffers();
        Ok(())
    }
}

// Blanket implementation for any engine that works on a DoubleBufferedState
impl<E, State, AuxState> DoubleBufferedEvolutionEngine<State, AuxState> for E
where
    E: EvolutionEngine<State, AuxState>,
    State: DoubleBufferedState,
{
}

use super::ode::traits::{OdeSystem, Solver, VectorOperations};

/// A unified wrapper that turns any OdeSystem and Solver pair into an EvolutionEngine.
pub struct SystemEvolver<Sys, Sol> {
    /// Value for `system` (system).
    pub system: Sys,
    /// Value for `solver` (solver).
    pub solver: Sol,
    /// Value for `time` (time).
    pub time: f64,
}

impl<State, Sys, Sol> EvolutionEngine<State, ()> for SystemEvolver<Sys, Sol>
where
    State: VectorOperations,
    Sys: OdeSystem<State>,
    Sol: Solver<State>,
{
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut State,
        _aux: &mut (),
        _rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        self.solver.step(&self.system, self.time, state, dt);
        self.time += dt;
        Ok(())
    }
}
