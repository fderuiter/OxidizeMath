#![allow(missing_docs)]
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]
use math_explorer::pure_math::analysis::ode::{OdeSystem, VectorOperations};
use std::ops::{Add, AddAssign, Mul, MulAssign};
use std::time::Instant;

#[derive(Clone, Debug)]
struct SimpleState {
    data: Vec<f64>,
}

impl VectorOperations for SimpleState {
    fn scale_add(&mut self, other: &Self, scale: f64) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b * scale;
        }
    }
    fn copy_from(&mut self, other: &Self) {
        // self.data.copy_from_slice(&other.data); // VectorOperations doesn't imply Vec, but we know it here.
        // Wait, copy_from_slice panics if lengths differ.
        if self.data.len() != other.data.len() {
            self.data.resize(other.data.len(), 0.0);
        }
        self.data.copy_from_slice(&other.data);
    }

    fn copy_from_scaled(&mut self, source: &Self, other: &Self, scale: f64) {
        if self.data.len() != source.data.len() {
            self.data.resize(source.data.len(), 0.0);
        }
        for ((dst, src), oth) in self
            .data
            .iter_mut()
            .zip(source.data.iter())
            .zip(other.data.iter())
        {
            *dst = *src + *oth * scale;
        }
    }
}

impl Add for SimpleState {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self.scale_add(&rhs, 1.0);
        self
    }
}
impl Mul<f64> for SimpleState {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self {
        for a in self.data.iter_mut() {
            *a *= rhs;
        }
        self
    }
}
impl AddAssign for SimpleState {
    fn add_assign(&mut self, rhs: Self) {
        self.scale_add(&rhs, 1.0);
    }
}
impl MulAssign<f64> for SimpleState {
    fn mul_assign(&mut self, rhs: f64) {
        for a in self.data.iter_mut() {
            *a *= rhs;
        }
    }
}

struct BenchmarkSystem;

impl OdeSystem<SimpleState> for BenchmarkSystem {
    fn derivative(&self, _t: f64, state: &SimpleState) -> SimpleState {
        let mut d = state.clone();
        for x in d.data.iter_mut() {
            *x = -*x;
        }
        d
    }
    fn derivative_in_place(&self, _t: f64, state: &SimpleState, out: &mut SimpleState) {
        if out.data.len() != state.data.len() {
            out.data.resize(state.data.len(), 0.0);
        }
        for (o, i) in out.data.iter_mut().zip(state.data.iter()) {
            *o = -*i;
        }
    }
}

fn main() {
    let size = 1_000_000;
    let system = BenchmarkSystem;
    let mut state = SimpleState {
        data: vec![1.0; size],
    };

    use math_explorer::pure_math::analysis::ode::{RungeKutta4, Solver};
    let mut solver = RungeKutta4::new(&state);

    // Warmup
    for _ in 0..2 {
        solver.step(&system, 0.0, &mut state, 0.01);
    }

    let start = Instant::now();
    let steps = 100;
    let dt = 0.01;

    for _ in 0..steps {
        solver.step(&system, 0.0, &mut state, dt);
    }

    let duration = start.elapsed();
    println!("Time for {} steps of size {}: {:?}", steps, size, duration);
}
