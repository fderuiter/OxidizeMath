//! Example demonstrating McKendrick-von Foerster cohort waves and Tribolium cannibalism cycles.
use math_explorer::applied::cannibalism::{
    AgeStructuredConfig, AgeStructuredSolver, CannibalismModel, TriboliumLpaModel,
};
use math_explorer::nalgebra::Vector2;
use math_explorer::pure_math::analysis::ode::{RungeKutta4, Solver};

fn main() {
    let sub = AgeStructuredSolver::new(AgeStructuredConfig::diekmann_subcritical());
    let eq = sub.diekmann_equilibrium().unwrap_or(0.0);
    println!("Diekmann subcritical equilibrium A*: {eq:.4}");

    let mut hopf = AgeStructuredSolver::new(AgeStructuredConfig::diekmann_hopf());
    hopf.step_n(1600);
    let mut wave_min = f64::INFINITY;
    let mut wave_max = 0.0_f64;
    for _ in 0..600 {
        hopf.step();
        let a = hopf.adult_population();
        wave_min = wave_min.min(a);
        wave_max = wave_max.max(a);
    }
    println!("Diekmann Hopf cohort wave envelope (beta0 = 45.0): [{wave_min:.4}, {wave_max:.4}]");

    let lpa = TriboliumLpaModel::two_cycle_preset(0.7);
    let mut state = [1.5, 0.5, 1.0];
    for _ in 0..1000 {
        state = lpa.step(state);
    }
    let branch_1 = state[2];
    let branch_2 = lpa.step(state)[2];
    println!(
        "Tribolium LPA 2-cycle adult branches (c_pa = 0.7): [{:.4}, {:.4}]",
        branch_1.min(branch_2),
        branch_1.max(branch_2)
    );

    let ode = CannibalismModel::with_handling_time(0.4, 0.15, 0.1, 0.8, 0.1, 0.2, 0.05);
    let mut y = Vector2::new(5.0_f64, 2.0_f64);
    let mut rk4 = RungeKutta4::new(&y);
    for step in 0..300 {
        rk4.step(&ode, f64::from(step) * 0.05, &mut y, 0.05);
    }
    println!(
        "Guarded 2D Holling ODE at t = 15.0: N = {:.4}, C = {:.4}",
        y[0], y[1]
    );
}
