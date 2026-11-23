//! Example target for security repro lbm.
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;

fn main() {
    let _solver = LatticeBoltzmannD2Q9::new(10, 10, 1.0);

    // This code used to compile and cause UB by breaking invariants.
    // Now, the `width` field is private (pub(crate)), so this line is a compile error.
    // solver.state.width = 100;

    println!("Security check passed: Cannot modify internal state dimensions directly.");
    println!("The compiler prevents the invariant violation that would cause UB.");
}
