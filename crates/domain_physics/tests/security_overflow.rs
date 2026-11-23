//! Integration tests for security overflow.
use domain_physics::physics::fluid_dynamics::lattice_boltzmann::state::LatticeState;

#[test]
#[should_panic(expected = "Grid dimensions too large")]
#[verified_engine::verified]
fn test_lattice_state_overflow() {
    // 2^32 * 2^32 = 2^64 which overflows u64 (usize on 64-bit)
    // We use slightly larger values to ensure overflow even if usize is larger (unlikely)
    // or just large enough to overflow.
    let width = usize::MAX / 2;
    let height = usize::MAX / 2;
    // This should panic due to explicit check we will add.
    // Currently in debug mode it panics with "attempt to multiply with overflow".
    // In release mode without check, it would succeed (return struct with wrapped size).
    // We want it to panic with custom message or standard overflow panic.
    let _ = LatticeState::<9>::new(width, height);
}
