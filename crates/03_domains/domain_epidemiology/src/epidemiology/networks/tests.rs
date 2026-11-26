use super::*;
use std::sync::Mutex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static ALLOC_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[verified_engine::verified]
fn test_heterogeneous_r0() {
    let beta = 0.5;
    let gamma = 0.1;
    let mean_k = 4.0;
    let var_k = 0.0;
    let r0 = heterogeneous_r0(beta, gamma, mean_k, var_k);
    assert!((r0 - 15.0).abs() < math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_vaccinated_node_blocks_infection_and_protects_patient_zero() {
    let mut model = NetworkEpidemicModel::builder()
        .num_nodes(3)
        .parameters(1.0, 0.0)
        .states(vec![
            NodeState::Infected,
            NodeState::Susceptible,
            NodeState::Susceptible,
        ])
        .adjacency(vec![vec![1], vec![0, 2], vec![1]])
        .build()
        .expect("valid chain network");

    assert_eq!(model.patient_zero(), Some(0));
    assert_eq!(
        model.set_node_vaccinated(0, true),
        Err(NetworkModelError::InvalidIntervention)
    );
    assert_eq!(
        model.set_node_vaccinated(99, true),
        Err(NetworkModelError::InvalidNodeIndex)
    );

    model
        .set_node_vaccinated(1, true)
        .expect("vaccinating bridge hub succeeds");
    assert_eq!(model.states()[1], NodeState::Vaccinated);

    let mut rng = oxidize_core::rng::OxidizeRng::new(7);
    for _ in 0..8 {
        model.step_with_rng(&mut rng);
    }
    assert_eq!(model.states()[1], NodeState::Vaccinated);
    assert_eq!(model.states()[2], NodeState::Susceptible);
}

#[test]
#[verified_engine::verified]
fn test_symmetric_edge_quarantine_blocks_contagion_both_directions() {
    let mut forward = NetworkEpidemicModel::builder()
        .num_nodes(2)
        .parameters(1.0, 0.0)
        .states(vec![NodeState::Infected, NodeState::Susceptible])
        .adjacency(vec![vec![1], vec![0]])
        .build()
        .expect("valid dyad");

    forward
        .set_edge_quarantined(0, 1, true)
        .expect("quarantine (0, 1)");
    assert!(forward.is_edge_quarantined(0, 1));
    assert!(forward.is_edge_quarantined(1, 0));

    let mut rng = oxidize_core::rng::OxidizeRng::new(11);
    forward.step_with_rng(&mut rng);
    assert_eq!(forward.states()[1], NodeState::Susceptible);

    let mut reverse = NetworkEpidemicModel::builder()
        .num_nodes(2)
        .parameters(1.0, 0.0)
        .states(vec![NodeState::Susceptible, NodeState::Infected])
        .adjacency(vec![vec![1], vec![0]])
        .build()
        .expect("valid reverse dyad");
    reverse
        .set_edge_quarantined(0, 1, true)
        .expect("symmetric quarantine blocks 1 -> 0");
    reverse.step_with_rng(&mut rng);
    assert_eq!(reverse.states()[0], NodeState::Susceptible);

    reverse
        .set_edge_quarantined(1, 0, false)
        .expect("unquarantine (1, 0)");
    assert!(!reverse.is_edge_quarantined(0, 1));
    reverse.step_with_rng(&mut rng);
    assert_eq!(reverse.states()[0], NodeState::Infected);
}

#[test]
#[verified_engine::verified]
fn test_active_degree_moments_collapse_r0_on_hub_vaccination() {
    let mut star = NetworkEpidemicModel::builder()
        .num_nodes(5)
        .parameters(0.5, 0.25)
        .states(vec![
            NodeState::Susceptible,
            NodeState::Infected,
            NodeState::Susceptible,
            NodeState::Susceptible,
            NodeState::Susceptible,
        ])
        .adjacency(vec![vec![1, 2, 3, 4], vec![0], vec![0], vec![0], vec![0]])
        .build()
        .expect("valid star network");

    let r0_before = star.effective_r0();
    assert!(r0_before > 1.5);

    star.set_node_vaccinated(0, true)
        .expect("vaccinate central hub");
    let (mean_after, var_after) = star.active_degree_moments();
    assert_eq!(mean_after, 0.0);
    assert_eq!(var_after, 0.0);
    assert_eq!(star.effective_r0(), 0.0);
}

#[test]
#[verified_engine::verified]
fn test_step_and_containment_zero_allocation_under_verified_allocator() {
    let mut rng = oxidize_core::rng::OxidizeRng::new(2026);
    let mut model = NetworkEpidemicModel::new_with_rng(64, 0.25, 0.08, &mut rng);
    let candidate_node = (0..model.num_nodes())
        .find(|&n| model.states()[n] == NodeState::Susceptible)
        .expect("at least one susceptible vertex");
    let first_edge = model.edges().first().copied();

    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ((), _metrics) = VerifiedEngine::run_verified(|| {
        model
            .set_node_vaccinated(candidate_node, true)
            .expect("zero-alloc vaccination");
        if let Some([u, v]) = first_edge {
            model
                .set_edge_quarantined(u, v, true)
                .expect("zero-alloc edge quarantine");
        }
        let _moments = model.active_degree_moments();
        let _r0 = model.effective_r0();
        for _ in 0..5 {
            model.step_with_rng(&mut rng);
        }
    });
}
