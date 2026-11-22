//! Comprehensive example demonstrating Markov Chain models.

use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::statistics::markov::TimeIndex;
use math_explorer::pure_math::statistics::markov::ctmc::ContinuousMarkovChain;
use math_explorer::pure_math::statistics::markov::dtmc::{MarkovChain, StateType};
use math_explorer::pure_math::statistics::markov::hmm::HiddenMarkovModel;
use math_explorer::pure_math::statistics::markov::tensor::TransitionTensor;
use math_explorer::rng::OxidizeRng;

fn demo_epv() {
    println!("1. Expected Possession Value (DTMC with Rewards)");
    let transitions = DMatrix::from_row_slice(
        4,
        4,
        &[
            0.50, 0.20, 0.10, 0.20, // Offense
            0.10, 0.60, 0.25, 0.05, // Advantage
            0.00, 0.00, 1.00, 0.00, // Score
            0.00, 0.00, 0.00, 1.00, // Turnover
        ],
    );

    let state_types = vec![
        StateType::Transient, // Offense
        StateType::Transient, // Advantage
        StateType::Absorbing, // Score
        StateType::Absorbing, // Turnover
    ];

    let chain = MarkovChain::new(transitions, state_types).unwrap();
    let rewards = DVector::from_vec(vec![2.0, 0.0]);
    let epv = chain.expected_possession_value(&rewards).unwrap();

    println!("   EPV from offense state: {:.3} points", epv[0]);
    println!("   EPV from advantage state: {:.3} points", epv[1]);

    let absorption = chain.absorption_probabilities().unwrap();
    println!("   P(score | start in offense) = {:.3}", absorption[(0, 0)]);
    println!(
        "   P(score | start in advantage) = {:.3}\n",
        absorption[(1, 0)]
    );
}

fn demo_shot_clock() {
    println!("2. Shot Clock Urgency (Time-Varying Transitions)");
    let mut tensor = TransitionTensor::new(
        3,
        TimeIndex::new(0.0).unwrap(),
        TimeIndex::new(24.0).unwrap(),
    );

    let p_24 = DMatrix::from_row_slice(
        3,
        3,
        &[0.85, 0.10, 0.05, 0.00, 1.00, 0.00, 0.00, 0.00, 1.00],
    );
    tensor
        .add_time_slice(TimeIndex::new(24.0).unwrap(), p_24)
        .unwrap();

    let p_3 = DMatrix::from_row_slice(
        3,
        3,
        &[0.30, 0.60, 0.10, 0.00, 1.00, 0.00, 0.00, 0.00, 1.00],
    );
    tensor
        .add_time_slice(TimeIndex::new(3.0).unwrap(), p_3)
        .unwrap();

    let p_12 = tensor
        .transition_matrix_at(TimeIndex::new(12.0).unwrap())
        .unwrap();
    println!("   Shot probability at 12 seconds: {:.3}\n", p_12[(0, 1)]);
}

fn demo_hot_hand() {
    println!("3. Hot Hand Detection (HMM)");
    let initial = DVector::from_vec(vec![0.5, 0.5]);
    let transitions = DMatrix::from_row_slice(2, 2, &[0.70, 0.30, 0.40, 0.60]);
    let emissions = DMatrix::from_row_slice(2, 2, &[0.70, 0.30, 0.20, 0.80]);
    let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();
    let shots = vec![1, 1, 1, 0, 1];

    let states = hmm.viterbi(&shots).unwrap();
    println!("   Most likely states: {:?}", states);

    let posterior = hmm.filter(&shots).unwrap();
    println!("   P(Hot | all shots) = {:.3}\n", posterior[1]);
}

fn demo_gamblers_ruin() {
    println!("4. Gambler's Ruin (Classic DTMC)");
    let transitions = DMatrix::from_row_slice(
        5,
        5,
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0,
            0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );

    let states = vec![
        StateType::Absorbing,
        StateType::Transient,
        StateType::Transient,
        StateType::Transient,
        StateType::Absorbing,
    ];

    let chain = MarkovChain::new(transitions, states).unwrap();
    let absorption = chain.absorption_probabilities().unwrap();
    for i in 0..3 {
        println!(
            "   Starting with ${}: P(reach $4) = {:.3}",
            i + 1,
            absorption[(i, 1)]
        );
    }
    let times = chain.expected_absorption_times().unwrap();
    println!("   Expected games from $2: {:.1}\n", times[1]);
}

fn demo_birth_death() {
    println!("5. Birth-Death Process (CTMC)");
    let generator = DMatrix::from_row_slice(2, 2, &[-2.0, 2.0, 3.0, -3.0]);
    let chain = ContinuousMarkovChain::new(generator).unwrap();

    if let Some(pi) = chain.steady_state() {
        println!("   π(0) = {:.3}, π(1) = {:.3}", pi[0], pi[1]);
    }

    let p_t = chain.transition_probabilities(1.0).unwrap();
    println!("   P(0→1, t=1) = {:.3}", p_t[(0, 1)]);

    let mut rng = OxidizeRng::default();
    let trajectory = chain.simulate_trajectory(0, 10.0, &mut rng).unwrap();
    println!("   Trajectory length: {}\n", trajectory.len());
}

fn demo_market_regime() {
    println!("6. Market Regime Detection (HMM)");
    let initial = DVector::from_vec(vec![0.33, 0.33, 0.34]);
    let transitions = DMatrix::from_row_slice(
        3,
        3,
        &[0.70, 0.20, 0.10, 0.20, 0.70, 0.10, 0.25, 0.25, 0.50],
    );
    let emissions = DMatrix::from_row_slice(
        3,
        5,
        &[
            0.30, 0.30, 0.20, 0.10, 0.10, 0.10, 0.10, 0.20, 0.30, 0.30, 0.10, 0.20, 0.40, 0.20,
            0.10,
        ],
    );
    let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();
    let observations = vec![1, 0, 2, 3, 4];

    let states = hmm.viterbi(&observations).unwrap();
    println!("   Inferred regimes: {:?}", states);

    let current_belief = hmm.filter(&observations).unwrap();
    println!(
        "   P(Bull) = {:.3}, P(Bear) = {:.3}, P(Sideways) = {:.3}\n",
        current_belief[0], current_belief[1], current_belief[2]
    );
}

fn main() {
    println!("=== Math Explorer: Markov Chain Models Demo ===\n");
    demo_epv();
    demo_shot_clock();
    demo_hot_hand();
    demo_gamblers_ruin();
    demo_birth_death();
    demo_market_regime();
}
