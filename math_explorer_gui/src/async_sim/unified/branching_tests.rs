#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{UnifiedBranchSharedState, UnifiedCheckpointSlot, UnifiedModel, UnifiedSimRunner};
use crate::async_sim::{BackgroundWorker, SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::MAX_TIMELINE_RING_BYTES;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Clone)]
struct SyntheticLinearModel {
    val: f64,
    step_idx: usize,
}

impl FiniteStateGuard for SyntheticLinearModel {}

impl TheoryDescribable for SyntheticLinearModel {
    fn theory_description(&self) -> String {
        "Synthetic linear growth model for core-only timeline branching verification.".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Synthetic linear growth model.".to_string()
    }

    fn theory_citation(&self) -> String {
        "Euler, L. (1768). Institutionum calculi integralis.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

impl UnifiedModel for SyntheticLinearModel {
    fn domain() -> &'static str {
        "core_test"
    }

    fn name() -> &'static str {
        "Synthetic Linear Branch Model"
    }

    fn supports_timeline_branch() -> bool {
        true
    }

    fn new(params: &HashMap<String, f64>) -> Self {
        Self {
            val: params.get("rate").copied().unwrap_or(1.0),
            step_idx: 0,
        }
    }

    fn step(&mut self, params: &HashMap<String, f64>) {
        let rate = params.get("rate").copied().unwrap_or(1.0);
        self.val += rate * 0.1;
        self.step_idx += 1;
    }

    fn get_snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(RwLock::new(Vec::new())),
            custom_data: vec![self.step_idx as f64 * 0.1, self.val],
            structured_data: None,
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn parameters() -> HashMap<String, ParameterConstraint> {
        HashMap::from([(
            "rate".to_string(),
            ParameterConstraint {
                min: 0.1,
                max: 5.0,
                step: 0.1,
            },
        )])
    }

    fn time_step_dt(&self, _params: &HashMap<String, f64>) -> f64 {
        0.1
    }

    fn write_checkpoint(&self, slot: &mut UnifiedCheckpointSlot) {
        slot.scalars[0] = self.val;
        slot.scalars[1] = self.step_idx as f64;
    }

    fn restore_checkpoint(&mut self, slot: &UnifiedCheckpointSlot, _params: &HashMap<String, f64>) {
        self.val = slot.scalars[0];
        self.step_idx = slot.scalars[1].max(0.0) as usize;
    }

    fn fork_shadow_twin(&self) -> Option<Self> {
        Some(self.clone())
    }

    fn scalar_field_len(&self) -> usize {
        1
    }

    fn read_scalar_field(&self, out: &mut [f64]) -> usize {
        if out.is_empty() {
            return 0;
        }
        out[0] = self.val;
        1
    }
}

type SharedParamMap = Arc<RwLock<HashMap<String, f64>>>;
type SharedBranchMap = Arc<RwLock<UnifiedBranchSharedState>>;
type SyncRunnerBundle<M> = (UnifiedSimRunner<M>, SharedParamMap, SharedBranchMap);

fn new_sync_runner<M: UnifiedModel>() -> SyncRunnerBundle<M> {
    let mut defaults = HashMap::new();
    for (k, c) in M::parameters() {
        defaults.insert(k, f64::midpoint(c.min, c.max));
    }
    let params = Arc::new(RwLock::new(defaults));
    let sculpt = Arc::new(RwLock::new(crate::async_sim::SculptStatePayload::default()));
    let formula = Arc::new(RwLock::new(None));
    let branch_shared = Arc::new(RwLock::new(UnifiedBranchSharedState::default()));
    let runner = UnifiedSimRunner::<M>::with_shared_and_branch(
        Arc::clone(&params),
        sculpt,
        formula,
        Arc::clone(&branch_shared),
    );
    (runner, params, branch_shared)
}

#[test]
fn test_core_worker_vs_cooperative_wasm_step_parity_for_shadow_twin() {
    // Cooperative single-threaded runner (identical to wasm32 execution slice)
    let (mut coop_runner, coop_params, coop_shared) = new_sync_runner::<SyntheticLinearModel>();
    assert!(coop_runner.branch_rt.timeline.memory_budget_bytes() <= MAX_TIMELINE_RING_BYTES);
    for _ in 0..8 {
        coop_runner.step_once();
    }
    coop_runner.process_command(SimCommand::ForkBranch);
    coop_params.write().unwrap().insert("rate".to_string(), 3.5);
    for _ in 0..12 {
        coop_runner.step_once();
    }
    let coop_state = coop_shared.read().unwrap().clone();

    // BackgroundWorker command-driven runner (native worker thread / wasm cooperative worker)
    let (worker_runner, worker_params, worker_shared) = new_sync_runner::<SyntheticLinearModel>();
    let mut worker = BackgroundWorker::new(worker_runner);
    let _ = worker.send_command(SimCommand::Step(8));
    let _ = worker.send_command(SimCommand::ForkBranch);
    for _ in 0..80 {
        let _ = worker.update();
        if worker_shared.read().unwrap().fork_active
            && worker_shared.read().unwrap().current_step == 8
        {
            break;
        }
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    worker_params
        .write()
        .unwrap()
        .insert("rate".to_string(), 3.5);
    let _ = worker.send_command(SimCommand::Step(12));
    for _ in 0..80 {
        let _ = worker.update();
        if worker_shared.read().unwrap().current_step == 20 {
            break;
        }
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let wk_state = worker_shared.read().unwrap().clone();

    assert_eq!(coop_state.current_step, 20);
    assert_eq!(wk_state.current_step, 20);
    assert!(
        (coop_state.divergence.instantaneous_l2 - wk_state.divergence.instantaneous_l2).abs()
            < 1.0e-12
    );
    assert!(
        (coop_state.divergence.peak_abs_delta - wk_state.divergence.peak_abs_delta).abs() < 1.0e-12
    );
    assert!(
        (coop_state.divergence.cumulative_area_delta - wk_state.divergence.cumulative_area_delta)
            .abs()
            < 1.0e-12
    );
}

#[cfg(feature = "pure_math")]
#[test]
fn test_wave_unified_ring_buffer_and_zero_perturbation_fork() {
    use crate::tabs::analysis::wave_unified::WaveUnified;

    let (mut wave_runner, _wave_params, wave_shared) = new_sync_runner::<WaveUnified>();
    assert!(wave_runner.branch_rt.timeline.memory_budget_bytes() <= MAX_TIMELINE_RING_BYTES);

    for _ in 0..24 {
        wave_runner.step();
    }
    wave_runner.process_command(SimCommand::ScrubToStep(12));
    let st = wave_shared.read().unwrap().clone();
    assert_eq!(st.current_step, 12);
    assert!(
        st.current_time > 0.0,
        "Scrubbing to k=12 must not reset to t=0"
    );
    assert_eq!(
        st.max_step, 12,
        "Scrubbing to k=12 must truncate future checkpoints"
    );
    let snap_at_12 = wave_runner.get_snapshot().custom_data[0];
    assert!((snap_at_12 - 12.0 * 0.008).abs() < 1.0e-9);

    wave_runner.process_command(SimCommand::ForkBranch);
    for _ in 0..100 {
        wave_runner.step();
    }
    let forked = wave_shared.read().unwrap().clone();
    assert!(forked.fork_active);
    assert_eq!(forked.fork_step, Some(12));
    assert_eq!(forked.divergence.steps_since_fork, 100);
    assert_eq!(forked.divergence.instantaneous_l2, 0.0);
    assert_eq!(forked.divergence.peak_abs_delta, 0.0);
    assert_eq!(forked.divergence.cumulative_area_delta, 0.0);
}

#[cfg(feature = "physics")]
#[test]
fn test_lbm_unified_ring_buffer_and_zero_perturbation_fork_over_100_steps() {
    use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;

    let (mut lbm_runner, _, lbm_shared) = new_sync_runner::<LbmUnified>();
    assert!(lbm_runner.branch_rt.timeline.memory_budget_bytes() <= MAX_TIMELINE_RING_BYTES);
    assert!(lbm_runner.branch_rt.timeline.capacity() >= 2);
    for _ in 0..10 {
        lbm_runner.step();
    }
    lbm_runner.process_command(SimCommand::ForkBranch);
    for _ in 0..100 {
        lbm_runner.step();
    }
    let lbm_st = lbm_shared.read().unwrap().clone();
    assert_eq!(lbm_st.divergence.steps_since_fork, 100);
    assert_eq!(lbm_st.divergence.instantaneous_l2, 0.0);
    assert_eq!(lbm_st.divergence.peak_abs_delta, 0.0);
}

#[cfg(feature = "biology")]
#[test]
fn test_hodgkin_huxley_scrub_truncates_trajectory_history_exactly() {
    use crate::tabs::neuroscience::hodgkin_huxley::HodgkinHuxleyUnified;

    let (mut hh_runner, _, hh_shared) = new_sync_runner::<HodgkinHuxleyUnified>();
    for _ in 0..30 {
        hh_runner.step();
    }
    let full_len = hh_runner.get_snapshot().custom_data.len();
    hh_runner.process_command(SimCommand::ScrubToStep(15));
    let truncated_len = hh_runner.get_snapshot().custom_data.len();
    assert!(truncated_len < full_len);
    assert_eq!(hh_shared.read().unwrap().current_step, 15);
    assert!((hh_shared.read().unwrap().current_time - 0.15).abs() < 1.0e-9);
}

#[cfg(feature = "pure_math")]
#[test]
fn test_wave_perturbed_fork_difference_heatmap_exports_and_url_bounds() {
    use super::UnifiedSimTool;
    use crate::accessibility::{all_announcements, clear_test_announcements};
    use crate::framework::timeline_branch::exports::assert_framework_exports_all_branch_formats;
    use crate::framework::{InteractiveTool, RasterBranchViewMode, TimelineBranchAction};
    use crate::tabs::analysis::wave_unified::WaveUnified;
    use crate::url_state::URL_JSON_BOUNDS;

    let mut tool = UnifiedSimTool::<WaveUnified>::new();
    let saved = tool
        .save_state()
        .expect("WaveUnified save_state should succeed");
    assert!(saved.len() <= URL_JSON_BOUNDS.max_bytes);

    clear_test_announcements();
    tool.apply_timeline_action(TimelineBranchAction::ToggleForkBranch, None);
    let anns = all_announcements();
    assert!(anns
        .iter()
        .any(|(msg, pri)| pri == "polite" && msg.contains("Forked")));

    let mut export_tool = UnifiedSimTool::<WaveUnified>::new();
    let (mut runner, params, shared) = new_sync_runner::<WaveUnified>();
    for _ in 0..8 {
        runner.step();
    }
    runner.process_command(SimCommand::ForkBranch);
    params
        .write()
        .unwrap()
        .insert("wave_speed".to_string(), 2.2);
    runner.process_command(SimCommand::SetRasterViewMode(
        RasterBranchViewMode::DifferenceHeatmap,
    ));
    for _ in 0..15 {
        runner.step();
    }
    let snap = runner.get_snapshot();
    let st = shared.read().unwrap().clone();
    assert!(st.divergence.instantaneous_l2 > 0.0 && st.divergence.peak_abs_delta > 0.0);
    assert!(!st.primary_profile.is_empty() && !st.shadow_profile.is_empty());

    *export_tool.branch_state.write().unwrap() = st;
    *export_tool.params.write().unwrap() = params.read().unwrap().clone();
    export_tool.stepped_snapshot = Some(snap);
    assert!(export_tool.export_capabilities().dataset && export_tool.export_capabilities().figure);

    let saved_forked = export_tool.save_state().expect("Forked save_state");
    assert!(saved_forked.len() <= URL_JSON_BOUNDS.max_bytes);
    assert!(!saved_forked.contains("shadow_profile"));
    assert_framework_exports_all_branch_formats(Box::new(export_tool), "Shadow Baseline");
}

#[cfg(all(feature = "pure_math", not(target_arch = "wasm32")))]
#[test]
fn test_wave_unified_hot_checkpoint_and_diff_loop_zero_heap_allocations() {
    use crate::framework::{compute_slice_divergence, render_perceptual_difference_rgb};
    use crate::tabs::analysis::wave_unified::WaveUnified;

    let params = HashMap::from([
        ("wave_speed".to_string(), 1.2),
        ("damping".to_string(), 0.02),
    ]);
    let shadow_params = HashMap::from([
        ("wave_speed".to_string(), 0.9),
        ("damping".to_string(), 0.02),
    ]);
    let mut primary = WaveUnified::new(&params);
    let mut shadow = primary.fork_shadow_twin().expect("Shadow twin clone");
    let mut slot = primary.checkpoint_template();
    let n = primary.scalar_field_len();
    let mut p_buf = vec![0.0_f64; n];
    let mut s_buf = vec![0.0_f64; n];
    let mut rgb_buf = vec![0_u8; n * 3];

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((l2, peak), _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        primary.step(&params);
        shadow.step(&shadow_params);
        primary.write_checkpoint(&mut slot);
        primary.read_scalar_field(&mut p_buf);
        shadow.read_scalar_field(&mut s_buf);
        let (_, max_abs) = render_perceptual_difference_rgb(&p_buf, &s_buf, &mut rgb_buf);
        let (l2, _) = compute_slice_divergence(&p_buf, &s_buf);
        (l2, max_abs)
    });
    assert!(l2 > 0.0 && peak > 0.0);
}
