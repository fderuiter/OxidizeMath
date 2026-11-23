#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    CheckpointRing, SimCommand, SimStateSender, SimWorkerState, SimulationRunner, WorkerStatusEvent,
};
use std::sync::mpsc::Receiver;
use std::time::Duration;

pub(super) enum FrameTickOutcome {
    ContinueRunning,
    CompletedOrFaulted,
    Disconnected,
}

pub(super) fn make_worker_ring<R: SimulationRunner + ?Sized>(runner: &R) -> Option<CheckpointRing> {
    let stride = runner.checkpoint_stride()?;
    let mut ring = CheckpointRing::new(stride);
    ring.reset_and_seed(runner);
    Some(ring)
}

pub(super) fn sync_worker_timeline(ring: &Option<CheckpointRing>, state_tx: &SimStateSender) {
    state_tx.set_timeline(ring.as_ref().map(CheckpointRing::metadata));
}

fn emit_worker_divergence<R: SimulationRunner + ?Sized>(runner: &R, state_tx: &SimStateSender) {
    let report = runner.diagnostic_report().unwrap_or_else(|| {
        crate::framework::recovery::DiagnosticReport::new(
            0.0,
            0,
            "state",
            "Non-finite floating-point state encountered",
        )
    });
    let _ = state_tx.send_status(WorkerStatusEvent::Faulted(report.message.clone()));
    let _ = state_tx.send_divergence(report);
    let snapshot = runner.get_snapshot();
    let _ = state_tx.send_snapshot(snapshot);
}

fn run_worker_step_batch<R: SimulationRunner + ?Sized>(
    runner: &mut R,
    ring: &mut Option<CheckpointRing>,
    step_count: usize,
    state: &mut SimWorkerState,
    state_tx: &SimStateSender,
) -> bool {
    *state = SimWorkerState::Stepping;
    if state_tx.send_status(WorkerStatusEvent::Stepping).is_err() {
        return false;
    }
    if let Some(r) = ring.as_mut() {
        r.prepare_forward_advance();
    }
    for _ in 0..step_count {
        if let Some(r) = ring.as_mut() {
            r.record_active_params(runner);
        }
        runner.step_once();
        if !runner.is_finite_state() {
            *state = SimWorkerState::Faulted;
            emit_worker_divergence(runner, state_tx);
            return true;
        }
        if let Some(r) = ring.as_mut() {
            r.on_substep_advanced(runner);
        }
    }
    sync_worker_timeline(ring, state_tx);
    let snapshot = runner.get_snapshot();
    *state = SimWorkerState::Ready;
    state_tx
        .send_snapshot_and_status(snapshot, WorkerStatusEvent::Ready)
        .is_ok()
}

fn run_worker_rewind_or_seek<R: SimulationRunner + ?Sized>(
    runner: &mut R,
    ring: &mut Option<CheckpointRing>,
    target_step: Option<usize>,
    back_steps: usize,
    emit_stepping: bool,
    state: &mut SimWorkerState,
    state_tx: &SimStateSender,
) -> bool {
    if *state == SimWorkerState::Running {
        runner.process_command(SimCommand::Pause);
    }
    if emit_stepping {
        *state = SimWorkerState::Stepping;
        if state_tx.send_status(WorkerStatusEvent::Stepping).is_err() {
            return false;
        }
    }
    if let Some(r) = ring.as_mut() {
        let target = target_step
            .unwrap_or_else(|| r.metadata().current_step.saturating_sub(back_steps.max(1)));
        r.seek_to_step(runner, target);
    }
    sync_worker_timeline(ring, state_tx);
    let snapshot = runner.get_snapshot();
    *state = SimWorkerState::Ready;
    state_tx
        .send_snapshot_and_status(snapshot, WorkerStatusEvent::Ready)
        .is_ok()
}

fn handle_worker_reset<R: SimulationRunner + ?Sized>(
    runner: &mut R,
    ring: &mut Option<CheckpointRing>,
    state: &mut SimWorkerState,
    emit_paused_snapshot: &mut bool,
    state_tx: &SimStateSender,
) -> bool {
    *state = SimWorkerState::Ready;
    *emit_paused_snapshot = false;
    runner.process_command(SimCommand::Reset);
    if let Some(r) = ring.as_mut() {
        r.reset_and_seed(runner);
    }
    sync_worker_timeline(ring, state_tx);
    state_tx
        .send_snapshot_and_status(runner.get_snapshot(), WorkerStatusEvent::Ready)
        .is_ok()
}

fn handle_worker_custom_mutation<R: SimulationRunner + ?Sized>(
    cmd: SimCommand,
    runner: &mut R,
    ring: &mut Option<CheckpointRing>,
    emit_paused_snapshot: &mut bool,
    state_tx: &SimStateSender,
) -> bool {
    let prev_step = runner.current_step();
    runner.process_command(cmd);
    if let Some(r) = ring.as_mut() {
        if runner.current_step() == 0 && (prev_step > 0 || r.keyframe_count() <= 1) {
            r.reset_and_seed(runner);
        } else if runner.current_step() > 0 {
            r.record_mutation_keyframe(runner);
        }
    }
    sync_worker_timeline(ring, state_tx);
    *emit_paused_snapshot = true;
    true
}

fn handle_worker_pause<R: SimulationRunner + ?Sized>(
    runner: &mut R,
    ring: &Option<CheckpointRing>,
    state: &mut SimWorkerState,
    emit_paused_snapshot: &mut bool,
    state_tx: &SimStateSender,
) -> bool {
    let was_running = *state == SimWorkerState::Running;
    runner.process_command(SimCommand::Pause);
    *state = SimWorkerState::Ready;
    sync_worker_timeline(ring, state_tx);
    if was_running {
        *emit_paused_snapshot = false;
        state_tx
            .send_snapshot_and_status(runner.get_snapshot(), WorkerStatusEvent::Ready)
            .is_ok()
    } else {
        state_tx.send_status(WorkerStatusEvent::Ready).is_ok()
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_worker_command<R: SimulationRunner + ?Sized>(
    cmd: SimCommand,
    runner: &mut R,
    ring: &mut Option<CheckpointRing>,
    state: &mut SimWorkerState,
    steps_per_frame: &mut usize,
    emit_paused_snapshot: &mut bool,
    state_tx: &SimStateSender,
) -> bool {
    match cmd {
        SimCommand::Start => {
            runner.process_command(SimCommand::Start);
            *state = SimWorkerState::Running;
            state_tx.send_status(WorkerStatusEvent::Running).is_ok()
        }
        SimCommand::Pause => {
            handle_worker_pause(runner, ring, state, emit_paused_snapshot, state_tx)
        }
        SimCommand::Step(count) => {
            *emit_paused_snapshot = false;
            run_worker_step_batch(runner, ring, count, state, state_tx)
        }
        SimCommand::StepBack(count) => {
            *emit_paused_snapshot = false;
            run_worker_rewind_or_seek(runner, ring, None, count, true, state, state_tx)
        }
        SimCommand::SeekStep(target) => {
            *emit_paused_snapshot = false;
            run_worker_rewind_or_seek(runner, ring, Some(target), 0, false, state, state_tx)
        }
        SimCommand::Reset => {
            handle_worker_reset(runner, ring, state, emit_paused_snapshot, state_tx)
        }
        SimCommand::SetSpeed(speed) => {
            *steps_per_frame = speed;
            runner.process_command(SimCommand::SetSpeed(speed));
            true
        }
        SimCommand::TrainEpisodes(_) => {
            *state = SimWorkerState::Running;
            runner.process_command(cmd);
            state_tx.send_status(WorkerStatusEvent::Running).is_ok()
        }
        SimCommand::ApplyBrush { .. }
        | SimCommand::ApplyBrushStroke(_)
        | SimCommand::SyncSculptState(_)
        | SimCommand::ClearObstacles
        | SimCommand::Custom(_) => {
            handle_worker_custom_mutation(cmd, runner, ring, emit_paused_snapshot, state_tx)
        }
    }
}

pub(super) fn drain_worker_commands<R: SimulationRunner + ?Sized>(
    first_cmd: SimCommand,
    cmd_rx: &Receiver<SimCommand>,
    runner: &mut R,
    ring: &mut Option<CheckpointRing>,
    state: &mut SimWorkerState,
    steps_per_frame: &mut usize,
    state_tx: &SimStateSender,
) -> bool {
    let mut cmd = first_cmd;
    let mut emit_paused_snapshot = false;
    loop {
        if !dispatch_worker_command(
            cmd,
            runner,
            ring,
            state,
            steps_per_frame,
            &mut emit_paused_snapshot,
            state_tx,
        ) {
            return false;
        }
        if state.is_faulted() {
            return true;
        }
        match cmd_rx.try_recv() {
            Ok(next) => cmd = next,
            Err(_) => break,
        }
    }
    if emit_paused_snapshot && *state == SimWorkerState::Ready {
        let snapshot = runner.get_snapshot();
        if state_tx.send_snapshot(snapshot).is_err() {
            return false;
        }
    }
    true
}

fn step_twin_sliced_frame_loop<R: SimulationRunner + ?Sized>(
    runner: &mut R,
    ring: &mut Option<CheckpointRing>,
    steps_per_frame: usize,
    budget: Duration,
    state: &mut SimWorkerState,
    state_tx: &SimStateSender,
) -> bool {
    let slice_start = crate::web_io::PlatformInstant::now();
    let mut twin_budget = super::TwinSliceBudget::from_total(budget);
    if let Some(r) = ring.as_mut() {
        r.prepare_forward_advance();
    }
    for _ in 0..steps_per_frame {
        if !twin_budget.can_advance_pair(slice_start.elapsed()) {
            break;
        }
        if let Some(r) = ring.as_mut() {
            r.record_active_params(runner);
        }
        let t_a = crate::web_io::PlatformInstant::now();
        runner.step_branch_a();
        let dur_a = t_a.elapsed();
        let t_b = crate::web_io::PlatformInstant::now();
        runner.step_branch_b();
        twin_budget.record_pair(dur_a, t_b.elapsed());
        if !runner.is_finite_state() {
            *state = SimWorkerState::Faulted;
            sync_worker_timeline(ring, state_tx);
            emit_worker_divergence(runner, state_tx);
            return false;
        }
        if let Some(r) = ring.as_mut() {
            r.on_substep_advanced(runner);
        }
        if runner.is_complete() {
            break;
        }
    }
    true
}

fn effective_step_budget(
    max_slice: Option<Duration>,
    state_tx: &SimStateSender,
) -> Option<Duration> {
    max_slice.map(|b| {
        let reserve = if state_tx.has_unread_snapshot() {
            Duration::ZERO
        } else {
            Duration::from_micros(500).min(b / 3)
        };
        b.saturating_sub(reserve)
    })
}

fn step_running_frame_loop<R: SimulationRunner + ?Sized>(
    runner: &mut R,
    ring: &mut Option<CheckpointRing>,
    steps_per_frame: usize,
    max_slice: Option<Duration>,
    state: &mut SimWorkerState,
    state_tx: &SimStateSender,
) -> bool {
    if let Some(budget) = max_slice.filter(|_| runner.has_counterfactual_twin()) {
        return step_twin_sliced_frame_loop(runner, ring, steps_per_frame, budget, state, state_tx);
    }
    let slice_start = max_slice.map(|_| crate::web_io::PlatformInstant::now());
    let step_budget = effective_step_budget(max_slice, state_tx);
    if let Some(r) = ring.as_mut() {
        r.prepare_forward_advance();
    }
    let substeps = ring
        .as_ref()
        .map_or(1, |_| runner.running_substeps_per_tick().max(1));
    let total_ticks = steps_per_frame.saturating_mul(substeps);
    let mut prev_elapsed = Duration::ZERO;
    for tick in 0..total_ticks {
        let remaining_slice = match (step_budget, slice_start) {
            (Some(budget), Some(start)) => {
                let elapsed = start.elapsed();
                let last_dur = elapsed.saturating_sub(prev_elapsed);
                prev_elapsed = elapsed;
                let projected = elapsed
                    .saturating_add(last_dur)
                    .saturating_add(Duration::from_micros(150));
                if tick > 0 && projected >= budget {
                    break;
                }
                Some(budget.saturating_sub(elapsed))
            }
            _ => None,
        };
        if let Some(r) = ring.as_mut() {
            r.record_active_params(runner);
            runner.step_running_substep();
        } else {
            runner.step_sliced(remaining_slice);
        }
        if !runner.is_finite_state() {
            *state = SimWorkerState::Faulted;
            sync_worker_timeline(ring, state_tx);
            emit_worker_divergence(runner, state_tx);
            return false;
        }
        if let Some(r) = ring.as_mut() {
            r.on_substep_advanced(runner);
        }
        if runner.is_complete() {
            break;
        }
    }
    true
}

pub(super) fn advance_running_frame<R: SimulationRunner + ?Sized>(
    runner: &mut R,
    ring: &mut Option<CheckpointRing>,
    steps_per_frame: usize,
    max_slice: Option<Duration>,
    state: &mut SimWorkerState,
    state_tx: &SimStateSender,
) -> FrameTickOutcome {
    if !step_running_frame_loop(runner, ring, steps_per_frame, max_slice, state, state_tx) {
        return FrameTickOutcome::CompletedOrFaulted;
    }
    sync_worker_timeline(ring, state_tx);
    if runner.is_complete() {
        let snapshot = runner.get_snapshot();
        *state = SimWorkerState::Ready;
        if state_tx
            .send_snapshot_and_status(snapshot, WorkerStatusEvent::Ready)
            .is_err()
        {
            return FrameTickOutcome::Disconnected;
        }
        FrameTickOutcome::CompletedOrFaulted
    } else if state_tx.is_closed() {
        FrameTickOutcome::Disconnected
    } else if state_tx.has_unread_snapshot() {
        FrameTickOutcome::ContinueRunning
    } else if state_tx.send_snapshot(runner.get_snapshot()).is_err() {
        FrameTickOutcome::Disconnected
    } else {
        FrameTickOutcome::ContinueRunning
    }
}
