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

fn step_running_frame_loop<R: SimulationRunner + ?Sized>(
    runner: &mut R,
    ring: &mut Option<CheckpointRing>,
    steps_per_frame: usize,
    max_slice: Option<Duration>,
    state: &mut SimWorkerState,
    state_tx: &SimStateSender,
) -> bool {
    let slice_start = max_slice.map(|_| crate::web_io::PlatformInstant::now());
    if let Some(r) = ring.as_mut() {
        r.prepare_forward_advance();
    }
    let substeps = if ring.is_some() {
        runner.running_substeps_per_tick().max(1)
    } else {
        1
    };
    let total_ticks = steps_per_frame.saturating_mul(substeps);
    for tick in 0..total_ticks {
        if tick > 0 {
            if let (Some(budget), Some(start)) = (max_slice, slice_start) {
                if start.elapsed() >= budget {
                    break;
                }
            }
        }
        if let Some(r) = ring.as_mut() {
            r.record_active_params(runner);
            runner.step_running_substep();
        } else {
            runner.step();
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
    let is_done = runner.is_complete();
    let snapshot = runner.get_snapshot();
    if is_done {
        *state = SimWorkerState::Ready;
        if state_tx
            .send_snapshot_and_status(snapshot, WorkerStatusEvent::Ready)
            .is_err()
        {
            return FrameTickOutcome::Disconnected;
        }
        FrameTickOutcome::CompletedOrFaulted
    } else if state_tx.send_snapshot(snapshot).is_err() {
        FrameTickOutcome::Disconnected
    } else {
        FrameTickOutcome::ContinueRunning
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn spawn_native_worker<T: SimulationRunner>(
    mut runner: T,
    cmd_rx: Receiver<SimCommand>,
    state_tx: SimStateSender,
) {
    std::thread::spawn(move || {
        let mut state = SimWorkerState::Ready;
        let mut steps_per_frame = runner.get_steps_per_frame();
        let mut ring = make_worker_ring(&runner);
        sync_worker_timeline(&ring, &state_tx);

        loop {
            let cmd_opt = if state.is_running() {
                cmd_rx.try_recv().ok()
            } else {
                cmd_rx.recv().ok()
            };
            let Some(maybe_cmd) = cmd_opt
                .map(Some)
                .or_else(|| state.is_running().then_some(None))
            else {
                break;
            };

            if let Some(first_cmd) = maybe_cmd {
                if !drain_worker_commands(
                    first_cmd,
                    &cmd_rx,
                    &mut runner,
                    &mut ring,
                    &mut state,
                    &mut steps_per_frame,
                    &state_tx,
                ) {
                    break;
                }
            }

            if state.is_running() {
                match advance_running_frame(
                    &mut runner,
                    &mut ring,
                    steps_per_frame,
                    None,
                    &mut state,
                    &state_tx,
                ) {
                    FrameTickOutcome::ContinueRunning => {
                        std::thread::sleep(Duration::from_millis(16));
                    }
                    FrameTickOutcome::CompletedOrFaulted => {}
                    FrameTickOutcome::Disconnected => break,
                }
            }
        }
    });
}
