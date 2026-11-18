#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{SimCommand, SimStateUpdate, SimWorkerState, SimulationRunner, WorkerStatusEvent};
use std::sync::mpsc::{self, Receiver, SyncSender};

pub(super) enum FrameTickOutcome {
    ContinueRunning,
    CompletedOrFaulted,
    Disconnected,
}

fn emit_worker_divergence<R: SimulationRunner + ?Sized>(
    runner: &R,
    state_tx: &SyncSender<SimStateUpdate>,
) {
    let report = runner.diagnostic_report().unwrap_or_else(|| {
        crate::framework::recovery::DiagnosticReport::new(
            0.0,
            0,
            "state",
            "Non-finite floating-point state encountered",
        )
    });
    let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Faulted(
        report.message.clone(),
    )));
    let _ = state_tx.send(SimStateUpdate::Divergence(report));
    let snapshot = runner.get_snapshot();
    let _ = state_tx.try_send(SimStateUpdate::Snapshot(snapshot));
}

fn run_worker_step_batch<R: SimulationRunner + ?Sized>(
    runner: &mut R,
    step_count: usize,
    state: &mut SimWorkerState,
    state_tx: &SyncSender<SimStateUpdate>,
) -> bool {
    *state = SimWorkerState::Stepping;
    let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Stepping));
    for _ in 0..step_count {
        runner.step_once();
        if !runner.is_finite_state() {
            *state = SimWorkerState::Faulted;
            emit_worker_divergence(runner, state_tx);
            return true;
        }
    }
    let snapshot = runner.get_snapshot();
    if state_tx.send(SimStateUpdate::Snapshot(snapshot)).is_err() {
        return false;
    }
    *state = SimWorkerState::Ready;
    let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
    true
}

fn dispatch_worker_command<R: SimulationRunner + ?Sized>(
    cmd: SimCommand,
    runner: &mut R,
    state: &mut SimWorkerState,
    steps_per_frame: &mut usize,
    emit_paused_snapshot: &mut bool,
    state_tx: &SyncSender<SimStateUpdate>,
) -> bool {
    match cmd {
        SimCommand::Start => {
            *state = SimWorkerState::Running;
            let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Running));
            true
        }
        SimCommand::Pause => {
            *state = SimWorkerState::Ready;
            let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
            true
        }
        SimCommand::Step(count) => {
            *emit_paused_snapshot = false;
            run_worker_step_batch(runner, count, state, state_tx)
        }
        SimCommand::Reset => {
            *state = SimWorkerState::Ready;
            *emit_paused_snapshot = false;
            runner.process_command(SimCommand::Reset);
            let snapshot = runner.get_snapshot();
            let ok = state_tx.send(SimStateUpdate::Snapshot(snapshot)).is_ok();
            let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
            ok
        }
        SimCommand::SetSpeed(speed) => {
            *steps_per_frame = speed;
            runner.process_command(SimCommand::SetSpeed(speed));
            true
        }
        SimCommand::TrainEpisodes(_) => {
            *state = SimWorkerState::Running;
            runner.process_command(cmd);
            let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Running));
            true
        }
        SimCommand::ApplyBrush { .. } | SimCommand::ClearObstacles | SimCommand::Custom(_) => {
            runner.process_command(cmd);
            *emit_paused_snapshot = true;
            true
        }
    }
}

pub(super) fn drain_worker_commands<R: SimulationRunner + ?Sized>(
    first_cmd: SimCommand,
    cmd_rx: &Receiver<SimCommand>,
    runner: &mut R,
    state: &mut SimWorkerState,
    steps_per_frame: &mut usize,
    state_tx: &SyncSender<SimStateUpdate>,
) -> bool {
    let mut cmd = first_cmd;
    let mut emit_paused_snapshot = false;
    loop {
        if !dispatch_worker_command(
            cmd,
            runner,
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
        let _ = state_tx.try_send(SimStateUpdate::Snapshot(snapshot));
    }
    true
}

pub(super) fn advance_running_frame<R: SimulationRunner + ?Sized>(
    runner: &mut R,
    steps_per_frame: usize,
    state: &mut SimWorkerState,
    state_tx: &SyncSender<SimStateUpdate>,
) -> FrameTickOutcome {
    for _ in 0..steps_per_frame {
        runner.step();
        if !runner.is_finite_state() {
            *state = SimWorkerState::Faulted;
            emit_worker_divergence(runner, state_tx);
            return FrameTickOutcome::CompletedOrFaulted;
        }
        if runner.is_complete() {
            break;
        }
    }
    let is_done = runner.is_complete();
    let snapshot = runner.get_snapshot();
    if is_done {
        *state = SimWorkerState::Ready;
        if state_tx.send(SimStateUpdate::Snapshot(snapshot)).is_err() {
            return FrameTickOutcome::Disconnected;
        }
        let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
        FrameTickOutcome::CompletedOrFaulted
    } else if matches!(
        state_tx.try_send(SimStateUpdate::Snapshot(snapshot)),
        Err(mpsc::TrySendError::Disconnected(_))
    ) {
        FrameTickOutcome::Disconnected
    } else {
        FrameTickOutcome::ContinueRunning
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn spawn_native_worker<T: SimulationRunner>(
    mut runner: T,
    cmd_rx: Receiver<SimCommand>,
    state_tx: SyncSender<SimStateUpdate>,
) {
    std::thread::spawn(move || {
        let mut state = SimWorkerState::Ready;
        let mut steps_per_frame = runner.get_steps_per_frame();

        loop {
            let cmd_opt = if state.is_running() {
                cmd_rx.try_recv().ok()
            } else {
                match cmd_rx.recv() {
                    Ok(c) => Some(c),
                    Err(_) => break,
                }
            };

            if let Some(first_cmd) = cmd_opt {
                if !drain_worker_commands(
                    first_cmd,
                    &cmd_rx,
                    &mut runner,
                    &mut state,
                    &mut steps_per_frame,
                    &state_tx,
                ) {
                    break;
                }
            }

            if state.is_running() {
                match advance_running_frame(&mut runner, steps_per_frame, &mut state, &state_tx) {
                    FrameTickOutcome::ContinueRunning => {
                        std::thread::sleep(std::time::Duration::from_millis(16));
                    }
                    FrameTickOutcome::CompletedOrFaulted => {}
                    FrameTickOutcome::Disconnected => break,
                }
            }
        }
    });
}
