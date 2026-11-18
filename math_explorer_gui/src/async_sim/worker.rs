#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{SimCommand, SimStateUpdate, SimWorkerState, SimulationRunner, WorkerStatusEvent};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;
use std::time::Duration;

fn process_worker_commands<T: SimulationRunner>(
    runner: &mut T,
    first_cmd: SimCommand,
    cmd_rx: &Receiver<SimCommand>,
    state_tx: &SyncSender<SimStateUpdate>,
    state: &mut SimWorkerState,
    steps_per_frame: &mut usize,
) {
    let mut cmd = first_cmd;
    let mut emit_paused_snapshot = false;

    loop {
        match cmd {
            SimCommand::Start => {
                *state = SimWorkerState::Running;
                let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Running));
            }
            SimCommand::Pause => {
                *state = SimWorkerState::Ready;
                let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
            }
            SimCommand::Reset => {
                *state = SimWorkerState::Ready;
                emit_paused_snapshot = false;
                runner.process_command(SimCommand::Reset);
                let snapshot = runner.get_snapshot();
                let _ = state_tx.send(SimStateUpdate::Snapshot(snapshot));
                let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
            }
            SimCommand::SetSpeed(speed) => {
                *steps_per_frame = speed;
                runner.process_command(SimCommand::SetSpeed(speed));
            }
            SimCommand::TrainEpisodes(_) => {
                *state = SimWorkerState::Running;
                runner.process_command(cmd);
                let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Running));
            }
            SimCommand::ApplyBrush { .. } | SimCommand::ClearObstacles | SimCommand::Custom(_) => {
                runner.process_command(cmd);
                emit_paused_snapshot = true;
            }
        }
        if let Ok(next_cmd) = cmd_rx.try_recv() {
            cmd = next_cmd;
        } else {
            break;
        }
    }

    if emit_paused_snapshot && *state == SimWorkerState::Ready {
        let snapshot = runner.get_snapshot();
        let _ = state_tx.try_send(SimStateUpdate::Snapshot(snapshot));
    }
}

fn step_running_worker<T: SimulationRunner>(
    runner: &mut T,
    state_tx: &SyncSender<SimStateUpdate>,
    state: &mut SimWorkerState,
    steps_per_frame: usize,
) -> bool {
    let mut diverged = false;
    for _ in 0..steps_per_frame {
        runner.step();
        if !runner.is_finite_state() {
            diverged = true;
            break;
        }
        if runner.is_complete() {
            break;
        }
    }

    if diverged {
        *state = SimWorkerState::Faulted;
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
        return true;
    }

    let is_done = runner.is_complete();
    let snapshot = runner.get_snapshot();
    if is_done {
        *state = SimWorkerState::Ready;
        if state_tx.send(SimStateUpdate::Snapshot(snapshot)).is_err() {
            return false;
        }
        let _ = state_tx.send(SimStateUpdate::Status(WorkerStatusEvent::Ready));
    } else {
        if let Err(mpsc::TrySendError::Disconnected(_)) =
            state_tx.try_send(SimStateUpdate::Snapshot(snapshot))
        {
            return false;
        }
        thread::sleep(Duration::from_millis(16));
    }
    true
}

pub(super) fn spawn_native_worker<T: SimulationRunner>(
    mut runner: T,
    cmd_rx: Receiver<SimCommand>,
    state_tx: SyncSender<SimStateUpdate>,
) {
    thread::spawn(move || {
        let mut state = SimWorkerState::Ready;
        let mut steps_per_frame = runner.get_steps_per_frame();

        loop {
            let cmd_opt = if state.is_running() {
                cmd_rx.try_recv().ok()
            } else if let Ok(c) = cmd_rx.recv() {
                Some(c)
            } else {
                break;
            };

            if let Some(first_cmd) = cmd_opt {
                process_worker_commands(
                    &mut runner,
                    first_cmd,
                    &cmd_rx,
                    &state_tx,
                    &mut state,
                    &mut steps_per_frame,
                );
            }

            if state.is_running()
                && !step_running_worker(&mut runner, &state_tx, &mut state, steps_per_frame)
            {
                break;
            }
        }
    });
}
