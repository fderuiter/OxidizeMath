//! Cooperative WebAssembly simulation worker loop driven by [`super::wasm_scheduler`].

use super::wasm_scheduler::{
    acknowledge_wake, allocate_worker_slice_budget, clear_worker_entries, current_frame_epoch,
    is_worker_suspended, record_worker_slice_elapsed, set_worker_running, suspend_idle_worker,
    wait_for_next_animation_frame,
};
use super::worker::{
    advance_running_frame, drain_worker_commands, make_worker_ring, sync_worker_timeline,
    FrameTickOutcome,
};
use super::{take_worker_context, SimCommand, SimWorkerState, WorkerStatusEvent};
use std::sync::mpsc::{Receiver, TryRecvError};

async fn poll_next_wasm_command(
    handle: u32,
    actively_running: bool,
    cmd_rx: &Receiver<SimCommand>,
) -> Result<Option<SimCommand>, ()> {
    if actively_running {
        return match cmd_rx.try_recv() {
            Ok(cmd) => Ok(Some(cmd)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(()),
        };
    }
    match cmd_rx.try_recv() {
        Ok(cmd) => Ok(Some(cmd)),
        Err(TryRecvError::Disconnected) => Err(()),
        Err(TryRecvError::Empty) => {
            suspend_idle_worker(handle).await;
            match cmd_rx.try_recv() {
                Ok(cmd) => Ok(Some(cmd)),
                Err(TryRecvError::Empty) => Ok(None),
                Err(TryRecvError::Disconnected) => Err(()),
            }
        }
    }
}

struct WasmWorkerLoop {
    handle: u32,
    runner: Box<dyn super::SimulationRunner>,
    cmd_rx: Receiver<SimCommand>,
    state_tx: super::SimStateSender,
    state: SimWorkerState,
    steps_per_frame: usize,
    ring: Option<super::CheckpointRing>,
    last_ticked_epoch: u64,
}

impl WasmWorkerLoop {
    fn tick_epoch_slice(&mut self) -> bool {
        let epoch = current_frame_epoch();
        if self.last_ticked_epoch == epoch {
            return true;
        }
        let budget = allocate_worker_slice_budget(self.handle);
        let slice_start = crate::web_io::PlatformInstant::now();
        let outcome = advance_running_frame(
            &mut *self.runner,
            &mut self.ring,
            self.steps_per_frame,
            Some(budget),
            &mut self.state,
            &self.state_tx,
        );
        record_worker_slice_elapsed(slice_start.elapsed());
        self.last_ticked_epoch = epoch;
        set_worker_running(self.handle, self.state.is_running());
        !matches!(outcome, FrameTickOutcome::Disconnected)
    }
}

/// Cooperative simulation worker entry point for WebAssembly.
pub(super) async fn run_simulation_worker(handle: u32) {
    let Some(ctx) = take_worker_context(handle) else {
        clear_worker_entries(handle);
        return;
    };
    let steps_per_frame = ctx.runner.get_steps_per_frame();
    let ring = make_worker_ring(&*ctx.runner);
    sync_worker_timeline(&ring, &ctx.state_tx);
    if !ctx.state_tx.mark_seeded()
        && ctx
            .state_tx
            .send_snapshot_and_status(ctx.runner.get_snapshot(), WorkerStatusEvent::Ready)
            .is_err()
    {
        clear_worker_entries(handle);
        return;
    }

    let mut worker = WasmWorkerLoop {
        handle,
        runner: ctx.runner,
        cmd_rx: ctx.cmd_rx,
        state_tx: ctx.state_tx,
        state: SimWorkerState::Ready,
        steps_per_frame,
        ring,
        last_ticked_epoch: 0,
    };
    loop {
        acknowledge_wake(handle);
        let actively_running = worker.state.is_running() && !is_worker_suspended(handle);
        set_worker_running(handle, worker.state.is_running());

        let Ok(cmd_opt) = poll_next_wasm_command(handle, actively_running, &worker.cmd_rx).await
        else {
            break;
        };
        if let Some(first_cmd) = cmd_opt {
            if !drain_worker_commands(
                first_cmd,
                &worker.cmd_rx,
                &mut *worker.runner,
                &mut worker.ring,
                &mut worker.state,
                &mut worker.steps_per_frame,
                &worker.state_tx,
            ) {
                break;
            }
            set_worker_running(handle, worker.state.is_running());
        }

        if worker.state.is_running() && !is_worker_suspended(handle) {
            if !worker.tick_epoch_slice() {
                break;
            }
            if worker.state.is_running() && !is_worker_suspended(handle) {
                wait_for_next_animation_frame(handle).await;
            }
        }
    }

    clear_worker_entries(handle);
}
