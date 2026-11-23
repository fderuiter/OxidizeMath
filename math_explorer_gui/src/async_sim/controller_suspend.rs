//! Visibility suspension primitives for [`super::SimulationController`].
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, Condvar, Mutex};

/// Shared synchronization handle allowing native simulation threads to park with zero CPU spin
/// while their owning viewport pane or tool is hidden.
#[cfg(not(target_arch = "wasm32"))]
pub(super) struct NativeSuspension {
    suspended: AtomicBool,
    closed: AtomicBool,
    gate: Mutex<u64>,
    cvar: Condvar,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeSuspension {
    pub(super) fn new(initial_suspended: bool) -> Arc<Self> {
        Arc::new(Self {
            suspended: AtomicBool::new(initial_suspended),
            closed: AtomicBool::new(false),
            gate: Mutex::new(0),
            cvar: Condvar::new(),
        })
    }

    #[inline]
    pub(super) fn is_suspended(&self) -> bool {
        self.suspended.load(Ordering::Acquire)
    }

    #[inline]
    pub(super) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    pub(super) fn set_suspended(&self, suspended: bool) {
        self.suspended.store(suspended, Ordering::Release);
        self.notify();
    }

    pub(super) fn mark_closed(&self) {
        self.closed.store(true, Ordering::Release);
        self.notify();
    }

    pub(super) fn notify(&self) {
        let mut epoch = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        *epoch = epoch.wrapping_add(1);
        drop(epoch);
        self.cvar.notify_all();
    }

    pub(super) fn wait_for_change(&self, observed_epoch: u64) {
        let guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        if *guard == observed_epoch && self.is_suspended() && !self.is_closed() {
            drop(self.cvar.wait(guard));
        }
    }

    pub(super) fn current_epoch(&self) -> u64 {
        *self.gate.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// RAII guard notifying a parked native worker thread when its [`super::SimulationController`] drops.
#[cfg(not(target_arch = "wasm32"))]
pub(super) struct NativeSuspensionDropGuard(pub(super) Arc<NativeSuspension>);

#[cfg(not(target_arch = "wasm32"))]
impl Drop for NativeSuspensionDropGuard {
    fn drop(&mut self) {
        self.0.mark_closed();
    }
}

impl super::SimulationController {
    #[cfg(not(target_arch = "wasm32"))]
    /// Spawns a simulation worker with an explicit initial visibility suspension state.
    pub fn new_with_suspended<T: super::SimulationRunner>(runner: T, suspended: bool) -> Self {
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<super::SimCommand>();
        let (state_tx, state_rx) = super::sim_state_channel();
        let (initial_snapshot, initial_timeline) =
            super::seed_controller_channel(&runner, &state_tx);
        let suspension = NativeSuspension::new(suspended);
        spawn_native_worker(runner, cmd_rx, state_tx, Arc::clone(&suspension));

        Self {
            cmd_tx,
            state_rx,
            latest_snapshot: Some(initial_snapshot),
            timeline_metadata: initial_timeline,
            pending_snapshot: false,
            awaiting_pause_ack: false,
            suspended,
            state: super::SimWorkerState::Ready,
            diagnostic_report: None,
            native_suspension: NativeSuspensionDropGuard(suspension),
        }
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    /// Spawns a cooperative WASM simulation worker with an initial visibility suspension state.
    pub fn new_with_suspended<T: super::SimulationRunner>(runner: T, suspended: bool) -> Self {
        Self::from_boxed_runner(Box::new(runner), suspended)
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    fn from_boxed_runner(runner: Box<dyn super::SimulationRunner>, suspended: bool) -> Self {
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<super::SimCommand>();
        let (state_tx, state_rx) = super::sim_state_channel();
        let (initial_snapshot, initial_timeline) =
            super::seed_controller_channel(&*runner, &state_tx);
        let handle = super::register_worker_context(super::WorkerContext {
            runner,
            cmd_rx,
            state_tx,
        });
        super::register_wasm_worker(handle, suspended);
        wasm_bindgen_futures::spawn_local(async move {
            super::run_simulation_worker(handle).await;
        });

        Self {
            cmd_tx,
            state_rx,
            latest_snapshot: Some(initial_snapshot),
            timeline_metadata: initial_timeline,
            pending_snapshot: false,
            awaiting_pause_ack: false,
            suspended,
            state: super::SimWorkerState::Ready,
            diagnostic_report: None,
            handle,
        }
    }

    /// Suspends or resumes worker execution while preserving logical [`super::SimWorkerState::Running`].
    pub fn set_suspended(&mut self, suspended: bool) {
        if self.suspended == suspended {
            return;
        }
        self.suspended = suspended;
        #[cfg(not(target_arch = "wasm32"))]
        self.native_suspension.0.set_suspended(suspended);
        #[cfg(target_arch = "wasm32")]
        super::sync_wasm_worker_state(self.handle, self.state.is_running(), suspended);
    }

    /// Returns `true` when visibility suspension is currently active on this controller.
    #[must_use]
    pub fn is_suspended(&self) -> bool {
        self.suspended
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    pub(super) fn wake_wasm_worker(&self) {
        super::sync_wasm_worker_state(self.handle, self.state.is_running(), self.suspended);
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeWorkerLoop<T: super::SimulationRunner> {
    runner: T,
    cmd_rx: std::sync::mpsc::Receiver<super::SimCommand>,
    state_tx: super::SimStateSender,
    suspension: Arc<NativeSuspension>,
    state: super::SimWorkerState,
    steps_per_frame: usize,
    ring: Option<super::CheckpointRing>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: super::SimulationRunner> NativeWorkerLoop<T> {
    fn step_suspended(&mut self, observed_epoch: u64) -> bool {
        match self.cmd_rx.try_recv() {
            Ok(cmd) => super::worker::drain_worker_commands(
                cmd,
                &self.cmd_rx,
                &mut self.runner,
                &mut self.ring,
                &mut self.state,
                &mut self.steps_per_frame,
                &self.state_tx,
            ),
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.suspension.wait_for_change(observed_epoch);
                true
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => false,
        }
    }

    fn step_iteration(&mut self) -> bool {
        if self.suspension.is_closed() {
            return false;
        }
        let epoch = self.suspension.current_epoch();
        if self.state.is_running() && self.suspension.is_suspended() {
            return self.step_suspended(epoch);
        }
        let cmd_opt = if self.state.is_running() {
            self.cmd_rx.try_recv().ok()
        } else {
            self.cmd_rx.recv().ok()
        };
        let Some(maybe_cmd) = cmd_opt
            .map(Some)
            .or_else(|| self.state.is_running().then_some(None))
        else {
            return false;
        };
        if let Some(cmd) = maybe_cmd {
            if !super::worker::drain_worker_commands(
                cmd,
                &self.cmd_rx,
                &mut self.runner,
                &mut self.ring,
                &mut self.state,
                &mut self.steps_per_frame,
                &self.state_tx,
            ) {
                return false;
            }
        }
        if !self.state.is_running() || self.suspension.is_suspended() {
            return true;
        }
        match super::worker::advance_running_frame(
            &mut self.runner,
            &mut self.ring,
            self.steps_per_frame,
            None,
            &mut self.state,
            &self.state_tx,
        ) {
            super::worker::FrameTickOutcome::ContinueRunning => {
                std::thread::sleep(std::time::Duration::from_millis(16));
                true
            }
            super::worker::FrameTickOutcome::CompletedOrFaulted => true,
            super::worker::FrameTickOutcome::Disconnected => false,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_native_worker<T: super::SimulationRunner>(
    runner: T,
    cmd_rx: std::sync::mpsc::Receiver<super::SimCommand>,
    state_tx: super::SimStateSender,
    suspension: Arc<NativeSuspension>,
) {
    std::thread::spawn(move || {
        let steps_per_frame = runner.get_steps_per_frame();
        let ring = super::worker::make_worker_ring(&runner);
        super::worker::sync_worker_timeline(&ring, &state_tx);
        let mut worker = NativeWorkerLoop {
            runner,
            cmd_rx,
            state_tx,
            suspension,
            state: super::SimWorkerState::Ready,
            steps_per_frame,
            ring,
        };
        while worker.step_iteration() {}
    });
}
