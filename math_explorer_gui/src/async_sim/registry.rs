use super::{SimCommand, SimStateUpdate, SimulationRunner};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Mutex;

#[allow(missing_docs)]
pub struct WorkerContext {
    #[allow(missing_docs)]
    pub runner: Box<dyn SimulationRunner>,
    #[allow(missing_docs)]
    pub cmd_rx: Receiver<SimCommand>,
    #[allow(missing_docs)]
    pub state_tx: Sender<SimStateUpdate>,
}

static NEXT_HANDLE: AtomicU32 = AtomicU32::new(1);
static WORKER_REGISTRY: Mutex<Option<HashMap<u32, WorkerContext>>> = Mutex::new(None);

#[allow(missing_docs)]
pub fn register_worker_context(ctx: WorkerContext) -> u32 {
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    let mut guard = WORKER_REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    guard.get_or_insert_with(HashMap::new).insert(handle, ctx);
    handle
}

#[allow(missing_docs)]
pub fn take_worker_context(handle: u32) -> Option<WorkerContext> {
    let mut guard = WORKER_REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_mut()?.remove(&handle)
}

#[allow(missing_docs)]
pub fn unregister_worker_context(handle: u32) -> Option<WorkerContext> {
    take_worker_context(handle)
}
