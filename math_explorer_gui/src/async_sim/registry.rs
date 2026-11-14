use super::{SimCommand, SimStateUpdate, SimulationRunner};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::Mutex;

#[allow(missing_docs)]
pub struct WorkerContext {
    #[allow(missing_docs)]
    pub runner: Box<dyn SimulationRunner>,
    #[allow(missing_docs)]
    pub cmd_rx: Receiver<SimCommand>,
    #[allow(missing_docs)]
    pub state_tx: SyncSender<SimStateUpdate>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::async_sim::StateSnapshot;
    use std::sync::mpsc;
    use std::sync::Arc;

    struct DummyRunner;

    impl SimulationRunner for DummyRunner {
        fn process_command(&mut self, _cmd: SimCommand) {}
        fn step(&mut self) {}
        fn get_snapshot(&self) -> StateSnapshot {
            StateSnapshot {
                width: 1,
                height: 1,
                pixels: Arc::new(std::sync::RwLock::new(vec![])),
                custom_data: vec![],
                structured_data: None,
            }
        }
        fn get_steps_per_frame(&self) -> usize {
            1
        }
    }

    fn create_dummy_context() -> WorkerContext {
        let (_cmd_tx, cmd_rx) = mpsc::channel::<SimCommand>();
        let (state_tx, _state_rx) = mpsc::sync_channel::<SimStateUpdate>(4);
        WorkerContext {
            runner: Box::new(DummyRunner),
            cmd_rx,
            state_tx,
        }
    }

    #[test]
    fn test_handle_registry_valid_retrieval() {
        let ctx = create_dummy_context();
        let handle = register_worker_context(ctx);
        assert!(handle > 0, "Registered handle must be positive integer");

        let retrieved = take_worker_context(handle);
        assert!(retrieved.is_some(), "Valid handle must retrieve context");
    }

    #[test]
    fn test_handle_registry_invalid_rejection() {
        assert!(take_worker_context(0).is_none());
        assert!(take_worker_context(999_999_999).is_none());
        assert!(take_worker_context(u32::MAX).is_none());
    }

    #[test]
    fn test_handle_registry_duplicate_consumption() {
        let ctx = create_dummy_context();
        let handle = register_worker_context(ctx);

        let first = take_worker_context(handle);
        assert!(first.is_some());

        let second = take_worker_context(handle);
        assert!(second.is_none());

        let third = take_worker_context(handle);
        assert!(third.is_none());
    }

    #[test]
    fn test_handle_registry_unregister_cleanup() {
        let ctx = create_dummy_context();
        let handle = register_worker_context(ctx);

        let removed = unregister_worker_context(handle);
        assert!(removed.is_some(), "Unregister must return existing context");

        assert!(take_worker_context(handle).is_none());
        assert!(unregister_worker_context(handle).is_none());
    }
}
