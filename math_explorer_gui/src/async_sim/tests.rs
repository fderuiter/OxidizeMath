use super::*;
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
    let (state_tx, _state_rx) = mpsc::channel::<SimStateUpdate>();
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
    // Forged or non-existent handle IDs must return None
    assert!(take_worker_context(0).is_none());
    assert!(take_worker_context(999_999_999).is_none());
    assert!(take_worker_context(u32::MAX).is_none());
}

#[test]
fn test_handle_registry_duplicate_consumption() {
    let ctx = create_dummy_context();
    let handle = register_worker_context(ctx);

    // First consumption succeeds
    let first = take_worker_context(handle);
    assert!(first.is_some());

    // Subsequent/duplicate retrieval returns None because context is consumed
    let second = take_worker_context(handle);
    assert!(second.is_none());

    let third = take_worker_context(handle);
    assert!(third.is_none());
}

#[test]
fn test_handle_registry_unregister_cleanup() {
    let ctx = create_dummy_context();
    let handle = register_worker_context(ctx);

    // Dropping simulation controller before worker starts cleans up registry
    let removed = unregister_worker_context(handle);
    assert!(removed.is_some(), "Unregister must return existing context");

    // Once cleaned up, worker attempting to take handle gets None
    assert!(take_worker_context(handle).is_none());
    assert!(unregister_worker_context(handle).is_none());
}
