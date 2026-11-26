//! Integration tests verifying `VerifiedAllocator` traps heap allocations inside locked
//! `VerifiedEngine::run_verified` regions while permitting stack-only verified computations.

use std::sync::Mutex;
use std::sync::atomic::Ordering;
use verified_engine::allocator::{MEMORY_LOCK, VerifiedAllocator};
use verified_engine::engine::VerifiedEngine;
use verified_engine::verified;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static ALLOC_TEST_LOCK: Mutex<()> = Mutex::new(());

#[verified]
fn stack_polynomial_step(x: f64, a: f64, b: f64, c: f64) -> f64 {
    (a * x + b) * x + c
}

#[test]
fn test_zero_alloc_closure_passes_under_verified_allocator() {
    let _guard = ALLOC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (val, metrics) =
        VerifiedEngine::run_verified(|| stack_polynomial_step(3.0, 2.0, -1.0, 5.0));
    assert!((val - 20.0).abs() < 1e-12);
    assert_eq!(metrics.function_calls, 1);
    assert!(metrics.arithmetic_ops >= 4);
    assert!(!MEMORY_LOCK.load(Ordering::SeqCst));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_verified_allocator_traps_heap_allocation_inside_locked_closure() {
    if std::env::var_os("VE_TRAP_CHILD").is_some() {
        let _ = VerifiedEngine::run_verified(|| {
            let buf: Vec<u8> = std::hint::black_box(Vec::with_capacity(64));
            std::hint::black_box(buf.capacity())
        });
        return;
    }

    let _guard = ALLOC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let exe = std::env::current_exe().expect("current test executable");
    let output = std::process::Command::new(exe)
        .env("VE_TRAP_CHILD", "1")
        .args([
            "--exact",
            "test_verified_allocator_traps_heap_allocation_inside_locked_closure",
            "--nocapture",
        ])
        .output()
        .expect("spawn child test process");

    assert!(
        !output.status.success(),
        "Child process should abort when allocating under VerifiedAllocator lock"
    );
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("VIOLATION: Dynamic memory allocation after initialization."),
        "Expected runtime_violation message in child output, got:\n{combined}"
    );
}
