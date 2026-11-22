use super::*;
use crate::tabs::chaos::attractors::{AttractorRunner, AttractorSharedParams};
use crate::tabs::morphogenesis::{MorphogenesisSharedParams, MorphogenesisTab};
use std::sync::RwLock;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

async fn tick_microtasks() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        if let Ok(win) = global.dyn_into::<web_sys::Window>() {
            let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0);
        } else {
            let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

async fn wait_wasm_step(controller: &mut SimulationController, expected: usize) {
    for _ in 0..100 {
        let _ = controller.update();
        if controller.state() == SimWorkerState::Ready
            && controller
                .timeline_metadata()
                .is_some_and(|m| m.current_step == expected)
        {
            return;
        }
        tick_microtasks().await;
    }
    panic!(
        "WASM worker did not settle at step {expected}; state={} meta={:?}",
        controller.state(),
        controller.timeline_metadata()
    );
}

#[wasm_bindgen_test]
async fn wasm_worker_checkpoint_step_stepback_seek_and_reset() {
    let params = Arc::new(RwLock::new(MorphogenesisSharedParams {
        a: 0.14,
        b: 0.86,
        d_u: 1.0,
        d_v: 50.0,
        dt: 0.002,
        high_contrast: false,
    }));
    let runner = MorphogenesisTab::create_runner(&params, 20, 20);
    let mut controller = SimulationController::new(runner);
    assert_eq!(controller.timeline_metadata().unwrap().stride, 16);

    controller.send_command(SimCommand::Step(20)).unwrap();
    wait_wasm_step(&mut controller, 20).await;

    controller.send_command(SimCommand::StepBack(1)).unwrap();
    wait_wasm_step(&mut controller, 19).await;

    controller.send_command(SimCommand::SeekStep(10)).unwrap();
    wait_wasm_step(&mut controller, 10).await;
    assert_eq!(controller.timeline_metadata().unwrap().max_step, 20);

    controller.send_command(SimCommand::Reset).unwrap();
    wait_wasm_step(&mut controller, 0).await;
    assert_eq!(controller.timeline_metadata().unwrap().max_step, 0);
}

#[wasm_bindgen_test]
async fn wasm_attractor_running_slice_and_seek_pauses_cleanly() {
    let params = Arc::new(RwLock::new(AttractorSharedParams {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0,
        dt: 0.01,
        simulation_speed: 5,
        max_points: 200,
    }));
    let mut controller = SimulationController::new(AttractorRunner::from_params(params));
    controller.send_command(SimCommand::Start).unwrap();
    for _ in 0..8 {
        tick_microtasks().await;
        let _ = controller.update();
    }
    let reached = controller.timeline_metadata().unwrap().max_step;
    assert!(
        reached >= 5,
        "Expected running frames to advance >= 5 steps"
    );

    controller.send_command(SimCommand::SeekStep(3)).unwrap();
    wait_wasm_step(&mut controller, 3).await;
    assert_eq!(controller.state(), SimWorkerState::Ready);
}
