//! Host and headless Chromium (`wasm32`) tests for [`super::VolumetricWorkerBridge`].

use super::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

#[test]
fn transport_selection_respects_cross_origin_isolation_matrix() {
    assert_eq!(
        select_wasm_transport(true, true, true),
        WorkerTransportMode::SharedArrayBufferRing
    );
    assert_eq!(
        select_wasm_transport(false, false, true),
        WorkerTransportMode::DedicatedWorkerPostMessage
    );
    assert_eq!(
        select_wasm_transport(false, true, true),
        WorkerTransportMode::DedicatedWorkerPostMessage
    );
    assert_eq!(
        select_wasm_transport(false, false, false),
        WorkerTransportMode::CooperativeMainThread
    );
    assert_eq!(
        WorkerTransportMode::SharedArrayBufferRing.label(),
        "SAB Worker"
    );
    assert_eq!(
        WorkerTransportMode::DedicatedWorkerPostMessage.label(),
        "Web Worker"
    );
}

#[test]
fn native_worker_bridge_roundtrips_3d_schnakenberg_step() {
    let mut bridge = VolumetricWorkerBridge::new();
    assert_eq!(
        bridge.transport_mode(),
        WorkerTransportMode::NativeBackgroundThread
    );
    let dim = 16;
    let total = dim * dim * dim;
    let mut u = vec![1.0_f32; total];
    let mut v = vec![0.9_f32; total];
    u[flatten_3d_index(8, 8, 8, dim, dim)] = 1.45;
    let job = VolumetricStepJob {
        dim,
        dt: 0.008,
        du: 1.0,
        dv: 18.0,
        a: 0.10,
        b: 0.90,
        substeps: 2,
    };
    assert!(bridge.submit_turing_step(job, &u, &v));
    let mut completed_ms = None;
    for _ in 0..200 {
        if let Some(ms) = bridge.poll_completed_step(&mut u, &mut v) {
            completed_ms = Some(ms);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert!(completed_ms.is_some(), "native worker step must complete");
    assert!(!bridge.is_step_in_flight());
    assert!(u.iter().all(|x| x.is_finite()) && v.iter().all(|x| x.is_finite()));
    assert!((u[flatten_3d_index(8, 8, 8, dim, dim)] - 1.45).abs() > 1e-4);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test]
async fn wasm_blob_worker_executes_off_thread_transferable_step_under_non_isolated_csp() {
    let caps = probe_web_worker_capabilities();
    assert!(!caps.cross_origin_isolated);
    assert!(caps.dedicated_worker_available);
    assert_eq!(
        caps.selected_transport,
        WorkerTransportMode::DedicatedWorkerPostMessage
    );

    let mut bridge = VolumetricWorkerBridge::new();
    let dim = 16;
    let n = dim * dim * dim;
    let mut u = vec![1.0_f32; n];
    let mut v = vec![0.9_f32; n];
    u[flatten_3d_index(8, 8, 8, dim, dim)] = 1.5;
    let mut u_ref = u.clone();
    let mut v_ref = v.clone();
    let mut u_tmp = vec![0.0_f32; n];
    let mut v_tmp = vec![0.0_f32; n];
    let job = VolumetricStepJob {
        dim,
        dt: 0.008,
        du: 1.0,
        dv: 18.0,
        a: 0.10,
        b: 0.90,
        substeps: 2,
    };
    step_schnakenberg_3d_f32(job, &mut u_ref, &mut v_ref, &mut u_tmp, &mut v_tmp);
    assert!(bridge.submit_turing_step(job, &u, &v));

    let mut elapsed = None;
    for _ in 0..80 {
        if let Some(ms) = bridge.poll_completed_step(&mut u, &mut v) {
            elapsed = Some(ms);
            break;
        }
        let p = js_sys::Promise::new(&mut |resolve, _| {
            if let Ok(win) = js_sys::global().dyn_into::<web_sys::Window>() {
                let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 5);
            } else {
                let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
            }
        });
        let _ = wasm_bindgen_futures::JsFuture::from(p).await;
    }
    assert!(
        elapsed.is_some(),
        "Blob Web Worker must resolve step under PRODUCTION_CSP"
    );
    for i in 0..n {
        assert!(
            (u[i] - u_ref[i]).abs() < 1e-5,
            "worker/rust u parity mismatch at {i}"
        );
        assert!(
            (v[i] - v_ref[i]).abs() < 1e-5,
            "worker/rust v parity mismatch at {i}"
        );
    }
}
