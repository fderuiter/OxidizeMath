use super::{
    copy_image_to_clipboard_with_callback, copy_text_to_clipboard_with_callback,
    request_browser_repaint,
};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen_test::wasm_bindgen_test;

async fn wait_for_slot<T>(slot: &Rc<RefCell<Option<T>>>) {
    for _ in 0..25 {
        if slot.borrow().is_some() {
            return;
        }
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            let win = web_sys::window().expect("window");
            let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 10);
        });
        let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    }
}

#[wasm_bindgen_test]
async fn wasm_clipboard_text_and_image_exports_deliver_callbacks() {
    request_browser_repaint();

    let text_slot: Rc<RefCell<Option<Result<(), String>>>> = Rc::new(RefCell::new(None));
    let text_cb = Rc::clone(&text_slot);
    let text_launch = copy_text_to_clipboard_with_callback("x,y\n1,2\n", move |res| {
        *text_cb.borrow_mut() = Some(res);
    });
    assert!(text_launch.is_ok());
    wait_for_slot(&text_slot).await;
    let text_outcome = text_slot
        .borrow_mut()
        .take()
        .expect("clipboard text callback must resolve");
    if let Err(err) = text_outcome {
        assert!(err.contains("Clipboard write failed"));
    }

    let img = egui::ColorImage::new([2, 2], vec![egui::Color32::from_rgb(30, 120, 220); 4]);
    let img_slot: Rc<RefCell<Option<Result<(), String>>>> = Rc::new(RefCell::new(None));
    let img_cb = Rc::clone(&img_slot);
    let img_launch = copy_image_to_clipboard_with_callback(&img, move |res| {
        *img_cb.borrow_mut() = Some(res);
    });
    assert!(img_launch.is_ok());
    wait_for_slot(&img_slot).await;
    let img_outcome = img_slot
        .borrow_mut()
        .take()
        .expect("clipboard image callback must resolve");
    if let Err(err) = img_outcome {
        assert!(err.contains("Clipboard write failed"));
    }
}
