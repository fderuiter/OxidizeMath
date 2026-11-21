use super::{DefaultVfs, VirtualFileSystem, WasmVfs, resolve_wasm_fetch_url, trigger_download};
use std::io::ErrorKind;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn wasm_vfs_lists_embedded_papers_and_rejects_unknown_directory() {
    let entries = WasmVfs
        .list_dir("papers")
        .await
        .expect("embedded papers directory should be listed by WasmVfs");
    assert!(
        !entries.is_empty(),
        "embedded papers directory should not be empty"
    );
    assert!(
        entries.iter().any(|name| name.ends_with(".tex")),
        "embedded papers directory should contain .tex manuscripts"
    );

    let missing_err = (DefaultVfs {})
        .list_dir("nonexistent_wasm_vfs_directory_404")
        .await
        .expect_err("unknown VFS directory should return an error");
    assert_eq!(missing_err.kind(), ErrorKind::NotFound);
}

#[wasm_bindgen_test]
async fn wasm_vfs_reads_http_resource_over_browser_fetch() {
    assert_eq!(
        resolve_wasm_fetch_url("__test_vfs_fixture.txt"),
        "./__test_vfs_fixture.txt"
    );
    let body = WasmVfs
        .read_to_string("__test_vfs_fixture.txt")
        .await
        .expect("WasmVfs should fetch HTTP 200 fixture over window.fetch");
    assert_eq!(body.trim(), "oxidize-wasm-vfs-ok");
}

#[wasm_bindgen_test]
async fn wasm_vfs_propagates_not_found_on_http_404() {
    let err = WasmVfs
        .read_to_string("__missing_vfs_fixture_404.txt")
        .await
        .expect_err("HTTP 404 response should produce an io::Error");
    assert_eq!(err.kind(), ErrorKind::NotFound);
    assert!(
        err.to_string().contains("404"),
        "error message should include HTTP 404 status: {err}"
    );
}

#[wasm_bindgen_test]
async fn wasm_vfs_write_to_file_triggers_browser_blob_download() {
    let payload = b"t,value\n0.0,1.25\n1.0,2.50\n";
    WasmVfs
        .write_to_file("simulation_trace.csv", payload)
        .await
        .expect("WasmVfs::write_to_file should succeed via Blob URL download");
    trigger_download("direct_download.json", br#"{"ok":true}"#);
}
