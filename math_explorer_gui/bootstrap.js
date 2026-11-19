// External WebAssembly bootstrap module for Math Explorer GUI under strict CSP ('self')
import createInitializer from './initializer.js';

const WASM_SOURCE_SIZE = 0;

const loadingStages = [];
window.__mathExplorerLoadingStages = loadingStages;

function recordLoadingStage(stage, loadedBytes, totalBytes) {
    const progress =
        typeof loadedBytes === 'number' && typeof totalBytes === 'number' && totalBytes > 0
            ? Math.min(100, Math.floor((loadedBytes * 100) / totalBytes))
            : null;
    loadingStages.push({
        stage,
        progress,
        loadedBytes: typeof loadedBytes === 'number' ? loadedBytes : null,
    });
}

function formatBootstrapError(error) {
    if (error instanceof Error) {
        return error.stack || `${error.name}: ${error.message}`;
    }
    return String(error);
}

function showBootstrapError(error) {
    console.error('Failed to initialize Math Explorer WebAssembly module:', error);
    const detailsText = formatBootstrapError(error);

    const applyDomErrorState = () => {
        const loadingIndicator = document.getElementById('loading_indicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }

        const canvas = document.getElementById('the_canvas_id');
        if (canvas) {
            canvas.style.display = 'none';
        }

        const errorContainer = document.getElementById('error_container');
        if (errorContainer) {
            errorContainer.style.display = 'flex';
        }

        const errorDetails = document.getElementById('error_details');
        if (errorDetails) {
            errorDetails.textContent = detailsText;
            errorDetails.style.display = 'block';
        }

        const reloadButton = document.getElementById('webgl_reload_button');
        if (reloadButton && typeof reloadButton.focus === 'function') {
            reloadButton.focus();
        }
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applyDomErrorState, { once: true });
    } else {
        applyDomErrorState();
    }
}

async function awaitStaticHostIsolationIfNeeded() {
    if (
        window.crossOriginIsolated === false &&
        window.isSecureContext &&
        navigator.serviceWorker &&
        !navigator.serviceWorker.controller &&
        !window.sessionStorage.getItem('coiReloadedBySelf')
    ) {
        recordLoadingStage('activating-isolation', 0, WASM_SOURCE_SIZE);
        await new Promise((resolve) => {
            let settled = false;
            const finish = () => {
                if (!settled) {
                    settled = true;
                    resolve();
                }
            };
            navigator.serviceWorker.addEventListener('controllerchange', finish, { once: true });
            window.setTimeout(finish, 2500);
        });
        if (window.sessionStorage.getItem('coiReloadedBySelf')) {
            return true;
        }
    }
    return false;
}

async function fetchWasmWithProgress(source, hooks) {
    const response = await fetch(source, { mode: 'cors', credentials: 'same-origin' });
    const status = response.status;
    const statusText = response.statusText;
    if (!response.ok) {
        throw new Error(
            `HTTP ${status} (${statusText || 'Request Failed'}) while fetching ${source}`,
        );
    }

    const contentEncoding = response.headers.get('content-encoding');
    const rawContentLength = Number(response.headers.get('content-length') || 0);
    const total =
        !contentEncoding && Number.isFinite(rawContentLength) && rawContentLength > 0
            ? rawContentLength
            : WASM_SOURCE_SIZE;

    if (!response.body || typeof response.body.getReader !== 'function') {
        recordLoadingStage('compiling-wasm', total, total);
        return response;
    }

    const reader = response.body.getReader();
    let current = 0;

    const stream = new ReadableStream({
        start(controller) {
            function push() {
                reader
                    .read()
                    .then(({ done, value }) => {
                        if (done) {
                            const finalBytes = total > 0 ? total : current;
                            recordLoadingStage('compiling-wasm', finalBytes, total);
                            hooks?.onProgress?.({ current: finalBytes, total });
                            controller.close();
                            return;
                        }

                        current += value.byteLength;
                        recordLoadingStage('downloading-wasm', current, total);
                        hooks?.onProgress?.({ current, total });
                        controller.enqueue(value);
                        push();
                    })
                    .catch((err) => {
                        controller.error(err);
                    });
            }

            push();
        },
        cancel(reason) {
            return reader.cancel(reason);
        },
    });

    const headers = new Headers(response.headers);
    if (!headers.has('Content-Type')) {
        headers.set('Content-Type', 'application/wasm');
    }

    return new Response(stream, {
        status,
        statusText,
        headers,
    });
}

async function bootstrap() {
    if (await awaitStaticHostIsolationIfNeeded()) {
        return;
    }

    const hooks = typeof createInitializer === 'function' ? createInitializer() : null;
    hooks?.onStart?.();

    try {
        recordLoadingStage('loading-js', 0, WASM_SOURCE_SIZE);
        const bindings = await import('./math_explorer_gui.js');
        const init = bindings.default;

        recordLoadingStage('fetching-wasm', 0, WASM_SOURCE_SIZE);
        const wasmResponse = await fetchWasmWithProgress('./math_explorer_gui_bg.wasm', hooks);
        const wasm = await init({ module_or_path: wasmResponse });

        hooks?.onComplete?.();
        hooks?.onSuccess?.(wasm);

        window.wasmBindings = bindings;
        window.dispatchEvent(
            new CustomEvent('TrunkApplicationStarted', { detail: { wasm } }),
        );
    } catch (error) {
        if (window.sessionStorage.getItem('coiReloadedBySelf')) {
            return;
        }
        hooks?.onComplete?.();
        if (hooks && typeof hooks.onFailure === 'function' && window.__mathExplorerBoot) {
            hooks.onFailure(error);
        } else if (
            window.__mathExplorerBoot &&
            typeof window.__mathExplorerBoot.handleFailure === 'function'
        ) {
            window.__mathExplorerBoot.handleFailure(error, './math_explorer_gui_bg.wasm');
        } else {
            showBootstrapError(error);
        }
    }
}

bootstrap();
