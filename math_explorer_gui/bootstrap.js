// CSP-compliant external WebAssembly bootstrap with progressive loading and error recovery
const loadingStages = [];
window.__mathExplorerLoadingStages = loadingStages;

function updateLoadingStage(stage, subtitleText, progressPct, loadedBytes) {
    loadingStages.push({
        stage,
        subtitle: subtitleText,
        progress: progressPct ?? null,
        loadedBytes: loadedBytes ?? null,
    });

    const indicator = document.getElementById('loading_indicator');
    if (!indicator) {
        return;
    }

    indicator.setAttribute('data-stage', stage);
    if (typeof progressPct === 'number') {
        indicator.setAttribute('data-progress', String(progressPct));
    }
    if (typeof loadedBytes === 'number') {
        indicator.setAttribute('data-loaded-bytes', String(loadedBytes));
    }

    const subtitle = document.getElementById('loading_subtitle') || indicator.querySelector('.loading-subtitle');
    if (subtitle && subtitleText) {
        subtitle.textContent = subtitleText;
    }
}

function reportStartupFailure(err) {
    const detailText =
        err && typeof err === 'object'
            ? err.stack || err.message || String(err)
            : String(err);

    if (typeof window.__mathExplorerShowStartupError === 'function') {
        window.__mathExplorerShowStartupError(
            'Application Startup Failed',
            'Math Explorer could not download or initialize its WebAssembly runtime. Verify your network connection and reload the page.',
            detailText
        );
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
        updateLoadingStage('activating-isolation', 'Activating cross-origin isolation...', 0, 0);
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

function createProgressResponse(response) {
    const contentLengthHeader = response.headers.get('Content-Length');
    const totalBytes = contentLengthHeader ? Number.parseInt(contentLengthHeader, 10) : 0;

    if (!response.body || typeof ReadableStream !== 'function') {
        updateLoadingStage('compiling-wasm', 'Compiling WebAssembly & initializing WebGL...', 95, 0);
        return response;
    }

    const reader = response.body.getReader();
    let loadedBytes = 0;

    const stream = new ReadableStream({
        async pull(controller) {
            const { done, value } = await reader.read();
            if (done) {
                updateLoadingStage(
                    'compiling-wasm',
                    'Compiling WebAssembly & initializing WebGL...',
                    100,
                    loadedBytes
                );
                controller.close();
                return;
            }

            loadedBytes += value.byteLength;
            const loadedKb = Math.max(1, Math.round(loadedBytes / 1024));
            if (totalBytes > 0) {
                const pct = Math.min(99, Math.max(1, Math.floor((loadedBytes * 100) / totalBytes)));
                const totalKb = Math.max(1, Math.round(totalBytes / 1024));
                updateLoadingStage(
                    'downloading-wasm',
                    `Downloading WebAssembly... ${pct}% (${loadedKb} KB / ${totalKb} KB)`,
                    pct,
                    loadedBytes
                );
            } else {
                updateLoadingStage(
                    'downloading-wasm',
                    `Downloading WebAssembly... (${loadedKb} KB)`,
                    null,
                    loadedBytes
                );
            }

            controller.enqueue(value);
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
        status: response.status,
        statusText: response.statusText,
        headers,
    });
}

async function bootstrapApplication() {
    try {
        const reloadingForIsolation = await awaitStaticHostIsolationIfNeeded();
        if (reloadingForIsolation) {
            return;
        }

        updateLoadingStage('loading-js', 'Loading JavaScript runtime...', 5, 0);
        const bindings = await import('./math_explorer_gui.js');

        updateLoadingStage('fetching-wasm', 'Requesting WebAssembly binary...', 10, 0);
        const wasmResponse = await fetch('./math_explorer_gui_bg.wasm');
        if (!wasmResponse.ok) {
            throw new Error(
                `Failed to load WebAssembly binary: HTTP ${wasmResponse.status} ${wasmResponse.statusText}`
            );
        }

        const trackedResponse = createProgressResponse(wasmResponse);
        const wasm = await bindings.default({ module_or_path: trackedResponse });

        window.wasmBindings = bindings;
        window.dispatchEvent(new CustomEvent('TrunkApplicationStarted', { detail: { wasm } }));
    } catch (err) {
        if (window.sessionStorage.getItem('coiReloadedBySelf')) {
            return;
        }
        reportStartupFailure(err);
    }
}

bootstrapApplication();
