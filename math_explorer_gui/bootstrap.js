// External WebAssembly bootstrap module for Math Explorer GUI under strict CSP ('self')

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

async function bootstrap() {
    try {
        const bindings = await import('./math_explorer_gui.js');
        const init = bindings.default;
        const wasm = await init({ module_or_path: './math_explorer_gui_bg.wasm' });

        window.wasmBindings = bindings;
        window.dispatchEvent(
            new CustomEvent('TrunkApplicationStarted', { detail: { wasm } }),
        );
    } catch (error) {
        showBootstrapError(error);
    }
}

bootstrap();
