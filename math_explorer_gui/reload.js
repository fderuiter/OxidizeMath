// External reload button and startup error boundary handler for Math Explorer GUI
(function () {
    function showStartupError(titleText, messageText, detailsText) {
        const loading = document.getElementById('loading_indicator');
        if (loading) {
            loading.remove();
        }

        const canvas = document.getElementById('the_canvas_id');
        if (canvas) {
            canvas.setAttribute('style', 'display: none;');
        }

        const errorContainer = document.getElementById('error_container');
        if (errorContainer) {
            if (titleText) {
                const titleEl = errorContainer.querySelector('.error-title');
                if (titleEl) {
                    titleEl.textContent = titleText;
                }
            }
            if (messageText) {
                const msgEl = errorContainer.querySelector('.error-message');
                if (msgEl) {
                    msgEl.textContent = messageText;
                }
            }
            errorContainer.setAttribute('style', 'display: flex;');
        }

        const detailsEl = document.getElementById('error_details');
        if (detailsEl) {
            detailsEl.textContent = detailsText || 'Unknown startup error.';
            detailsEl.setAttribute('style', 'display: block;');
        }

        const reloadButton = document.getElementById('webgl_reload_button');
        if (reloadButton) {
            reloadButton.focus();
        }
    }

    window.__mathExplorerShowStartupError = showStartupError;

    function attachHandlers() {
        const reloadButton = document.querySelector('.reload-btn');
        if (reloadButton && reloadButton.dataset.reloadBound !== 'true') {
            reloadButton.dataset.reloadBound = 'true';
            reloadButton.addEventListener('click', () => {
                window.location.reload();
            });
        }

        const bootstrapScript = document.getElementById('wasm_bootstrap_script');
        if (bootstrapScript && bootstrapScript.dataset.errorBound !== 'true') {
            bootstrapScript.dataset.errorBound = 'true';
            bootstrapScript.addEventListener('error', () => {
                showStartupError(
                    'Application Startup Failed',
                    'Math Explorer could not load its startup module. Check your network connection and reload the page.',
                    'Failed to load external module script: bootstrap.js'
                );
            });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachHandlers);
    } else {
        attachHandlers();
    }
})();
