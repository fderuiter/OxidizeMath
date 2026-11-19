// External reload button handler for Math Explorer GUI error boundary
(function () {
    function clearStaleIsolationFlags() {
        try {
            window.sessionStorage.removeItem('coiReloadedBySelf');
            window.sessionStorage.removeItem('coiCoepHasFailed');
        } catch (_) {
            // Ignore storage errors in restricted browser modes
        }
    }

    function attachReloadHandler() {
        const reloadButton = document.querySelector('.reload-btn');
        if (reloadButton && !reloadButton.dataset.reloadBound) {
            reloadButton.dataset.reloadBound = 'true';
            reloadButton.addEventListener('click', () => {
                clearStaleIsolationFlags();
                window.location.reload();
            });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachReloadHandler, { once: true });
    } else {
        attachReloadHandler();
    }
})();
