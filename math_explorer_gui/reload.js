// External reload button handler for Math Explorer GUI error boundary
(function () {
    function attachReloadHandler() {
        const reloadButton = document.querySelector('.reload-btn');
        if (reloadButton) {
            reloadButton.addEventListener('click', () => {
                window.location.reload();
            });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachReloadHandler);
    } else {
        attachReloadHandler();
    }
})();
