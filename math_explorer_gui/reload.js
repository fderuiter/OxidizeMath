// External reload button handler for Math Explorer GUI error boundary
(function () {
    function initReloadButton() {
        var reloadBtn = document.querySelector('.reload-btn');
        if (reloadBtn) {
            reloadBtn.addEventListener('click', function () {
                window.location.reload();
            });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initReloadButton);
    } else {
        initReloadButton();
    }
})();
