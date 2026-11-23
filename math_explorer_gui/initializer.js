// Trunk initializer hooks for Math Explorer WebAssembly download progress and recovery
function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) {
        return '0 B';
    }
    if (bytes < 1000) {
        return Math.round(bytes) + ' B';
    }
    if (bytes < 1000000) {
        return (bytes / 1000).toFixed(1) + ' KB';
    }
    return (bytes / 1000000).toFixed(2) + ' MB';
}

function setIndeterminateProgress(progressEl, stageLabel) {
    if (!progressEl) {
        return;
    }
    progressEl.removeAttribute('value');
    progressEl.removeAttribute('aria-valuenow');
    progressEl.setAttribute('aria-valuemin', '0');
    progressEl.setAttribute('aria-valuemax', '100');
    progressEl.setAttribute('aria-valuetext', stageLabel);
}

const POLITE_MILESTONES = [25, 50, 75];
const POLITE_TASK_PACE_MS = 0;

export default function mathExplorerInitializer() {
    let maxLoadedBytes = 0;
    let compileStarted = false;
    let settled = false;
    let lastAnnouncedBucket = 0;
    let pendingPoliteMessages = [];
    let politeTimerId = null;
    let boundAbortSignal = null;

    function getElements() {
        return {
            subtitleEl: document.getElementById('loading_subtitle'),
            progressEl: document.getElementById('loading_progress'),
            bytesEl: document.getElementById('loading_bytes'),
        };
    }

    function isStartupFailed() {
        const boot = typeof window !== 'undefined' ? window.__mathExplorerBoot : null;
        if (boot && typeof boot.isBootFailed === 'function' && boot.isBootFailed()) {
            return true;
        }
        const root = typeof document !== 'undefined' ? document.documentElement : null;
        if (root && root.getAttribute('data-boot-failed') === 'true') {
            return true;
        }
        const errorContainer =
            typeof document !== 'undefined' ? document.getElementById('error_container') : null;
        if (
            errorContainer &&
            (errorContainer.getAttribute('data-error-active') === 'true' ||
                (errorContainer.style && errorContainer.style.display === 'flex'))
        ) {
            return true;
        }
        return false;
    }

    function clearPendingPoliteAnnouncements() {
        if (politeTimerId !== null && typeof window !== 'undefined') {
            window.clearTimeout(politeTimerId);
            politeTimerId = null;
        }
        pendingPoliteMessages = [];
    }

    function bindAbortCleanup() {
        const boot = typeof window !== 'undefined' ? window.__mathExplorerBoot : null;
        const signal =
            boot && typeof boot.getAbortSignal === 'function' ? boot.getAbortSignal() : null;
        if (!signal) {
            return;
        }
        if (signal.aborted) {
            clearPendingPoliteAnnouncements();
            return;
        }
        if (boundAbortSignal !== signal) {
            boundAbortSignal = signal;
            signal.addEventListener('abort', clearPendingPoliteAnnouncements, { once: true });
        }
    }

    function drainNextPoliteAnnouncement() {
        politeTimerId = null;
        if (isStartupFailed()) {
            pendingPoliteMessages = [];
            return;
        }
        if (pendingPoliteMessages.length === 0) {
            return;
        }
        const nextMessage = pendingPoliteMessages.shift();
        const boot = typeof window !== 'undefined' ? window.__mathExplorerBoot : null;
        if (boot && typeof boot.announce === 'function') {
            boot.announce(nextMessage, 'polite');
        }
        if (pendingPoliteMessages.length > 0 && typeof window !== 'undefined') {
            politeTimerId = window.setTimeout(drainNextPoliteAnnouncement, POLITE_TASK_PACE_MS);
        }
    }

    function enqueuePoliteAnnouncement(message, immediateIfIdle) {
        if (!message) {
            return;
        }
        if (isStartupFailed()) {
            clearPendingPoliteAnnouncements();
            return;
        }
        bindAbortCleanup();
        const boot = typeof window !== 'undefined' ? window.__mathExplorerBoot : null;
        if (!boot || typeof boot.announce !== 'function') {
            return;
        }
        if (immediateIfIdle && politeTimerId === null && pendingPoliteMessages.length === 0) {
            boot.announce(message, 'polite');
            if (typeof window !== 'undefined') {
                politeTimerId = window.setTimeout(drainNextPoliteAnnouncement, POLITE_TASK_PACE_MS);
            }
            return;
        }
        pendingPoliteMessages.push(message);
        if (politeTimerId === null && typeof window !== 'undefined') {
            politeTimerId = window.setTimeout(drainNextPoliteAnnouncement, POLITE_TASK_PACE_MS);
        }
    }

    function enqueueCrossedMilestones(pct) {
        for (let i = 0; i < POLITE_MILESTONES.length; i += 1) {
            const milestone = POLITE_MILESTONES[i];
            if (pct >= milestone && milestone > lastAnnouncedBucket) {
                lastAnnouncedBucket = milestone;
                enqueuePoliteAnnouncement(
                    'WebAssembly download ' + milestone + '% complete.',
                    true,
                );
            }
        }
    }

    return {
        onStart() {
            clearPendingPoliteAnnouncements();
            maxLoadedBytes = 0;
            compileStarted = false;
            settled = false;
            lastAnnouncedBucket = 0;
            bindAbortCleanup();

            const boot = window.__mathExplorerBoot;
            if (boot) {
                if (typeof boot.enterPhase === 'function') {
                    boot.enterPhase('transfer', 'Starting WebAssembly download');
                } else {
                    boot.recordProgress('Starting WebAssembly download');
                }
                enqueuePoliteAnnouncement(
                    'Downloading Math Explorer WebAssembly module.',
                    true,
                );
            }

            const { subtitleEl, progressEl, bytesEl } = getElements();
            if (subtitleEl) {
                subtitleEl.textContent = 'Downloading WebAssembly module...';
            }
            setIndeterminateProgress(progressEl, 'Starting WebAssembly download');
            if (bytesEl) {
                bytesEl.textContent = 'Starting download...';
            }
        },

        onProgress({ current, total } = {}) {
            if (isStartupFailed()) {
                clearPendingPoliteAnnouncements();
                return;
            }
            const safeCurrent = Number.isFinite(current) && current > 0 ? Math.floor(current) : 0;
            const safeTotal = Number.isFinite(total) && total > 0 ? Math.floor(total) : 0;

            const boot = window.__mathExplorerBoot;
            const { subtitleEl, progressEl, bytesEl } = getElements();

            // Indeterminate fallback when uncompressed sourceSize is unknown/zero
            if (safeTotal <= 0) {
                if (boot) {
                    boot.recordProgress('Downloading WebAssembly (' + safeCurrent + ' bytes)');
                }
                maxLoadedBytes = Math.max(maxLoadedBytes, safeCurrent);
                const statusCopy = maxLoadedBytes > 0
                    ? formatBytes(maxLoadedBytes) + ' downloaded'
                    : 'Downloading...';
                if (subtitleEl) {
                    subtitleEl.textContent = 'Downloading WebAssembly module...';
                }
                setIndeterminateProgress(progressEl, statusCopy);
                if (bytesEl) {
                    bytesEl.textContent = statusCopy;
                }
                return;
            }

            // Transition to indeterminate compilation stage once all bytes arrive
            if (safeCurrent >= safeTotal || compileStarted) {
                maxLoadedBytes = safeTotal;
                const firstCompileTransition = !compileStarted;
                compileStarted = true;

                if (boot) {
                    if (typeof boot.enterPhase === 'function') {
                        boot.enterPhase('compile', 'Compiling WebAssembly module');
                    } else {
                        boot.recordProgress('Compiling WebAssembly module');
                    }
                }

                if (subtitleEl) {
                    subtitleEl.textContent = 'Compiling WebAssembly module...';
                }
                setIndeterminateProgress(progressEl, 'Compiling WebAssembly module');
                if (bytesEl) {
                    bytesEl.textContent =
                        formatBytes(safeTotal) + ' / ' + formatBytes(safeTotal) + ' (100%) — compiling...';
                }
                if (firstCompileTransition && boot) {
                    enqueueCrossedMilestones(100);
                    enqueuePoliteAnnouncement(
                        'Download complete. Compiling WebAssembly module.',
                        true,
                    );
                }
                return;
            }

            if (boot) {
                boot.recordProgress('Downloading WebAssembly (' + safeCurrent + ' bytes)');
            }

            // Monotonic determinate byte progress during transfer
            const clampedCurrent = Math.min(safeCurrent, safeTotal);
            maxLoadedBytes = Math.max(maxLoadedBytes, clampedCurrent);
            const pct = Math.min(99, Math.floor((maxLoadedBytes / safeTotal) * 100));
            const formattedLoaded = formatBytes(maxLoadedBytes);
            const formattedTotal = formatBytes(safeTotal);

            if (subtitleEl) {
                subtitleEl.textContent = 'Downloading WebAssembly module...';
            }
            if (progressEl) {
                progressEl.max = safeTotal;
                progressEl.value = maxLoadedBytes;
                progressEl.setAttribute('aria-valuemin', '0');
                progressEl.setAttribute('aria-valuemax', String(safeTotal));
                progressEl.setAttribute('aria-valuenow', String(maxLoadedBytes));
                progressEl.setAttribute(
                    'aria-valuetext',
                    pct + '% (' + formattedLoaded + ' of ' + formattedTotal + ')'
                );
            }
            if (bytesEl) {
                bytesEl.textContent = formattedLoaded + ' / ' + formattedTotal + ' (' + pct + '%)';
            }

            enqueueCrossedMilestones(pct);
        },

        onComplete() {
            settled = true;
            const boot = window.__mathExplorerBoot;
            if (boot) {
                if (typeof boot.enterPhase === 'function') {
                    boot.enterPhase('compile', 'WebAssembly initialization settled');
                } else {
                    boot.recordProgress('WebAssembly initialization settled');
                }
            }
        },

        onSuccess() {
            settled = true;
            const boot = window.__mathExplorerBoot;
            if (boot) {
                if (typeof boot.enterPhase === 'function') {
                    boot.enterPhase('compile', 'Initializing WebGL graphics context');
                } else {
                    boot.recordProgress('Initializing WebGL graphics context');
                }
            }
            const { subtitleEl, progressEl, bytesEl } = getElements();
            if (subtitleEl) {
                subtitleEl.textContent = 'Initializing WebGL & graphics engine...';
            }
            setIndeterminateProgress(progressEl, 'Initializing WebGL and graphics engine');
            if (bytesEl && maxLoadedBytes > 0) {
                bytesEl.textContent = formatBytes(maxLoadedBytes) + ' loaded — starting graphics...';
            }
        },

        onFailure(error) {
            settled = true;
            clearPendingPoliteAnnouncements();
            const boot = window.__mathExplorerBoot;
            if (boot) {
                boot.handleFailure(error);
            }
        },

        cancelPendingAnnouncements() {
            clearPendingPoliteAnnouncements();
        },

        isSettled() {
            return settled;
        },
    };
}
