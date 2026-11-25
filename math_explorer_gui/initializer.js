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

function setIndeterminateProgress(progressEl, stageLabel, phase) {
    if (!progressEl) {
        return;
    }
    if (phase) {
        progressEl.setAttribute('data-phase', phase);
    } else {
        progressEl.removeAttribute('data-phase');
    }
    progressEl.removeAttribute('value');
    progressEl.removeAttribute('aria-valuenow');
    progressEl.setAttribute('aria-valuemin', '0');
    progressEl.setAttribute('aria-valuemax', '100');
    progressEl.setAttribute('aria-valuetext', stageLabel);
}

const POLITE_MILESTONES = [25, 50, 75];
const POLITE_TASK_PACE_MS = 0;

export default function mathExplorerInitializer(options = {}) {
    let maxLoadedBytes = 0;
    let compileStarted = false;
    let settled = false;
    let lastAnnouncedBucket = 0;
    let pendingPoliteMessages = [];
    let politeTimerId = null;
    let boundAbortSignal = null;
    // Start fallback polite toggle at slot 1 so slot 0 remains empty for Rust's AtomicBool::new(false).
    let fallbackPoliteSlotToggle = 1;

    function getElements() {
        return {
            subtitleEl: document.getElementById('loading_subtitle'),
            progressEl: document.getElementById('loading_progress'),
            bytesEl: document.getElementById('loading_bytes'),
        };
    }

    function isStartupFailed() {
        const optSignal = options && options.signal;
        if ((optSignal && optSignal.aborted) || (boundAbortSignal && boundAbortSignal.aborted)) {
            return true;
        }
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
            (options && options.signal) ||
            (boot && typeof boot.getAbortSignal === 'function' ? boot.getAbortSignal() : null);
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

    function announcePolite(message) {
        if (!message) {
            return;
        }
        const boot = typeof window !== 'undefined' ? window.__mathExplorerBoot : null;
        if (boot && typeof boot.announce === 'function') {
            boot.announce(message, 'polite');
            return;
        }
        if (typeof document === 'undefined') {
            return;
        }
        const writePoliteSlot = () => {
            const slot0 = document.getElementById('aria-live-polite-slot-0');
            const slot1 = document.getElementById('aria-live-polite-slot-1');
            let slotIdx = fallbackPoliteSlotToggle;
            if (slot0 && slot1) {
                const slot0Busy = Boolean(slot0.textContent && slot0.textContent.length > 0);
                const slot1Busy = Boolean(slot1.textContent && slot1.textContent.length > 0);
                if (slot0Busy && !slot1Busy) {
                    slotIdx = 1;
                } else if (slot1Busy && !slot0Busy) {
                    slotIdx = 0;
                }
            }
            fallbackPoliteSlotToggle = 1 - slotIdx;
            const activeSlot = slotIdx === 0 ? slot0 : slot1;
            const siblingSlot = slotIdx === 0 ? slot1 : slot0;
            if (siblingSlot) {
                siblingSlot.textContent = '';
            }
            if (activeSlot) {
                activeSlot.textContent = String(message);
            }
        };
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', writePoliteSlot, { once: true });
        } else {
            writePoliteSlot();
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
        announcePolite(nextMessage);
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
        if (immediateIfIdle && politeTimerId === null && pendingPoliteMessages.length === 0) {
            announcePolite(message);
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
            }
            enqueuePoliteAnnouncement(
                'Downloading Math Explorer WebAssembly module.',
                true,
            );

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
                setIndeterminateProgress(progressEl, 'Compiling WebAssembly module', 'compile');
                if (bytesEl) {
                    bytesEl.textContent =
                        formatBytes(safeTotal) + ' / ' + formatBytes(safeTotal) + ' (100%) — compiling...';
                }
                if (firstCompileTransition) {
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
                progressEl.removeAttribute('data-phase');
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
            setIndeterminateProgress(progressEl, 'Initializing WebGL and graphics engine', 'compile');
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
