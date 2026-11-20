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

export default function mathExplorerInitializer() {
    let maxLoadedBytes = 0;
    let compileStarted = false;
    let lastAnnouncedBucket = 0;

    function getElements() {
        return {
            subtitleEl: document.getElementById('loading_subtitle'),
            progressEl: document.getElementById('loading_progress'),
            bytesEl: document.getElementById('loading_bytes'),
        };
    }

    return {
        onStart() {
            maxLoadedBytes = 0;
            compileStarted = false;
            lastAnnouncedBucket = 0;

            const boot = window.__mathExplorerBoot;
            if (boot) {
                if (typeof boot.enterPhase === 'function') {
                    boot.enterPhase('transfer', 'Starting WebAssembly download');
                } else {
                    boot.recordProgress('Starting WebAssembly download');
                }
                boot.announce('Downloading Math Explorer WebAssembly module.', 'polite');
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
                    boot.announce('Download complete. Compiling WebAssembly module.', 'polite');
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

            const bucket = Math.floor(pct / 25) * 25;
            if (bucket > lastAnnouncedBucket && bucket < 100 && boot) {
                lastAnnouncedBucket = bucket;
                boot.announce('WebAssembly download ' + bucket + '% complete.', 'polite');
            }
        },

        onComplete() {
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
            const boot = window.__mathExplorerBoot;
            if (boot) {
                boot.handleFailure(error);
            }
        },
    };
}
