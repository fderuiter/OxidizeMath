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

function getLoadingElements() {
    return {
        subtitleEl: document.getElementById('loading_subtitle'),
        progressEl: document.getElementById('loading_progress'),
        bytesEl: document.getElementById('loading_bytes'),
    };
}

function notifyCompilePhase(boot, label) {
    if (!boot) {
        return;
    }
    if (typeof boot.enterPhase === 'function') {
        boot.enterPhase('compile', label);
    } else if (typeof boot.recordProgress === 'function') {
        boot.recordProgress(label);
    }
}

export default function mathExplorerInitializer() {
    let maxLoadedBytes = 0;
    let compileStarted = false;
    let lastAnnouncedBucket = 0;

    function updateIndeterminateTransfer(safeCurrent, boot, els) {
        if (boot) {
            boot.recordProgress('Downloading WebAssembly (' + safeCurrent + ' bytes)');
        }
        maxLoadedBytes = Math.max(maxLoadedBytes, safeCurrent);
        const statusCopy =
            maxLoadedBytes > 0 ? formatBytes(maxLoadedBytes) + ' downloaded' : 'Downloading...';
        if (els.subtitleEl) {
            els.subtitleEl.textContent = 'Downloading WebAssembly module...';
        }
        setIndeterminateProgress(els.progressEl, statusCopy);
        if (els.bytesEl) {
            els.bytesEl.textContent = statusCopy;
        }
    }

    function updateCompileTransition(safeTotal, boot, els) {
        maxLoadedBytes = safeTotal;
        const firstCompileTransition = !compileStarted;
        compileStarted = true;
        notifyCompilePhase(boot, 'Compiling WebAssembly module');
        if (els.subtitleEl) {
            els.subtitleEl.textContent = 'Compiling WebAssembly module...';
        }
        setIndeterminateProgress(els.progressEl, 'Compiling WebAssembly module');
        if (els.bytesEl) {
            els.bytesEl.textContent =
                formatBytes(safeTotal) + ' / ' + formatBytes(safeTotal) + ' (100%) — compiling...';
        }
        if (firstCompileTransition && boot) {
            boot.announce('Download complete. Compiling WebAssembly module.', 'polite');
        }
    }

    function updateDeterminateTransfer(safeCurrent, safeTotal, boot, els) {
        if (boot) {
            boot.recordProgress('Downloading WebAssembly (' + safeCurrent + ' bytes)');
        }
        maxLoadedBytes = Math.max(maxLoadedBytes, Math.min(safeCurrent, safeTotal));
        const pct = Math.min(99, Math.floor((maxLoadedBytes / safeTotal) * 100));
        const formattedLoaded = formatBytes(maxLoadedBytes);
        const formattedTotal = formatBytes(safeTotal);
        if (els.subtitleEl) {
            els.subtitleEl.textContent = 'Downloading WebAssembly module...';
        }
        if (els.progressEl) {
            els.progressEl.max = safeTotal;
            els.progressEl.value = maxLoadedBytes;
            els.progressEl.setAttribute('aria-valuemin', '0');
            els.progressEl.setAttribute('aria-valuemax', String(safeTotal));
            els.progressEl.setAttribute('aria-valuenow', String(maxLoadedBytes));
            els.progressEl.setAttribute(
                'aria-valuetext',
                pct + '% (' + formattedLoaded + ' of ' + formattedTotal + ')'
            );
        }
        if (els.bytesEl) {
            els.bytesEl.textContent = formattedLoaded + ' / ' + formattedTotal + ' (' + pct + '%)';
        }
        const bucket = Math.floor(pct / 25) * 25;
        if (bucket > lastAnnouncedBucket && bucket < 100 && boot) {
            lastAnnouncedBucket = bucket;
            boot.announce('WebAssembly download ' + bucket + '% complete.', 'polite');
        }
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
            const els = getLoadingElements();
            if (els.subtitleEl) {
                els.subtitleEl.textContent = 'Downloading WebAssembly module...';
            }
            setIndeterminateProgress(els.progressEl, 'Starting WebAssembly download');
            if (els.bytesEl) {
                els.bytesEl.textContent = 'Starting download...';
            }
        },

        onProgress({ current, total } = {}) {
            const safeCurrent = Number.isFinite(current) && current > 0 ? Math.floor(current) : 0;
            const safeTotal = Number.isFinite(total) && total > 0 ? Math.floor(total) : 0;
            const boot = window.__mathExplorerBoot;
            const els = getLoadingElements();
            if (safeTotal <= 0) {
                updateIndeterminateTransfer(safeCurrent, boot, els);
                return;
            }
            if (safeCurrent >= safeTotal || compileStarted) {
                updateCompileTransition(safeTotal, boot, els);
                return;
            }
            updateDeterminateTransfer(safeCurrent, safeTotal, boot, els);
        },

        onComplete() {
            notifyCompilePhase(window.__mathExplorerBoot, 'WebAssembly initialization settled');
        },

        onSuccess() {
            notifyCompilePhase(window.__mathExplorerBoot, 'Initializing WebGL graphics context');
            const els = getLoadingElements();
            if (els.subtitleEl) {
                els.subtitleEl.textContent = 'Initializing WebGL & graphics engine...';
            }
            setIndeterminateProgress(els.progressEl, 'Initializing WebGL and graphics engine');
            if (els.bytesEl && maxLoadedBytes > 0) {
                els.bytesEl.textContent =
                    formatBytes(maxLoadedBytes) + ' loaded — starting graphics...';
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
