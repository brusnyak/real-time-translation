document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const initBtn = document.getElementById('initBtn');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusIndicator = document.getElementById('statusIndicator');
    const statusLabel = document.getElementById('statusLabel');
    const inputLevelSegmentsContainer = document.getElementById('inputLevelSegments');
    const NUM_LEVEL_SEGMENTS = 10;
    let levelSegments = [];
    const ttsModelSelect = document.getElementById('ttsModelSelect');
    const inputLanguageSelect = document.getElementById('inputLanguageSelect');
    const outputLanguageSelect = document.getElementById('outputLanguageSelect');
    const inputLanguageBadge = document.getElementById('inputLanguageBadge');
    const outputLanguageBadge = document.getElementById('outputLanguageBadge');
    const transcriptionBox = document.getElementById('transcriptionBox');
    const translationBox = document.getElementById('translationBox');
    const sttTime = document.getElementById('sttTime');
    const mtTime = document.getElementById('mtTime');
    const ttsTime = document.getElementById('ttsTime');
    const totalTime = document.getElementById('totalTime');
    const stateListening = document.getElementById('stateListening');
    const stateTranslating = document.getElementById('stateTranslating');
    const stateSpeaking = document.getElementById('stateSpeaking');
    const settingsStatusText = document.getElementById('settingsStatusText'); // Existing status text
    const activityLog = document.getElementById('activityLog'); // New activity log container

    // --- Global State ---
    let websocket = null;
    let audioContext = null;
    let mediaStream = null;
    let audioProcessor = null;
    let isInitialized = false;
    let isRecording = false;

    // Build API / WS URLs using current page protocol and host (safer on dev vs prod)
    const proto = window.location.protocol === 'https:' ? 'https' : 'http';
    const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const API_BASE_URL = `${proto}://${window.location.hostname}${window.location.port ? ':' + window.location.port : ''}`;
    const WS_URL = `${wsProto}://${window.location.hostname}${window.location.port ? ':' + window.location.port : ''}/ws`;
    const API_URL = `${API_BASE_URL}`;

    // Initialize level segments
    for (let i = 0; i < NUM_LEVEL_SEGMENTS; i++) {
        const segment = document.createElement('div');
        segment.classList.add('level-segment');
        inputLevelSegmentsContainer.appendChild(segment);
        levelSegments.push(segment);
    }

    const SAMPLE_RATE = 16000;
    const BUFFER_SIZE = 4096;

    // --- Input level smoothing state ----
    let emaLevel = 0.0; // exponential moving average for smoothing
    const EMA_ALPHA = 0.3; // Increased for more responsiveness

    // --- Throttle state for debug logs (avoid spamming console) ---
    let lastLogAt = 0;
    const LOG_THROTTLE_MS = 1000; // log at most once per second

    // --- Utility Functions ---
    function updateStatus(indicatorClass, labelText) {
        statusIndicator.className = `indicator ${indicatorClass}`;
        statusLabel.textContent = labelText;
    }

    function updateStateIcon(element, isActive) {
        if (isActive) {
            element.classList.remove('idle');
            element.classList.add('active');
        } else {
            element.classList.remove('active');
            element.classList.add('idle');
        }
    }

    function setInputLevel(level) {
        // Expect 'level' to be a 0..1 RMS-like magnitude (server-side should send RMS or average absolute)
        // Apply sqrt to approximate perceptual loudness, then EMA smoothing
        const scaled = Math.sqrt(Math.min(1, Math.max(0, level)));
        emaLevel = emaLevel * (1 - EMA_ALPHA) + scaled * EMA_ALPHA;

        const normalizedLevel = Math.min(1, Math.max(0, emaLevel));
        const activeSegments = Math.ceil(normalizedLevel * NUM_LEVEL_SEGMENTS);

        levelSegments.forEach((segment, index) => {
            if (index < activeSegments) {
                segment.classList.add('active');

                // style classes for medium/high ranges
                if (normalizedLevel > 0.7) {
                    segment.classList.add('high');
                    segment.classList.remove('medium');
                } else if (normalizedLevel > 0.3) {
                    segment.classList.add('medium');
                    segment.classList.remove('high');
                } else {
                    segment.classList.remove('medium', 'high');
                }
            } else {
                segment.classList.remove('active', 'medium', 'high');
            }
        });
    }

    function resetUI() {
        transcriptionBox.innerHTML = '<p class="placeholder">Waiting for speech...</p>';
        translationBox.innerHTML = '<p class="placeholder">Translation will appear here...</p>';
        sttTime.textContent = '0.0s';
        mtTime.textContent = '0.0s';
        ttsTime.textContent = '0.0s';
        totalTime.textContent = '0.0s';
        emaLevel = 0;
        setInputLevel(0);
        updateStateIcon(stateListening, false);
        updateStateIcon(stateTranslating, false);
        updateStateIcon(stateSpeaking, false);
        settingsStatusText.textContent = ''; // Clear existing status text
        activityLog.innerHTML = ''; // Clear activity log
    }

    function appendLog(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('p');
        logEntry.classList.add('log-entry', `log-${type}`);
        logEntry.innerHTML = `[${timestamp}] ${escapeHtml(message)}`;
        activityLog.prepend(logEntry); // Add to top
        // Optional: Limit log entries to prevent UI clutter
        while (activityLog.children.length > 50) {
            activityLog.removeChild(activityLog.lastChild);
        }
    }

    // --- WebSocket Handling ---
    function connectWebSocket() {
        try {
            websocket = new WebSocket(WS_URL);
        } catch (err) {
            settingsStatusText.textContent = `Invalid WS URL: ${WS_URL}`;
            updateStatus('error', 'WS Error');
            return;
        }

        websocket.onopen = () => {
            console.log('WebSocket connected.');
            updateStatus('on', 'Connected');
            appendLog('WebSocket connected.', 'success');
        };

        websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === 'audio_level') {
                    setInputLevel(typeof data.level === 'number' ? data.level : 0);
                } else if (data.type === 'transcription_result') {
                    if (data.transcribed) {
                        transcriptionBox.innerHTML = `<p>${escapeHtml(data.transcribed)}</p>`;
                        sttTime.textContent = `${(data.metrics.stt_time || 0).toFixed(2)}s`;
                        updateStateIcon(stateTranslating, true);
                    }
                } else if (data.type === 'translation_result') {
                    if (data.translated) {
                        translationBox.innerHTML = `<p>${escapeHtml(data.translated)}</p>`;
                        mtTime.textContent = `${(data.metrics.mt_time || 0).toFixed(2)}s`;
                        updateStateIcon(stateTranslating, false);
                        updateStateIcon(stateSpeaking, true);
                    }
                } else if (data.type === 'final_metrics') {
                    if (data.metrics) {
                        ttsTime.textContent = `${(data.metrics.tts_time || 0).toFixed(2)}s`;
                        totalTime.textContent = `${(data.metrics.total_latency || 0).toFixed(2)}s`;
                        updateStateIcon(stateSpeaking, false);
                    }
                } else if (data.type === 'status') {
                    // Backend status messages go to the new activity log
                    appendLog(data.message || '', 'info');
                } else if (data.status === 'error') {
                    console.error('Server Error:', data.message);
                    appendLog(`Server Error: ${data.message}`, 'error');
                    settingsStatusText.textContent = `Error: ${data.message}`; // Keep critical error in settingsStatusText
                    stopRecording();
                }
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
                appendLog(`Error parsing WebSocket message: ${error.message}`, 'error');
            }
        };

        websocket.onclose = () => {
            console.log('WebSocket disconnected.');
            updateStatus('off', 'Disconnected');
            appendLog('WebSocket disconnected.', 'warning');
            stopRecording();
        };

        websocket.onerror = (error) => {
            console.error('WebSocket Error:', error);
            settingsStatusText.textContent = 'WebSocket connection error.';
            updateStatus('error', 'Error');
            appendLog(`WebSocket Error: ${error.message}`, 'error');
            stopRecording();
        };
    }

    function closeWebSocket() {
        if (websocket) {
            websocket.close();
            websocket = null;
        }
    }

    // --- Audio Recording Handling ---
    async function startRecording() {
        if (isRecording) return;

        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                }
            });

            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: SAMPLE_RATE
            });

            const source = audioContext.createMediaStreamSource(mediaStream);

            // NOTE: createScriptProcessor is deprecated in favor of AudioWorklet.
            // For simplicity and compatibility we use it here but consider AudioWorklet for production.
            audioProcessor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);
            audioProcessor.onaudioprocess = (event) => {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    const audioData = event.inputBuffer.getChannelData(0);
                    const audioClone = new Float32Array(audioData); // copy before sending

                    // Send as binary message (server must accept Float32Array buffer)
                    websocket.send(audioClone.buffer);

                    // Throttled debug log to avoid heavy console spam
                    const now = Date.now();
                    if (now - lastLogAt > LOG_THROTTLE_MS) {
                        console.debug(`Sent ${audioClone.length} audio samples`);
                        lastLogAt = now;
                    }
                }
            };

            source.connect(audioProcessor);
            // audioProcessor -> destination is required for some browsers to keep the processing alive
            audioProcessor.connect(audioContext.destination);

            isRecording = true;
            updateStatus('on', 'Listening...');
            startBtn.disabled = true;
            stopBtn.disabled = false;
            updateStateIcon(stateListening, true);

            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({ type: 'start' }));
            }
        } catch (error) {
            console.error('Error accessing microphone:', error);
            settingsStatusText.textContent = `Microphone access denied: ${error.message}`;
            updateStatus('error', 'Mic Error');
            stopRecording();
        }
    }

    function stopRecording() {
        if (!isRecording) return;

        isRecording = false;
        updateStatus('off', 'Ready');
        startBtn.disabled = false;
        stopBtn.disabled = true;
        updateStateIcon(stateListening, false);
        updateStateIcon(stateTranslating, false);
        updateStateIcon(stateSpeaking, false);
        resetUI();

        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        if (audioProcessor) {
            try {
                audioProcessor.disconnect();
            } catch (_) { /* ignore */ }
            audioProcessor = null;
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }

        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({ type: 'stop' }));
        }
    }

    // --- Helpers & Events ---
    initBtn.addEventListener('click', async () => {
        if (isInitialized) {
            settingsStatusText.textContent = 'Pipeline already initialized.';
            return;
        }
        initBtn.disabled = true;
        updateStatus('prepping', 'Initializing Models...');
        appendLog('Initializing models...', 'info');
        settingsStatusText.textContent = 'Loading models (this may take ~1 minute on first run)...'; // Keep this for prominent display

        try {
            const sourceLang = inputLanguageSelect.value;
            const targetLang = outputLanguageSelect.value;
            const ttsModel = ttsModelSelect.value;

            appendLog(`Attempting to initialize pipeline with: Source=${sourceLang.toUpperCase()}, Target=${targetLang.toUpperCase()}, TTS Model=${ttsModel}.`, 'info');

            const response = await fetch(`${API_URL}/initialize?source_lang=${sourceLang}&target_lang=${targetLang}&tts_model_choice=${ttsModel}`, { method: 'POST' });
            const data = await response.json();

            if (data.status === 'success') {
                isInitialized = true;
                startBtn.disabled = false;
                updateStatus('off', 'Ready');
                settingsStatusText.textContent = data.message;
                appendLog(`Pipeline initialized successfully: ${data.message}`, 'success');
                connectWebSocket();
            } else {
                settingsStatusText.textContent = `Initialization failed: ${data.message}`;
                updateStatus('error', 'Init Error');
                appendLog(`Initialization failed: ${data.message}`, 'error');
                initBtn.disabled = false;
            }
        } catch (error) {
            console.error('Initialization API error:', error);
            settingsStatusText.textContent = `Initialization failed: ${error.message}`;
            updateStatus('error', 'Init Error');
            appendLog(`Initialization API error: ${error.message}`, 'error');
            initBtn.disabled = false;
        }
    });

    startBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);

    function sendConfigUpdate() {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            const config = {
                type: 'config_update',
                source_lang: inputLanguageSelect.value,
                target_lang: outputLanguageSelect.value,
                tts_model_choice: ttsModelSelect.value
            };
            websocket.send(JSON.stringify(config));
            appendLog(`Configuration updated: Source=${config.source_lang.toUpperCase()}, Target=${config.target_lang.toUpperCase()}, TTS Model=${config.tts_model_choice}.`, 'info');
            settingsStatusText.textContent = 'Configuration updated.'; // Keep this for immediate feedback
        } else {
            settingsStatusText.textContent = 'Not connected to server. Cannot update config.';
            appendLog('Not connected to server. Cannot update config.', 'warning');
        }
    }

    inputLanguageSelect.addEventListener('change', (event) => {
        inputLanguageBadge.textContent = event.target.value.toUpperCase();
        appendLog(`Input language changed to ${event.target.value.toUpperCase()}.`, 'info');
        sendConfigUpdate();
    });

    outputLanguageSelect.addEventListener('change', (event) => {
        outputLanguageBadge.textContent = event.target.value.toUpperCase();
        appendLog(`Output language changed to ${event.target.value.toUpperCase()}.`, 'info');
        sendConfigUpdate();
    });

    ttsModelSelect.addEventListener('change', (event) => {
        appendLog(`TTS Model changed to ${event.target.value}.`, 'info');
        sendConfigUpdate();
    });

    // Basic HTML-escape helper to avoid injection when updating innerHTML
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Initial UI setup
    resetUI();
    updateStatus('off', 'Awaiting Initialization');
    startBtn.disabled = true;
    stopBtn.disabled = true;
});
