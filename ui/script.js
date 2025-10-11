const initBtn = document.getElementById('initBtn');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const languageSelect = document.getElementById('languageSelect');
const transcriptionBox = document.getElementById('transcriptionBox');
const translationBox = document.getElementById('translationBox');
const sttTime = document.getElementById('sttTime');
const mtTime = document.getElementById('mtTime');
const ttsTime = document.getElementById('ttsTime');
const totalTime = document.getElementById('totalTime');
const statusText = document.getElementById('statusText');
const statusIndicator = document.getElementById('statusIndicator');
const statusLabel = document.getElementById('statusLabel');

let ws;
let mediaRecorder;
let audioContext;
let audioStream;
let isInitialized = false;

// Function to update status and indicator
function updateStatus(message, isOn = false) {
    statusText.textContent = message;
    statusLabel.textContent = message;
    if (isOn) {
        statusIndicator.classList.remove('off');
        statusIndicator.classList.add('on');
    } else {
        statusIndicator.classList.remove('on');
        statusIndicator.classList.add('off');
    }
}

// WebSocket connection
function connectWebSocket() {
    ws = new WebSocket("ws://localhost:8000/ws");

    ws.onopen = () => {
        updateStatus("Connected", true);
        initBtn.disabled = false;
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        console.log("Received:", message);

        if (message.type === "status") {
            updateStatus(message.data, message.data.includes("Ready") || message.data.includes("Recording"));
        } else if (message.type === "init_complete") {
            isInitialized = true;
            startBtn.disabled = false;
            initBtn.disabled = true;
            updateStatus("Ready", true);
        } else if (message.type === "transcription") {
            transcriptionBox.innerHTML = `<p>${message.data.text}</p>`;
            translationBox.innerHTML = `<p>${message.data.translation}</p>`;
            sttTime.textContent = `${message.data.metrics.stt_time.toFixed(2)}s`;
            mtTime.textContent = `${message.data.metrics.mt_time.toFixed(2)}s`;
            ttsTime.textContent = `${message.data.metrics.tts_time.toFixed(2)}s`;
            totalTime.textContent = `${message.data.metrics.total_latency.toFixed(2)}s`;
            transcriptionBox.scrollTop = transcriptionBox.scrollHeight;
            translationBox.scrollTop = translationBox.scrollHeight;
        } else if (message.type === "error") {
            updateStatus(`Error: ${message.data}`, false);
            stopRecording();
        }
    };

    ws.onclose = () => {
        updateStatus("Disconnected", false);
        initBtn.disabled = true;
        startBtn.disabled = true;
        stopBtn.disabled = true;
        isInitialized = false;
        console.log("WebSocket disconnected. Reconnecting in 5 seconds...");
        setTimeout(connectWebSocket, 5000);
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        updateStatus("Connection Error", false);
        ws.close();
    };
}

// Start recording audio from microphone
async function startRecording() {
    if (!isInitialized) {
        updateStatus("Please initialize first.", false);
        return;
    }
    if (ws.readyState !== WebSocket.OPEN) {
        updateStatus("WebSocket not connected.", false);
        return;
    }

    try {
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(audioStream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1); // 4096 buffer size, 1 input, 1 output

        processor.onaudioprocess = (event) => {
            if (ws.readyState === WebSocket.OPEN) {
                const audioData = event.inputBuffer.getChannelData(0);
                // Convert Float32Array to Int16Array for sending
                const int16Array = new Int16Array(audioData.length);
                for (let i = 0; i < audioData.length; i++) {
                    int16Array[i] = Math.max(-1, Math.min(1, audioData[i])) * 0x7FFF;
                }
                ws.send(int16Array.buffer);
            }
        };

        source.connect(processor);
        processor.connect(audioContext.destination);

        ws.send(JSON.stringify({ type: "start" }));
        startBtn.disabled = true;
        stopBtn.disabled = false;
        updateStatus("Recording...", true);
        transcriptionBox.innerHTML = '<p class="placeholder">Waiting for speech...</p>';
        translationBox.innerHTML = '<p class="placeholder">Translation will appear here...</p>';

    } catch (err) {
        console.error("Error accessing microphone:", err);
        updateStatus(`Mic error: ${err.message}`, false);
        stopRecording();
    }
}

// Stop recording
function stopRecording() {
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "stop" }));
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
    updateStatus("Stopped", false);
}

// Event Listeners
initBtn.addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "init" }));
        initBtn.disabled = true;
        updateStatus("Initializing...", true);
    } else {
        updateStatus("WebSocket not connected. Attempting to reconnect...", false);
        connectWebSocket();
    }
});

startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

languageSelect.addEventListener('change', (event) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "language", data: event.target.value }));
    }
});

// Initial connection
connectWebSocket();
