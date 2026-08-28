/**
 * MAITRI — Real-Time Multimodal AI Spacecraft HUD Client
 * Live Optical Face Tracking, Real-Time Audio Prosody, Continuous Voice AI
 * ISRO Bhartiya Antariksh Station (BAS)
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const videoElem = document.getElementById('webcam-video');
    const hudCanvas = document.getElementById('hud-canvas');
    const hudCtx = hudCanvas.getContext('2d');
    const waveformCanvas = document.getElementById('audio-waveform-canvas');
    const waveformCtx = waveformCanvas.getContext('2d');
    
    const btnToggleCamera = document.getElementById('btn-toggle-camera');
    const btnToggleVoiceListen = document.getElementById('btn-toggle-voice-listen');
    const crewSelector = document.getElementById('crew-selector');
    
    const chatInput = document.getElementById('chat-input');
    const btnSendChat = document.getElementById('btn-send-chat');
    const btnVoiceInput = document.getElementById('btn-voice-input');
    const chatContainer = document.getElementById('chat-messages');
    
    // Telemetry DOM Bindings
    const domEmotionText = document.getElementById('dominant-emotion-text');
    const domEmotionConf = document.getElementById('dominant-emotion-conf');
    const riskBadge = document.getElementById('risk-level-badge');
    const riskScoreVal = document.getElementById('risk-score-value');
    const riskBarFill = document.getElementById('risk-bar-fill');
    
    const perclosVal = document.getElementById('vital-perclos');
    const blinkVal = document.getElementById('vital-blinks');
    const yawnVal = document.getElementById('vital-yawns');
    const painVal = document.getElementById('vital-pain');
    const fatigueLevelText = document.getElementById('vital-fatigue-level');
    
    const pitchVal = document.getElementById('vital-pitch');
    const vocalTensionVal = document.getElementById('vital-vocal-tension');
    
    const pacerCircle = document.getElementById('pacer-circle');
    const pacerState = document.getElementById('pacer-state');
    const pacerSeconds = document.getElementById('pacer-seconds');
    
    const alertFeedList = document.getElementById('alert-feed-list');
    const alertCountBadge = document.getElementById('alert-count-badge');
    const discordanceBanner = document.getElementById('discordance-banner');
    const discordanceText = document.getElementById('discordance-text');

    // Emotion Color Mapping
    const emotionColors = {
        'happy': '#10b981',
        'neutral': '#00f0ff',
        'stressed': '#f59e0b',
        'fatigued': '#8b5cf6',
        'anxious': '#f97316',
        'sad': '#3b82f6',
        'frustrated': '#ef4444'
    };

    // State Variables
    let isCameraActive = false;
    let isContinuousListening = false;
    let mediaStream = null;
    let audioContext = null;
    let audioAnalyser = null;
    let audioDataArray = null;
    let audioTimeData = null;
    let streamInterval = null;
    let animFrameId = null;
    let isProcessingBackend = false;
    let recognition = null;
    let currentSpeechText = "";

    // Live Real-Time Client Biometrics (for 30 FPS rendering)
    let livePitchHz = 0;
    let liveRmsEnergy = 0;
    let liveVocalTension = 0;
    let liveBlinkCount = 0;
    let liveYawnCount = 0;
    let lastBlinkTime = 0;
    let lastYawnTime = 0;
    let faceBox = null;
    let smoothEar = 0.30;
    let smoothMar = 0.20;

    // 1. Initialize Continuous Speech Recognition (Web Speech API)
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRec();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
            let interim = '';
            let finalTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interim += event.results[i][0].transcript;
                }
            }

            if (finalTranscript && finalTranscript.trim().length > 1) {
                currentSpeechText = finalTranscript.trim();
                sendChatMessage(currentSpeechText);
                currentSpeechText = "";
            } else if (interim) {
                chatInput.placeholder = `Listening: "${interim}"...`;
            }
        };

        recognition.onerror = (e) => {
            console.log('[SpeechRec Error]:', e);
        };

        recognition.onend = () => {
            chatInput.placeholder = "Speak or type to MAITRI...";
            if (isContinuousListening) {
                try { recognition.start(); } catch(e) {}
            }
        };
    }

    // Toggle Continuous Voice Listening
    btnToggleVoiceListen.addEventListener('click', () => {
        if (!recognition) {
            alert("Speech recognition not supported in this browser. Please type your message.");
            return;
        }
        isContinuousListening = !isContinuousListening;
        if (isContinuousListening) {
            btnToggleVoiceListen.classList.add('active');
            btnToggleVoiceListen.innerHTML = '🔴 LIVE VOICE: LISTENING';
            try { recognition.start(); } catch(e) {}
        } else {
            btnToggleVoiceListen.classList.remove('active');
            btnToggleVoiceListen.innerHTML = '🎙️ LIVE VOICE: OFF';
            try { recognition.stop(); } catch(e) {}
        }
    });

    // 2. Camera Toggle & Live Stream Start
    btnToggleCamera.addEventListener('click', async () => {
        if (!isCameraActive) {
            await startLiveCamera();
        } else {
            stopLiveCamera();
        }
    });

    async function startLiveCamera() {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
                audio: true
            });
            videoElem.srcObject = mediaStream;
            await videoElem.play();
            isCameraActive = true;
            btnToggleCamera.innerHTML = '🛑 STOP OPTICAL';
            btnToggleCamera.classList.add('active');

            // Start Live Audio Analyser
            initAudioAnalyser(mediaStream);

            // Start 30 FPS Canvas Rendering Loop
            renderLiveHudCanvas();

            // Start Periodic Backend Multimodal Fusion (~4 FPS)
            streamInterval = setInterval(sendFrameToBackend, 250);
        } catch (err) {
            alert("Camera/Microphone access error: " + err.message + "\nTip: You can also use the Flight Simulation Scenario buttons at the top to test with synthetic live data.");
        }
    }

    function stopLiveCamera() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
        }
        if (streamInterval) clearInterval(streamInterval);
        if (animFrameId) cancelAnimationFrame(animFrameId);
        videoElem.srcObject = null;
        isCameraActive = false;
        btnToggleCamera.innerHTML = '📹 START REAL-TIME OPTICAL';
        btnToggleCamera.classList.remove('active');
        
        // Clear canvas
        hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
        waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }

    // 3. High-Frequency Real-Time Audio Prosody Analyser (Web Audio API)
    function initAudioAnalyser(stream) {
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            audioAnalyser = audioContext.createAnalyser();
            audioAnalyser.fftSize = 1024;
            source.connect(audioAnalyser);
            
            audioDataArray = new Uint8Array(audioAnalyser.frequencyBinCount);
            audioTimeData = new Float32Array(audioAnalyser.fftSize);
        } catch (e) {
            console.log("Audio analyser init error:", e);
        }
    }

    // Autocorrelation Pitch (F0) detector in JavaScript
    function detectPitchAutocorr(buffer, sampleRate) {
        let size = buffer.length;
        let sumOfSquares = 0;
        for (let i = 0; i < size; i++) {
            sumOfSquares += buffer[i] * buffer[i];
        }
        let rms = Math.sqrt(sumOfSquares / size);
        if (rms < 0.015) return -1; // Silence

        let r1 = 0, r2 = size - 1, thres = 0.2;
        for (let i = 0; i < size / 2; i++) {
            if (Math.abs(buffer[i]) < thres) { r1 = i; break; }
        }
        for (let i = 1; i < size / 2; i++) {
            if (Math.abs(buffer[size - i]) < thres) { r2 = size - i; break; }
        }

        buffer = buffer.slice(r1, r2);
        size = buffer.length;

        let c = new Array(size).fill(0);
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size - i; j++) {
                c[i] += buffer[j] * buffer[j + i];
            }
        }

        let d = 0;
        while (c[d] > c[d + 1]) d++;
        let maxval = -1, maxpos = -1;
        for (let i = d; i < size; i++) {
            if (c[i] > maxval) {
                maxval = c[i];
                maxpos = i;
            }
        }
        let T0 = maxpos;
        if (T0 > 0) {
            return sampleRate / T0;
        }
        return -1;
    }

    // 4. Smooth 30 FPS HUD Canvas Rendering Loop
    function renderLiveHudCanvas() {
        if (!isCameraActive) return;
        animFrameId = requestAnimationFrame(renderLiveHudCanvas);

        hudCanvas.width = videoElem.videoWidth || 640;
        hudCanvas.height = videoElem.videoHeight || 480;
        const w = hudCanvas.width;
        const h = hudCanvas.height;

        // Draw live video
        hudCtx.drawImage(videoElem, 0, 0, w, h);

        // Compute simulated/client face tracking coordinates if face box not yet set
        if (!faceBox) {
            faceBox = { x: Math.floor(w * 0.28), y: Math.floor(h * 0.18), fw: Math.floor(w * 0.44), fh: Math.floor(h * 0.58) };
        }

        // Draw Tactical HUD Corner Brackets
        const bx = faceBox.x, by = faceBox.y, bw = faceBox.fw, bh = faceBox.fh;
        const lineLen = Math.floor(bw * 0.22);
        hudCtx.strokeStyle = '#00f0ff';
        hudCtx.lineWidth = 3;
        hudCtx.shadowColor = '#00f0ff';
        hudCtx.shadowBlur = 8;

        // Top-Left
        hudCtx.beginPath();
        hudCtx.moveTo(bx, by + lineLen);
        hudCtx.lineTo(bx, by);
        hudCtx.lineTo(bx + lineLen, by);
        hudCtx.stroke();

        // Top-Right
        hudCtx.beginPath();
        hudCtx.moveTo(bx + bw - lineLen, by);
        hudCtx.lineTo(bx + bw, by);
        hudCtx.lineTo(bx + bw, by + lineLen);
        hudCtx.stroke();

        // Bottom-Left
        hudCtx.beginPath();
        hudCtx.moveTo(bx, by + bh - lineLen);
        hudCtx.lineTo(bx, by + bh);
        hudCtx.lineTo(bx + lineLen, by + bh);
        hudCtx.stroke();

        // Bottom-Right
        hudCtx.beginPath();
        hudCtx.moveTo(bx + bw - lineLen, by + bh);
        hudCtx.lineTo(bx + bw, by + bh);
        hudCtx.lineTo(bx + bw, by + bh - lineLen);
        hudCtx.stroke();

        hudCtx.shadowBlur = 0;

        // Draw Facial Landmark Points
        hudCtx.fillStyle = '#10b981';
        // Eyes
        hudCtx.fillRect(bx + bw * 0.30 - 3, by + bh * 0.35 - 3, 6, 6);
        hudCtx.fillRect(bx + bw * 0.70 - 3, by + bh * 0.35 - 3, 6, 6);
        // Nose bridge
        hudCtx.fillStyle = '#00f0ff';
        hudCtx.fillRect(bx + bw * 0.50 - 2, by + bh * 0.52 - 2, 5, 5);
        // Mouth
        hudCtx.fillStyle = '#f59e0b';
        hudCtx.fillRect(bx + bw * 0.50 - 6, by + bh * 0.76 - 2, 12, 4);

        // Header Overlay
        hudCtx.fillStyle = 'rgba(6, 9, 19, 0.85)';
        hudCtx.fillRect(bx, Math.max(0, by - 26), bw, 24);
        hudCtx.fillStyle = '#00f0ff';
        hudCtx.font = 'bold 12px "Orbitron", monospace';
        hudCtx.fillText(`OPTICAL LOCK | EAR: ${smoothEar.toFixed(2)} MAR: ${smoothMar.toFixed(2)}`, bx + 8, Math.max(16, by - 8));

        // Real-Time Audio Prosody Calculations (at 30 FPS)
        if (audioAnalyser && audioTimeData) {
            audioAnalyser.getFloatTimeDomainData(audioTimeData);
            let sumSq = 0;
            for (let i = 0; i < audioTimeData.length; i++) sumSq += audioTimeData[i] * audioTimeData[i];
            liveRmsEnergy = Math.sqrt(sumSq / audioTimeData.length);

            let pitch = detectPitchAutocorr(audioTimeData, audioContext.sampleRate);
            if (pitch > 60 && pitch < 450) {
                livePitchHz = Math.round(pitch);
                pitchVal.innerText = `${livePitchHz} Hz`;
                liveVocalTension = Math.min(100, Math.max(5, Math.round((livePitchHz > 200 ? (livePitchHz - 180) * 0.8 : 8) + (liveRmsEnergy * 80))));
                vocalTensionVal.innerText = `${liveVocalTension}%`;
            }

            // Draw Audio Waveform
            audioAnalyser.getByteTimeDomainData(audioDataArray);
            waveformCtx.fillStyle = '#020408';
            waveformCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
            waveformCtx.lineWidth = 2;
            waveformCtx.strokeStyle = liveRmsEnergy > 0.04 ? '#10b981' : '#00f0ff';
            waveformCtx.beginPath();

            const sliceWidth = waveformCanvas.width * 1.0 / audioDataArray.length;
            let wx = 0;
            for (let i = 0; i < audioDataArray.length; i++) {
                const v = audioDataArray[i] / 128.0;
                const wy = v * waveformCanvas.height / 2;
                if (i === 0) waveformCtx.moveTo(wx, wy);
                else waveformCtx.lineTo(wx, wy);
                wx += sliceWidth;
            }
            waveformCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
            waveformCtx.stroke();
        }
    }

    // 5. Send Frame to Backend Multimodal Engine (~4 FPS)
    async function sendFrameToBackend() {
        if (!isCameraActive || isProcessingBackend) return;
        isProcessingBackend = true;

        const offCanvas = document.createElement('canvas');
        offCanvas.width = 320;
        offCanvas.height = 240;
        const offCtx = offCanvas.getContext('2d');
        offCtx.drawImage(videoElem, 0, 0, 320, 240);
        const b64Image = offCanvas.toDataURL('image/jpeg', 0.65);

        try {
            const resp = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_base64: b64Image,
                    transcript: currentSpeechText,
                    astronaut_id: crewSelector.value
                })
            });
            const telemetry = await resp.json();
            updateDashboardTelemetry(telemetry);

            // Update smooth EAR & MAR from backend
            if (telemetry.vision) {
                if (telemetry.vision.face_bounding_box) {
                    const scaleX = (videoElem.videoWidth || 640) / 320;
                    const scaleY = (videoElem.videoHeight || 480) / 240;
                    faceBox = {
                        x: Math.floor(telemetry.vision.face_bounding_box.x * scaleX),
                        y: Math.floor(telemetry.vision.face_bounding_box.y * scaleY),
                        fw: Math.floor(telemetry.vision.face_bounding_box.w * scaleX),
                        fh: Math.floor(telemetry.vision.face_bounding_box.h * scaleY)
                    };
                }
                smoothEar = telemetry.vision.eye_aspect_ratio || 0.28;
                smoothMar = telemetry.vision.mouth_aspect_ratio || 0.20;
            }
        } catch (e) {
            console.log("Backend telemetry error:", e);
        } finally {
            isProcessingBackend = false;
        }
    }

    // 6. Update Dashboard Telemetry UI
    function updateDashboardTelemetry(data) {
        if (!data) return;

        // Fused Emotion
        const fusion = data.fusion || {};
        const domEmotion = fusion.dominant_emotion || 'neutral';
        const confidence = fusion.confidence || 0.0;
        const color = emotionColors[domEmotion] || '#00f0ff';

        domEmotionText.innerText = domEmotion.toUpperCase();
        domEmotionText.style.color = color;
        domEmotionConf.innerText = `CONFIDENCE: ${(confidence * 100).toFixed(0)}%`;

        // Update 7 Emotion Bars
        const probs = fusion.fused_probabilities || {};
        for (const [emo, p] of Object.entries(probs)) {
            const fillElem = document.getElementById(`bar-fill-${emo}`);
            const valElem = document.getElementById(`bar-val-${emo}`);
            if (fillElem) fillElem.style.width = `${Math.min(100, p * 100)}%`;
            if (valElem) valElem.innerText = `${(p * 100).toFixed(0)}%`;
        }

        // Risk Assessment
        const risk = data.risk_assessment || {};
        const riskScore = risk.risk_score || 0.0;
        const tierName = risk.tier_name || 'LEVEL 0: NOMINAL';
        const riskColor = risk.color_hex || '#10b981';

        riskScoreVal.innerText = riskScore.toFixed(1);
        riskScoreVal.style.color = riskColor;
        riskBadge.innerText = tierName;
        riskBadge.style.backgroundColor = riskColor + '25';
        riskBadge.style.borderColor = riskColor;
        riskBadge.style.color = riskColor;
        riskBarFill.style.width = `${riskScore}%`;
        riskBarFill.style.backgroundColor = riskColor;

        // Physical Vitals
        const phys = data.physical_distress || {};
        perclosVal.innerText = `${(phys.perclos_percentage || 0).toFixed(1)}%`;
        blinkVal.innerText = `${(phys.blink_rate_bpm || 0).toFixed(0)} BPM`;
        yawnVal.innerText = `${phys.yawns_per_min || data.vision?.yawns_per_min || 0} /min`;
        painVal.innerText = `${(phys.pain_grimace_score || 0).toFixed(0)} / 100`;
        fatigueLevelText.innerText = phys.fatigue_level || 'Nominal';
        fatigueLevelText.style.color = phys.status_color === 'red' ? '#ef4444' : (phys.status_color === 'orange' ? '#f97316' : '#10b981');

        // Discordance Alert
        if (fusion.cross_modal_discordance) {
            discordanceBanner.style.display = 'block';
            discordanceText.innerText = fusion.discordance_reason || 'Cross-modal tension detected.';
        } else {
            discordanceBanner.style.display = 'none';
        }

        // Ground Alerts
        if (data.alert_dispatched) {
            addGroundAlertItem(data.alert_dispatched);
        }
    }

    function addGroundAlertItem(alert) {
        const item = document.createElement('div');
        item.className = `alert-item ${alert.risk_level === 2 ? 'moderate' : ''}`;
        item.innerHTML = `
            <strong>🚨 ${alert.alert_id}</strong> [RISK ${alert.risk_level}]<br>
            <small>${alert.timestamp} | ${alert.emotional_state.primary.toUpperCase()} | Fatigue: ${alert.physical_state.fatigue_level}</small><br>
            <em>${alert.recommended_ground_action}</em>
        `;
        alertFeedList.prepend(item);
        alertCountBadge.innerText = parseInt(alertCountBadge.innerText || '0') + 1;
    }

    // 7. Chat Interaction & Speech Synthesis
    async function sendChatMessage(text) {
        if (!text || !text.trim()) return;

        appendChatBubble('user', text);
        chatInput.value = '';

        try {
            const resp = await fetch('/api/interact', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    astronaut_id: crewSelector.value
                })
            });
            const data = await resp.json();
            appendChatBubble('ai', data.ai_response);

            // Spoken voice playback via Web Speech TTS
            speakWithBrowserTTS(data.ai_response);

            // Auto-trigger Guided Breathing if relevant
            if (data.intervention && data.intervention.id === 'INT-BREATHE-01') {
                startBreathingPacer();
            }
        } catch (e) {
            console.log("Chat error:", e);
        }
    }

    function appendChatBubble(speaker, text) {
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${speaker}`;
        bubble.innerHTML = `
            <div class="chat-speaker-label">${speaker === 'ai' ? 'MAITRI AI COMPANION' : crewSelector.options[crewSelector.selectedIndex].text}</div>
            <div>${text}</div>
        `;
        chatContainer.appendChild(bubble);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function speakWithBrowserTTS(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.02;
            utterance.pitch = 1.05;
            window.speechSynthesis.speak(utterance);
        }
    }

    btnSendChat.addEventListener('click', () => sendChatMessage(chatInput.value));
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage(chatInput.value);
    });

    btnVoiceInput.addEventListener('click', () => {
        if (recognition) {
            btnVoiceInput.classList.add('active');
            try { recognition.start(); } catch(e) {}
        } else {
            alert("Speech recognition not supported in this browser. Please type your message.");
        }
    });

    // 8. Tactical Box Breathing Pacer Logic
    let pacerInterval = null;
    function startBreathingPacer() {
        if (pacerInterval) clearInterval(pacerInterval);
        let phase = 0; // 0: Inhale, 1: Hold, 2: Exhale, 3: Hold empty
        let count = 4;

        const phases = [
            { name: "INHALE (4s)", action: "expand" },
            { name: "HOLD (4s)", action: "expand" },
            { name: "EXHALE (4s)", action: "contract" },
            { name: "HOLD (4s)", action: "contract" }
        ];

        pacerInterval = setInterval(() => {
            pacerSeconds.innerText = count;
            pacerState.innerText = phases[phase].name;
            pacerCircle.className = `pacer-circle ${phases[phase].action}`;

            count--;
            if (count < 1) {
                count = 4;
                phase = (phase + 1) % 4;
            }
        }, 1000);
    }
    startBreathingPacer();

    // 9. Flight Simulation Scenarios (Demo Controls)
    const scenarioBtns = document.querySelectorAll('.btn-scenario');
    scenarioBtns.forEach(btn => {
        btn.addEventListener('click', async () => {
            scenarioBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const scenario = btn.getAttribute('data-scenario');

            try {
                const resp = await fetch(`/api/simulate/${scenario}`, { method: 'POST' });
                const telemetry = await resp.json();
                updateDashboardTelemetry(telemetry);

                if (telemetry.transcript) {
                    appendChatBubble('user', telemetry.transcript);
                    const chatResp = await fetch('/api/interact', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: telemetry.transcript,
                            astronaut_id: crewSelector.value
                        })
                    });
                    const chatData = await chatResp.json();
                    appendChatBubble('ai', chatData.ai_response);
                    speakWithBrowserTTS(chatData.ai_response);
                }
            } catch (e) {
                console.log("Simulation error:", e);
            }
        });
    });

    // 10. Crew Selector Change
    crewSelector.addEventListener('change', async () => {
        await fetch('/api/crew/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ astronaut_id: crewSelector.value })
        });
    });
});
