/**
 * MAITRI — Spacecraft HUD & Multimodal AI Assistant Client Logic
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
    const btnToggleMic = document.getElementById('btn-toggle-mic');
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

    // Emotion Keys and Color Mapping
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
    let isMicActive = false;
    let mediaStream = null;
    let audioContext = null;
    let audioAnalyser = null;
    let audioDataArray = null;
    let streamInterval = null;
    let isProcessing = false;
    let recognition = null;
    let currentSpeechText = "";

    // 1. Initialize Web Speech API for Astronaut Voice Recognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRec();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            chatInput.value = transcript;
            currentSpeechText = transcript;
            sendChatMessage(transcript);
        };

        recognition.onerror = (e) => {
            console.log('[SpeechRec Error]:', e);
            btnVoiceInput.classList.remove('active');
        };

        recognition.onend = () => {
            btnVoiceInput.classList.remove('active');
        };
    }

    // 2. Camera Toggle & Frame Streaming
    btnToggleCamera.addEventListener('click', async () => {
        if (!isCameraActive) {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: true
                });
                videoElem.srcObject = mediaStream;
                await videoElem.play();
                isCameraActive = true;
                btnToggleCamera.innerHTML = '🛑 STOP OPTICAL';
                btnToggleCamera.classList.add('active');

                // Start Web Audio Analyser
                initAudioVisualizer(mediaStream);

                // Start Video Capture & Telemetry Loop (~3.5 FPS)
                streamInterval = setInterval(captureAndSendFrame, 280);
            } catch (err) {
                alert("Camera/Microphone access error: " + err.message + ". You can use Flight Simulation Scenarios to test without hardware.");
            }
        } else {
            stopCamera();
        }
    });

    function stopCamera() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
        }
        if (streamInterval) clearInterval(streamInterval);
        videoElem.srcObject = null;
        isCameraActive = false;
        btnToggleCamera.innerHTML = '📹 START OPTICAL';
        btnToggleCamera.classList.remove('active');
    }

    // 3. Audio Waveform Visualizer
    function initAudioVisualizer(stream) {
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            audioAnalyser = audioContext.createAnalyser();
            audioAnalyser.fftSize = 256;
            source.connect(audioAnalyser);
            audioDataArray = new Uint8Array(audioAnalyser.frequencyBinCount);
            drawWaveform();
        } catch (e) {
            console.log("Audio visualizer error:", e);
        }
    }

    function drawWaveform() {
        requestAnimationFrame(drawWaveform);
        if (!audioAnalyser) return;
        audioAnalyser.getByteTimeDomainData(audioDataArray);

        waveformCtx.fillStyle = '#020408';
        waveformCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        waveformCtx.lineWidth = 2;
        waveformCtx.strokeStyle = '#00f0ff';
        waveformCtx.beginPath();

        const sliceWidth = waveformCanvas.width * 1.0 / audioDataArray.length;
        let x = 0;
        for (let i = 0; i < audioDataArray.length; i++) {
            const v = audioDataArray[i] / 128.0;
            const y = v * waveformCanvas.height / 2;
            if (i === 0) waveformCtx.moveTo(x, y);
            else waveformCtx.lineTo(x, y);
            x += sliceWidth;
        }
        waveformCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
        waveformCtx.stroke();
    }

    // 4. Capture Canvas Frame & Send to API
    async function captureAndSendFrame() {
        if (!isCameraActive || isProcessing) return;
        isProcessing = true;

        hudCanvas.width = videoElem.videoWidth || 640;
        hudCanvas.height = videoElem.videoHeight || 480;
        hudCtx.drawImage(videoElem, 0, 0, hudCanvas.width, hudCanvas.height);

        const b64Image = hudCanvas.toDataURL('image/jpeg', 0.7);

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
            currentSpeechText = ""; // reset after ingest
        } catch (e) {
            console.log("Frame processing error:", e);
        } finally {
            isProcessing = false;
        }
    }

    // 5. Update Telemetry Dashboard
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
        const riskLevel = risk.risk_level || 0;
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

        // Audio Vitals
        const audio = data.audio || {};
        pitchVal.innerText = audio.pitch_f0_hz ? `${audio.pitch_f0_hz.toFixed(0)} Hz` : '-- Hz';
        vocalTensionVal.innerText = audio.vocal_tension_score ? `${(audio.vocal_tension_score * 100).toFixed(0)}%` : '0%';

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

    // 6. Chat Interaction
    async function sendChatMessage(text) {
        if (!text || !text.trim()) return;

        // Append user bubble
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

            // Trigger Browser Web Speech TTS for high-fidelity audio
            speakWithBrowserTTS(data.ai_response);

            // If guided breathing triggered, start pacer
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
            utterance.rate = 1.0;
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
            recognition.start();
        } else {
            alert("Speech recognition not supported in this browser. Please type your message.");
        }
    });

    // 7. Guided Box Breathing Pacer Logic
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

    // 8. Flight Simulation Scenarios
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

                // Add synthetic dialogue if transcript exists
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

    // 9. Crew Selector Change
    crewSelector.addEventListener('change', async () => {
        await fetch('/api/crew/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ astronaut_id: crewSelector.value })
        });
    });
});
