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
    
    // Telemetry DOM Bindings (Dashboard Tab)
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
    
    const pacerState = document.getElementById('pacer-state');
    const pacerSeconds = document.getElementById('pacer-seconds');
    
    const alertFeedList = document.getElementById('alert-feed-list');
    const alertCountBadge = document.getElementById('alert-count-badge');
    const discordanceBanner = document.getElementById('discordance-banner');
    const discordanceText = document.getElementById('discordance-text');
    const opticalLockBadge = document.getElementById('optical-lock-badge');
    const alertsTableBody = document.getElementById('alerts-table-body');
    const liveMissionClock = document.getElementById('live-mission-clock');

    // Analysis Tab Bindings
    const serPitchDisplay = document.getElementById('ser-pitch-display');
    const serTensionDisplay = document.getElementById('ser-tension-display');
    const sentimentValenceDisplay = document.getElementById('sentiment-valence-display');
    const attnWF = document.getElementById('attn-w-f');
    const attnWV = document.getElementById('attn-w-v');
    const attnWT = document.getElementById('attn-w-t');
    const barFillFacial = document.getElementById('bar-fill-facial');
    const barValFacial = document.getElementById('bar-val-facial');
    const barFillSpeech = document.getElementById('bar-fill-speech');
    const barValSpeech = document.getElementById('bar-val-speech');
    const barFillText = document.getElementById('bar-fill-text');
    const barValText = document.getElementById('bar-val-text');

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
    let faceBox = null;
    let smoothEar = 0.30;
    let smoothMar = 0.20;
    let missionStartTime = Date.now() - (4 * 3600 + 12 * 60 + 33) * 1000;

    // 1. Live Mission Clock Ticker
    setInterval(() => {
        const elapsed = Math.floor((Date.now() - missionStartTime) / 1000);
        const hrs = String(Math.floor(elapsed / 3600)).padStart(2, '0');
        const mins = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
        const secs = String(elapsed % 60).padStart(2, '0');
        if (liveMissionClock) liveMissionClock.innerText = `T+${hrs}:${mins}:${secs}`;
    }, 1000);

    // 2. Initialize Continuous Speech Recognition (Web Speech API)
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
    if (btnToggleVoiceListen) {
        btnToggleVoiceListen.addEventListener('click', () => {
            if (!recognition) {
                alert("Speech recognition not supported in this browser. Please type your message.");
                return;
            }
            isContinuousListening = !isContinuousListening;
            if (isContinuousListening) {
                btnToggleVoiceListen.classList.add('bg-error/20', 'border-error', 'text-error');
                btnToggleVoiceListen.innerHTML = '<span class="material-symbols-outlined text-[16px]">mic</span><span>VOICE: ON</span>';
                try { recognition.start(); } catch(e) {}
            } else {
                btnToggleVoiceListen.classList.remove('bg-error/20', 'border-error', 'text-error');
                btnToggleVoiceListen.innerHTML = '<span class="material-symbols-outlined text-[16px]">mic</span><span>VOICE: OFF</span>';
                try { recognition.stop(); } catch(e) {}
            }
        });
    }

    // 3. Camera Toggle & Live Stream Start
    if (btnToggleCamera) {
        btnToggleCamera.addEventListener('click', async () => {
            if (!isCameraActive) {
                await startLiveCamera();
            } else {
                stopLiveCamera();
            }
        });
    }

    async function startLiveCamera() {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
                audio: true
            });
            videoElem.srcObject = mediaStream;
            await videoElem.play();
            isCameraActive = true;
            btnToggleCamera.innerHTML = '<span class="material-symbols-outlined text-[16px]">videocam_off</span><span>STOP OPTICAL</span>';
            btnToggleCamera.classList.add('bg-error/20', 'border-error', 'text-error');

            initAudioAnalyser(mediaStream);
            renderLiveHudCanvas();
            streamInterval = setInterval(sendFrameToBackend, 250);
            if (opticalLockBadge) opticalLockBadge.innerText = "OPTICAL TRACKING LOCKED";
        } catch (err) {
            alert("Camera/Microphone access error: " + err.message + "\nTip: Use the Flight Simulation Scenarios at the top to test live multimodal data!");
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
        btnToggleCamera.innerHTML = '<span class="material-symbols-outlined text-[16px]">videocam</span><span>START OPTICAL</span>';
        btnToggleCamera.classList.remove('bg-error/20', 'border-error', 'text-error');
        
        hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
        waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        if (opticalLockBadge) opticalLockBadge.innerText = "OPTICAL STANDBY";
    }

    // 4. Real-Time Audio Prosody Analyser
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

    function detectPitchAutocorr(buffer, sampleRate) {
        let size = buffer.length;
        let sumOfSquares = 0;
        for (let i = 0; i < size; i++) sumOfSquares += buffer[i] * buffer[i];
        let rms = Math.sqrt(sumOfSquares / size);
        if (rms < 0.015) return -1;

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
            for (let j = 0; j < size - i; j++) c[i] += buffer[j] * buffer[j + i];
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
        if (T0 > 0) return sampleRate / T0;
        return -1;
    }

    // 5. Smooth 30 FPS Canvas Rendering
    function renderLiveHudCanvas() {
        if (!isCameraActive) return;
        animFrameId = requestAnimationFrame(renderLiveHudCanvas);

        hudCanvas.width = videoElem.videoWidth || 640;
        hudCanvas.height = videoElem.videoHeight || 480;
        const w = hudCanvas.width;
        const h = hudCanvas.height;

        hudCtx.drawImage(videoElem, 0, 0, w, h);

        if (!faceBox) {
            faceBox = { x: Math.floor(w * 0.28), y: Math.floor(h * 0.18), fw: Math.floor(w * 0.44), fh: Math.floor(h * 0.58) };
        }

        const bx = faceBox.x, by = faceBox.y, bw = faceBox.fw, bh = faceBox.fh;
        const lineLen = Math.floor(bw * 0.22);
        hudCtx.strokeStyle = '#4de082';
        hudCtx.lineWidth = 2.5;

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

        // Landmark points
        hudCtx.fillStyle = '#4de082';
        hudCtx.fillRect(bx + bw * 0.32 - 2, by + bh * 0.36 - 2, 5, 5);
        hudCtx.fillRect(bx + bw * 0.68 - 2, by + bh * 0.36 - 2, 5, 5);
        hudCtx.fillStyle = '#c6bfff';
        hudCtx.fillRect(bx + bw * 0.50 - 2, by + bh * 0.54 - 2, 4, 4);
        hudCtx.fillStyle = '#eec200';
        hudCtx.fillRect(bx + bw * 0.50 - 5, by + bh * 0.74 - 2, 10, 4);

        // Header Overlay Box
        hudCtx.fillStyle = 'rgba(11, 14, 20, 0.85)';
        hudCtx.fillRect(bx, Math.max(0, by - 24), bw, 22);
        hudCtx.fillStyle = '#4de082';
        hudCtx.font = '500 11px "JetBrains Mono", monospace';
        hudCtx.fillText(`68_LNDMK // EAR:${smoothEar.toFixed(2)} MAR:${smoothMar.toFixed(2)}`, bx + 6, Math.max(15, by - 8));

        // Audio calculations
        if (audioAnalyser && audioTimeData) {
            audioAnalyser.getFloatTimeDomainData(audioTimeData);
            let sumSq = 0;
            for (let i = 0; i < audioTimeData.length; i++) sumSq += audioTimeData[i] * audioTimeData[i];
            liveRmsEnergy = Math.sqrt(sumSq / audioTimeData.length);

            let pitch = detectPitchAutocorr(audioTimeData, audioContext.sampleRate);
            if (pitch > 60 && pitch < 450) {
                livePitchHz = Math.round(pitch);
                if (pitchVal) pitchVal.innerText = `${livePitchHz} Hz`;
                if (serPitchDisplay) serPitchDisplay.innerText = `${livePitchHz} Hz`;
                liveVocalTension = Math.min(100, Math.max(5, Math.round((livePitchHz > 200 ? (livePitchHz - 180) * 0.8 : 8) + (liveRmsEnergy * 80))));
                if (vocalTensionVal) vocalTensionVal.innerText = `${liveVocalTension}%`;
                if (serTensionDisplay) serTensionDisplay.innerText = `${liveVocalTension}%`;
            }

            // Audio Waveform
            audioAnalyser.getByteTimeDomainData(audioDataArray);
            waveformCtx.fillStyle = '#0B0E14';
            waveformCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
            waveformCtx.lineWidth = 1.5;
            waveformCtx.strokeStyle = liveRmsEnergy > 0.04 ? '#4de082' : '#c6bfff';
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

    // 6. Send Frame to Backend Multimodal Engine (~4 FPS)
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
                    astronaut_id: crewSelector ? crewSelector.value : 'CREW-BAS-01'
                })
            });
            const telemetry = await resp.json();
            updateDashboardTelemetry(telemetry);

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

    // 7. Update Dashboard Telemetry UI
    function updateDashboardTelemetry(data) {
        if (!data) return;

        // Fused Emotion
        const fusion = data.fusion || {};
        const domEmotion = fusion.dominant_emotion || 'neutral';
        const confidence = fusion.confidence || 0.0;

        if (domEmotionText) {
            domEmotionText.innerText = domEmotion.toUpperCase();
            domEmotionText.className = `font-headline-md text-xl font-bold leading-none ${domEmotion === 'stressed' || domEmotion === 'frustrated' ? 'text-error' : (domEmotion === 'fatigued' ? 'text-tertiary' : 'text-primary')}`;
        }
        if (domEmotionConf) domEmotionConf.innerText = `${(confidence * 100).toFixed(0)}%`;

        // Update Multimodal Fusion Bars
        const fer = data.fer || {};
        const ser = data.ser || {};
        const textNlp = data.text_sentiment || {};

        const ferConf = Math.round((fer.confidence || 0.78) * 100);
        const serConf = Math.round((ser.confidence || 0.65) * 100);
        const textConf = Math.round((textNlp.confidence || 0.60) * 100);

        if (barFillFacial) barFillFacial.style.width = `${ferConf}%`;
        if (barValFacial) barValFacial.innerText = `${ferConf}%`;
        if (barFillSpeech) barFillSpeech.style.width = `${serConf}%`;
        if (barValSpeech) barValSpeech.innerText = `${serConf}%`;
        if (barFillText) barFillText.style.width = `${textConf}%`;
        if (barValText) barValText.innerText = `${textConf}%`;

        // Update Attention Weights
        const attn = fusion.attention_weights || {};
        if (attnWF) attnWF.innerText = `F:${Math.round((attn.facial_alpha || 0.4) * 100)}%`;
        if (attnWV) attnWV.innerText = `V:${Math.round((attn.speech_beta || 0.35) * 100)}%`;
        if (attnWT) attnWT.innerText = `T:${Math.round((attn.linguistic_gamma || 0.25) * 100)}%`;

        // Analysis Tab Breakdown
        const probs = fusion.fused_probabilities || {};
        for (const [emo, p] of Object.entries(probs)) {
            const valEl = document.getElementById(`analysis-val-${emo}`);
            const barEl = document.getElementById(`analysis-bar-${emo}`);
            if (valEl) valEl.innerText = `${(p * 100).toFixed(1)}%`;
            if (barEl) barEl.style.width = `${Math.min(100, p * 100)}%`;
        }

        // Sentiment Valence
        if (sentimentValenceDisplay && textNlp.valence !== undefined) {
            sentimentValenceDisplay.innerText = textNlp.valence >= 0 ? `+${textNlp.valence.toFixed(2)}` : `${textNlp.valence.toFixed(2)}`;
        }

        // Risk Assessment
        const risk = data.risk_assessment || {};
        const riskScore = risk.risk_score || 0.0;
        const tierName = risk.tier_name || 'LEVEL 0: NOMINAL';
        const riskColor = risk.color_hex || '#4de082';

        if (riskScoreVal) {
            riskScoreVal.innerText = riskScore.toFixed(1);
            riskScoreVal.className = `font-display-lg text-3xl font-bold ${riskScore > 70 ? 'text-error' : (riskScore > 30 ? 'text-tertiary' : 'text-secondary')}`;
        }
        if (riskBadge) {
            riskBadge.innerHTML = `<span class="w-2 h-2 rounded-full ${riskScore > 70 ? 'bg-error animate-pulse' : (riskScore > 30 ? 'bg-tertiary' : 'bg-secondary')}"></span><span class="font-label-caps text-[11px] font-semibold">${tierName}</span>`;
        }
        if (riskBarFill) {
            riskBarFill.style.width = `${Math.max(5, riskScore)}%`;
            riskBarFill.className = `h-full transition-all duration-400 ${riskScore > 70 ? 'bg-error' : (riskScore > 30 ? 'bg-tertiary' : 'bg-secondary')}`;
        }

        // Physical Vitals
        const phys = data.physical_distress || {};
        if (perclosVal) perclosVal.innerText = `${(phys.perclos_percentage || 0).toFixed(1)}%`;
        if (blinkVal) blinkVal.innerText = `${(phys.blink_rate_bpm || 16).toFixed(0)} BPM`;
        if (yawnVal) yawnVal.innerText = `${phys.yawns_per_min || 0} /min`;
        if (painVal) painVal.innerText = `${(phys.pain_grimace_score || 0).toFixed(0)} / 100`;
        if (fatigueLevelText) {
            fatigueLevelText.innerText = phys.fatigue_level ? phys.fatigue_level.toUpperCase() : 'NOMINAL';
            fatigueLevelText.className = `${phys.status_color === 'red' ? 'text-error' : (phys.status_color === 'orange' ? 'text-tertiary' : 'text-secondary')} font-bold`;
        }

        // Discordance Alert
        if (discordanceBanner) {
            if (fusion.cross_modal_discordance) {
                discordanceBanner.classList.remove('hidden');
                if (discordanceText) discordanceText.innerText = fusion.discordance_reason || 'Masked stress detected.';
            } else {
                discordanceBanner.classList.add('hidden');
            }
        }

        // Ground Alerts
        if (data.alert_dispatched) {
            addGroundAlertItem(data.alert_dispatched);
        }
    }

    function addGroundAlertItem(alert) {
        if (!alertFeedList) return;
        const item = document.createElement('div');
        item.className = `p-2 bg-surface border-l-2 ${alert.risk_level >= 3 ? 'border-error' : 'border-tertiary'} flex flex-col gap-0.5`;
        item.innerHTML = `
            <div class="flex justify-between items-center text-[10px]">
                <strong class="${alert.risk_level >= 3 ? 'text-error' : 'text-tertiary'}">${alert.alert_id}</strong>
                <span class="text-on-surface-variant">${alert.timestamp}</span>
            </div>
            <div class="text-[11px] text-on-surface">${alert.emotional_state.primary.toUpperCase()} | LVL ${alert.risk_level} (${alert.risk_score})</div>
            <div class="text-[10px] text-on-surface-variant italic">${alert.recommended_ground_action}</div>
        `;
        alertFeedList.prepend(item);
        if (alertCountBadge) alertCountBadge.innerText = `${alertFeedList.children.length} QUEUED`;

        // Also append to Alerts Table in Tab 4
        if (alertsTableBody) {
            const tr = document.createElement('tr');
            tr.className = 'data-row border-b border-outline-variant/20';
            tr.innerHTML = `
                <td class="py-2.5 text-primary">${alert.alert_id}</td>
                <td class="py-2.5 text-on-surface-variant">${alert.timestamp}</td>
                <td class="py-2.5 ${alert.risk_level >= 3 ? 'text-error' : 'text-tertiary'}">LVL ${alert.risk_level} (${alert.risk_score})</td>
                <td class="py-2.5">${alert.emotional_state.primary.toUpperCase()}</td>
                <td class="py-2.5 text-right text-tertiary">QUEUED_S-BAND</td>
            `;
            alertsTableBody.prepend(tr);
        }
    }

    // 8. Chat Interaction & Speech Synthesis
    async function sendChatMessage(text) {
        if (!text || !text.trim()) return;

        appendChatBubble('user', text);
        if (chatInput) chatInput.value = '';

        try {
            const resp = await fetch('/api/interact', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    astronaut_id: crewSelector ? crewSelector.value : 'CREW-BAS-01'
                })
            });
            const data = await resp.json();
            appendChatBubble('ai', data.ai_response);

            // Spoken TTS playback
            speakWithBrowserTTS(data.ai_response);

            if (data.intervention && data.intervention.id === 'INT-BREATHE-01') {
                startBreathingPacer();
            }
        } catch (e) {
            console.log("Chat error:", e);
        }
    }

    function appendChatBubble(speaker, text) {
        if (!chatContainer) return;
        const bubble = document.createElement('div');
        const isAi = speaker === 'ai';
        bubble.className = `${isAi ? 'bg-surface border-outline-variant' : 'bg-surface-container-low border-outline-variant/60'} border rounded-DEFAULT p-2.5 ${isAi ? 'self-end' : 'self-start'} w-5/6`;
        bubble.innerHTML = `
            <div class="flex justify-between items-center mb-1">
                <span class="font-label-caps text-[10px] ${isAi ? 'text-primary' : 'text-secondary'} font-bold">${isAi ? 'MAITRI AI' : 'CREW-01 (AUDIO)'}</span>
                <span class="font-data-mono text-[10px] text-on-surface-variant">NOW</span>
            </div>
            <p class="text-on-surface text-xs ${isAi ? '' : 'italic'}">${text}</p>
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

    if (btnSendChat) btnSendChat.addEventListener('click', () => sendChatMessage(chatInput.value));
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendChatMessage(chatInput.value);
        });
    }

    if (btnVoiceInput) {
        btnVoiceInput.addEventListener('click', () => {
            if (recognition) {
                try { recognition.start(); } catch(e) {}
            }
        });
    }

    // 9. Tactical Box Breathing Pacer Logic
    let pacerInterval = null;
    function startBreathingPacer() {
        if (pacerInterval) clearInterval(pacerInterval);
        let phase = 0;
        let count = 4;

        const phases = [
            { name: "INHALE", action: "expand" },
            { name: "HOLD", action: "expand" },
            { name: "EXHALE", action: "contract" },
            { name: "HOLD", action: "contract" }
        ];

        pacerInterval = setInterval(() => {
            if (pacerSeconds) pacerSeconds.innerText = `${count}s`;
            if (pacerState) pacerState.innerText = phases[phase].name;

            count--;
            if (count < 1) {
                count = 4;
                phase = (phase + 1) % 4;
            }
        }, 1000);
    }
    startBreathingPacer();

    // 10. Flight Simulation Scenarios (Demo Controls)
    const scenarioBtns = document.querySelectorAll('.btn-scenario');
    scenarioBtns.forEach(btn => {
        btn.addEventListener('click', async () => {
            scenarioBtns.forEach(b => b.classList.remove('border-primary', 'bg-primary/20'));
            btn.classList.add('border-primary', 'bg-primary/20');
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
                            astronaut_id: crewSelector ? crewSelector.value : 'CREW-BAS-01'
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

    // 11. Crew Selector Change
    if (crewSelector) {
        crewSelector.addEventListener('change', async () => {
            await fetch('/api/crew/select', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ astronaut_id: crewSelector.value })
            });
        });
    }
});
