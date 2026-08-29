/**
 * MAITRI — AI Well-Being System Client Logic
 * ISRO Bhartiya Antariksh Station (BAS)
 * Clean, Modern Web Application Engine
 */

document.addEventListener('DOMContentLoaded', () => {
    // ---------------------------------------------------------
    // DOM Element Bindings
    // ---------------------------------------------------------
    const videoElem = document.getElementById('webcam-video');
    const hudCanvas = document.getElementById('hud-canvas');
    const hudCtx = hudCanvas ? hudCanvas.getContext('2d') : null;
    const waveformCanvas = document.getElementById('audio-waveform-canvas');
    const waveformCtx = waveformCanvas ? waveformCanvas.getContext('2d') : null;
    
    const btnToggleCamera = document.getElementById('btn-toggle-camera');
    const btnToggleVoiceListen = document.getElementById('btn-toggle-voice-listen');
    const crewSelector = document.getElementById('crew-selector');
    
    const chatInput = document.getElementById('chat-input');
    const btnSendChat = document.getElementById('btn-send-chat');
    const btnVoiceInput = document.getElementById('btn-voice-input');
    const chatContainer = document.getElementById('chat-messages');
    
    // Telemetry Bindings (Dashboard)
    const domEmotionText = document.getElementById('dominant-emotion-text');
    const domEmotionConf = document.getElementById('dominant-emotion-conf');
    const emotionTrendBadge = document.getElementById('emotion-trend-badge');
    const stateDurationText = document.getElementById('state-duration-text');
    const gracefulDegradeStatus = document.getElementById('graceful-degrade-status');

    const riskBadge = document.getElementById('risk-level-badge');
    const riskScoreVal = document.getElementById('risk-score-value');
    const riskBarFill = document.getElementById('risk-bar-fill');
    const riskHorizonStatus = document.getElementById('risk-horizon-status');
    
    const perclosVal = document.getElementById('vital-perclos');
    const blinkVal = document.getElementById('vital-blinks');
    const yawnVal = document.getElementById('vital-yawns');
    const painVal = document.getElementById('vital-pain');
    const fatigueLevelText = document.getElementById('vital-fatigue-level');
    
    const pitchVal = document.getElementById('vital-pitch');
    const vocalTensionVal = document.getElementById('vital-vocal-tension');
    const hudEarMetric = document.getElementById('hud-ear-metric');
    
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

    // 24H Timeline SVG Elements
    const timelineValencePath = document.getElementById('timeline-valence-path');
    const timelineActiveNode = document.getElementById('timeline-active-node');

    // ---------------------------------------------------------
    // Operational State Variables
    // ---------------------------------------------------------
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
    let ws = null; // Real-time WebSocket connection
    const valenceBuffer = [58, 55, 60, 52, 48, 65, 50, 58, 62, 58]; // Rolling emotion buffer

    // Real-Time Biometrics Buffer
    let livePitchHz = 0;
    let liveRmsEnergy = 0;
    let liveVocalTension = 0;
    let faceBox = null;
    let smoothEar = 0.31;
    let smoothMar = 0.20;
    let missionStartTime = Date.now() - (4 * 3600 + 12 * 60 + 33) * 1000;
    let stateStartTimestamp = Date.now() - (14 * 60 + 32) * 1000;

    // ---------------------------------------------------------
    // 1. Mission Clock & State Duration Timers
    // ---------------------------------------------------------
    setInterval(() => {
        const elapsed = Math.floor((Date.now() - missionStartTime) / 1000);
        const hrs = String(Math.floor(elapsed / 3600)).padStart(2, '0');
        const mins = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
        const secs = String(elapsed % 60).padStart(2, '0');
        if (liveMissionClock) liveMissionClock.innerText = `MET T+${hrs}:${mins}:${secs}`;

        const stateElapsed = Math.floor((Date.now() - stateStartTimestamp) / 1000);
        const sMins = Math.floor(stateElapsed / 60);
        const sSecs = stateElapsed % 60;
        if (stateDurationText) stateDurationText.innerText = `${sMins}m ${String(sSecs).padStart(2, '0')}s`;
    }, 1000);

    // ---------------------------------------------------------
    // 2. Speech Recognition (Web Speech API)
    // ---------------------------------------------------------
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
            } else if (interim && chatInput) {
                chatInput.placeholder = `Listening: "${interim}"...`;
            }
        };

        recognition.onerror = (e) => {
            console.log('[SpeechRec Error]:', e);
        };

        recognition.onend = () => {
            if (chatInput) chatInput.placeholder = "Type or speak to MAITRI...";
            if (isContinuousListening) {
                try { recognition.start(); } catch(e) {}
            }
        };
    }

    if (btnToggleVoiceListen) {
        btnToggleVoiceListen.addEventListener('click', () => {
            if (!recognition) {
                alert("Speech recognition not supported in this browser. Please type your message.");
                return;
            }
            isContinuousListening = !isContinuousListening;
            if (isContinuousListening) {
                btnToggleVoiceListen.classList.remove('bg-dark-850', 'text-slate-200', 'border-slate-700');
                btnToggleVoiceListen.classList.add('bg-red-600/20', 'border-red-500', 'text-red-400');
                btnToggleVoiceListen.innerHTML = '<span class="material-symbols-outlined text-[18px]">mic</span><span>Voice: On</span>';
                try { recognition.start(); } catch(e) {}
            } else {
                btnToggleVoiceListen.classList.add('bg-dark-850', 'text-slate-200', 'border-slate-700');
                btnToggleVoiceListen.classList.remove('bg-red-600/20', 'border-red-500', 'text-red-400');
                btnToggleVoiceListen.innerHTML = '<span class="material-symbols-outlined text-[18px]">mic</span><span>Voice: Off</span>';
                try { recognition.stop(); } catch(e) {}
            }
        });
    }

    // ---------------------------------------------------------
    // 3. Live Optical Camera & Audio Prosody Capture
    // ---------------------------------------------------------
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
            
            btnToggleCamera.innerHTML = '<span class="material-symbols-outlined text-[18px]">videocam_off</span><span>Stop Camera</span>';
            btnToggleCamera.classList.remove('bg-indigo-600', 'hover:bg-indigo-500');
            btnToggleCamera.classList.add('bg-red-600', 'hover:bg-red-500');

            initAudioAnalyser(mediaStream);
            renderLiveHudCanvas();
            streamInterval = setInterval(sendFrameToBackend, 250);
            
            if (opticalLockBadge) {
                opticalLockBadge.innerText = "Camera Active";
                opticalLockBadge.className = "px-2.5 py-1 text-xs font-semibold bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 rounded-full";
            }
            if (gracefulDegradeStatus) {
                gracefulDegradeStatus.innerText = "3 of 3 Modalities Active";
                gracefulDegradeStatus.className = "text-xs font-semibold text-emerald-400";
            }
        } catch (err) {
            alert("Camera/Microphone access error: " + err.message + "\nTip: You can use the Simulation Scenarios buttons at the top for instant testing!");
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
        
        btnToggleCamera.innerHTML = '<span class="material-symbols-outlined text-[18px]">videocam</span><span>Start Camera</span>';
        btnToggleCamera.classList.add('bg-indigo-600', 'hover:bg-indigo-500');
        btnToggleCamera.classList.remove('bg-red-600', 'hover:bg-red-500');
        
        if (hudCtx) hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
        if (waveformCtx) waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        
        if (opticalLockBadge) {
            opticalLockBadge.innerText = "Camera Standby";
            opticalLockBadge.className = "px-2.5 py-1 text-xs font-semibold bg-dark-850 border border-slate-800 text-slate-400 rounded-full";
        }
        if (gracefulDegradeStatus) {
            gracefulDegradeStatus.innerText = "Audio + Text Active (Camera Standby)";
            gracefulDegradeStatus.className = "text-xs font-semibold text-amber-400";
        }
    }

    // ---------------------------------------------------------
    // 4. Acoustic Prosody & Real-Time Pitch Extraction
    // ---------------------------------------------------------
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

    // ---------------------------------------------------------
    // 5. HUD Canvas Overlay Rendering
    // ---------------------------------------------------------
    function renderLiveHudCanvas() {
        if (!isCameraActive || !hudCtx) return;
        animFrameId = requestAnimationFrame(renderLiveHudCanvas);

        hudCanvas.width = videoElem.videoWidth || 640;
        hudCanvas.height = videoElem.videoHeight || 480;
        const w = hudCanvas.width;
        const h = hudCanvas.height;

        // Draw live video frame
        hudCtx.drawImage(videoElem, 0, 0, w, h);

        if (!faceBox) {
            faceBox = { x: Math.floor(w * 0.28), y: Math.floor(h * 0.18), fw: Math.floor(w * 0.44), fh: Math.floor(h * 0.58) };
        }

        const bx = faceBox.x, by = faceBox.y, bw = faceBox.fw, bh = faceBox.fh;
        const lineLen = Math.floor(bw * 0.2);

        // Clean Modern Green/Indigo Target Box
        hudCtx.strokeStyle = '#6366F1';
        hudCtx.lineWidth = 2.5;

        // Top-Left Corner
        hudCtx.beginPath();
        hudCtx.moveTo(bx, by + lineLen);
        hudCtx.lineTo(bx, by);
        hudCtx.lineTo(bx + lineLen, by);
        hudCtx.stroke();

        // Top-Right Corner
        hudCtx.beginPath();
        hudCtx.moveTo(bx + bw - lineLen, by);
        hudCtx.lineTo(bx + bw, by);
        hudCtx.lineTo(bx + bw, by + lineLen);
        hudCtx.stroke();

        // Bottom-Left Corner
        hudCtx.beginPath();
        hudCtx.moveTo(bx, by + bh - lineLen);
        hudCtx.lineTo(bx, by + bh);
        hudCtx.lineTo(bx + lineLen, by + bh);
        hudCtx.stroke();

        // Bottom-Right Corner
        hudCtx.beginPath();
        hudCtx.moveTo(bx + bw - lineLen, by + bh);
        hudCtx.lineTo(bx + bw, by + bh);
        hudCtx.lineTo(bx + bw, by + bh - lineLen);
        hudCtx.stroke();

        // Facial Landmark Dots
        hudCtx.fillStyle = '#34D399';
        hudCtx.fillRect(bx + bw * 0.33 - 2, by + bh * 0.36 - 2, 4, 4);
        hudCtx.fillRect(bx + bw * 0.67 - 2, by + bh * 0.36 - 2, 4, 4);
        hudCtx.fillStyle = '#818CF8';
        hudCtx.fillRect(bx + bw * 0.50 - 2, by + bh * 0.54 - 2, 4, 4);
        hudCtx.fillStyle = '#FBBF24';
        hudCtx.fillRect(bx + bw * 0.50 - 5, by + bh * 0.74 - 2, 10, 3);

        if (hudEarMetric) hudEarMetric.innerText = smoothEar.toFixed(2);

        // Audio Waveform Render
        if (audioAnalyser && audioTimeData && waveformCtx) {
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

            audioAnalyser.getByteTimeDomainData(audioDataArray);
            waveformCtx.fillStyle = '#090D16';
            waveformCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
            waveformCtx.lineWidth = 2;
            waveformCtx.strokeStyle = liveRmsEnergy > 0.03 ? '#34D399' : '#6366F1';
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

    // ---------------------------------------------------------
    // 6. Backend Multimodal Processing (~4 FPS)
    // ---------------------------------------------------------
    async function sendFrameToBackend() {
        if (!isCameraActive || isProcessingBackend) return;
        isProcessingBackend = true;

        const offCanvas = document.createElement('canvas');
        offCanvas.width = 320;
        offCanvas.height = 240;
        const offCtx = offCanvas.getContext('2d');
        offCtx.drawImage(videoElem, 0, 0, 320, 240);
        const b64Image = offCanvas.toDataURL('image/jpeg', 0.65);

        // If WebSocket is active and open, send frame over WS for sub-50ms latency
        if (ws && ws.readyState === WebSocket.OPEN) {
            try {
                ws.send(JSON.stringify({
                    action: 'frame',
                    image_base64: b64Image,
                    transcript: currentSpeechText,
                    astronaut_id: crewSelector ? crewSelector.value : 'CREW-BAS-01'
                }));
            } catch (wsErr) {
                console.warn('[WS Send Error]:', wsErr);
            } finally {
                isProcessingBackend = false;
            }
            return;
        }

        // Fallback: REST API
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
                smoothEar = telemetry.vision.eye_aspect_ratio || 0.31;
                smoothMar = telemetry.vision.mouth_aspect_ratio || 0.20;
            }
        } catch (e) {
            console.log("Backend telemetry error:", e);
        } finally {
            isProcessingBackend = false;
        }
    }

    // ---------------------------------------------------------
    // 7. Update Dashboard Telemetry UI
    // ---------------------------------------------------------
    function updateDashboardTelemetry(data) {
        if (!data) return;

        // Fused Emotion
        const fusion = data.fusion || {};
        const domEmotion = fusion.dominant_emotion || 'neutral';
        const confidence = fusion.confidence || 0.85;

        if (domEmotionText) {
            domEmotionText.innerText = domEmotion.charAt(0).toUpperCase() + domEmotion.slice(1);
            domEmotionText.className = `text-3xl lg:text-4xl font-extrabold tracking-tight ${domEmotion === 'stressed' || domEmotion === 'frustrated' ? 'text-red-400' : (domEmotion === 'fatigued' ? 'text-amber-400' : 'text-white')}`;
        }
        if (domEmotionConf) domEmotionConf.innerText = `${(confidence * 100).toFixed(0)}%`;

        // Trend Badge
        if (emotionTrendBadge) {
            if (domEmotion === 'stressed' || domEmotion === 'frustrated') {
                emotionTrendBadge.innerHTML = '<span class="material-symbols-outlined text-lg">trending_up</span><span>Worsening</span>';
                emotionTrendBadge.className = 'px-4 py-2 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 font-semibold text-sm flex items-center gap-1.5';
            } else if (domEmotion === 'fatigued') {
                emotionTrendBadge.innerHTML = '<span class="material-symbols-outlined text-lg">trending_up</span><span>Elevating</span>';
                emotionTrendBadge.className = 'px-4 py-2 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400 font-semibold text-sm flex items-center gap-1.5';
            } else {
                emotionTrendBadge.innerHTML = '<span class="material-symbols-outlined text-lg">trending_flat</span><span>Stable</span>';
                emotionTrendBadge.className = 'px-4 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-semibold text-sm flex items-center gap-1.5';
            }
        }

        // Modality Bars & Weights
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

        // Attention Weights
        const attn = fusion.attention_weights || {};
        if (attnWF) attnWF.innerText = `Weight: ${Math.round((attn.facial_alpha || 0.4) * 100)}%`;
        if (attnWV) attnWV.innerText = `Weight: ${Math.round((attn.speech_beta || 0.35) * 100)}%`;
        if (attnWT) attnWT.innerText = `Weight: ${Math.round((attn.linguistic_gamma || 0.25) * 100)}%`;

        // Analysis Page Probabilities
        const probs = fusion.fused_probabilities || {};
        for (const [emo, p] of Object.entries(probs)) {
            const valEl = document.getElementById(`analysis-val-${emo}`);
            const barEl = document.getElementById(`analysis-bar-${emo}`);
            if (valEl) valEl.innerText = `${(p * 100).toFixed(1)}%`;
            if (barEl) barEl.style.width = `${Math.min(100, p * 100)}%`;
        }

        // Risk Assessment
        const risk = data.risk_assessment || {};
        const riskScore = risk.risk_score !== undefined ? risk.risk_score : 12.5;
        const tierName = risk.tier_name || 'Level 0: Nominal';

        if (riskScoreVal) {
            riskScoreVal.innerText = riskScore.toFixed(1);
            riskScoreVal.className = `text-4xl lg:text-5xl font-extrabold tracking-tight ${riskScore > 70 ? 'text-red-400' : (riskScore > 50 ? 'text-orange-400' : (riskScore > 30 ? 'text-amber-400' : 'text-emerald-400'))}`;
        }
        if (riskBadge) {
            const isL3 = riskScore > 70;
            const isL2 = riskScore > 50 && riskScore <= 70;
            const isL1 = riskScore > 30 && riskScore <= 50;
            const badgeClass = isL3 ? 'bg-red-500/10 border-red-500/30 text-red-400' : (isL2 ? 'bg-orange-500/10 border-orange-500/30 text-orange-400' : (isL1 ? 'bg-amber-500/10 border-amber-500/30 text-amber-400' : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'));
            const dotClass = isL3 ? 'bg-red-400' : (isL2 ? 'bg-orange-400' : (isL1 ? 'bg-amber-400' : 'bg-emerald-400'));
            riskBadge.className = `px-4 py-2 rounded-xl border font-bold text-sm flex items-center gap-2 ${badgeClass}`;
            riskBadge.innerHTML = `<span class="w-2.5 h-2.5 rounded-full ${dotClass}"></span><span>${tierName}</span>`;
        }
        if (riskBarFill) {
            riskBarFill.style.width = `${Math.max(5, riskScore)}%`;
            riskBarFill.className = `h-full rounded-full transition-all duration-300 ${riskScore > 70 ? 'bg-red-500' : (riskScore > 50 ? 'bg-orange-500' : (riskScore > 30 ? 'bg-amber-500' : 'bg-emerald-500'))}`;
        }
        if (riskHorizonStatus) {
            riskHorizonStatus.innerText = riskScore > 70 ? 'Critical Severity' : (riskScore > 50 ? 'Moderate Alert' : 'Within Tolerance');
            riskHorizonStatus.className = `font-bold ${riskScore > 70 ? 'text-red-400' : (riskScore > 50 ? 'text-orange-400' : 'text-emerald-400')}`;
        }

        // Physical Vitals
        const phys = data.physical_distress || {};
        if (perclosVal) perclosVal.innerText = `${(phys.perclos_percentage || 4.2).toFixed(1)}%`;
        if (blinkVal) blinkVal.innerText = `${(phys.blink_rate_bpm || 16).toFixed(0)} BPM (Healthy)`;
        if (yawnVal) yawnVal.innerText = `${phys.yawns_per_min || 0} / min (Normal)`;
        if (painVal) painVal.innerText = `${(phys.pain_grimace_score || 0).toFixed(0)} / 100`;
        if (fatigueLevelText) {
            fatigueLevelText.innerText = phys.fatigue_level ? phys.fatigue_level : 'Nominal / Rested';
            fatigueLevelText.className = `font-bold ${phys.status_color === 'red' ? 'text-red-400' : (phys.status_color === 'orange' ? 'text-amber-400' : 'text-emerald-400')}`;
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

        // Dynamic Rolling Emotion Timeline
        updateValenceTimeline(domEmotion, fusion.valence);

        // Ground Alerts
        if (data.alert_dispatched) {
            addGroundAlertItem(data.alert_dispatched);
        }
    }

    function addGroundAlertItem(alert) {
        if (!alertFeedList) return;
        const item = document.createElement('div');
        const isCritical = alert.risk_level >= 3;
        item.className = `p-4 bg-dark-850 border-l-4 ${isCritical ? 'border-red-500 bg-red-950/20' : 'border-amber-500 bg-amber-950/20'} rounded-xl border border-slate-800 flex flex-col gap-1.5 text-sm`;
        item.innerHTML = `
            <div class="flex justify-between items-center font-bold">
                <span class="${isCritical ? 'text-red-400' : 'text-amber-400'}">${alert.alert_id} (Level ${alert.risk_level})</span>
                <span class="text-xs px-2 py-0.5 rounded ${isCritical ? 'bg-red-500/10 text-red-400' : 'bg-amber-500/10 text-amber-400'} font-semibold">Queued</span>
            </div>
            <p class="text-xs text-slate-300 leading-relaxed">${alert.emotional_state.primary.toUpperCase()} · Fatigue: ${alert.physical_state.fatigue_level}</p>
            <span class="text-[11px] text-slate-500 font-mono">${alert.timestamp} · S-Band Relay</span>
        `;
        alertFeedList.prepend(item);
        if (alertCountBadge) alertCountBadge.innerText = `${alertFeedList.children.length} Queued Alerts`;

        if (alertsTableBody) {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="py-4 font-bold text-indigo-400">${alert.alert_id}</td>
                <td class="py-4 text-slate-400">${alert.timestamp}</td>
                <td class="py-4"><span class="px-2.5 py-0.5 rounded-full text-xs font-semibold ${isCritical ? 'bg-red-500/10 text-red-400' : 'bg-amber-500/10 text-amber-400'}">Level ${alert.risk_level}</span></td>
                <td class="py-4 text-slate-200">${alert.emotional_state.primary.toUpperCase()}</td>
                <td class="py-4 text-amber-400 font-semibold" id="status-${alert.alert_id}">Queued_S-Band</td>
                <td class="py-4 text-right">
                    <button onclick="acknowledgeAlert('${alert.alert_id}', this)" class="btn-ack-alert px-3 py-1.5 text-xs font-semibold rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-all shadow-sm">Acknowledge</button>
                </td>
            `;
            alertsTableBody.prepend(tr);
        }
    }

    // ---------------------------------------------------------
    // 8. Conversational Companion AI Engine
    // ---------------------------------------------------------
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

            // Audio speech playback
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
        const isAi = speaker === 'ai';
        const bubble = document.createElement('div');
        bubble.className = `${isAi ? 'bg-dark-850 border-slate-800' : 'bg-indigo-950/40 border-indigo-500/30'} border rounded-2xl p-4 ${isAi ? 'self-end' : 'self-start'} w-11/12 text-sm leading-relaxed`;
        bubble.innerHTML = `
            <div class="flex justify-between items-center mb-1 text-xs ${isAi ? 'text-indigo-300 font-bold' : 'text-emerald-400 font-semibold'}">
                <span>${isAi ? 'MAITRI AI Companion' : 'Astronaut (Spoken)'}</span>
                <span class="text-slate-500 font-normal">Now</span>
            </div>
            <p class="text-slate-200">${text}</p>
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

    // ---------------------------------------------------------
    // 9. Guided Tactical Box Breathing Pacer
    // ---------------------------------------------------------
    let pacerInterval = null;
    function startBreathingPacer() {
        if (pacerInterval) clearInterval(pacerInterval);
        let phase = 0;
        let count = 4;

        const phases = [
            { name: "Inhale (4s)", action: "expand" },
            { name: "Hold (4s)", action: "expand" },
            { name: "Exhale (4s)", action: "contract" },
            { name: "Hold (4s)", action: "contract" }
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

    // ---------------------------------------------------------
    // 10. Flight Simulation Scenario Triggers
    // ---------------------------------------------------------
    const scenarioBtns = document.querySelectorAll('.btn-scenario');
    scenarioBtns.forEach(btn => {
        btn.addEventListener('click', async () => {
            scenarioBtns.forEach(b => b.classList.remove('bg-indigo-600', 'text-white', 'border-indigo-500'));
            btn.classList.add('bg-indigo-600', 'text-white', 'border-indigo-500');
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

    // ---------------------------------------------------------
    // 11. Crew Selector Change
    // ---------------------------------------------------------
    if (crewSelector) {
        crewSelector.addEventListener('change', async () => {
            await fetch('/api/crew/select', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ astronaut_id: crewSelector.value })
            });
        });
    }

    // ---------------------------------------------------------
    // 12. WebSocket Real-Time Telemetry Stream
    // ---------------------------------------------------------
    function initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
        try {
            ws = new WebSocket(wsUrl);
            ws.onopen = () => {
                console.log('✅ [MAITRI WS] Connected to live flight telemetry stream');
            };
            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'telemetry' && msg.payload) {
                        updateDashboardTelemetry(msg.payload);
                        if (msg.payload.vision) {
                            smoothEar = msg.payload.vision.eye_aspect_ratio || smoothEar;
                            smoothMar = msg.payload.vision.mouth_aspect_ratio || smoothMar;
                        }
                    } else if (msg.type === 'chat_response' && msg.payload) {
                        appendChatBubble('ai', msg.payload.ai_response);
                        speakWithBrowserTTS(msg.payload.ai_response);
                    }
                } catch (err) {
                    console.error('[MAITRI WS] Message parse error:', err);
                }
            };
            ws.onerror = (err) => {
                console.warn('[MAITRI WS] Socket error, falling back to HTTP:', err);
            };
            ws.onclose = () => {
                console.log('[MAITRI WS] Connection closed. Auto-reconnecting in 3s...');
                setTimeout(initWebSocket, 3000);
            };
        } catch (e) {
            console.warn('[MAITRI WS] Init exception:', e);
        }
    }

    // ---------------------------------------------------------
    // 13. Dynamic Emotion Valence Waveform (Real-Time SVG)
    // ---------------------------------------------------------
    function updateValenceTimeline(domEmotion, valenceScore) {
        if (!timelineValencePath || !timelineActiveNode) return;
        
        let y = 58;
        if (domEmotion === 'happy') y = 25;
        else if (domEmotion === 'calm' || domEmotion === 'neutral') y = 55;
        else if (domEmotion === 'fatigued') y = 78;
        else if (domEmotion === 'sad' || domEmotion === 'isolated') y = 84;
        else if (domEmotion === 'stressed' || domEmotion === 'frustrated') y = 96;

        if (valenceScore !== undefined && valenceScore !== null) {
            y = Math.min(105, Math.max(15, Math.round(60 - (valenceScore * 40))));
        }

        valenceBuffer.push(y);
        if (valenceBuffer.length > 20) valenceBuffer.shift();

        const step = 500 / (valenceBuffer.length - 1);
        let d = `M 0,${valenceBuffer[0]}`;
        for (let i = 1; i < valenceBuffer.length; i++) {
            const x = Math.round(i * step);
            const prevX = Math.round((i - 1) * step);
            const midX = Math.round((prevX + x) / 2);
            d += ` C ${midX},${valenceBuffer[i-1]} ${midX},${valenceBuffer[i]} ${x},${valenceBuffer[i]}`;
        }

        timelineValencePath.setAttribute('d', d);
        timelineActiveNode.setAttribute('cx', '500');
        timelineActiveNode.setAttribute('cy', y);
        timelineActiveNode.setAttribute('fill', y > 75 ? '#F87171' : (y > 65 ? '#FBBF24' : '#34D399'));
    }

    // ---------------------------------------------------------
    // 14. Alert Acknowledge Action Handler
    // ---------------------------------------------------------
    window.acknowledgeAlert = async function(alertId, btnEl) {
        if (!alertId) return;
        try {
            const resp = await fetch(`/api/alerts/acknowledge/${alertId}`, { method: 'POST' });
            const res = await resp.json();
            if (res.status === 'ACKNOWLEDGED') {
                if (btnEl) {
                    btnEl.disabled = true;
                    btnEl.className = 'px-3 py-1.5 text-xs font-semibold rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 cursor-default';
                    btnEl.innerText = 'Acknowledged';
                }
                const statusEl = document.getElementById(`status-${alertId}`);
                if (statusEl) {
                    statusEl.className = 'py-4 text-emerald-400 font-semibold';
                    statusEl.innerText = 'Acknowledged';
                }
            }
        } catch (e) {
            console.error('Ack alert error:', e);
        }
    };

    // ---------------------------------------------------------
    // 15. Live Session History Loader
    // ---------------------------------------------------------
    window.loadSessionHistory = async function() {
        try {
            const resp = await fetch('/api/history/telemetry');
            const data = await resp.json();
            const container = document.getElementById('session-telemetry-logs');
            if (!container) return;

            const list = data.history || [];
            if (list.length > 0) {
                container.innerHTML = '';
                list.slice(0, 15).forEach((record, idx) => {
                    const item = document.createElement('div');
                    const dom = record.dominant_emotion || 'neutral';
                    const risk = record.risk_score !== undefined ? record.risk_score.toFixed(1) : '12.0';
                    const isHigh = record.risk_score > 50;

                    let timeStr = 'Recent';
                    if (record.timestamp) {
                        if (typeof record.timestamp === 'number') {
                            const d = new Date(record.timestamp > 1e11 ? record.timestamp : record.timestamp * 1000);
                            timeStr = d.toTimeString().split(' ')[0];
                        } else if (typeof record.timestamp === 'string') {
                            timeStr = record.timestamp.includes('T') ? record.timestamp.split('T')[1]?.slice(0, 8) : record.timestamp.slice(-8);
                        }
                    }

                    item.className = `p-4 bg-dark-850 border-l-4 ${isHigh ? 'border-amber-500' : 'border-emerald-500'} rounded-xl border border-slate-800 flex justify-between items-center`;
                    item.innerHTML = `
                        <div>
                            <div class="flex items-center gap-2">
                                <strong class="text-white">Record #${record.session_id || idx + 1} — ${dom.toUpperCase()}</strong>
                                <span class="text-xs font-semibold px-2 py-0.5 rounded-full ${isHigh ? 'bg-amber-500/10 text-amber-400' : 'bg-emerald-500/10 text-emerald-400'}">Risk: ${risk}</span>
                            </div>
                            <p class="text-xs text-slate-400 mt-0.5">PERCLOS: ${record.perclos ? record.perclos.toFixed(1) + '%' : '4.0%'} · Pitch F0: ${record.pitch_f0 ? record.pitch_f0.toFixed(0) + ' Hz' : '130 Hz'} · Vocal Tension: ${record.vocal_tension ? record.vocal_tension.toFixed(0) + '%' : '12%'}</p>
                        </div>
                        <span class="text-xs font-mono text-slate-400">${timeStr}</span>
                    `;
                    container.appendChild(item);
                });
            }
        } catch (e) {
            console.error('Session history load error:', e);
        }
    };

    // Initialize Real-Time WebSocket & Session Data on load
    initWebSocket();
    window.loadSessionHistory();
});
