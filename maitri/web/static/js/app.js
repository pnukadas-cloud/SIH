/**
 * MAITRI — AI Well-Being System Client Engine
 * ISRO Bhartiya Antariksh Station (BAS)
 * Multimodal Perception, Biometric Tracking, RBAC Console & Real-Time HUD
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
    const chatTypingIndicator = document.getElementById('chat-typing-indicator');
    
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

    // Mathematical Breakdown Components
    const compAffect = document.getElementById('comp-affect');
    const compFatigue = document.getElementById('comp-fatigue');
    const compTension = document.getElementById('comp-tension');

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
    let ws = null;
    const valenceBuffer = [58, 55, 60, 52, 48, 65, 50, 58, 62, 58];

    let livePitchHz = 0;
    let liveRmsEnergy = 0;
    let liveVocalTension = 0;
    let faceBox = null;
    let isFaceDetected = false;
    let smoothEar = 0.28;
    let smoothMar = 0.20;
    let missionStartTime = Date.now() - (4 * 3600 + 12 * 60 + 33) * 1000;
    let stateStartTimestamp = Date.now() - (14 * 60 + 32) * 1000;

    // Standby State Reset across HUD
    function setSensorStandbyState(isSearchingFace = false) {
        isFaceDetected = false;
        faceBox = null;

        if (isSearchingFace) {
            if (domEmotionText) {
                domEmotionText.innerText = "No Face Detected";
                domEmotionText.className = "text-2xl font-extrabold text-amber-400 tracking-tight";
            }
            if (domEmotionConf) domEmotionConf.innerText = "0%";
            if (emotionTrendBadge) {
                emotionTrendBadge.innerHTML = '<span class="material-symbols-outlined text-base">center_focus_weak</span><span>Align Face</span>';
                emotionTrendBadge.className = 'px-3.5 py-1.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400 font-semibold text-xs flex items-center gap-1.5';
            }
            if (riskScoreVal) {
                riskScoreVal.innerText = "--";
                riskScoreVal.className = "text-4xl font-extrabold tracking-tight text-slate-500";
            }
            if (riskBadge) {
                riskBadge.className = 'px-3 py-1.5 rounded-xl border font-bold text-xs flex items-center gap-1.5 bg-amber-500/10 border-amber-500/30 text-amber-400';
                riskBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span><span>Awaiting Face Lock</span>';
            }
            if (riskBarFill) {
                riskBarFill.style.width = "0%";
                riskBarFill.className = "h-full rounded-full bg-slate-700 transition-all duration-300";
            }
            if (riskHorizonStatus) {
                riskHorizonStatus.innerText = "Awaiting Face Lock";
                riskHorizonStatus.className = "font-bold text-amber-400";
            }
            if (fatigueLevelText) {
                fatigueLevelText.innerText = "Awaiting Face Lock";
                fatigueLevelText.className = "text-xs font-semibold text-amber-400";
            }
            if (perclosVal) perclosVal.innerText = "0.0%";
            if (hudEarMetric) hudEarMetric.innerText = "--";
            if (blinkVal) blinkVal.innerText = "0 / min";
            if (yawnVal) yawnVal.innerText = "0";
            if (painVal) painVal.innerText = "0%";
            if (compAffect) compAffect.innerText = "-- pts";
            if (compFatigue) compFatigue.innerText = "-- pts";
            if (compTension) compTension.innerText = "-- pts";
            if (hudAstronautWellbeing) {
                hudAstronautWellbeing.innerText = "ALIGN FACE";
                hudAstronautWellbeing.className = "text-amber-400 font-semibold";
            }
            if (hudAstronautRisk) hudAstronautRisk.innerText = "-- / 100";
        } else {
            // Camera completely offline
            if (domEmotionText) {
                domEmotionText.innerText = "Standby";
                domEmotionText.className = "text-3xl font-extrabold text-slate-400 tracking-tight";
            }
            if (domEmotionConf) domEmotionConf.innerText = "--";
            if (emotionTrendBadge) {
                emotionTrendBadge.innerHTML = '<span class="material-symbols-outlined text-base">sensors_off</span><span>Camera Standby</span>';
                emotionTrendBadge.className = 'px-3.5 py-1.5 rounded-xl bg-slate-800 border border-slate-700 text-slate-400 font-semibold text-xs flex items-center gap-1.5';
            }
            if (riskScoreVal) {
                riskScoreVal.innerText = "--";
                riskScoreVal.className = "text-4xl font-extrabold tracking-tight text-slate-500";
            }
            if (riskBadge) {
                riskBadge.className = 'px-3 py-1.5 rounded-xl border font-bold text-xs flex items-center gap-1.5 bg-slate-800 border border-slate-700 text-slate-400';
                riskBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-slate-500"></span><span>Standby: Camera Offline</span>';
            }
            if (riskBarFill) {
                riskBarFill.style.width = "0%";
                riskBarFill.className = "h-full rounded-full bg-slate-700 transition-all duration-300";
            }
            if (riskHorizonStatus) {
                riskHorizonStatus.innerText = "System Standby";
                riskHorizonStatus.className = "font-bold text-slate-400";
            }
            if (fatigueLevelText) {
                fatigueLevelText.innerText = "Standby (Camera Offline)";
                fatigueLevelText.className = "text-xs font-semibold text-slate-400";
            }
            if (perclosVal) perclosVal.innerText = "--";
            if (hudEarMetric) hudEarMetric.innerText = "--";
            if (blinkVal) blinkVal.innerText = "--";
            if (yawnVal) yawnVal.innerText = "--";
            if (painVal) painVal.innerText = "--";
            if (compAffect) compAffect.innerText = "-- pts";
            if (compFatigue) compFatigue.innerText = "-- pts";
            if (compTension) compTension.innerText = "-- pts";
            if (hudAstronautWellbeing) {
                hudAstronautWellbeing.innerText = "STANDBY";
                hudAstronautWellbeing.className = "text-slate-400 font-semibold";
            }
            if (hudAstronautRisk) hudAstronautRisk.innerText = "-- / 100";
            if (stateDurationText) stateDurationText.innerText = "--";
        }
    }

    // ---------------------------------------------------------
    // 1. Mission Clock & Timers
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
    // 2. Camera & Microphone Permission State Management
    // ---------------------------------------------------------
    function updatePermissionUI(state, message) {
        const banner = document.getElementById('media-permission-banner');
        const iconContainer = document.getElementById('permission-icon-container');
        const icon = document.getElementById('permission-icon');
        const title = document.getElementById('permission-title');
        const statusTag = document.getElementById('permission-status-tag');
        const desc = document.getElementById('permission-description');
        if (!banner) return;

        if (state === 'granted') {
            banner.className = 'web-card p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-l-4 border-emerald-500';
            if (iconContainer) iconContainer.className = 'w-10 h-10 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0';
            if (icon) icon.innerText = 'check_circle';
            if (title) title.innerText = 'Biometric Sensors: Active & Locked';
            if (statusTag) {
                statusTag.className = 'px-2.5 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30';
                statusTag.innerText = 'Permission Granted';
            }
            if (desc) desc.innerText = 'Continuous multimodal perception active (FACS Action Units AU04/06/12/20/43, EAR/MAR, PERCLOS, and F0 pitch).';
        } else if (state === 'denied') {
            banner.className = 'web-card p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-l-4 border-red-500';
            if (iconContainer) iconContainer.className = 'w-10 h-10 rounded-xl bg-red-500/20 text-red-400 flex items-center justify-center shrink-0';
            if (icon) icon.innerText = 'gpp_bad';
            if (title) title.innerText = 'Sensor Permissions Blocked';
            if (statusTag) {
                statusTag.className = 'px-2.5 py-0.5 text-xs font-semibold rounded-full bg-red-500/10 text-red-400 border border-red-500/30';
                statusTag.innerText = 'Blocked in Browser';
            }
            if (desc) desc.innerText = 'Camera/Microphone access was denied. To enable: Click the lock icon next to the URL in your browser address bar, set Camera & Microphone to "Allow", and reload this page.';
        } else if (state === 'unavailable') {
            banner.className = 'web-card p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-l-4 border-amber-500';
            if (iconContainer) iconContainer.className = 'w-10 h-10 rounded-xl bg-amber-500/20 text-amber-400 flex items-center justify-center shrink-0';
            if (icon) icon.innerText = 'videocam_off';
            if (title) title.innerText = 'No Camera/Microphone Device Detected';
            if (statusTag) {
                statusTag.className = 'px-2.5 py-0.5 text-xs font-semibold rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30';
                statusTag.innerText = 'Device Unavailable';
            }
            if (desc) desc.innerText = 'Hardware capture devices not detected. You can use the 1-Click Flight Demonstration Scenarios above to test all perception pipelines.';
        } else {
            // Default / Not yet requested
            banner.className = 'web-card p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-l-4 border-indigo-500';
            if (iconContainer) iconContainer.className = 'w-10 h-10 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center shrink-0';
            if (icon) icon.innerText = 'perm_camera_mic';
            if (title) title.innerText = 'Biometric Sensor Stream';
            if (statusTag) {
                statusTag.className = 'px-2.5 py-0.5 text-xs font-semibold rounded-full bg-slate-800 text-slate-300 border border-slate-700';
                statusTag.innerText = 'Ready to Request';
            }
            if (desc) desc.innerText = 'Connect your webcam & microphone for real-time facial Action Units (FACS), blink tracking, and acoustic F0 prosody.';
        }
    }

    async function checkExistingPermissions() {
        if (navigator.permissions && navigator.permissions.query) {
            try {
                const camStatus = await navigator.permissions.query({ name: 'camera' });
                if (camStatus.state === 'granted') {
                    updatePermissionUI('granted');
                } else if (camStatus.state === 'denied') {
                    updatePermissionUI('denied');
                }
            } catch(e) {}
        }
    }
    checkExistingPermissions();

    let isVoiceActive = false;
    let micAudioStream = null;

    // ---------------------------------------------------------
    // 3. Speech Recognition & Microphone Stream Management
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

            if (interim && chatInput) {
                chatInput.value = interim;
                chatInput.placeholder = `Listening: "${interim}"...`;
            }

            if (finalTranscript && finalTranscript.trim().length > 1) {
                currentSpeechText = finalTranscript.trim();
                if (chatInput) chatInput.value = "";
                sendChatMessage(currentSpeechText);
                currentSpeechText = "";
            }
        };

        recognition.onerror = (e) => {
            console.log('[SpeechRec Status]:', e.error);
            if (e.error === 'not-allowed') {
                updatePermissionUI('denied');
            }
        };

        recognition.onend = () => {
            if (isVoiceActive) {
                try { recognition.start(); } catch(e) {}
            } else {
                if (chatInput) chatInput.placeholder = "Type or speak to MAITRI...";
            }
        };
    }

    async function toggleVoiceListening() {
        if (!isVoiceActive) {
            // ACTIVATE MICROPHONE
            try {
                if (!mediaStream && !micAudioStream) {
                    micAudioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    initAudioAnalyser(micAudioStream);
                } else if (mediaStream) {
                    mediaStream.getAudioTracks().forEach(track => track.enabled = true);
                    initAudioAnalyser(mediaStream);
                } else if (micAudioStream) {
                    micAudioStream.getAudioTracks().forEach(track => track.enabled = true);
                    initAudioAnalyser(micAudioStream);
                }

                if (recognition) {
                    try { recognition.start(); } catch(e) {}
                }

                isVoiceActive = true;
                if (btnToggleVoiceListen) {
                    btnToggleVoiceListen.classList.remove('bg-dark-850', 'text-slate-200', 'border-slate-700');
                    btnToggleVoiceListen.classList.add('bg-red-600/20', 'border-red-500', 'text-red-400');
                    btnToggleVoiceListen.innerHTML = '<span class="material-symbols-outlined text-[16px] animate-pulse">mic</span><span>Voice: Live</span>';
                }
                if (btnVoiceInput) {
                    btnVoiceInput.classList.add('bg-red-600/20', 'border-red-500', 'text-red-400');
                }
                if (chatInput) {
                    chatInput.placeholder = "🎙️ Microphone Live: Speak naturally to MAITRI...";
                }
                updatePermissionUI('granted');
            } catch (err) {
                console.warn("Microphone access error:", err);
                updatePermissionUI('denied', err.message);
                alert("Microphone access could not be enabled: " + err.message + "\nPlease check that your microphone is allowed in browser settings.");
            }
        } else {
            // MUTE MICROPHONE
            isVoiceActive = false;
            if (mediaStream) {
                mediaStream.getAudioTracks().forEach(track => track.enabled = false);
            }
            if (micAudioStream) {
                micAudioStream.getAudioTracks().forEach(track => { track.enabled = false; track.stop(); });
                micAudioStream = null;
            }
            if (recognition) {
                try { recognition.stop(); } catch(e) {}
            }

            if (btnToggleVoiceListen) {
                btnToggleVoiceListen.classList.add('bg-dark-850', 'text-slate-200', 'border-slate-700');
                btnToggleVoiceListen.classList.remove('bg-red-600/20', 'border-red-500', 'text-red-400');
                btnToggleVoiceListen.innerHTML = '<span class="material-symbols-outlined text-[16px]">mic_off</span><span>Voice: Off</span>';
            }
            if (btnVoiceInput) {
                btnVoiceInput.classList.remove('bg-red-600/20', 'border-red-500', 'text-red-400');
            }
            if (chatInput) {
                chatInput.placeholder = "Type or speak to MAITRI...";
            }

            // Immediately reset audio UI readings
            livePitchHz = 0;
            liveVocalTension = 0;
            if (pitchVal) pitchVal.innerText = "0 Hz (Muted)";
            if (vocalTensionVal) vocalTensionVal.innerText = "0%";
            if (serPitchDisplay) serPitchDisplay.innerText = "0 Hz (Muted)";
            if (serTensionDisplay) serTensionDisplay.innerText = "0%";

            // Draw calm flat baseline
            if (waveformCanvas && waveformCtx) {
                waveformCtx.fillStyle = "#090D16";
                waveformCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
                waveformCtx.strokeStyle = "rgba(99, 102, 241, 0.2)";
                waveformCtx.lineWidth = 1;
                waveformCtx.beginPath();
                waveformCtx.moveTo(0, waveformCanvas.height / 2);
                waveformCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
                waveformCtx.stroke();
            }
        }
    }

    if (btnToggleVoiceListen) {
        btnToggleVoiceListen.addEventListener('click', toggleVoiceListening);
    }
    if (btnVoiceInput) {
        btnVoiceInput.addEventListener('click', toggleVoiceListening);
    }

    // ---------------------------------------------------------
    // 4. Live Optical Camera & Video Capture
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
                audio: isVoiceActive
            });
            videoElem.srcObject = mediaStream;
            await videoElem.play();
            isCameraActive = true;
            
            btnToggleCamera.innerHTML = '<span class="material-symbols-outlined text-[16px]">videocam_off</span><span>Stop Camera</span>';
            btnToggleCamera.classList.remove('bg-indigo-600', 'hover:bg-indigo-500');
            btnToggleCamera.classList.add('bg-red-600', 'hover:bg-red-500');

            if (isVoiceActive) {
                initAudioAnalyser(mediaStream);
            }
            renderLiveHudCanvas();
            streamInterval = setInterval(sendFrameToBackend, 250);
            
            if (opticalLockBadge) {
                opticalLockBadge.innerText = "Camera Active";
                opticalLockBadge.className = "px-2.5 py-0.5 text-xs font-semibold bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 rounded-full";
            }
            if (gracefulDegradeStatus) {
                gracefulDegradeStatus.innerText = isVoiceActive ? "3 of 3 Modalities Active" : "Vision + Text Active (Voice Muted)";
                gracefulDegradeStatus.className = "text-xs font-semibold text-emerald-400";
            }
            updatePermissionUI('granted');
        } catch (err) {
            console.warn("Camera access error:", err.name, err.message);
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                updatePermissionUI('denied');
            } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                updatePermissionUI('unavailable');
            } else {
                updatePermissionUI('denied', err.message);
            }
        }
    }

    function stopLiveCamera() {
        if (mediaStream) {
            mediaStream.getVideoTracks().forEach(track => track.stop());
            if (!isVoiceActive) {
                mediaStream.getAudioTracks().forEach(track => track.stop());
            }
        }
        if (streamInterval) clearInterval(streamInterval);
        if (animFrameId) cancelAnimationFrame(animFrameId);
        videoElem.srcObject = null;
        isCameraActive = false;
        
        btnToggleCamera.innerHTML = '<span class="material-symbols-outlined text-[16px]">videocam</span><span>Start Camera</span>';
        btnToggleCamera.classList.add('bg-indigo-600', 'hover:bg-indigo-500');
        btnToggleCamera.classList.remove('bg-red-600', 'hover:bg-red-500');
        
        if (hudCtx) hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
        
        if (opticalLockBadge) {
            opticalLockBadge.innerText = "Camera Standby";
            opticalLockBadge.className = "px-2.5 py-0.5 text-xs font-semibold bg-dark-850 border border-slate-800 text-slate-400 rounded-full";
        }
        if (gracefulDegradeStatus) {
            gracefulDegradeStatus.innerText = isVoiceActive ? "Voice + Text Active (Camera Standby)" : "System Standby";
            gracefulDegradeStatus.className = "text-xs font-semibold text-amber-400";
        }

        // Clean standby reset across all well-being and affect cards
        setSensorStandbyState(false);
    }

    // ---------------------------------------------------------
    // 5. Audio Waveform & Autocorrelation Pitch Extraction
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
        return sampleRate / T0;
    }

    function renderLiveHudCanvas() {
        if (!isCameraActive) return;
        animFrameId = requestAnimationFrame(renderLiveHudCanvas);

        if (hudCanvas && hudCtx && videoElem.readyState >= 2) {
            const w = videoElem.videoWidth || 640;
            const h = videoElem.videoHeight || 480;
            if (hudCanvas.width !== w || hudCanvas.height !== h) {
                hudCanvas.width = w;
                hudCanvas.height = h;
            }

            // Clear transparent overlay so native video plays underneath
            hudCtx.clearRect(0, 0, w, h);

            if (!isFaceDetected || !faceBox) {
                // Draw searching / alignment reticle
                const scanW = Math.floor(w * 0.45);
                const scanH = Math.floor(h * 0.60);
                const scanX = Math.floor((w - scanW) / 2);
                const scanY = Math.floor((h - scanH) / 2.2);

                hudCtx.save();
                hudCtx.setLineDash([8, 6]);
                hudCtx.strokeStyle = "rgba(245, 158, 11, 0.75)";
                hudCtx.lineWidth = 2;
                hudCtx.strokeRect(scanX, scanY, scanW, scanH);
                hudCtx.restore();

                // Corner accents in amber
                const len = 20;
                hudCtx.strokeStyle = "#F59E0B";
                hudCtx.lineWidth = 3;
                hudCtx.beginPath();
                hudCtx.moveTo(scanX, scanY + len); hudCtx.lineTo(scanX, scanY); hudCtx.lineTo(scanX + len, scanY);
                hudCtx.moveTo(scanX + scanW - len, scanY); hudCtx.lineTo(scanX + scanW, scanY); hudCtx.lineTo(scanX + scanW, scanY + len);
                hudCtx.moveTo(scanX, scanY + scanH - len); hudCtx.lineTo(scanX, scanY + scanH); hudCtx.lineTo(scanX + len, scanY + scanH);
                hudCtx.moveTo(scanX + scanW - len, scanY + scanH); hudCtx.lineTo(scanX + scanW, scanY + scanH); hudCtx.lineTo(scanX + scanW, scanY + scanH - len);
                hudCtx.stroke();

                // Tag
                hudCtx.fillStyle = "rgba(9, 13, 22, 0.85)";
                hudCtx.fillRect(scanX, scanY - 26, 210, 22);
                hudCtx.font = "bold 11px Inter, sans-serif";
                hudCtx.fillStyle = "#F59E0B";
                hudCtx.fillText("OPTICAL SCANNER: SEARCHING FACE", scanX + 8, scanY - 11);
                return;
            }

            const boxW = faceBox.fw;
            const boxH = faceBox.fh;
            const boxX = faceBox.x;
            const boxY = faceBox.y;

            // Reticle Target Box (Cyan / Emerald Optical Lock)
            hudCtx.strokeStyle = "rgba(52, 211, 153, 0.85)";
            hudCtx.lineWidth = 2;
            hudCtx.strokeRect(boxX, boxY, boxW, boxH);

            // Corner Accents
            const len = 20;
            hudCtx.strokeStyle = "#34D399";
            hudCtx.lineWidth = 3;
            hudCtx.beginPath();
            hudCtx.moveTo(boxX, boxY + len); hudCtx.lineTo(boxX, boxY); hudCtx.lineTo(boxX + len, boxY);
            hudCtx.moveTo(boxX + boxW - len, boxY); hudCtx.lineTo(boxX + boxW, boxY); hudCtx.lineTo(boxX + boxW, boxY + len);
            hudCtx.moveTo(boxX, boxY + boxH - len); hudCtx.lineTo(boxX, boxY + boxH); hudCtx.lineTo(boxX + len, boxY + boxH);
            hudCtx.moveTo(boxX + boxW - len, boxY + boxH); hudCtx.lineTo(boxX + boxW, boxY + boxH); hudCtx.lineTo(boxX + boxW, boxY + boxH - len);
            hudCtx.stroke();

            // Eye Landmark Circles
            const eyeY = Math.floor(boxY + boxH * 0.35);
            const leftEyeX = Math.floor(boxX + boxW * 0.30);
            const rightEyeX = Math.floor(boxX + boxW * 0.70);
            const eyeRad = Math.max(4, Math.floor(smoothEar * 20));

            hudCtx.fillStyle = smoothEar < 0.16 ? "rgba(239, 68, 68, 0.9)" : "rgba(52, 211, 153, 0.9)";
            hudCtx.beginPath();
            hudCtx.arc(leftEyeX, eyeY, eyeRad, 0, 2 * Math.PI);
            hudCtx.arc(rightEyeX, eyeY, eyeRad, 0, 2 * Math.PI);
            hudCtx.fill();

            // Mouth Landmark
            const mouthY = Math.floor(boxY + boxH * 0.76);
            const mouthX = Math.floor(boxX + boxW * 0.50);
            const mouthRad = Math.max(4, Math.floor(smoothMar * 22));
            hudCtx.fillStyle = smoothMar > 0.45 ? "rgba(245, 158, 11, 0.9)" : "rgba(99, 102, 241, 0.9)";
            hudCtx.beginPath();
            hudCtx.ellipse(mouthX, mouthY, mouthRad * 1.6, mouthRad, 0, 0, 2 * Math.PI);
            hudCtx.fill();

            // Real-time Facial Biomarker HUD Tag
            hudCtx.fillStyle = "rgba(9, 13, 22, 0.85)";
            hudCtx.fillRect(boxX, boxY - 26, 185, 22);
            hudCtx.font = "bold 11px Inter, sans-serif";
            hudCtx.fillStyle = "#34D399";
            hudCtx.fillText(`OPTICAL LOCK: EAR ${smoothEar.toFixed(2)} | MAR ${smoothMar.toFixed(2)}`, boxX + 6, boxY - 11);

            if (hudEarMetric) hudEarMetric.innerText = smoothEar.toFixed(2);
        }
    }

    // ---------------------------------------------------------
    // 5. Audio Waveform & Autocorrelation Pitch Extraction
    // ---------------------------------------------------------
    let audioAnimFrameId = null;

    function renderAudioWaveform() {
        if (!waveformCanvas || !waveformCtx) return;

        if (!isVoiceActive || !audioAnalyser) {
            // Microphone is OFF / Muted -> Draw clean flat baseline and clear pitch
            waveformCtx.fillStyle = "#090D16";
            waveformCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
            waveformCtx.strokeStyle = "rgba(99, 102, 241, 0.25)";
            waveformCtx.lineWidth = 1;
            waveformCtx.beginPath();
            waveformCtx.moveTo(0, waveformCanvas.height / 2);
            waveformCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
            waveformCtx.stroke();

            if (pitchVal) pitchVal.innerText = "0 Hz (Muted)";
            if (vocalTensionVal) vocalTensionVal.innerText = "0%";
            if (serPitchDisplay) serPitchDisplay.innerText = "0 Hz (Muted)";
            if (serTensionDisplay) serTensionDisplay.innerText = "0%";
            return;
        }

        audioAnimFrameId = requestAnimationFrame(renderAudioWaveform);

        audioAnalyser.getByteTimeDomainData(audioDataArray);
        audioAnalyser.getFloatTimeDomainData(audioTimeData);

        const pitch = detectPitchAutocorr(audioTimeData, audioContext.sampleRate);
        if (pitch > 60 && pitch < 450) {
            livePitchHz = Math.round(pitch);
            if (pitchVal) pitchVal.innerText = `${livePitchHz} Hz`;
            if (serPitchDisplay) serPitchDisplay.innerText = `${livePitchHz} Hz`;
            
            liveVocalTension = Math.min(100, Math.max(0, Math.round(((livePitchHz - 128) / 80) * 100)));
            if (vocalTensionVal) vocalTensionVal.innerText = `${liveVocalTension}%`;
            if (serTensionDisplay) serTensionDisplay.innerText = `${liveVocalTension}%`;
        } else {
            // Microphone is active, but user is paused/quiet
            if (pitchVal) pitchVal.innerText = "Quiet (Listening...)";
            if (vocalTensionVal) vocalTensionVal.innerText = "0%";
        }

        waveformCtx.fillStyle = "#090D16";
        waveformCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        waveformCtx.lineWidth = 1.5;
        waveformCtx.strokeStyle = "#34D399";
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
    function updateDashboardTelemetry(data, isForced = false) {
        if (!data) return;

        const isSim = !!data.is_simulation || !!data.scenario;
        const hasFace = !!(data.vision && data.vision.face_detected);

        // Check if camera is inactive or no face detected
        if (!isSim) {
            if (!isCameraActive) {
                setSensorStandbyState(false);
                return;
            } else if (!hasFace && !isVoiceActive) {
                setSensorStandbyState(true);
                return;
            }
        }

        // Face is present: calculate dynamic faceBox for HUD
        if (data.vision && data.vision.face_bounding_box && videoElem) {
            const scaleX = (videoElem.videoWidth || 640) / 320;
            const scaleY = (videoElem.videoHeight || 480) / 240;
            faceBox = {
                x: Math.floor(data.vision.face_bounding_box.x * scaleX),
                y: Math.floor(data.vision.face_bounding_box.y * scaleY),
                fw: Math.floor(data.vision.face_bounding_box.w * scaleX),
                fh: Math.floor(data.vision.face_bounding_box.h * scaleY)
            };
            isFaceDetected = true;
            smoothEar = data.vision.eye_aspect_ratio || smoothEar;
            smoothMar = data.vision.mouth_aspect_ratio || smoothMar;
        } else if (isSim) {
            isFaceDetected = true;
        }

        // Fused Emotion
        const fusion = data.fusion || {};
        const domEmotion = fusion.dominant_emotion || 'neutral';
        const confidence = fusion.confidence || 0.85;

        if (domEmotionText) {
            const emotionDisplayNames = {
                'happy': 'Happy / Expressive',
                'neutral': 'Calm / Neutral',
                'stressed': 'Stressed / Strained',
                'frustrated': 'Frustrated / Tense',
                'anxious': 'Anxious / Vigilant',
                'sad': 'Sad / Withdrawn',
                'fatigued': 'Fatigued / Drowsy'
            };
            const emotionColors = {
                'happy': 'text-emerald-400',
                'neutral': 'text-cyan-300',
                'stressed': 'text-red-400',
                'frustrated': 'text-rose-500',
                'anxious': 'text-amber-400',
                'sad': 'text-blue-400',
                'fatigued': 'text-purple-400'
            };
            domEmotionText.innerText = emotionDisplayNames[domEmotion] || (domEmotion.charAt(0).toUpperCase() + domEmotion.slice(1));
            domEmotionText.className = `text-3xl font-extrabold tracking-tight ${emotionColors[domEmotion] || 'text-white'}`;
        }
        if (domEmotionConf) domEmotionConf.innerText = `${(confidence * 100).toFixed(0)}%`;

        // Trend Badge
        if (emotionTrendBadge) {
            if (domEmotion === 'stressed' || domEmotion === 'frustrated') {
                emotionTrendBadge.innerHTML = '<span class="material-symbols-outlined text-base">trending_up</span><span>Worsening</span>';
                emotionTrendBadge.className = 'px-3.5 py-1.5 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 font-semibold text-xs flex items-center gap-1.5';
            } else if (domEmotion === 'anxious') {
                emotionTrendBadge.innerHTML = '<span class="material-symbols-outlined text-base">warning</span><span>Elevated</span>';
                emotionTrendBadge.className = 'px-3.5 py-1.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400 font-semibold text-xs flex items-center gap-1.5';
            } else if (domEmotion === 'sad') {
                emotionTrendBadge.innerHTML = '<span class="material-symbols-outlined text-base">trending_down</span><span>Low Affect</span>';
                emotionTrendBadge.className = 'px-3.5 py-1.5 rounded-xl bg-blue-500/10 border border-blue-500/30 text-blue-400 font-semibold text-xs flex items-center gap-1.5';
            } else if (domEmotion === 'fatigued') {
                emotionTrendBadge.innerHTML = '<span class="material-symbols-outlined text-base">bedtime</span><span>Fatigue Buildup</span>';
                emotionTrendBadge.className = 'px-3.5 py-1.5 rounded-xl bg-purple-500/10 border border-purple-500/30 text-purple-400 font-semibold text-xs flex items-center gap-1.5';
            } else if (domEmotion === 'happy') {
                emotionTrendBadge.innerHTML = '<span class="material-symbols-outlined text-base">mood</span><span>Positive Flow</span>';
                emotionTrendBadge.className = 'px-3.5 py-1.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-semibold text-xs flex items-center gap-1.5';
            } else {
                emotionTrendBadge.innerHTML = '<span class="material-symbols-outlined text-base">trending_flat</span><span>Stable</span>';
                emotionTrendBadge.className = 'px-3.5 py-1.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-semibold text-xs flex items-center gap-1.5';
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
            riskScoreVal.className = `text-4xl font-extrabold tracking-tight ${riskScore > 70 ? 'text-red-400' : (riskScore > 50 ? 'text-orange-400' : (riskScore > 30 ? 'text-amber-400' : 'text-emerald-400'))}`;
        }
        if (riskBadge) {
            const isL3 = riskScore > 70;
            const isL2 = riskScore > 50 && riskScore <= 70;
            const isL1 = riskScore > 30 && riskScore <= 50;
            const badgeClass = isL3 ? 'bg-red-500/10 border-red-500/30 text-red-400' : (isL2 ? 'bg-orange-500/10 border-orange-500/30 text-orange-400' : (isL1 ? 'bg-amber-500/10 border-amber-500/30 text-amber-400' : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'));
            const dotClass = isL3 ? 'bg-red-400' : (isL2 ? 'bg-orange-400' : (isL1 ? 'bg-amber-400' : 'bg-emerald-400'));
            riskBadge.className = `px-3 py-1.5 rounded-xl border font-bold text-xs flex items-center gap-1.5 ${badgeClass}`;
            riskBadge.innerHTML = `<span class="w-2 h-2 rounded-full ${dotClass}"></span><span>${tierName}</span>`;
        }
        if (riskBarFill) {
            riskBarFill.style.width = `${Math.max(5, riskScore)}%`;
            riskBarFill.className = `h-full rounded-full transition-all duration-300 ${riskScore > 70 ? 'bg-red-500' : (riskScore > 50 ? 'bg-orange-500' : (riskScore > 30 ? 'bg-amber-500' : 'bg-emerald-500'))}`;
        }
        if (riskHorizonStatus) {
            riskHorizonStatus.innerText = riskScore > 70 ? 'Critical Severity' : (riskScore > 50 ? 'Moderate Alert' : 'Within Tolerance');
            riskHorizonStatus.className = `font-bold ${riskScore > 70 ? 'text-red-400' : (riskScore > 50 ? 'text-orange-400' : 'text-emerald-400')}`;
        }

        // Live Facial Biomarkers & Physical Distress Updates
        const vision = data.vision || {};
        const phys = data.physical_distress || {};

        if (perclosVal) {
            const pPct = (vision.perclos !== undefined ? (vision.perclos * 100) : (phys.perclos_percentage || 0.0)).toFixed(1);
            perclosVal.innerText = `${pPct}%`;
        }
        if (hudEarMetric) {
            hudEarMetric.innerText = (vision.eye_aspect_ratio || smoothEar || 0.28).toFixed(2);
        }
        if (blinkVal) {
            const blinks = Math.round(vision.blinks_per_min || phys.blink_rate_bpm || 16);
            blinkVal.innerText = `${blinks} / min`;
        }
        if (yawnVal) {
            const yawns = vision.yawns_per_min !== undefined ? vision.yawns_per_min : (phys.yawns_per_min || 0);
            yawnVal.innerText = `${yawns}`;
        }
        if (painVal) {
            const au4 = (vision.action_units && vision.action_units.AU04_brow_furrow) || 0.0;
            painVal.innerText = `${Math.round(au4 * 100)}%`;
        }
        if (fatigueLevelText) {
            fatigueLevelText.innerText = phys.fatigue_level || "Nominal / Rested";
            fatigueLevelText.className = `text-xs font-semibold ${phys.fatigue_level === 'Severe' ? 'text-red-400' : (phys.fatigue_level === 'Moderate' ? 'text-amber-400' : 'text-emerald-400')}`;
        }
        const lightingTag = document.getElementById('lighting-status-tag');
        if (lightingTag && vision.lighting) {
            const lStatus = vision.lighting.status || 'OPTIMAL';
            const lColor = lStatus === 'OPTIMAL' ? 'text-emerald-400' : 'text-amber-400';
            lightingTag.innerHTML = `<span class="text-slate-400">Illumination:</span> <strong class="${lColor} font-semibold ml-1">${lStatus}</strong>`;
        }

        // Mathematical Breakdown Component Updates
        const components = (data.wellbeing && data.wellbeing.components) || {};
        if (compAffect) compAffect.innerText = `${(components.negative_affect_penalty !== undefined ? components.negative_affect_penalty : (components.negative_affect || 0)).toFixed(1)} pts`;
        if (compFatigue) compFatigue.innerText = `${(components.ocular_fatigue_score !== undefined ? components.ocular_fatigue_score : (components.ocular_fatigue || 0)).toFixed(1)} pts`;
        if (compTension) compTension.innerText = `${(components.autonomic_tension_score !== undefined ? components.autonomic_tension_score : (components.autonomic_tension || 0)).toFixed(1)} pts`;

        // Update HUD Header Astronaut Status
        if (hudAstronautWellbeing) {
            if (riskScore > 70) {
                hudAstronautWellbeing.innerText = "CRITICAL";
                hudAstronautWellbeing.className = "text-red-400 font-semibold";
            } else if (riskScore > 50) {
                hudAstronautWellbeing.innerText = "ELEVATED";
                hudAstronautWellbeing.className = "text-orange-400 font-semibold";
            } else if (riskScore > 30) {
                hudAstronautWellbeing.innerText = "MILD LOAD";
                hudAstronautWellbeing.className = "text-amber-400 font-semibold";
            } else {
                hudAstronautWellbeing.innerText = "NOMINAL";
                hudAstronautWellbeing.className = "text-emerald-400 font-semibold";
            }
        }
        if (hudAstronautRisk) {
            hudAstronautRisk.innerText = `${riskScore.toFixed(1)} / 100`;
        }

        // Voice & Acoustic Vitals
        if (!isVoiceActive) {
            if (pitchVal) pitchVal.innerText = "0 Hz (Muted)";
            if (vocalTensionVal) vocalTensionVal.innerText = "0%";
            if (serPitchDisplay) serPitchDisplay.innerText = "0 Hz (Muted)";
            if (serTensionDisplay) serTensionDisplay.innerText = "0%";
        } else if (data.ser && !mediaStream && !micAudioStream) {
            if (pitchVal && data.ser.pitch_hz) pitchVal.innerText = `${Math.round(data.ser.pitch_hz)} Hz`;
            if (vocalTensionVal && data.ser.vocal_tension !== undefined) vocalTensionVal.innerText = `${Math.round(data.ser.vocal_tension * 100)}%`;
            if (serPitchDisplay && data.ser.pitch_hz) serPitchDisplay.innerText = `${Math.round(data.ser.pitch_hz)} Hz`;
            if (serTensionDisplay && data.ser.vocal_tension !== undefined) serTensionDisplay.innerText = `${Math.round(data.ser.vocal_tension * 100)}%`;
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

        // Dynamic Rolling Emotion Timeline with Authentic Score (Throttled unless forced)
        updateValenceTimeline(domEmotion, fusion.valence, isForced);

        // Ground Alerts
        if (data.alert_dispatched) {
            addGroundAlertItem(data.alert_dispatched);
        }
    }

    function addGroundAlertItem(alert) {
        if (!alertFeedList) return;
        const item = document.createElement('div');
        const isCritical = alert.risk_level >= 3;
        item.className = `p-3 bg-dark-850 border-l-4 ${isCritical ? 'border-red-500' : 'border-amber-500'} rounded-xl border border-slate-800 flex flex-col gap-1 text-xs`;
        item.innerHTML = `
            <div class="flex justify-between items-center font-bold">
                <span class="text-slate-100">${alert.alert_id}</span>
                <span class="text-[10px] px-2 py-0.5 rounded ${isCritical ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'} font-semibold">
                    ${isCritical ? 'CRITICAL' : 'ALERT'}
                </span>
            </div>
            <p class="text-[11px] text-slate-300 leading-relaxed">${alert.reason || 'Telemetry threshold exceeded.'}</p>
            <span class="text-[10px] text-slate-500 font-mono">Dispatched to Flight Surgeon</span>
        `;
        alertFeedList.prepend(item);
    }

    // ---------------------------------------------------------
    // 8. Conversational Companion AI Interaction
    // ---------------------------------------------------------
    async function sendChatMessage(text) {
        if (!text || text.trim().length === 0) return;
        appendChatBubble('user', text);
        if (chatInput) chatInput.value = '';

        if (chatTypingIndicator) chatTypingIndicator.classList.remove('hidden');

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
            if (chatTypingIndicator) chatTypingIndicator.classList.add('hidden');
            appendChatBubble('ai', data.ai_response);

            // Voice synthesis
            speakWithBrowserTTS(data.ai_response);

            if (data.intervention && data.intervention.id === 'INT-BREATHE-01') {
                startBreathingPacer();
            }
        } catch (e) {
            if (chatTypingIndicator) chatTypingIndicator.classList.add('hidden');
            console.log("Chat error:", e);
            appendChatBubble('ai', "MAITRI telemetry sync nominal. Offline autonomous companion standing by.");
        }
    }

    function appendChatBubble(speaker, text) {
        if (!chatContainer) return;
        const isAi = speaker === 'ai';
        const bubble = document.createElement('div');
        bubble.className = `${isAi ? 'bg-dark-850 border-slate-800' : 'bg-indigo-950/40 border-indigo-500/30'} border rounded-xl p-3 ${isAi ? 'self-end' : 'self-start'} w-11/12 text-xs leading-relaxed`;
        bubble.innerHTML = `
            <div class="flex justify-between items-center mb-1 text-xs ${isAi ? 'text-indigo-300 font-bold' : 'text-emerald-400 font-semibold'}">
                <span>${isAi ? 'MAITRI AI Companion' : 'Astronaut (Spoken)'}</span>
                <span class="text-slate-500 font-normal text-[10px]">Now</span>
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
                updateDashboardTelemetry(telemetry, true);

                if (telemetry.transcript) {
                    appendChatBubble('user', telemetry.transcript);
                    if (chatTypingIndicator) chatTypingIndicator.classList.remove('hidden');
                    const chatResp = await fetch('/api/interact', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: telemetry.transcript,
                            astronaut_id: crewSelector ? crewSelector.value : 'CREW-BAS-01'
                        })
                    });
                    const chatData = await chatResp.json();
                    if (chatTypingIndicator) chatTypingIndicator.classList.add('hidden');
                    appendChatBubble('ai', chatData.ai_response);
                    speakWithBrowserTTS(chatData.ai_response);
                }
            } catch (e) {
                if (chatTypingIndicator) chatTypingIndicator.classList.add('hidden');
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
                setTimeout(initWebSocket, 3000);
            };
        } catch (e) {
            console.warn('[MAITRI WS] Init exception:', e);
        }
    }

    // ---------------------------------------------------------
    // 13. Dynamic Emotion Valence Waveform (Accurate SVG)
    // ---------------------------------------------------------
    let lastTimelineShiftTime = 0;

    function updateValenceTimeline(domEmotion, valenceScore, isForced = false) {
        if (!timelineValencePath || !timelineActiveNode) return;
        
        let y = 58;
        if (valenceScore !== undefined && valenceScore !== null) {
            // Map valence (-1.0 to +1.0) into Y coordinates [15 to 105]
            // +1.0 -> y=20 (Top Green Zone)
            //  0.0 -> y=58 (Middle Neutral Zone)
            // -1.0 -> y=98 (Bottom Red Zone)
            y = Math.min(105, Math.max(15, Math.round(58 - (valenceScore * 40))));
        } else {
            if (domEmotion === 'happy') y = 25;
            else if (domEmotion === 'calm' || domEmotion === 'neutral') y = 55;
            else if (domEmotion === 'fatigued') y = 78;
            else if (domEmotion === 'sad' || domEmotion === 'isolated') y = 84;
            else if (domEmotion === 'stressed' || domEmotion === 'frustrated') y = 96;
        }

        const now = Date.now();
        // Only scroll/shift timeline if it is an explicit event (scenario/chat) OR 5+ seconds have elapsed
        if (isForced || (now - lastTimelineShiftTime > 5000)) {
            valenceBuffer.push(y);
            if (valenceBuffer.length > 20) valenceBuffer.shift();
            lastTimelineShiftTime = now;

            const step = 500 / (valenceBuffer.length - 1);
            let d = `M 0,${valenceBuffer[0]}`;
            for (let i = 1; i < valenceBuffer.length; i++) {
                const x = Math.round(i * step);
                const prevX = Math.round((i - 1) * step);
                const midX = Math.round((prevX + x) / 2);
                d += ` C ${midX},${valenceBuffer[i-1]} ${midX},${valenceBuffer[i]} ${x},${valenceBuffer[i]}`;
            }
            timelineValencePath.setAttribute('d', d);
        }

        timelineActiveNode.setAttribute('cx', '500');
        timelineActiveNode.setAttribute('cy', y);
        timelineActiveNode.setAttribute('fill', y > 75 ? '#F87171' : (y > 45 ? '#94A3B8' : '#34D399'));
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
                    btnEl.className = 'px-3 py-1 text-xs font-semibold rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 cursor-default';
                    btnEl.innerText = 'Acknowledged';
                }
                const statusEl = document.getElementById(`status-${alertId}`);
                if (statusEl) {
                    statusEl.className = 'py-3 text-emerald-400 font-semibold';
                    statusEl.innerText = 'Acknowledged';
                }
            }
        } catch (e) {
            console.error('Ack alert error:', e);
        }
    };

    // ---------------------------------------------------------
    // 15. Live Session History Loader (Astronaut Isolated)
    // ---------------------------------------------------------
    window.loadSessionHistory = async function() {
        const crewSel = document.getElementById('crew-selector');
        const astId = crewSel ? crewSel.value : 'CREW-BAS-01';
        try {
            const resp = await fetch(`/api/history/telemetry?astronaut_id=${astId}`);
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

                    item.className = `p-3 bg-dark-850 border-l-4 ${isHigh ? 'border-amber-500' : 'border-emerald-500'} rounded-xl border border-slate-800 flex justify-between items-center text-xs`;
                    item.innerHTML = `
                        <div>
                            <div class="flex items-center gap-2">
                                <strong class="text-white">Crew: ${record.astronaut_id || astId} — ${dom.toUpperCase()}</strong>
                                <span class="text-[10px] font-semibold px-2 py-0.5 rounded-full ${isHigh ? 'bg-amber-500/10 text-amber-400' : 'bg-emerald-500/10 text-emerald-400'}">Risk: ${risk}</span>
                            </div>
                            <p class="text-[11px] text-slate-400 mt-0.5">PERCLOS: ${record.perclos ? record.perclos.toFixed(1) + '%' : '4.0%'} · Pitch F0: ${record.pitch_f0 ? record.pitch_f0.toFixed(0) + ' Hz' : '130 Hz'} · Vocal Tension: ${record.vocal_tension ? record.vocal_tension.toFixed(0) + '%' : '12%'}</p>
                        </div>
                        <span class="text-xs font-mono text-slate-400">${timeStr}</span>
                    `;
                    container.appendChild(item);
                });
            } else {
                container.innerHTML = `<div class="p-6 text-center text-slate-400 text-xs">No prior session records logged for ${astId}. Real-time monitoring active.</div>`;
            }
        } catch (e) {
            console.error('Session history load error:', e);
        }
    };

    // ---------------------------------------------------------
    // 16. Admin / Flight Surgeon Console Loader
    // ---------------------------------------------------------
    window.loadAdminConsole = async function() {
        try {
            const resp = await fetch('/api/admin/crew-summary');
            if (resp.status === 403) {
                console.warn("User does not have Admin clearance. Prompting role toggle.");
                return;
            }
            const data = await resp.json();
            const triageContainer = document.getElementById('admin-triage-list');
            if (!triageContainer) return;

            const alerts = data.recent_alerts || [];
            if (alerts.length > 0) {
                triageContainer.innerHTML = '';
                alerts.forEach(alt => {
                    const isCrit = alt.risk_level >= 3;
                    const item = document.createElement('div');
                    item.className = `p-3 bg-dark-850 border-l-4 ${isCrit ? 'border-red-500' : 'border-amber-500'} rounded-xl border border-slate-800 flex justify-between items-center text-xs`;
                    item.innerHTML = `
                        <div>
                            <div class="flex items-center gap-2">
                                <strong class="text-white">${alt.alert_id}</strong>
                                <span class="px-2 py-0.5 rounded ${isCrit ? 'bg-red-500/10 text-red-400' : 'bg-amber-500/10 text-amber-400'} font-semibold">
                                    Level ${alt.risk_level} · Risk ${alt.risk_score ? alt.risk_score.toFixed(1) : '55.0'}
                                </span>
                            </div>
                            <p class="text-slate-400 mt-0.5">Primary Affect: ${alt.dominant_emotion || 'Distress'} · Crew: ${alt.astronaut_id || 'CREW-BAS-01'}</p>
                        </div>
                        <button onclick="acknowledgeAlert('${alt.alert_id}', this)" class="px-3 py-1 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold">
                            Acknowledge
                        </button>
                    `;
                    triageContainer.appendChild(item);
                });
            }
        } catch(e) {
            console.error("Admin console load error:", e);
        }
    };

    // ---------------------------------------------------------
    // 13. Biometric Face Recognition & Identity Engine (Upgrade)
    // ---------------------------------------------------------
    window.currentAstronautId = "AST-001";
    let enrollCapturedFrame = null;

    window.dismissFaceIdBanner = function() {
        const banner = document.getElementById('face-id-status-banner');
        if (banner) banner.classList.add('hidden');
    };

    window.openEnrollModal = function() {
        const modal = document.getElementById('enroll-astronaut-modal');
        if (modal) modal.classList.remove('hidden');
    };

    window.closeEnrollModal = function() {
        const modal = document.getElementById('enroll-astronaut-modal');
        if (modal) modal.classList.add('hidden');
    };

    window.openManualLoginModal = function() {
        const modal = document.getElementById('manual-login-modal');
        if (modal) modal.classList.remove('hidden');
    };

    window.closeManualLoginModal = function() {
        const modal = document.getElementById('manual-login-modal');
        if (modal) modal.classList.add('hidden');
    };

    window.captureEnrollmentSnapshot = function() {
        if (!videoElem || !isCameraActive || videoElem.videoWidth === 0) {
            alert("Please start the webcam camera first to capture a live face sample.");
            return;
        }
        const offscreen = document.createElement('canvas');
        offscreen.width = 320;
        offscreen.height = 240;
        const ctx = offscreen.getContext('2d');
        ctx.drawImage(videoElem, 0, 0, 320, 240);
        enrollCapturedFrame = offscreen.toDataURL('image/jpeg', 0.85);

        const statusLabel = document.getElementById('enroll-face-status');
        if (statusLabel) {
            statusLabel.innerHTML = `<span class="text-emerald-400 font-bold">✓ Biometric frame captured (${offscreen.width}x${offscreen.height})</span>`;
        }
    };

    window.handleEnrollSubmit = async function(e) {
        e.preventDefault();
        const feedback = document.getElementById('enroll-feedback');
        feedback.className = 'p-2.5 rounded-xl text-center bg-indigo-500/10 text-indigo-300 border border-indigo-500/30';
        feedback.innerText = "Extracting 128-D biometric signature & enrolling into database...";
        feedback.classList.remove('hidden');

        const payload = {
            astronaut_id: document.getElementById('enroll-id').value.trim(),
            name: document.getElementById('enroll-name').value.trim(),
            callsign: document.getElementById('enroll-callsign').value.trim(),
            role: document.getElementById('enroll-role').value.trim(),
            username: document.getElementById('enroll-username').value.trim(),
            password: document.getElementById('enroll-password').value.trim(),
            frame: enrollCapturedFrame
        };

        try {
            const resp = await fetch('/api/auth/enroll', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();
            if (resp.ok && data.status === 'SUCCESS') {
                feedback.className = 'p-2.5 rounded-xl text-center bg-emerald-500/10 text-emerald-300 border border-emerald-500/30';
                feedback.innerText = `✓ ${data.message}`;
                await window.loadEnrolledAstronauts();
                await window.selectCrewById(payload.astronaut_id);
                setTimeout(() => {
                    window.closeEnrollModal();
                    feedback.classList.add('hidden');
                }, 1400);
            } else {
                feedback.className = 'p-2.5 rounded-xl text-center bg-red-500/10 text-red-300 border border-red-500/30';
                feedback.innerText = `Enrollment failed: ${data.detail?.error || "Unknown error"}`;
            }
        } catch (err) {
            feedback.className = 'p-2.5 rounded-xl text-center bg-red-500/10 text-red-300 border border-red-500/30';
            feedback.innerText = `Network error during enrollment: ${err.message}`;
        }
    };

    window.handleManualLoginSubmit = async function(e) {
        e.preventDefault();
        const feedback = document.getElementById('login-feedback');
        feedback.className = 'p-2.5 rounded-xl text-center bg-indigo-500/10 text-indigo-300 border border-indigo-500/30';
        feedback.innerText = "Authenticating credentials against flight database...";
        feedback.classList.remove('hidden');

        const username = document.getElementById('login-username').value.trim();
        const password = document.getElementById('login-password').value.trim();

        try {
            const resp = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username_or_id: username, password: password })
            });
            const data = await resp.json();
            if (resp.ok && data.status === 'SUCCESS') {
                feedback.className = 'p-2.5 rounded-xl text-center bg-emerald-500/10 text-emerald-300 border border-emerald-500/30';
                feedback.innerText = `✓ Welcome, ${data.user.name}. Opening mission session.`;
                await window.selectCrewById(data.user.user_id);
                setTimeout(() => {
                    window.closeManualLoginModal();
                    feedback.classList.add('hidden');
                }, 1000);
            } else {
                feedback.className = 'p-2.5 rounded-xl text-center bg-red-500/10 text-red-300 border border-red-500/30';
                feedback.innerText = `Login failed: ${data.detail?.error || "Invalid credentials."}`;
            }
        } catch (err) {
            feedback.className = 'p-2.5 rounded-xl text-center bg-red-500/10 text-red-300 border border-red-500/30';
            feedback.innerText = `Network error: ${err.message}`;
        }
    };

    window.syncLoginUsername = function(selectedId) {
        const usernameInput = document.getElementById('login-username');
        if (usernameInput && selectedId) {
            usernameInput.value = selectedId;
        }
    };

    window.startFaceRecognitionScan = async function() {
        const banner = document.getElementById('face-id-status-banner');
        const icon = document.getElementById('face-id-status-icon');
        const title = document.getElementById('face-id-status-title');
        const msg = document.getElementById('face-id-status-msg');

        if (banner) {
            banner.className = 'web-card p-3 rounded-xl flex items-center justify-between gap-3 text-xs border border-indigo-500/40 bg-indigo-950/30';
            banner.classList.remove('hidden');
            if (icon) icon.innerText = 'sync';
            if (title) title.innerText = 'Scanning Facial Biometrics';
            if (msg) msg.innerText = 'Capturing camera frame and evaluating 128-D spatial LBP embedding...';
        }

        // Ensure camera is active
        if (!isCameraActive || !videoElem || videoElem.videoWidth === 0) {
            if (btnToggleCamera) {
                btnToggleCamera.click();
                await new Promise(r => setTimeout(r, 1200));
            }
        }

        // Grab current frame
        const offscreen = document.createElement('canvas');
        offscreen.width = 320;
        offscreen.height = 240;
        const ctx = offscreen.getContext('2d');
        if (videoElem && videoElem.videoWidth > 0) {
            ctx.drawImage(videoElem, 0, 0, 320, 240);
        } else {
            // Draw synthetic test face if physical camera hardware not attached
            ctx.fillStyle = "#1e293b";
            ctx.fillRect(0, 0, 320, 240);
            ctx.fillStyle = "#f5d0b5";
            ctx.beginPath();
            ctx.ellipse(160, 120, 50, 65, 0, 0, Math.PI * 2);
            ctx.fill();
        }

        const base64Frame = offscreen.toDataURL('image/jpeg', 0.85);

        try {
            const resp = await fetch('/api/auth/recognize-face', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame: base64Frame })
            });
            const data = await resp.json();

            if (!banner) return;

            if (data.status === 'IDENTIFIED') {
                banner.className = 'web-card p-3 rounded-xl flex items-center justify-between gap-3 text-xs border border-emerald-500/40 bg-emerald-950/30';
                if (icon) icon.innerText = 'verified_user';
                if (title) title.innerText = `IDENTIFIED: ${data.name} (${data.astronaut_id}) · ${(data.confidence * 100).toFixed(0)}% Confidence`;
                if (msg) msg.innerText = `Welcome, ${data.name}. Personal baseline and isolated mission session activated.`;

                window.currentAstronautId = data.astronaut_id;
                const selector = document.getElementById('crew-selector');
                if (selector) selector.value = data.astronaut_id;

                await window.loadActiveAstronautProfile();
                window.loadSessionHistory();
            } else if (data.status === 'LOW_CONFIDENCE') {
                banner.className = 'web-card p-3 rounded-xl flex items-center justify-between gap-3 text-xs border border-amber-500/40 bg-amber-950/30';
                if (icon) icon.innerText = 'warning';
                if (title) title.innerText = `IDENTITY UNCERTAIN (${((data.confidence || 0.65) * 100).toFixed(0)}%)`;
                if (msg) msg.innerText = data.message || 'Optical conditions degraded — please adjust lighting or use manual login.';
            } else if (data.status === 'UNKNOWN') {
                banner.className = 'web-card p-3 rounded-xl flex items-center justify-between gap-3 text-xs border border-amber-500/40 bg-amber-950/30';
                if (icon) icon.innerText = 'person_search';
                if (title) title.innerText = 'UNKNOWN ASTRONAUT';
                if (msg) msg.innerText = 'No matching biometric profile found. Please use manual login or enroll first.';
            } else if (data.status === 'MULTIPLE_FACES') {
                banner.className = 'web-card p-3 rounded-xl flex items-center justify-between gap-3 text-xs border border-red-500/40 bg-red-950/30';
                if (icon) icon.innerText = 'group';
                if (title) title.innerText = 'MULTIPLE ASTRONAUTS DETECTED';
                if (msg) msg.innerText = 'Multiple faces in viewport — biometric confirmation aborted for safety.';
            } else {
                banner.className = 'web-card p-3 rounded-xl flex items-center justify-between gap-3 text-xs border border-slate-700 bg-dark-900';
                if (icon) icon.innerText = 'visibility_off';
                if (title) title.innerText = 'NO ASTRONAUT DETECTED';
                if (msg) msg.innerText = 'Please center your face in the camera viewport and try again.';
            }
        } catch (err) {
            console.error("Face recognition error:", err);
            if (banner) {
                banner.className = 'web-card p-3 rounded-xl flex items-center justify-between gap-3 text-xs border border-red-500/40 bg-red-950/30';
                if (icon) icon.innerText = 'error';
                if (title) title.innerText = 'Biometric Scan Error';
                if (msg) msg.innerText = 'Could not reach identity recognition service.';
            }
        }
    };

    window.selectCrewById = async function(astronautId) {
        try {
            const resp = await fetch('/api/crew/select', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ astronaut_id: astronautId })
            });
            const data = await resp.json();
            window.currentAstronautId = astronautId;
            const selector = document.getElementById('crew-selector');
            if (selector) selector.value = astronautId;
            await window.loadActiveAstronautProfile();
            window.loadSessionHistory();
        } catch (err) {
            console.error("Error switching crew profile:", err);
        }
    };

    window.loadEnrolledAstronauts = async function() {
        try {
            const resp = await fetch('/api/auth/astronauts');
            const data = await resp.json();
            if (data.status === 'SUCCESS' && Array.isArray(data.astronauts)) {
                const crewSelector = document.getElementById('crew-selector');
                const loginSelect = document.getElementById('login-crew-select');

                if (crewSelector) {
                    crewSelector.innerHTML = '';
                    data.astronauts.forEach(a => {
                        const opt = document.createElement('option');
                        opt.value = a.astronaut_id;
                        opt.className = 'bg-dark-900 text-slate-100';
                        opt.innerText = `${a.callsign || a.astronaut_id} (${a.name})`;
                        if (a.astronaut_id === window.currentAstronautId) opt.selected = true;
                        crewSelector.appendChild(opt);
                    });
                }

                if (loginSelect) {
                    loginSelect.innerHTML = '<option value="">-- Choose Enrolled Astronaut --</option>';
                    data.astronauts.forEach(a => {
                        const opt = document.createElement('option');
                        opt.value = a.astronaut_id;
                        opt.innerText = `${a.astronaut_id} — ${a.name} (${a.role})`;
                        loginSelect.appendChild(opt);
                    });
                }
            }
        } catch (e) {
            console.warn("Could not load enrolled astronauts from DB:", e);
        }
    };

    window.loadActiveAstronautProfile = async function() {
        try {
            const resp = await fetch(`/api/astronaut/profile?astronaut_id=${window.currentAstronautId}`);
            const data = await resp.json();
            if (data && data.profile) {
                const p = data.profile;
                const hudName = document.getElementById('hud-astronaut-name');
                const hudId = document.getElementById('hud-astronaut-id-badge');
                const hudRole = document.getElementById('hud-astronaut-role-badge');
                const hudCallsign = document.getElementById('hud-astronaut-callsign');
                const hudSession = document.getElementById('hud-astronaut-session');

                if (hudName) hudName.innerText = p.name || data.astronaut_id;
                if (hudId) hudId.innerText = data.astronaut_id;
                if (hudRole) hudRole.innerText = p.role || 'Mission Specialist';
                if (hudCallsign) hudCallsign.innerText = p.callsign || 'Flight-Crew';
                if (hudSession && data.active_session_id) hudSession.innerText = data.active_session_id;

                // Update Profile Tab Active Section if present
                const profName = document.getElementById('profile-current-name');
                const profId = document.getElementById('profile-current-id');
                const profRole = document.getElementById('profile-current-role');
                const profSessions = document.getElementById('profile-current-sessions');
                const profAlerts = document.getElementById('profile-current-alerts');
                if (profName) profName.innerText = p.name;
                if (profId) profId.innerText = data.astronaut_id;
                if (profRole) profRole.innerText = p.role;
                if (profSessions) profSessions.innerText = data.recent_sessions_count || 0;
                if (profAlerts) profAlerts.innerText = data.recent_alerts_count || 0;
            }
        } catch (e) {
            console.error("Error loading active astronaut profile:", e);
        }
    };

    // Crew Selector Change Event (Strict Astronaut Switching)
    if (crewSelector) {
        crewSelector.addEventListener('change', async (e) => {
            await window.selectCrewById(e.target.value);
        });
    }

    // Crew Session Logout
    window.logoutCrewSession = async function() {
        try {
            await fetch('/api/auth/logout', { method: 'POST' });
        } catch(e) {
            console.warn("Logout error:", e);
        }
        window.location.href = '/login';
    };

    // Initialize Real-Time WebSocket, Sessions, Audio Waveform & Enrolled Astronauts
    initWebSocket();
    window.loadEnrolledAstronauts();
    window.loadActiveAstronautProfile();
    window.loadSessionHistory();
    renderAudioWaveform();
    setSensorStandbyState(false);
});
