/**
 * MAITRI — Futuristic Translucent 3D Holographic Torus Background
 * Pure WebGL Hardware-Accelerated Shader Implementation (Zero external dependencies)
 * ISRO BAS Aesthetic: Cyan / Indigo / Electric Violet Neon Glass Grid
 */

(function() {
    'use strict';

    const canvas = document.getElementById('bg-torus-canvas');
    if (!canvas) return;

    // Check prefers-reduced-motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Initialize WebGL context with transparency
    const gl = canvas.getContext('webgl', {
        alpha: true,
        antialias: true,
        depth: true,
        premultipliedAlpha: false
    }) || canvas.getContext('experimental-webgl', {
        alpha: true,
        antialias: true,
        depth: true,
        premultipliedAlpha: false
    });

    if (!gl) {
        console.warn('[MAITRI] WebGL not available for 3D Torus background.');
        return;
    }

    // Enable extension for derivatives (fwidth in fragment shader)
    const ext = gl.getExtension('OES_standard_derivatives');
    const hasDerivatives = !!ext;

    // -------------------------------------------------------------
    // Shaders
    // -------------------------------------------------------------
    const vsSource = `
        attribute vec3 aPosition;
        attribute vec3 aNormal;
        attribute vec2 aUV;

        uniform mat4 uProjection;
        uniform mat4 uView;
        uniform mat4 uModel;

        varying vec3 vNormal;
        varying vec3 vWorldPos;
        varying vec2 vUV;

        void main() {
            vUV = aUV;
            vec4 worldPos = uModel * vec4(aPosition, 1.0);
            vWorldPos = worldPos.xyz;
            vNormal = normalize(mat3(uModel) * aNormal);
            gl_Position = uProjection * uView * worldPos;
        }
    `;

    const fsSource = `
        #ifdef GL_OES_standard_derivatives
        #extension GL_OES_standard_derivatives : enable
        #endif

        precision mediump float;

        varying vec3 vNormal;
        varying vec3 vWorldPos;
        varying vec2 vUV;

        uniform vec3 uCameraPos;
        uniform float uTime;
        uniform float uHasDeriv;

        void main() {
            vec3 N = normalize(vNormal);
            vec3 V = normalize(uCameraPos - vWorldPos);

            // Double-sided lighting for translucent glass
            float NdotV = abs(dot(N, V));

            // 1. Holographic Fresnel rim glow
            float fresnel = pow(1.0 - NdotV, 2.0);

            // 2. Futuristic cybernetic wireframe / coordinate grid
            vec2 gridUV = vUV * vec2(48.0, 24.0);
            float gridFactor = 0.0;
            
            #ifdef GL_OES_standard_derivatives
            if (uHasDeriv > 0.5) {
                vec2 gridDist = abs(fract(gridUV - 0.5) - 0.5);
                vec2 gridDelta = fwidth(gridUV);
                vec2 gridLine = smoothstep(gridDelta * 1.5, vec2(0.0), gridDist);
                gridFactor = max(gridLine.x, gridLine.y);
            } else {
                vec2 gridDist = abs(fract(gridUV) - 0.5);
                gridFactor = step(0.42, max(gridDist.x, gridDist.y));
            }
            #else
            vec2 gridDist = abs(fract(gridUV) - 0.5);
            gridFactor = step(0.42, max(gridDist.x, gridDist.y));
            #endif

            // 3. Subtle traveling pulse wave
            float scanWave = sin(vUV.x * 6.28318 * 3.0 - uTime * 1.2) * 0.5 + 0.5;
            scanWave = pow(scanWave, 4.0) * 0.35;

            // 4. Color Palette (MAITRI Cyan / Indigo / Neon Violet)
            vec3 colDeepIndigo   = vec3(0.22, 0.16, 0.72); // #3829B8
            vec3 colElectricViolet = vec3(0.58, 0.20, 0.95); // #9433F2
            vec3 colNeonCyan     = vec3(0.04, 0.72, 0.88); // #0AB8E0
            vec3 colBrightCyan   = vec3(0.38, 0.92, 1.00); // #61EAFF

            float colorShift = sin(vWorldPos.y * 0.35 + vUV.x * 3.14 + uTime * 0.25) * 0.5 + 0.5;
            vec3 baseColor = mix(colDeepIndigo, colElectricViolet, colorShift);
            vec3 glowColor = mix(colElectricViolet, colNeonCyan, fresnel);

            vec3 finalColor = baseColor * 0.5 + glowColor * (fresnel * 1.3 + scanWave) + colBrightCyan * (gridFactor * 0.8);

            // Subtle translucent opacity — stays behind all UI without obscuring
            float baseAlpha = 0.08;
            float fresnelAlpha = fresnel * 0.28;
            float gridAlpha = gridFactor * 0.18;
            float waveAlpha = scanWave * 0.10;

            float totalAlpha = clamp(baseAlpha + fresnelAlpha + gridAlpha + waveAlpha, 0.0, 0.60);

            gl_FragColor = vec4(finalColor * totalAlpha, totalAlpha);
        }
    `;

    function createShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.warn('[MAITRI Shader Error]', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vsSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
    if (!vertexShader || !fragmentShader) return;

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.warn('[MAITRI Program Error]', gl.getProgramInfoLog(program));
        return;
    }

    const aPosLoc = gl.getAttribLocation(program, 'aPosition');
    const aNormLoc = gl.getAttribLocation(program, 'aNormal');
    const aUvLoc = gl.getAttribLocation(program, 'aUV');

    const uProjLoc = gl.getUniformLocation(program, 'uProjection');
    const uViewLoc = gl.getUniformLocation(program, 'uView');
    const uModelLoc = gl.getUniformLocation(program, 'uModel');
    const uCamPosLoc = gl.getUniformLocation(program, 'uCameraPos');
    const uTimeLoc = gl.getUniformLocation(program, 'uTime');
    const uHasDerivLoc = gl.getUniformLocation(program, 'uHasDeriv');

    // -------------------------------------------------------------
    // Torus Geometry Generation
    // -------------------------------------------------------------
    function generateTorus(majorR, minorR, radialSegs, tubularSegs) {
        const positions = [];
        const normals = [];
        const uvs = [];
        const indices = [];

        for (let j = 0; j <= radialSegs; j++) {
            const v = (j / radialSegs) * Math.PI * 2;
            const cosV = Math.cos(v);
            const sinV = Math.sin(v);

            for (let i = 0; i <= tubularSegs; i++) {
                const u = (i / tubularSegs) * Math.PI * 2;
                const cosU = Math.cos(u);
                const sinU = Math.sin(u);

                const x = (majorR + minorR * cosV) * cosU;
                const y = (majorR + minorR * cosV) * sinU;
                const z = minorR * sinV;

                positions.push(x, y, z);

                // Normal vector
                normals.push(cosV * cosU, cosV * sinU, sinV);

                uvs.push(i / tubularSegs, j / radialSegs);
            }
        }

        for (let j = 1; j <= radialSegs; j++) {
            for (let i = 1; i <= tubularSegs; i++) {
                const a = (tubularSegs + 1) * j + i - 1;
                const b = (tubularSegs + 1) * (j - 1) + i - 1;
                const c = (tubularSegs + 1) * (j - 1) + i;
                const d = (tubularSegs + 1) * j + i;

                indices.push(a, b, d);
                indices.push(b, c, d);
            }
        }

        return {
            positions: new Float32Array(positions),
            normals: new Float32Array(normals),
            uvs: new Float32Array(uvs),
            indices: new Uint16Array(indices),
            indexCount: indices.length
        };
    }

    // Main Large Torus: majorR = 6.2, minorR = 1.9
    const torusData = generateTorus(6.2, 1.9, 32, 64);

    // Buffers
    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, torusData.positions, gl.STATIC_DRAW);

    const normBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, torusData.normals, gl.STATIC_DRAW);

    const uvBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, torusData.uvs, gl.STATIC_DRAW);

    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, torusData.indices, gl.STATIC_DRAW);

    // -------------------------------------------------------------
    // Thin Satellite Coordinate Ring
    // -------------------------------------------------------------
    const outerRingData = generateTorus(8.8, 0.08, 16, 64);
    const posBufferRing = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBufferRing);
    gl.bufferData(gl.ARRAY_BUFFER, outerRingData.positions, gl.STATIC_DRAW);

    const normBufferRing = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normBufferRing);
    gl.bufferData(gl.ARRAY_BUFFER, outerRingData.normals, gl.STATIC_DRAW);

    const uvBufferRing = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBufferRing);
    gl.bufferData(gl.ARRAY_BUFFER, outerRingData.uvs, gl.STATIC_DRAW);

    const indexBufferRing = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBufferRing);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, outerRingData.indices, gl.STATIC_DRAW);

    // -------------------------------------------------------------
    // Matrix Math Helper Functions
    // -------------------------------------------------------------
    function mat4Create() {
        return new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]);
    }

    function mat4Perspective(out, fovy, aspect, near, far) {
        const f = 1.0 / Math.tan(fovy / 2);
        out[0] = f / aspect;
        out[1] = 0;
        out[2] = 0;
        out[3] = 0;
        out[4] = 0;
        out[5] = f;
        out[6] = 0;
        out[7] = 0;
        out[8] = 0;
        out[9] = 0;
        out[10] = (far + near) / (near - far);
        out[11] = -1;
        out[12] = 0;
        out[13] = 0;
        out[14] = (2 * far * near) / (near - far);
        out[15] = 0;
        return out;
    }

    function mat4LookAt(out, eye, center, up) {
        let x0, x1, x2, y0, y1, y2, z0, z1, z2, len;
        let eyex = eye[0], eyey = eye[1], eyez = eye[2];
        let upx = up[0], upy = up[1], upz = up[2];
        let centerx = center[0], centery = center[1], centerz = center[2];

        z0 = eyex - centerx; z1 = eyey - centery; z2 = eyez - centerz;
        len = 1 / Math.hypot(z0, z1, z2);
        z0 *= len; z1 *= len; z2 *= len;

        x0 = upy * z2 - upz * z1;
        x1 = upz * z0 - upx * z2;
        x2 = upx * z1 - upy * z0;
        len = 1 / Math.hypot(x0, x1, x2);
        x0 *= len; x1 *= len; x2 *= len;

        y0 = z1 * x2 - z2 * x1;
        y1 = z2 * x0 - z0 * x2;
        y2 = z0 * x1 - z1 * x0;

        out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
        out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
        out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
        out[12] = -(x0 * eyex + x1 * eyey + x2 * eyez);
        out[13] = -(y0 * eyex + y1 * eyey + y2 * eyez);
        out[14] = -(z0 * eyex + z1 * eyey + z2 * eyez);
        out[15] = 1;
        return out;
    }

    function mat4Identity(out) {
        for (let i = 0; i < 16; i++) out[i] = (i % 5 === 0) ? 1 : 0;
        return out;
    }

    function mat4RotateX(out, a, rad) {
        const s = Math.sin(rad), c = Math.cos(rad);
        const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
        const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
        if (a !== out) {
            out[0] = a[0]; out[1] = a[1]; out[2] = a[2]; out[3] = a[3];
            out[12] = a[12]; out[13] = a[13]; out[14] = a[14]; out[15] = a[15];
        }
        out[4] = a10 * c + a20 * s;
        out[5] = a11 * c + a21 * s;
        out[6] = a12 * c + a22 * s;
        out[7] = a13 * c + a23 * s;
        out[8] = a20 * c - a10 * s;
        out[9] = a21 * c - a11 * s;
        out[10] = a22 * c - a12 * s;
        out[11] = a23 * c - a13 * s;
        return out;
    }

    function mat4RotateY(out, a, rad) {
        const s = Math.sin(rad), c = Math.cos(rad);
        const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
        const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
        if (a !== out) {
            out[4] = a[4]; out[5] = a[5]; out[6] = a[6]; out[7] = a[7];
            out[12] = a[12]; out[13] = a[13]; out[14] = a[14]; out[15] = a[15];
        }
        out[0] = a00 * c - a20 * s;
        out[1] = a01 * c - a21 * s;
        out[2] = a02 * c - a22 * s;
        out[3] = a03 * c - a23 * s;
        out[8] = a00 * s + a20 * c;
        out[9] = a01 * s + a21 * c;
        out[10] = a02 * s + a22 * c;
        out[11] = a03 * s + a23 * c;
        return out;
    }

    function mat4RotateZ(out, a, rad) {
        const s = Math.sin(rad), c = Math.cos(rad);
        const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
        const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
        if (a !== out) {
            out[8] = a[8]; out[9] = a[9]; out[10] = a[10]; out[11] = a[11];
            out[12] = a[12]; out[13] = a[13]; out[14] = a[14]; out[15] = a[15];
        }
        out[0] = a00 * c + a10 * s;
        out[1] = a01 * c + a11 * s;
        out[2] = a02 * c + a12 * s;
        out[3] = a03 * c + a13 * s;
        out[4] = a10 * c - a00 * s;
        out[5] = a11 * c - a01 * s;
        out[6] = a12 * c - a02 * s;
        out[7] = a13 * c - a03 * s;
        return out;
    }

    function mat4Scale(out, a, v) {
        const x = v[0], y = v[1], z = v[2];
        out[0] = a[0] * x; out[1] = a[1] * x; out[2] = a[2] * x; out[3] = a[3] * x;
        out[4] = a[4] * y; out[5] = a[5] * y; out[6] = a[6] * y; out[7] = a[7] * y;
        out[8] = a[8] * z; out[9] = a[9] * z; out[10] = a[10] * z; out[11] = a[11] * z;
        out[12] = a[12]; out[13] = a[13]; out[14] = a[14]; out[15] = a[15];
        return out;
    }

    // -------------------------------------------------------------
    // State & Responsiveness
    // -------------------------------------------------------------
    const projMatrix = mat4Create();
    const viewMatrix = mat4Create();
    const modelMatrix = mat4Create();
    const modelMatrixRing = mat4Create();

    const cameraPos = [0, 0, 16.5];
    let rotX = 0.55;
    let rotY = 0.35;
    let rotZ = 0.15;

    let targetMouseX = 0;
    let targetMouseY = 0;
    let curMouseX = 0;
    let curMouseY = 0;

    // Mouse parallax tracking
    window.addEventListener('mousemove', (e) => {
        targetMouseX = (e.clientX / window.innerWidth - 0.5) * 0.45;
        targetMouseY = (e.clientY / window.innerHeight - 0.5) * 0.35;
    }, { passive: true });

    function resize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
        const w = window.innerWidth;
        const h = window.innerHeight;

        if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
            canvas.width = Math.floor(w * dpr);
            canvas.height = Math.floor(h * dpr);
        }

        gl.viewport(0, 0, canvas.width, canvas.height);
        const aspect = w / h;
        mat4Perspective(projMatrix, Math.PI / 4, aspect, 0.1, 100.0);
    }

    window.addEventListener('resize', resize, { passive: true });
    resize();

    // -------------------------------------------------------------
    // Render Loop
    // -------------------------------------------------------------
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    let lastTime = performance.now();
    let animId = null;

    function render(now) {
        animId = requestAnimationFrame(render);

        if (document.hidden) return; // Pause when tab not visible

        const dt = Math.min((now - lastTime) / 1000, 0.1);
        lastTime = now;

        const timeSec = now * 0.001;

        // Smooth continuous 3D rotation (15-20s period)
        if (!prefersReducedMotion) {
            rotX += dt * 0.28;
            rotY += dt * 0.42;
            rotZ += dt * 0.14;
        }

        // Smooth mouse parallax lerp
        curMouseX += (targetMouseX - curMouseX) * 0.05;
        curMouseY += (targetMouseY - curMouseY) * 0.05;

        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.useProgram(program);

        // Update Camera View Matrix with parallax
        mat4LookAt(viewMatrix, [cameraPos[0] + curMouseX * 3.0, cameraPos[1] - curMouseY * 2.5, cameraPos[2]], [0, 0, 0], [0, 1, 0]);

        gl.uniformMatrix4fv(uProjLoc, false, projMatrix);
        gl.uniformMatrix4fv(uViewLoc, false, viewMatrix);
        gl.uniform3f(uCamPosLoc, cameraPos[0] + curMouseX * 3.0, cameraPos[1] - curMouseY * 2.5, cameraPos[2]);
        gl.uniform1f(uTimeLoc, timeSec);
        gl.uniform1f(uHasDerivLoc, hasDerivatives ? 1.0 : 0.0);

        // Screen-size responsive scaling
        const isMobile = window.innerWidth < 768;
        const scaleFactor = isMobile ? 0.68 : (window.innerWidth < 1200 ? 0.88 : 1.05);

        // 1. Draw Main Translucent Torus
        mat4Identity(modelMatrix);
        mat4RotateX(modelMatrix, modelMatrix, rotX);
        mat4RotateY(modelMatrix, modelMatrix, rotY);
        mat4RotateZ(modelMatrix, modelMatrix, rotZ);
        mat4Scale(modelMatrix, modelMatrix, [scaleFactor, scaleFactor, scaleFactor]);

        gl.uniformMatrix4fv(uModelLoc, false, modelMatrix);

        // Bind attributes for main torus
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        gl.vertexAttribPointer(aPosLoc, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(aPosLoc);

        gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer);
        gl.vertexAttribPointer(aNormLoc, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(aNormLoc);

        gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
        gl.vertexAttribPointer(aUvLoc, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(aUvLoc);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.drawElements(gl.TRIANGLES, torusData.indexCount, gl.UNSIGNED_SHORT, 0);

        // 2. Draw Outer Slender Gimbal Ring (counter-rotating for gyroscopic depth)
        mat4Identity(modelMatrixRing);
        mat4RotateX(modelMatrixRing, modelMatrixRing, -rotX * 0.7 + 0.6);
        mat4RotateY(modelMatrixRing, modelMatrixRing, -rotY * 0.5);
        mat4RotateZ(modelMatrixRing, modelMatrixRing, rotZ * 1.2);
        mat4Scale(modelMatrixRing, modelMatrixRing, [scaleFactor * 0.95, scaleFactor * 0.95, scaleFactor * 0.95]);

        gl.uniformMatrix4fv(uModelLoc, false, modelMatrixRing);

        gl.bindBuffer(gl.ARRAY_BUFFER, posBufferRing);
        gl.vertexAttribPointer(aPosLoc, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, normBufferRing);
        gl.vertexAttribPointer(aNormLoc, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, uvBufferRing);
        gl.vertexAttribPointer(aUvLoc, 2, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBufferRing);
        gl.drawElements(gl.TRIANGLES, outerRingData.indexCount, gl.UNSIGNED_SHORT, 0);
    }

    animId = requestAnimationFrame(render);
})();
