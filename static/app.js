// --- State Management ---
let state = {
    agent_x: -0.5,
    agent_y: 0.0,
    target_x: 0.5,
    target_y: 0.0
};

let cachedOriginalProbs = [0.25, 0.25, 0.25, 0.25];
let cachedOriginalPrediction = "--";
let currentPrediction = "--";
let currentProbs = [0.25, 0.25, 0.25, 0.25];
let currentNarrative = "";
let currentNodes = null;
let currentLinks = null;

let socket = null;
let isDragging = false;
let dragTarget = null; // 'agent' or 'target'
let lastSendTime = 0;
const THROTTLE_MS = 25; // 40 FPS real-time updates for WebSockets

let intervenedFeatures = new Set();

// Sankey control state
let showAll = false;
let topK = 3;

// --- DOM Elements ---
const statusIndicator = document.getElementById("statusIndicator");
const statusText = document.getElementById("statusText");
const arenaCanvas = document.getElementById("arenaCanvas");
const ctx = arenaCanvas.getContext("2d");
let W = 400; // Will be set dynamically by setupHiDPICanvas
let H = 400;

function setupHiDPICanvas(canvas, context) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    context.scale(dpr, dpr);
    return { w: rect.width, h: rect.height };
}

// Metric displays
const metricLoss = document.getElementById("metricLoss");
const metricAcc = document.getElementById("metricAcc");
const metricSaeLoss = document.getElementById("metricSaeLoss");
const epochBadge = document.getElementById("epochBadge");
const valAgentX = document.getElementById("val-agent-x");
const valAgentY = document.getElementById("val-agent-y");
const valTargetX = document.getElementById("val-target-x");
const valTargetY = document.getElementById("val-target-y");
const predictionOverlay = document.getElementById("predictionOverlay");

// Tabs
const tabBtns = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

// Sankey inputs
const topKSlider = document.getElementById("topKSlider");
const topKVal = document.getElementById("topKVal");
const btnToggleAllNeurons = document.getElementById("btnToggleAllNeurons");
const sankeyCaption = document.getElementById("sankeyCaption");

// Intervention elements
const neuronsGrid = document.getElementById("neuronsGrid");
const btnResetNeurons = document.getElementById("btnResetNeurons");
const btnZeroAllNeurons = document.getElementById("btnZeroAllNeurons");
const interveneOrigPred = document.getElementById("interveneOrigPred");
const interveneModPred = document.getElementById("interveneModPred");

// Feature Observatory element
const featureObservatoryGrid = document.getElementById("featureObservatoryGrid");

// Feature Inversion (Dream) elements
const btnDream = document.getElementById("btnDream");
const dreamSpinner = document.getElementById("dreamSpinner");
const dreamFeatureId = document.getElementById("dreamFeatureId");
const dreamCoordsDisplay = document.getElementById("dreamCoordsDisplay");
const dreamAgentPos = document.getElementById("dreamAgentPos");
const dreamTargetPos = document.getElementById("dreamTargetPos");
const dreamCanvas = document.getElementById("dreamCanvas");
const dreamCtx = dreamCanvas.getContext("2d");
let DW = 350;
let DH = 350;

// Training buttons
const btnTrain1 = document.getElementById("btnTrain1");
const btnTrain50 = document.getElementById("btnTrain50");
const trainSpinner = document.getElementById("trainSpinner");
const progressBarContainer = document.getElementById("progressBarContainer");
const progressBar = document.getElementById("progressBar");

// --- WebSocket Connection ---
// --- WebSocket Setup ---
let reconnectDelay = 2000;

function connectWebSocket() {
    // Use wss:// when the page is served over HTTPS (required for Render / Railway / etc.)
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);
    
    socket.onopen = () => {
        statusIndicator.className = "status-indicator connected";
        statusText.innerText = "Connected";
        reconnectDelay = 2000; // Reset on successful connection
        document.getElementById("wakingUpOverlay").style.display = "none";
        sendState();
    };
    
    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.error) {
                console.error("Server error:", data.error);
                return;
            }
            handleInferenceResponse(data);
        } catch (err) {
            console.error("Error parsing WS message:", err);
        }
    };
    
    socket.onclose = () => {
        statusIndicator.className = "status-indicator disconnected";
        statusText.innerText = "Disconnected. Reconnecting...";
        document.getElementById("wakingUpOverlay").style.display = "flex";
        setTimeout(connectWebSocket, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 1.5, 30000); // Cap at 30s
    };
    
    socket.onerror = (err) => {
        console.error("WebSocket error", err);
        socket.close();
    };
}

// --- Coordinate Conversions ---
function toScreen(logicalX, maxPx) { return ((logicalX + 1) / 2) * maxPx; }
function toLogical(screenX, maxPx) { return (screenX / maxPx) * 2 - 1; }
function toScreenY(logicalY, maxPx) { return (1 - ((logicalY + 1) / 2)) * maxPx; }
function toLogicalY(screenY, maxPx) { return (1 - (screenY / maxPx)) * 2 - 1; }

// --- Draw Game Arena ---
function drawArena() {
    ctx.clearRect(0, 0, W, H);
    
    // Draw fine grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let i = -1.0; i <= 1.0; i += 0.25) {
        const x = toScreen(i, W);
        const y = toScreenY(i, H);
        
        ctx.beginPath();
        ctx.moveTo(x, 0); ctx.lineTo(x, H);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(0, y); ctx.lineTo(W, y);
        ctx.stroke();
    }
    
    // Draw major axes
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(W/2, 0); ctx.lineTo(W/2, H);
    ctx.moveTo(0, H/2); ctx.lineTo(W, H/2);
    ctx.stroke();
    
    // Convert states to screen coords
    const ax = toScreen(state.agent_x, W);
    const ay = toScreenY(state.agent_y, H);
    const tx = toScreen(state.target_x, W);
    const ty = toScreenY(state.target_y, H);
    
    // Draw connection line
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw Target Dot
    ctx.fillStyle = varColor("--accent-coral");
    ctx.beginPath(); ctx.arc(tx, ty, 8, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw Canvas Visual Feedback (Prediction Arrow)
    if (currentPrediction !== "--" && currentProbs) {
        const maxProb = Math.max(...currentProbs);
        let dirX = 0, dirY = 0;
        if (currentPrediction === "Up") dirY = -1;
        if (currentPrediction === "Down") dirY = 1;
        if (currentPrediction === "Left") dirX = -1;
        if (currentPrediction === "Right") dirX = 1;
        
        if (dirX !== 0 || dirY !== 0) {
            const arrowLength = 20 + (maxProb * 20); // 20px to 40px
            const endX = ax + dirX * arrowLength;
            const endY = ay + dirY * arrowLength;
            
            // Draw line
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(endX, endY);
            ctx.strokeStyle = `rgba(88, 166, 255, ${0.3 + maxProb * 0.7})`; // matches accent-cyan roughly
            ctx.lineWidth = 2 + (maxProb * 4);
            ctx.stroke();
            
            // Draw arrowhead
            const angle = Math.atan2(dirY, dirX);
            ctx.beginPath();
            ctx.moveTo(endX, endY);
            ctx.lineTo(endX - 10 * Math.cos(angle - Math.PI / 6), endY - 10 * Math.sin(angle - Math.PI / 6));
            ctx.lineTo(endX - 10 * Math.cos(angle + Math.PI / 6), endY - 10 * Math.sin(angle + Math.PI / 6));
            ctx.lineTo(endX, endY);
            ctx.fillStyle = `rgba(88, 166, 255, ${0.3 + maxProb * 0.7})`;
            ctx.fill();
        }
    }
    
    // Draw Agent Dot
    ctx.fillStyle = varColor("--accent-cyan");
    ctx.beginPath(); ctx.arc(ax, ay, 10, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
}

function varColor(cssVarName) {
    return getComputedStyle(document.documentElement).getPropertyValue(cssVarName).trim();
}

// --- Drag & Drop Handlers ---
arenaCanvas.addEventListener("mousedown", handlePointerDown);
arenaCanvas.addEventListener("mousemove", handlePointerMove);
arenaCanvas.addEventListener("mouseup", handlePointerUp);
arenaCanvas.addEventListener("mouseleave", handlePointerUp);

arenaCanvas.addEventListener("touchstart", (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    if (!touch) return;
    handlePointerDown(convertTouch(touch));
}, { passive: false });

arenaCanvas.addEventListener("touchmove", (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    if (!touch) return;
    handlePointerMove(convertTouch(touch));
}, { passive: false });

arenaCanvas.addEventListener("touchend", (e) => {
    e.preventDefault();
    handlePointerUp();
}, { passive: false });

// touchcancel fires when the OS interrupts a touch gesture (e.g. an
// incoming call, a system gesture, or the browser deciding to cancel it).
// Without handling it, isDragging could stay stuck "true" with no matching
// touchend ever arriving, making the Agent/Target appear to follow the
// pointer forever on the next unrelated mousemove/touchmove.
arenaCanvas.addEventListener("touchcancel", (e) => {
    handlePointerUp();
}, { passive: false });

function convertTouch(touch) {
    return {
        clientX: touch.clientX,
        clientY: touch.clientY
    };
}

function handlePointerDown(e) {
    const rect = arenaCanvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const ax = toScreen(state.agent_x, W);
    const ay = toScreenY(state.agent_y, H);
    const tx = toScreen(state.target_x, W);
    const ty = toScreenY(state.target_y, H);
    
    const distAgent = Math.hypot(mouseX - ax, mouseY - ay);
    const distTarget = Math.hypot(mouseX - tx, mouseY - ty);
    
    if (distAgent < 25 && distAgent < distTarget) {
        isDragging = true;
        dragTarget = "agent";
    } else if (distTarget < 25) {
        isDragging = true;
        dragTarget = "target";
    }
}

function handlePointerMove(e) {
    if (!isDragging) return;
    const rect = arenaCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const logX = Math.max(-1.0, Math.min(1.0, toLogical(x, W)));
    const logY = Math.max(-1.0, Math.min(1.0, toLogicalY(y, H)));
    
    if (dragTarget === "agent") {
        state.agent_x = logX;
        state.agent_y = logY;
    } else {
        state.target_x = logX;
        state.target_y = logY;
    }
    
    // Update local displays
    valAgentX.innerText = state.agent_x.toFixed(2);
    valAgentY.innerText = state.agent_y.toFixed(2);
    valTargetX.innerText = state.target_x.toFixed(2);
    valTargetY.innerText = state.target_y.toFixed(2);
    
    drawArena();
    
    const now = Date.now();
    if (now - lastSendTime > THROTTLE_MS) {
        sendState();
        lastSendTime = now;
    }
}

function handlePointerUp() {
    if (isDragging) {
        sendState();
    }
    isDragging = false;
    dragTarget = null;
}

// --- WebSocket Sender ---
function sendState() {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    
    // Check if there are active interventions
    const interveneNeurons = {};
    let hasInterventions = false;
    for (let i = 0; i < 32; i++) {
        const chk = document.getElementById(`chk-neuron-${i}`);
        if (chk && !chk.checked) {
            interveneNeurons[i] = 0.0;
            hasInterventions = true;
        }
    }
    
    const payload = {
        agent_x: state.agent_x,
        agent_y: state.agent_y,
        target_x: state.target_x,
        target_y: state.target_y,
        intervene_neurons: hasInterventions ? interveneNeurons : null,
        intervene_features: intervenedFeatures.size > 0 ? Object.fromEntries([...intervenedFeatures].map(f => [f, 0.0])) : null,
        top_k: showAll ? 32 : topK
    };
    
    socket.send(JSON.stringify(payload));
}

function handleInferenceResponse(data) {
    currentPrediction = data.prediction;
    currentProbs = data.probs;
    currentNarrative = data.narrative;
    currentNodes = data.nodes;
    currentLinks = data.links;

    // 1. Update Game Arena Prediction Overlay
    predictionOverlay.innerText = `Prediction: ${data.prediction}`;
    
    // 2. Cache original probabilities only when there are truly no active
    // interventions of either kind (neuron pruning OR SAE feature zeroing).
    // Previously this only checked neuron checkboxes, so zeroing a feature
    // in the Feature Observatory tab (without touching any neuron checkbox)
    // would get silently cached as the "original" prediction, corrupting
    // the Intervention Sandbox's probability-shift comparison.
    const hasNeuronInterventions = Array.from(document.querySelectorAll(".checkbox-grid input")).some(chk => !chk.checked);
    const hasInterventions = hasNeuronInterventions || intervenedFeatures.size > 0;
    if (!hasInterventions) {
        cachedOriginalProbs = [...data.probs];
        cachedOriginalPrediction = data.prediction;
        
        interveneOrigPred.innerText = data.prediction;
        interveneModPred.innerText = data.prediction;
    } else {
        interveneModPred.innerText = data.prediction;
    }
    
    // 3. Render decision confidence chart (Tab 1)
    renderConfidenceChart(data.probs);
    
    // 4. Render narrative explanation
    renderNarrative(data.narrative);
    
    // 5. Render Sankey diagram
    renderSankeyChart(data.nodes, data.links);
    
    // 6. Render Feature Observatory (Tab 3)
    renderFeatureObservatory(data.features);
    
    // 7. Update Intervention Sandbox logit shift plot (Tab 2)
    renderInterventionChart(data.probs);
}

// --- Narrative Formatting ---
function renderNarrative(mdText) {
    // Basic Markdown to HTML converter
    let html = mdText
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/- (.*?)(<br>|$)/g, '<li>$1</li>');
        
    if (html.includes('<li>')) {
        // Wrap adjacent list items in <ul>
        html = html.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
    }
    
    document.getElementById("narrativeText").innerHTML = `<p>${html}</p>`;
}

// --- Plotly Chart Drawers ---

function renderConfidenceChart(probs) {
    const data = [{
        type: 'bar',
        x: probs,
        y: ["Up", "Down", "Left", "Right"],
        orientation: 'h',
        marker: {
            color: probs,
            colorscale: 'Viridis',
            line: { width: 0 }
        }
    }];
    
    const layout = {
        font: { size: 10, color: '#a0aab8' },
        height: 170,
        margin: { l: 45, r: 10, t: 5, b: 20 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { range: [0, 1], gridcolor: 'rgba(255,255,255,0.05)', zeroline: false },
        yaxis: { zeroline: false }
    };
    
    Plotly.react("confidenceChart", data, layout, { displayModeBar: false, responsive: true });
}

function renderSankeyChart(nodes, links) {
    // Generate nodes flat array mapping unique identifier strings to human readable names and colors
    const allNodes = [];
    
    nodes.inputs.forEach((label, i) => {
        allNodes.push({ id: `input_${i}`, label: label, color: '#4facfe' });
    });
    
    nodes.neurons.forEach((label) => {
        const idVal = label.split(" ")[1];
        allNodes.push({ id: `neuron_${idVal}`, label: label, color: '#66fcf1' });
    });
    
    nodes.features.forEach((label) => {
        const idVal = label.split(" ")[1];
        allNodes.push({ id: `feature_${idVal}`, label: label, color: '#4ecdc4' });
    });
    
    nodes.outputs.forEach((label, i) => {
        allNodes.push({ id: `output_${i}`, label: label, color: '#ff6b6b' });
    });
    
    const nodeIndices = {};
    allNodes.forEach((node, index) => {
        nodeIndices[node.id] = index;
    });
    
    // Resolve link targets & sources. Throw away any links pointing to missing items
    const source = [];
    const target = [];
    const value = [];
    const linkLabels = [];
    
    links.forEach((link) => {
        const sIdx = nodeIndices[link.source];
        const tIdx = nodeIndices[link.target];
        if (sIdx !== undefined && tIdx !== undefined) {
            source.push(sIdx);
            target.push(tIdx);
            value.push(link.value);
            linkLabels.push(`Contribution: ${link.value.toFixed(3)}`);
        }
    });
    
    const trace = {
        type: 'sankey',
        orientation: 'h',
        node: {
            pad: 12,
            thickness: 18,
            line: { color: 'rgba(0,0,0,0.5)', width: 0.5 },
            label: allNodes.map(n => n.label),
            color: allNodes.map(n => n.color)
        },
        link: {
            source: source,
            target: target,
            value: value,
            label: linkLabels,
            color: 'rgba(102, 252, 241, 0.15)',
            line: { color: 'rgba(0,0,0,0.1)', width: 0.5 }
        }
    };
    
    const layout = {
        font: { size: 9, color: 'white' },
        height: 380,
        margin: { l: 5, r: 5, t: 5, b: 5 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };
    
    Plotly.react("sankeyChart", [trace], layout, { displayModeBar: false, responsive: true });
    
    // Update caption
    sankeyCaption.innerText = `Showing ${links.length} causal connections across ${nodes.neurons.length} neurons and ${nodes.features.length} features.`;
}

function renderInterventionChart(probs) {
    // Probability Shift = current (intervened) probs minus cached original (unintervened) probs
    const shifts = probs.map((prob, i) => prob - cachedOriginalProbs[i]);
    const dirs = ["Up", "Down", "Left", "Right"];
    
    // Colors based on shift positive (green) or negative (coral)
    const colors = shifts.map(val => val >= 0 ? '#00e676' : '#ff6b6b');
    
    const data = [{
        type: 'bar',
        x: dirs,
        y: shifts,
        marker: {
            color: colors
        }
    }];
    
    const layout = {
        font: { size: 10, color: '#a0aab8' },
        margin: { l: 30, r: 10, t: 10, b: 25 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        yaxis: { range: [-1.0, 1.0], gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.2)' },
        xaxis: { zeroline: false }
    };
    
    Plotly.react("interventionChart", data, layout, { displayModeBar: false, responsive: true });
}

function renderFeatureObservatory(features) {
    // Zip indexes and values, then include a feature if either:
    //   (a) it's actively firing above the display threshold, or
    //   (b) the user has zeroed it — even though zeroing suppresses its
    //       activation below the threshold, we still need to show it so
    //       the "Zeroed" toggle remains reachable to undo the intervention.
    const activeFeats = features
        .map((val, idx) => ({ id: idx, val: val }))
        .filter(f => f.val > 0.05 || intervenedFeatures.has(f.id))
        .sort((a, b) => b.val - a.val);
        
    featureObservatoryGrid.innerHTML = "";
    
    if (activeFeats.length === 0) {
        featureObservatoryGrid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--text-secondary); padding: 40px; font-size: 0.9rem;">No features active above threshold (0.05). Move the Agent or Target in the Game Arena to see features fire.</div>`;
        return;
    }
    
    activeFeats.slice(0, 12).forEach((f) => {
        const isZeroed = intervenedFeatures.has(f.id);
        const btnClass = isZeroed ? "btn-zero-feature active" : "btn-zero-feature";
        const btnText = isZeroed ? "Zeroed" : "Zero";
        
        const card = document.createElement("div");
        card.className = "feature-card";
        card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h4 style="margin: 0;">Feature ${f.id}</h4>
                <button class="${btnClass}" onclick="toggleFeatureIntervention(${f.id}, event)">${btnText}</button>
            </div>
            <div class="feature-val">${f.val.toFixed(2)}</div>
        `;
        featureObservatoryGrid.appendChild(card);
    });
}

window.toggleFeatureIntervention = function(featIdx, event) {
    if (intervenedFeatures.has(featIdx)) {
        intervenedFeatures.delete(featIdx);
    } else {
        intervenedFeatures.add(featIdx);
    }
    sendState(); // Instantly trigger a refresh
};

const btnResetFeatures = document.getElementById("btnResetFeatures");
if (btnResetFeatures) {
    btnResetFeatures.addEventListener("click", () => {
        intervenedFeatures.clear();
        sendState();
    });
}

// --- Training Interface Handler ---
async function fetchMetrics() {
    try {
        const res = await fetch("/api/metrics");
        const data = await res.json();
        updateTrainingUI(data);
    } catch (err) {
        console.error("Error fetching metrics:", err);
    }
}

function updateTrainingUI(data) {
    epochBadge.innerText = `Epoch: ${data.epoch}`;
    
    const lossHist = data.metrics.loss;
    const saeLossHist = data.metrics.sae_loss;
    const accHist = data.metrics.acc;
    
    if (lossHist.length > 0) {
        metricLoss.innerText = lossHist[lossHist.length - 1].toFixed(4);
        metricAcc.innerText = `${(accHist[accHist.length - 1] * 100).toFixed(1)}%`;
        metricSaeLoss.innerText = saeLossHist[saeLossHist.length - 1].toFixed(4);
        
        // Render history chart
        renderTrainingChart(data.metrics);
        
        // Fetch and update Heatmap
        fetchFeatureLogits();
    }
}

function renderTrainingChart(metrics) {
    const epochs = metrics.loss.map((_, i) => i + 1);
    
    const traceLoss = {
        x: epochs,
        y: metrics.loss,
        name: 'Base Loss',
        type: 'scatter',
        line: { color: varColor('--accent-cyan'), width: 1.5 }
    };
    
    const traceSaeSparsity = {
        x: epochs,
        y: metrics.sparsity,
        name: 'SAE Sparsity',
        type: 'scatter',
        line: { color: varColor('--accent-purple'), width: 1.5 }
    };
    
    const layout = {
        font: { size: 8, color: '#a0aab8' },
        height: 120,
        margin: { l: 25, r: 10, t: 10, b: 20 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        showlegend: false,
        xaxis: { gridcolor: 'rgba(255,255,255,0.03)', zeroline: false },
        yaxis: { gridcolor: 'rgba(255,255,255,0.03)', zeroline: false }
    };
    
    Plotly.react("trainingChart", [traceLoss, traceSaeSparsity], layout, { displayModeBar: false, responsive: true });
}

// Training click actions
btnTrain1.addEventListener("click", () => runTraining(1));
btnTrain50.addEventListener("click", () => runTraining(50));

async function runTraining(epochs) {
    setTrainingLoading(true);
    
    // Start progress mock animation for visual polish
    let progressVal = 0;
    progressBarContainer.style.display = "block";
    progressBar.style.width = "0%";
    
    const progressInterval = setInterval(() => {
        progressVal = Math.min(90, progressVal + (epochs === 1 ? 40 : 2));
        progressBar.style.width = `${progressVal}%`;
    }, 30);
    
    try {
        const response = await fetch("/api/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ epochs: epochs })
        });
        const data = await response.json();
        
        clearInterval(progressInterval);
        progressBar.style.width = "100%";
        
        setTimeout(() => {
            progressBarContainer.style.display = "none";
            progressBar.style.width = "0%";
            updateTrainingUI(data);
            setTrainingLoading(false);
            sendState(); // update inference
        }, 150);
        
    } catch (err) {
        clearInterval(progressInterval);
        progressBarContainer.style.display = "none";
        console.error("Training error:", err);
        setTrainingLoading(false);
    }
}

function setTrainingLoading(loading) {
    if (loading) {
        btnTrain1.disabled = true;
        btnTrain50.disabled = true;
        trainSpinner.style.display = "inline-block";
    } else {
        btnTrain1.disabled = false;
        btnTrain50.disabled = false;
        trainSpinner.style.display = "none";
    }
}

// --- Sankey Controls ---
topKSlider.addEventListener("input", (e) => {
    topK = parseInt(e.target.value);
    topKVal.innerText = topK;
    showAll = false;
    btnToggleAllNeurons.innerText = "Show All Neurons";
    btnToggleAllNeurons.classList.remove("active");
    sendState();
});

btnToggleAllNeurons.addEventListener("click", () => {
    showAll = !showAll;
    if (showAll) {
        btnToggleAllNeurons.innerText = "Show Top K Only";
        btnToggleAllNeurons.classList.add("active");
    } else {
        btnToggleAllNeurons.innerText = "Show All Neurons";
        btnToggleAllNeurons.classList.remove("active");
    }
    sendState();
});

// --- Tab Switching ---
tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        tabBtns.forEach(b => b.classList.remove("active"));
        tabContents.forEach(c => c.classList.remove("active"));
        
        btn.classList.add("active");
        const targetTab = btn.getAttribute("data-tab");
        document.getElementById(targetTab).classList.add("active");
        
        // Plotly resize fallback for tab switching
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
        }, 50);
    });
});

// --- Intervention Sandbox Grid Populator ---
function populateNeuronsGrid() {
    neuronsGrid.innerHTML = "";
    for (let i = 0; i < 32; i++) {
        const label = document.createElement("label");
        label.className = "neuron-chk-label";
        label.id = `lbl-neuron-${i}`;
        label.innerHTML = `<input type="checkbox" id="chk-neuron-${i}" checked> N-${i}`;
        
        const chk = label.querySelector("input");
        chk.addEventListener("change", () => {
            if (!chk.checked) {
                label.classList.add("pruned");
            } else {
                label.classList.remove("pruned");
            }
            sendState();
        });
        
        neuronsGrid.appendChild(label);
    }
}

btnResetNeurons.addEventListener("click", () => {
    for (let i = 0; i < 32; i++) {
        const chk = document.getElementById(`chk-neuron-${i}`);
        if (chk) {
            chk.checked = true;
            document.getElementById(`lbl-neuron-${i}`).classList.remove("pruned");
        }
    }
    sendState();
});

btnZeroAllNeurons.addEventListener("click", () => {
    for (let i = 0; i < 32; i++) {
        const chk = document.getElementById(`chk-neuron-${i}`);
        if (chk) {
            chk.checked = false;
            document.getElementById(`lbl-neuron-${i}`).classList.add("pruned");
        }
    }
    sendState();
});

// --- Feature Inversion (Dreaming) Logic ---
btnDream.addEventListener("click", async () => {
    const featureIdx = parseInt(dreamFeatureId.value);
    if (isNaN(featureIdx) || featureIdx < 0 || featureIdx > 127) {
        alert("Please enter a valid feature index between 0 and 127.");
        return;
    }
    
    btnDream.disabled = true;
    dreamSpinner.style.display = "inline-block";
    dreamCoordsDisplay.style.display = "none";
    
    try {
        const response = await fetch("/api/dream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ feature_idx: featureIdx })
        });
        const data = await response.json();
        
        if (data.status === "success") {
            const coords = data.dream_state; // [agent_x, agent_y, target_x, target_y]
            dreamAgentPos.innerText = `${coords[0].toFixed(2)}, ${coords[1].toFixed(2)}`;
            dreamTargetPos.innerText = `${coords[2].toFixed(2)}, ${coords[3].toFixed(2)}`;
            dreamCoordsDisplay.style.display = "block";
            
            drawDreamCanvas(coords);
        }
    } catch (err) {
        console.error("Dream error:", err);
    } finally {
        btnDream.disabled = false;
        dreamSpinner.style.display = "none";
    }
});

function drawDreamCanvas(coords) {
    // Use the module-level logical dimensions (set by setupHiDPICanvas), NOT
    // dreamCanvas.width/.height which are the raw pixel-buffer sizes scaled
    // by devicePixelRatio — using those causes all draws to appear in the
    // top-left corner on retina / HiDPI screens.
    
    dreamCtx.clearRect(0, 0, DW, DH);
    
    // Draw major axes
    dreamCtx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    dreamCtx.lineWidth = 1;
    dreamCtx.beginPath();
    dreamCtx.moveTo(DW/2, 0); dreamCtx.lineTo(DW/2, DH);
    dreamCtx.moveTo(0, DH/2); dreamCtx.lineTo(DW, DH/2);
    dreamCtx.stroke();
    
    // Convert dream states to screen coords
    const ax = toScreen(coords[0], DW);
    const ay = toScreenY(coords[1], DH);
    const tx = toScreen(coords[2], DW);
    const ty = toScreenY(coords[3], DH);
    
    // Connect
    dreamCtx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    dreamCtx.lineWidth = 1.5;
    dreamCtx.setLineDash([4, 4]);
    dreamCtx.beginPath(); dreamCtx.moveTo(ax, ay); dreamCtx.lineTo(tx, ty); dreamCtx.stroke();
    dreamCtx.setLineDash([]);
    
    // Draw Dream Agent
    dreamCtx.fillStyle = varColor("--accent-cyan");
    dreamCtx.beginPath(); dreamCtx.arc(ax, ay, 8, 0, Math.PI*2); dreamCtx.fill();
    dreamCtx.strokeStyle = "#fff";
    dreamCtx.lineWidth = 1.5;
    dreamCtx.stroke();
    
    // Label Agent
    // Draw Dream Target
    dreamCtx.fillStyle = varColor("--accent-coral");
    dreamCtx.beginPath(); dreamCtx.arc(tx, ty, 8, 0, Math.PI*2); dreamCtx.fill();
    dreamCtx.strokeStyle = "#fff";
    dreamCtx.lineWidth = 1.5;
    dreamCtx.stroke();
    
    // Label Target
    dreamCtx.fillStyle = "rgba(255,255,255,0.7)";
    dreamCtx.font = "9px monospace";
    dreamCtx.fillText("Target", tx + 12, ty + 3);
}

function drawInitialDreamCanvas() {
    const dims = setupHiDPICanvas(dreamCanvas, dreamCtx);
    DW = dims.w;
    DH = dims.h;
    dreamCtx.clearRect(0, 0, DW, DH);
    dreamCtx.fillStyle = "rgba(255,255,255,0.25)";
    dreamCtx.font = "11px var(--font-body)";
    dreamCtx.textAlign = "center";
    dreamCtx.fillText("Enter feature ID and click Dream to visualize.", DW/2, DH/2);
}

async function fetchFeatureLogits() {
    try {
        const res = await fetch("/api/feature_logits");
        const data = await res.json();
        renderHeatmap(data.matrix);
    } catch(err) {
        console.error("Error fetching feature logits:", err);
    }
}

function renderHeatmap(matrix) {
    // matrix is [4][128] for [Logits (Up, Down, Left, Right)][Features (0-127)]
    const yLabels = ["Up", "Down", "Left", "Right"];
    const xLabels = Array.from({length: 128}, (_, i) => `Feature ${i}`);
    
    const data = [{
        z: matrix,
        x: xLabels,
        y: yLabels,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0 // Center at 0
    }];
    
    const layout = {
        margin: { l: 50, r: 10, t: 10, b: 50 },
        font: { size: 10, color: '#a0aab8' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { showticklabels: false }
    };
    
    Plotly.react('heatmapChart', data, layout, { displayModeBar: false, responsive: true });
}

// --- How-To Guide ---
const btnHowTo = document.getElementById("btnHowTo");
if (btnHowTo) {
    btnHowTo.addEventListener("click", () => {
        window.open("how-to.html", "_blank");
    });
}

// --- Download Notebook ---
document.getElementById("btnExportNotebook").addEventListener("click", async () => {
    try {
        const btn = document.getElementById("btnExportNotebook");
        const originalText = btn.innerText;
        btn.innerText = "Exporting...";
        
        const res = await fetch("/api/export_notebook");
        const data = await res.json();
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/x-ipynb+json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `NeuroScope_Export_${Date.now()}.ipynb`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        
        btn.innerText = originalText;
    } catch (err) {
        console.error("Failed to export notebook:", err);
        alert("Failed to export notebook");
    }
});

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
    populateNeuronsGrid();
    
    // Setup initial canvases
    const arenaDims = setupHiDPICanvas(arenaCanvas, ctx);
    W = arenaDims.w;
    H = arenaDims.h;
    
    drawArena();
    drawInitialDreamCanvas();
    fetchMetrics();
    fetchFeatureLogits();
    connectWebSocket();
    
    // Handle window resize for Plotly responsiveness and canvas scaling
    window.addEventListener("resize", () => {
        const arenaDims = setupHiDPICanvas(arenaCanvas, ctx);
        W = arenaDims.w;
        H = arenaDims.h;
        drawArena();
        
        const dreamDims = setupHiDPICanvas(dreamCanvas, dreamCtx);
        DW = dreamDims.w;
        DH = dreamDims.h;
        // Optionally redraw dream canvas if it has data, for now just clear/redraw initial
        drawInitialDreamCanvas();

        const plots = ["confidenceChart", "sankeyChart", "interventionChart", "trainingChart"];
        plots.forEach(id => {
            const el = document.getElementById(id);
            if (el && el.classList.contains("js-plotly-plot")) {
                Plotly.Plots.resize(el);
            }
        });
    });

    // --- Export Insights Logic ---
    const btnExport = document.getElementById("btnExport");
    if (btnExport) {
        btnExport.addEventListener("click", () => {
            const report = {
                timestamp: new Date().toISOString(),
                game_state: {
                    agent: { x: state.agent_x, y: state.agent_y },
                    target: { x: state.target_x, y: state.target_y }
                },
                prediction: currentPrediction,
                probabilities: currentProbs,
                narrative: currentNarrative,
                sankey: {
                    nodes: currentNodes,
                    links: currentLinks
                }
            };
            
            const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "neuroscope_report.json";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    }

    // --- Guided Onboarding Tour ---
    const tourModal = document.getElementById("tourModal");
    const tourBody = document.getElementById("tourBody");
    const tourIndicators = document.getElementById("tourIndicators");
    const btnTourBack = document.getElementById("btnTourBack");
    const btnTourNext = document.getElementById("btnTourNext");
    const btnTourClose = document.getElementById("btnTourClose");

    const tourSteps = [
        {
            title: "The Task",
            body: "<p>Welcome to NeuroScope! This agent is learning a simple navigation task in the <strong>Game Arena</strong>.</p><p>Its goal is to predict the correct direction (Up, Down, Left, or Right) to reach the Target from the Agent's position.</p>"
        },
        {
            title: "The Black Box",
            body: "<p>A neural network makes these decisions. But neural networks are usually 'black boxes'—matrices of confusing numbers.</p><p>In the <strong>Intervention Sandbox</strong>, you can view the 32 raw 'MLP Neurons' that act as the network's brain. You can even 'prune' them by unchecking boxes to see how the decision changes.</p>"
        },
        {
            title: "The X-Ray",
            body: "<p>To understand the network, we use a <strong>Sparse Autoencoder (SAE)</strong>.</p><p>Think of it as a translator that converts the AI's dense, alien thoughts into human concepts. Check the <strong>Decision Backtrace</strong> to see exactly which features drove the final decision, or browse them in the <strong>Feature Observatory</strong>.</p>"
        }
    ];

    let currentTourStep = 0;

    function renderTourStep() {
        const step = tourSteps[currentTourStep];
        document.getElementById("tourTitle").innerText = step.title;
        tourBody.innerHTML = step.body;
        
        // Update Indicators
        tourIndicators.innerHTML = "";
        tourSteps.forEach((_, i) => {
            const dot = document.createElement("div");
            dot.className = "tour-dot" + (i === currentTourStep ? " active" : "");
            tourIndicators.appendChild(dot);
        });
        
        // Update Buttons
        btnTourBack.style.display = currentTourStep === 0 ? "none" : "block";
        btnTourNext.innerText = currentTourStep === tourSteps.length - 1 ? "Get Started" : "Next";
    }

    function closeTour() {
        tourModal.style.display = "none";
        localStorage.setItem("neuroscopeTourSeen", "true");
    }

    if (tourModal) {
        if (!localStorage.getItem("neuroscopeTourSeen")) {
            tourModal.style.display = "flex";
            renderTourStep();
        }
        
        btnTourClose.addEventListener("click", closeTour);
        
        btnTourNext.addEventListener("click", () => {
            if (currentTourStep < tourSteps.length - 1) {
                currentTourStep++;
                renderTourStep();
            } else {
                closeTour();
            }
        });
        
        btnTourBack.addEventListener("click", () => {
            if (currentTourStep > 0) {
                currentTourStep--;
                renderTourStep();
            }
        });
    }
});

// --- Custom Weight Upload ---
(function initUploadWeights() {
    const uploadDropzone = document.getElementById("uploadDropzone");
    const uploadBaseInput = document.getElementById("uploadBaseWeights");
    const uploadSaeInput = document.getElementById("uploadSaeWeights");
    const uploadBaseLabel = document.getElementById("uploadBaseLabel");
    const uploadSaeLabel = document.getElementById("uploadSaeLabel");
    const btnUploadWeights = document.getElementById("btnUploadWeights");
    const uploadStatus = document.getElementById("uploadStatus");
    const uploadSpinner = document.getElementById("uploadSpinnerEl");

    if (!btnUploadWeights) return; // guard if element missing

    function setFileLabel(labelEl, file) {
        if (labelEl) labelEl.textContent = file ? `✅ ${file.name}` : (labelEl.dataset.default || "Choose file…");
    }

    uploadBaseInput && uploadBaseInput.addEventListener("change", () => {
        setFileLabel(uploadBaseLabel, uploadBaseInput.files[0]);
    });
    uploadSaeInput && uploadSaeInput.addEventListener("change", () => {
        setFileLabel(uploadSaeLabel, uploadSaeInput.files[0]);
    });

    // Drag-over visual feedback on the entire dropzone
    if (uploadDropzone) {
        uploadDropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            uploadDropzone.classList.add("drag-over");
        });
        uploadDropzone.addEventListener("dragleave", () => uploadDropzone.classList.remove("drag-over"));
        uploadDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            uploadDropzone.classList.remove("drag-over");
            const ptFiles = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith(".pt"));
            if (ptFiles[0] && uploadBaseInput) {
                const dt = new DataTransfer(); dt.items.add(ptFiles[0]);
                uploadBaseInput.files = dt.files;
                setFileLabel(uploadBaseLabel, ptFiles[0]);
            }
            if (ptFiles[1] && uploadSaeInput) {
                const dt = new DataTransfer(); dt.items.add(ptFiles[1]);
                uploadSaeInput.files = dt.files;
                setFileLabel(uploadSaeLabel, ptFiles[1]);
            }
        });
    }

    function showUploadStatus(msg, type) {
        if (!uploadStatus) return;
        uploadStatus.textContent = msg;
        uploadStatus.className = `upload-status upload-status-${type}`;
        uploadStatus.style.display = "block";
    }

    btnUploadWeights.addEventListener("click", async () => {
        const baseFile = uploadBaseInput && uploadBaseInput.files[0];
        const saeFile = uploadSaeInput && uploadSaeInput.files[0];

        if (!baseFile && !saeFile) {
            showUploadStatus("Please select at least one .pt file.", "error");
            return;
        }

        btnUploadWeights.disabled = true;
        if (uploadSpinner) uploadSpinner.style.display = "inline-block";
        showUploadStatus("Uploading & validating weights…", "info");

        const formData = new FormData();
        if (baseFile) formData.append("base_weights", baseFile);
        if (saeFile) formData.append("sae_weights", saeFile);

        try {
            const res = await fetch("/api/upload_weights", { method: "POST", body: formData });
            const data = await res.json();

            if (res.ok) {
                showUploadStatus(data.message, "success");
                sendState();          // refresh inference with new weights
                fetchFeatureLogits(); // refresh heatmap
            } else {
                showUploadStatus(`❌ ${data.detail || "Upload failed."}`, "error");
            }
        } catch (err) {
            showUploadStatus(`❌ Network error: ${err.message}`, "error");
        } finally {
            btnUploadWeights.disabled = false;
            if (uploadSpinner) uploadSpinner.style.display = "none";
        }
    });
})();