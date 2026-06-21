import os
import io
import base64
import json
import logging
import asyncio
from typing import Optional

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from neuroscope_engine import NeuroEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuroScopeApp")

# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------
app = FastAPI(title="NeuroScope API", version="1.1.0")

# CORS origins are configurable via the ALLOWED_ORIGINS env var (comma
# separated). Defaults to "*" for local development / quick deploys, but you
# should set this explicitly in production, e.g.:
#   ALLOWED_ORIGINS=https://your-domain.com,https://www.your-domain.com
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = (
    ["*"] if _allowed_origins_env.strip() == "*"
    else [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
)
if ALLOWED_ORIGINS == ["*"]:
    logging.getLogger("NeuroScopeApp").warning(
        "CORS allow_origins is '*'. Set the ALLOWED_ORIGINS env var to a "
        "comma-separated allowlist before deploying to production."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Security headers on every response
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# ---------------------------------------------------------------------------
# Engine (single shared instance)
# ---------------------------------------------------------------------------
engine = NeuroEngine()
engine_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Max upload size: 50 MB  (PyTorch state-dicts for tiny models are <1 MB,
# but we give headroom for user experimentation)
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    epochs: int = Field(..., ge=1, le=500)

class DreamRequest(BaseModel):
    feature_idx: int = Field(..., ge=0, le=127)


# ---------------------------------------------------------------------------
# Health check (for uptime monitors / load balancers / platform health probes)
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {"status": "ok", "epoch": engine.epoch}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@app.post("/api/train")
async def train(req: TrainRequest):
    """Run N training epochs on the base model and SAE.

    Note: this holds engine_lock for the full duration of all N epochs
    (training itself runs in a worker thread via asyncio.to_thread so it
    doesn't block the event loop, but other lock-dependent endpoints —
    /ws inference, /api/metrics, /api/dream, weight upload — will queue
    behind it). At 500 epochs (the max) this is on the order of a couple
    of seconds on CPU, which is an acceptable bound; epochs is capped via
    the TrainRequest validator above specifically to keep this bounded.
    """
    def _do_train():
        last_res = None
        for _ in range(req.epochs):
            last_res = engine.train_step()
        return last_res

    async with engine_lock:
        last_res = await asyncio.to_thread(_do_train)

    return {
        "status": "success",
        "epoch": engine.epoch,
        "last_step": last_res,
        "metrics": {
            "loss": engine.metrics["loss"],
            "acc": engine.metrics["acc"],
            "sae_loss": engine.metrics["sae_loss"],
            "sparsity": engine.metrics["sparsity"],
        },
    }


# ---------------------------------------------------------------------------
# Feature inversion
# ---------------------------------------------------------------------------
@app.post("/api/dream")
async def dream(req: DreamRequest):
    """Gradient ascent to find the input that maximises a given SAE feature."""
    logger.info(f"Dreaming for feature {req.feature_idx}…")

    def _do_dream():
        return engine.feature_inversion(req.feature_idx)

    async with engine_lock:
        dream_coords = await asyncio.to_thread(_do_dream)

    return {
        "status": "success",
        "feature_idx": req.feature_idx,
        "dream_state": dream_coords.tolist(),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@app.get("/api/metrics")
async def get_metrics():
    """Current training metrics history."""
    # Acquire the lock before reading: train_step() runs in a worker thread
    # under this same lock and appends to these lists / reassigns cached
    # arrays, so reading without the lock risks observing a torn update.
    async with engine_lock:
        return {
            "epoch": engine.epoch,
            "metrics": {
                "loss": list(engine.metrics["loss"]),
                "acc": list(engine.metrics["acc"]),
                "sae_loss": list(engine.metrics["sae_loss"]),
                "sparsity": list(engine.metrics["sparsity"]),
            },
        }


# ---------------------------------------------------------------------------
# Feature-to-logit mapping
# ---------------------------------------------------------------------------
@app.get("/api/feature_logits")
async def get_feature_logits():
    async with engine_lock:
        return {"matrix": engine.raw_direct_path.tolist()}


# ---------------------------------------------------------------------------
# Custom weight upload  (NEW)
# ---------------------------------------------------------------------------
@app.post("/api/upload_weights")
async def upload_weights(
    base_weights: Optional[UploadFile] = File(None),
    sae_weights: Optional[UploadFile] = File(None),
):
    """
    Upload custom .pt weight files for the base model and/or the SAE.

    The endpoint validates that the uploaded state-dict matches the exact
    architecture before touching the live engine, so a bad file cannot
    corrupt the running model.
    """
    if base_weights is None and sae_weights is None:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Read bytes (with size cap)
    base_bytes: Optional[bytes] = None
    sae_bytes: Optional[bytes] = None

    if base_weights is not None:
        if base_weights.filename and not base_weights.filename.lower().endswith(".pt"):
            raise HTTPException(
                status_code=422,
                detail=f"Base weights file must be a .pt file (got '{base_weights.filename}').",
            )
        raw = await base_weights.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Base weights file too large (max 50 MB).")
        if len(raw) == 0:
            raise HTTPException(status_code=422, detail="Base weights file is empty.")
        base_bytes = raw

    if sae_weights is not None:
        if sae_weights.filename and not sae_weights.filename.lower().endswith(".pt"):
            raise HTTPException(
                status_code=422,
                detail=f"SAE weights file must be a .pt file (got '{sae_weights.filename}').",
            )
        raw = await sae_weights.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="SAE weights file too large (max 50 MB).")
        if len(raw) == 0:
            raise HTTPException(status_code=422, detail="SAE weights file is empty.")
        sae_bytes = raw

    try:
        async with engine_lock:
            message = await asyncio.to_thread(
                engine.load_weights_from_bytes, base_bytes, sae_bytes
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {"status": "success", "message": message}


# ---------------------------------------------------------------------------
# Export Jupyter notebook
# ---------------------------------------------------------------------------
@app.get("/api/export_notebook")
async def export_notebook():
    """Generate a Jupyter Notebook with model code + current weights."""
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    def add_markdown(source: str):
        nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [source]})

    def add_code(source: str):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [source],
        })

    add_markdown("# NeuroScope Interpretability Notebook\n\nExported from NeuroScope v1.1")

    add_code(
        "import torch\n"
        "import torch.nn as nn\n"
        "import base64, io\n"
        "from typing import Dict, Tuple, Optional\n\n"
        "class CoordinateCatcher(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(4, 32)\n"
        "        self.relu = nn.ReLU()\n"
        "        self.fc2 = nn.Linear(32, 4)\n"
        "    def forward(self, x, intervene_neurons=None, sae=None, intervene_features=None):\n"
        "        h = self.relu(self.fc1(x))\n"
        "        if intervene_neurons:\n"
        "            mask = torch.ones_like(h)\n"
        "            for n, v in intervene_neurons.items(): mask[:, int(n)] = v\n"
        "            h = h * mask\n"
        "        if sae is not None:\n"
        "            h, _ = sae(h, intervene_features=intervene_features)\n"
        "        return self.fc2(h), h\n\n"
        "class SparseAutoencoder(nn.Module):\n"
        "    def __init__(self, input_dim=32, hidden_dim=128):\n"
        "        super().__init__()\n"
        "        self.encoder = nn.Linear(input_dim, hidden_dim)\n"
        "        self.relu = nn.ReLU()\n"
        "        self.decoder = nn.Linear(hidden_dim, input_dim)\n"
        "    def forward(self, x, intervene_features=None):\n"
        "        enc = self.relu(self.encoder(x))\n"
        "        if intervene_features:\n"
        "            mask = torch.ones_like(enc)\n"
        "            for f, v in intervene_features.items(): mask[:, int(f)] = v\n"
        "            enc = enc * mask\n"
        "        return self.decoder(enc), enc\n"
        "    def encode(self, x): return self.relu(self.encoder(x))\n"
    )

    # Embed weights as base64 — write to a separate cell so the code cell stays readable.
    # Hold the lock while reading state_dict(): train_step() mutates these
    # parameters in a worker thread under the same lock, and PyTorch does not
    # guarantee state_dict() is atomic across all parameters mid-optimizer-step.
    async with engine_lock:
        base_buf = io.BytesIO()
        torch.save(engine.model.state_dict(), base_buf)
        base_b64 = base64.b64encode(base_buf.getvalue()).decode("utf-8")

        sae_buf = io.BytesIO()
        torch.save(engine.sae.state_dict(), sae_buf)
        sae_b64 = base64.b64encode(sae_buf.getvalue()).decode("utf-8")

    add_markdown("## Load Weights")
    add_code(
        f"BASE_B64 = '{base_b64}'\n"
        f"SAE_B64  = '{sae_b64}'\n\n"
        "model = CoordinateCatcher()\n"
        "model.load_state_dict(torch.load(io.BytesIO(base64.b64decode(BASE_B64)), weights_only=True))\n"
        "model.eval()\n\n"
        "sae = SparseAutoencoder()\n"
        "sae.load_state_dict(torch.load(io.BytesIO(base64.b64decode(SAE_B64)), weights_only=True))\n"
        "sae.eval()\n"
        "print('Models loaded!')\n"
    )

    add_markdown("## Quick Inference Example")
    add_code(
        "import torch\n"
        "x = torch.tensor([[-0.5, 0.0, 0.5, 0.0]])  # [agent_x, agent_y, target_x, target_y]\n"
        "with torch.no_grad():\n"
        "    out, h = model(x, sae=sae)\n"
        "    feats = sae.encode(h)\n"
        "    probs = torch.softmax(out, dim=1)\n"
        "dirs = ['Up', 'Down', 'Left', 'Right']\n"
        "print('Prediction:', dirs[probs.argmax().item()])\n"
        "print('Probabilities:', {d: f'{p:.3f}' for d, p in zip(dirs, probs[0].tolist())})\n"
        "print('Active SAE features:', [(i, f'{v:.3f}') for i, v in enumerate(feats[0].tolist()) if v > 0.1])\n"
    )

    return nb


# ---------------------------------------------------------------------------
# WebSocket — real-time inference
# ---------------------------------------------------------------------------
def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    try:
        while True:
            data = await websocket.receive_text()

            # Each message is handled independently: a single malformed or
            # out-of-range message sends an {"error": ...} reply and the
            # loop continues, instead of tearing down the whole connection.
            try:
                payload = json.loads(data)
                if not isinstance(payload, dict):
                    raise ValueError("Payload must be a JSON object.")

                # Coordinates are clamped to the [-1, 1] domain the model was
                # trained on. Without this, a buggy/malicious client could
                # send extreme values (e.g. 1e300) straight into the model.
                state = {
                    "agent_x":  _clamp(float(payload.get("agent_x", 0.0)), -1.0, 1.0),
                    "agent_y":  _clamp(float(payload.get("agent_y", 0.0)), -1.0, 1.0),
                    "target_x": _clamp(float(payload.get("target_x", 0.0)), -1.0, 1.0),
                    "target_y": _clamp(float(payload.get("target_y", 0.0)), -1.0, 1.0),
                }

                intervene_neurons = payload.get("intervene_neurons")
                if intervene_neurons is not None:
                    if not isinstance(intervene_neurons, dict):
                        raise ValueError("intervene_neurons must be an object.")
                    intervene_neurons = {int(k): float(v) for k, v in intervene_neurons.items()}

                intervene_features = payload.get("intervene_features")
                if intervene_features is not None:
                    if not isinstance(intervene_features, dict):
                        raise ValueError("intervene_features must be an object.")
                    intervene_features = {int(k): float(v) for k, v in intervene_features.items()}

                # top_k drives how many Sankey nodes are rendered; bound it
                # to sane values (the SAE has 128 features / model has 32
                # neurons, so anything beyond that is meaningless and an
                # unbounded value could make the Sankey payload huge).
                top_k = int(payload.get("top_k", 3))
                top_k = int(_clamp(top_k, 1, 32))

            except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as e:
                await websocket.send_text(json.dumps({"error": f"Invalid request: {e}"}))
                continue

            try:
                # Run the actual (synchronous, CPU-bound) inference work in a
                # worker thread while holding the lock. Previously this ran
                # directly on the event loop, which meant one client dragging
                # the agent would block every other client's WebSocket
                # messages and all HTTP requests for the duration of each
                # inference + Sankey + narrative computation.
                def _do_inference():
                    inf_res = engine.inference(
                        state,
                        intervene_neurons=intervene_neurons,
                        intervene_features=intervene_features,
                    )
                    nodes, links = engine.calculate_sankey_data(state, inf_res, top_k=top_k)
                    narrative = engine.generate_narrative(state, inf_res)
                    return inf_res, nodes, links, narrative

                async with engine_lock:
                    inf_res, nodes, links, narrative = await asyncio.to_thread(_do_inference)

                await websocket.send_text(json.dumps({
                    "prediction": inf_res["prediction"],
                    "probs": inf_res["probs"],
                    "hidden": inf_res["hidden"],
                    "features": inf_res["features"],
                    "nodes": nodes,
                    "links": links,
                    "narrative": narrative,
                }))
            except (ValueError, IndexError, RuntimeError) as e:
                # e.g. an out-of-range neuron/feature index raised by the engine
                logger.warning(f"WebSocket inference rejected: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"error": "Internal server error."}))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Static files  (must be last — catches everything not matched above)
# ---------------------------------------------------------------------------
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")