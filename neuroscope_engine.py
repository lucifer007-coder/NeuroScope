import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import io
import logging
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger("NeuroScopeEngine")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Hard cap on how many metric data-points to keep in memory (prevents OOM on long runs)
MAX_METRIC_HISTORY = 2000

# Hard cap on how many periodic checkpoint files to keep on disk (prevents
# unbounded disk growth on long-running deployments — older checkpoints are
# pruned, keeping only the most recent N for both base model and SAE).
MAX_CHECKPOINTS_TO_KEEP = 10


def generate_coordinate_dataset(num_samples=1000):
    inputs = torch.rand(num_samples, 4) * 2 - 1
    agent_x, agent_y, target_x, target_y = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    dx = target_x - agent_x
    dy = target_y - agent_y
    labels = torch.zeros(num_samples, dtype=torch.long)
    abs_dx, abs_dy = torch.abs(dx), torch.abs(dy)
    horiz_mask = abs_dx > abs_dy
    labels[horiz_mask & (dx > 0)] = 3   # Right
    labels[horiz_mask & (dx <= 0)] = 2  # Left
    vert_mask = ~horiz_mask
    labels[vert_mask & (dy > 0)] = 0    # Up
    labels[vert_mask & (dy <= 0)] = 1   # Down
    return inputs, labels


class CoordinateCatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 4)

    def forward(
        self,
        x: torch.Tensor,
        intervene_neurons: Optional[Dict[int, float]] = None,
        sae: Optional[nn.Module] = None,
        intervene_features: Optional[Dict[int, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.relu(self.fc1(x))
        if intervene_neurons is not None:
            mask = torch.ones_like(h)
            num_neurons = h.shape[1]
            for n_idx, val in intervene_neurons.items():
                n_idx = int(n_idx)
                # Reject negative/out-of-range indices instead of silently
                # wrapping (Python negative indexing) or raising an opaque
                # IndexError deep inside a tensor op.
                if not (0 <= n_idx < num_neurons):
                    raise ValueError(
                        f"intervene_neurons index {n_idx} out of range "
                        f"[0, {num_neurons - 1}]"
                    )
                mask[:, n_idx] = val
            h = h * mask
        if sae is not None:
            h_decoded, _ = sae(h, intervene_features=intervene_features)
            h = h_decoded
        out = self.fc2(h)
        return out, h


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        intervene_features: Optional[Dict[int, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.relu(self.encoder(x))
        if intervene_features is not None:
            mask = torch.ones_like(encoded)
            num_features = encoded.shape[1]
            for f_idx, val in intervene_features.items():
                f_idx = int(f_idx)
                if not (0 <= f_idx < num_features):
                    raise ValueError(
                        f"intervene_features index {f_idx} out of range "
                        f"[0, {num_features - 1}]"
                    )
                mask[:, f_idx] = val
            encoded = encoded * mask
        decoded = self.decoder(encoded)
        return decoded, encoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.encoder(x))


def _validate_state_dict(state_dict: dict, model: nn.Module, name: str) -> None:
    """Raise ValueError with a human-readable message if shapes don't match."""
    ref_sd = model.state_dict()
    missing = set(ref_sd.keys()) - set(state_dict.keys())
    extra = set(state_dict.keys()) - set(ref_sd.keys())
    if missing or extra:
        raise ValueError(
            f"{name} key mismatch — missing: {missing or '∅'}, unexpected: {extra or '∅'}"
        )
    for key in ref_sd:
        if state_dict[key].shape != ref_sd[key].shape:
            raise ValueError(
                f"{name} shape mismatch at '{key}': "
                f"expected {tuple(ref_sd[key].shape)}, got {tuple(state_dict[key].shape)}"
            )


class NeuroEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = CoordinateCatcher().to(self.device)
        self.sae = SparseAutoencoder().to(self.device)

        # Training state
        self.optimizer_base = optim.Adam(self.model.parameters(), lr=0.01)
        self.optimizer_sae = optim.Adam(self.sae.parameters(), lr=0.005)
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.epoch = 0
        self.metrics: Dict[str, List[float]] = {
            "loss": [], "acc": [], "sae_loss": [], "sparsity": []
        }

        # Try loading persisted weights
        self.load_models()
        self._cache_weights()

    def _cache_weights(self):
        self.cached_fc1_w = self.model.fc1.weight.data.abs().cpu().numpy()
        self.cached_decoder_w = self.sae.decoder.weight.data.abs().cpu().numpy()
        self.cached_fc2_w = self.model.fc2.weight.data.abs().cpu().numpy()
        self.cached_direct_path = self.cached_fc2_w @ self.cached_decoder_w

        self.raw_decoder_w = self.sae.decoder.weight.data
        self.raw_fc2_w = self.model.fc2.weight.data
        self.raw_direct_path = self.raw_fc2_w @ self.raw_decoder_w

    def load_models(self):
        if os.path.exists("models/base_live.pt"):
            try:
                self.model.load_state_dict(
                    torch.load("models/base_live.pt", weights_only=True)
                )
                logger.info("Loaded base model from disk.")
            except Exception as e:
                logger.warning(f"Could not load base model: {e}")
        if os.path.exists("models/sae_live.pt"):
            try:
                self.sae.load_state_dict(
                    torch.load("models/sae_live.pt", weights_only=True)
                )
                logger.info("Loaded SAE from disk.")
            except Exception as e:
                logger.warning(f"Could not load SAE: {e}")

    # ------------------------------------------------------------------
    # Custom weight upload
    # ------------------------------------------------------------------
    def load_weights_from_bytes(self, base_bytes: Optional[bytes], sae_bytes: Optional[bytes]) -> str:
        """
        Load user-supplied .pt weight bytes into the engine.
        Validates architecture match before touching live models.
        Returns a human-readable status string.
        """
        messages = []

        if base_bytes is not None:
            try:
                sd = torch.load(io.BytesIO(base_bytes), weights_only=True, map_location="cpu")
                _validate_state_dict(sd, self.model, "Base model")
                self.model.load_state_dict(sd)
                self.optimizer_base = optim.Adam(self.model.parameters(), lr=0.01)
                messages.append("✅ Base model weights loaded successfully.")
                logger.info("User-uploaded base model weights loaded.")
            except Exception as e:
                raise ValueError(f"Base model upload failed: {e}")

        if sae_bytes is not None:
            try:
                sd = torch.load(io.BytesIO(sae_bytes), weights_only=True, map_location="cpu")
                _validate_state_dict(sd, self.sae, "SAE")
                self.sae.load_state_dict(sd)
                self.optimizer_sae = optim.Adam(self.sae.parameters(), lr=0.005)
                messages.append("✅ SAE weights loaded successfully.")
                logger.info("User-uploaded SAE weights loaded.")
            except Exception as e:
                raise ValueError(f"SAE upload failed: {e}")

        self._cache_weights()
        return " ".join(messages) if messages else "No files uploaded."

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _trim_metrics(self):
        """Keep metrics lists bounded to avoid unbounded memory growth."""
        for key in self.metrics:
            if len(self.metrics[key]) > MAX_METRIC_HISTORY:
                self.metrics[key] = self.metrics[key][-MAX_METRIC_HISTORY:]

    def _prune_old_checkpoints(self):
        """Delete old periodic checkpoint files, keeping only the most
        recent MAX_CHECKPOINTS_TO_KEEP for each of base/SAE. Without this,
        a long-running deployment accumulates a checkpoint pair every 10
        epochs forever and eventually fills the disk."""
        for prefix in ("base_ep", "sae_ep"):
            try:
                files = [f for f in os.listdir("checkpoints") if f.startswith(prefix)]
            except FileNotFoundError:
                continue

            def _epoch_of(fname: str) -> int:
                try:
                    return int(fname[len(prefix):-3])  # strip prefix and ".pt"
                except ValueError:
                    return -1

            files.sort(key=_epoch_of)
            excess = len(files) - MAX_CHECKPOINTS_TO_KEEP
            for fname in files[:max(excess, 0)]:
                try:
                    os.remove(os.path.join("checkpoints", fname))
                except OSError as e:
                    logger.warning(f"Could not prune checkpoint {fname}: {e}")

    def train_step(self):
        # 1. Train Base Model
        X, y = generate_coordinate_dataset(1024)
        X, y = X.to(self.device), y.to(self.device)

        self.optimizer_base.zero_grad()
        out, hidden = self.model(X)
        loss = self.criterion(out, y)
        loss.backward()
        self.optimizer_base.step()

        acc = (out.argmax(dim=1) == y).float().mean().item()
        self.metrics["loss"].append(loss.item())
        self.metrics["acc"].append(acc)

        # 2. Train SAE on the hidden activations
        hidden_detached = hidden.detach()
        self.optimizer_sae.zero_grad()

        reconstructed, encoded = self.sae(hidden_detached)
        mse = self.mse_loss(reconstructed, hidden_detached)
        l1 = torch.norm(encoded, 1, dim=1).mean()

        # NOTE: this coefficient was previously 0.05, which is far too strong
        # relative to the reconstruction term for a 32->128 SAE — it drove
        # every one of the 128 latent features to zero (100% "dead") within
        # ~20 training epochs. Once every feature is dead, the Feature
        # Observatory has nothing to show, the Decision Backtrace narrative
        # has no features to cite, and feature_inversion ("dreaming") can
        # never move away from its zero-initialized input because the
        # gradient of a permanently-zero ReLU unit is itself always zero.
        # 0.01 keeps a healthy sparsity level (~90-96% of features inactive
        # per sample) while leaving the task-relevant features alive and
        # steerable.
        l1_lambda = 0.01
        sae_loss = mse + l1_lambda * l1
        sae_loss.backward()
        self.optimizer_sae.step()

        self.metrics["sae_loss"].append(sae_loss.item())
        self.metrics["sparsity"].append((encoded <= 0.1).float().mean().item())

        self.epoch += 1
        self._trim_metrics()

        # Save checkpoint occasionally
        if self.epoch % 10 == 0:
            torch.save(self.model.state_dict(), f"checkpoints/base_ep{self.epoch}.pt")
            torch.save(self.sae.state_dict(), f"checkpoints/sae_ep{self.epoch}.pt")
            torch.save(self.model.state_dict(), "models/base_live.pt")
            torch.save(self.sae.state_dict(), "models/sae_live.pt")
            self._prune_old_checkpoints()

        self._cache_weights()

        return {
            "epoch": self.epoch,
            "loss": loss.item(),
            "acc": acc,
            "sae_loss": sae_loss.item(),
        }

    # ------------------------------------------------------------------
    # Feature inversion / dreaming
    # ------------------------------------------------------------------
    def feature_inversion(self, feature_idx: int, steps: int = 100) -> np.ndarray:
        x = torch.zeros((1, 4), requires_grad=True, device=self.device)
        opt = optim.Adam([x], lr=0.1)

        for _ in range(steps):
            opt.zero_grad()
            _, h = self.model(x)
            feats = self.sae.encode(h)
            l2_penalty = 0.05 * (x ** 2).sum()
            loss = -feats[0, feature_idx] + l2_penalty
            loss.backward()
            opt.step()
            x.data.clamp_(-1, 1)

        return x.detach().cpu().numpy()[0]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference(
        self,
        state_dict: Dict[str, float],
        intervene_neurons: Optional[Dict[int, float]] = None,
        intervene_features: Optional[Dict[int, float]] = None,
    ) -> Dict:
        x = torch.tensor(
            [[
                state_dict["agent_x"], state_dict["agent_y"],
                state_dict["target_x"], state_dict["target_y"],
            ]],
            dtype=torch.float32,
            device=self.device,
        )

        out, h = self.model(
            x,
            intervene_neurons=intervene_neurons,
            sae=self.sae,
            intervene_features=intervene_features,
        )
        feats = self.sae.encode(h)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        pred_idx = out.argmax(dim=1).item()

        dirs = ["Up", "Down", "Left", "Right"]
        return {
            "prediction": dirs[pred_idx],
            "probs": probs.tolist(),
            "hidden": h[0].cpu().numpy().tolist(),
            "features": feats[0].cpu().numpy().tolist(),
        }

    # ------------------------------------------------------------------
    # Sankey data
    # ------------------------------------------------------------------
    def calculate_sankey_data(
        self, state: Dict[str, float], inference_res: Dict, top_k: int = 3
    ) -> Tuple[Dict, List[Dict]]:
        hidden = np.array(inference_res["hidden"])
        features = np.array(inference_res["features"])
        probs = np.array(inference_res["probs"])

        top_neurons = np.argsort(hidden)[::-1][:top_k]
        top_features = np.argsort(features)[::-1][:top_k]

        nodes = {
            "inputs": ["Agent X", "Agent Y", "Target X", "Target Y"],
            "neurons": [f"Neuron {i}" for i in top_neurons],
            "features": [f"Feature {i}" for i in top_features],
            "outputs": ["Up", "Down", "Left", "Right"],
        }

        links = []
        for n_idx in top_neurons:
            for inp_idx in range(4):
                weight = self.cached_fc1_w[n_idx, inp_idx]
                if weight > 0.01:
                    links.append({
                        "source": f"input_{inp_idx}",
                        "target": f"neuron_{n_idx}",
                        "value": float(weight * hidden[n_idx]),
                    })

        for f_idx in top_features:
            for n_idx in top_neurons:
                weight = self.cached_decoder_w[n_idx, f_idx]
                if weight > 0.01:
                    links.append({
                        "source": f"neuron_{n_idx}",
                        "target": f"feature_{f_idx}",
                        "value": float(weight * features[f_idx]),
                    })

        for f_idx in top_features:
            for out_idx in range(4):
                weight = self.cached_direct_path[out_idx, f_idx]
                if weight > 0.01:
                    links.append({
                        "source": f"feature_{f_idx}",
                        "target": f"output_{out_idx}",
                        "value": float(weight * probs[out_idx]),
                    })

        return nodes, links

    # ------------------------------------------------------------------
    # Narrative
    # ------------------------------------------------------------------
    def generate_narrative(self, state: Dict[str, float], inference_res: Dict) -> str:
        feats = np.array(inference_res["features"])
        top_idx = np.argsort(feats)[::-1][:3]

        dx = state["target_x"] - state["agent_x"]
        dy = state["target_y"] - state["agent_y"]

        nar = f"The model chose **{inference_res['prediction']}**.\n\n"
        nar += (
            f"**Game State:** The target is {dx:.2f} units horizontally "
            f"and {dy:.2f} units vertically from the agent.\n\n"
        )
        nar += "**Causal Trace (SAE Features):**\n"

        for idx in top_idx:
            val = feats[idx]
            if val > 0.1:
                impact = self.raw_direct_path[:, idx]
                favored_dir = ["Up", "Down", "Left", "Right"][impact.argmax().item()]
                nar += f"- **Feature #{idx}** (Activates at {val:.2f}): Strongly votes for **{favored_dir}**.\n"

        return nar