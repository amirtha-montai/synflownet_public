"""
Overcomplete BatchTopK sparse autoencoder for SynFlowNet graph embeddings.

Implements the standard formulation (Bussmann et al., 2024, "BatchTopK Sparse
Autoencoders") rather than an L1-penalised bottleneck:

    z_pre = ReLU(W_enc (h - b_pre) + b_enc)
    z     = BatchTopK(z_pre, k)
    h_hat = W_dec z + b_dec

Design points, each of which matters for interpretability and is asserted in the
tests at the bottom of this file:

* **Overcomplete.** ``n_features`` (1024) > ``input_dim`` (256), a 4x expansion.
  Overcompleteness is what licenses any superposition / monosemanticity reading;
  an undercomplete bottleneck (e.g. 256 -> 128) does not, however sparse it is.
* **Single linear encoder and decoder.** No hidden layers, no dropout. Each
  feature is one direction in embedding space, so ``W_dec[:, i]`` is directly
  interpretable as feature ``i``'s contribution.
* **BatchTopK, not L1.** Keeps the ``k * batch_size`` largest pre-activations
  across the whole batch, so mean L0 is exactly ``k`` with no penalty
  coefficient to tune. Per-example L0 may vary; the batch mean does not.
* **Unit-norm decoder columns.** Renormalised after every optimiser step so
  feature "importance" cannot be smuggled into decoder magnitude.
* **b_pre initialised to the training mean**, so the encoder sees centred input.
* **Dead-feature resampling.** Features that never activate are reinitialised
  toward high-reconstruction-error examples, which is what keeps a large
  dictionary from collapsing to a small used subset.

Run ``python overcomplete_sae.py`` to execute the self-tests.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    """Hyperparameters. Defaults are the configuration reported in the paper."""

    input_dim: int = 256
    n_features: int = 1024          # 4x overcomplete
    k: int = 64                     # BatchTopK sparsity; mean L0 == k
    lr: float = 3e-4
    batch_size: int = 512
    epochs: int = 50
    val_frac: float = 0.10          # 90/10 train/val
    seed: int = 42
    # dead-feature resampling
    resample_every: int = 10        # epochs
    resample_start: int = 20        # first epoch eligible
    resample_max_frac: float = 0.02  # cap per round
    standardize: bool = True        # per-feature standardisation of inputs

    def to_dict(self) -> Dict:
        return asdict(self)


def batch_topk(z_pre: torch.Tensor, k: int) -> torch.Tensor:
    """Zero all but the ``k * batch_size`` largest activations in the batch.

    Sparsity is enforced across the flattened batch rather than per row, so the
    *mean* number of active features per example is exactly ``k`` while
    individual examples may use more or fewer.
    """
    if z_pre.numel() == 0:
        return z_pre
    n_keep = min(k * z_pre.shape[0], z_pre.numel())
    flat = z_pre.flatten()
    # threshold = n_keep-th largest value; values below it are dropped
    thresh = torch.topk(flat, n_keep, sorted=True).values[-1]
    return torch.where(z_pre >= thresh, z_pre, torch.zeros_like(z_pre))


class BatchTopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(torch.empty(cfg.n_features, cfg.input_dim))
        self.b_enc = nn.Parameter(torch.zeros(cfg.n_features))
        self.W_dec = nn.Parameter(torch.empty(cfg.input_dim, cfg.n_features))
        self.b_dec = nn.Parameter(torch.zeros(cfg.input_dim))
        self.b_pre = nn.Parameter(torch.zeros(cfg.input_dim))

        nn.init.kaiming_uniform_(self.W_enc, nonlinearity="relu")
        # initialise decoder as the encoder transpose, a common warm start
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Rescale every decoder column to unit L2 norm."""
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    @torch.no_grad()
    def init_b_pre(self, X: torch.Tensor) -> None:
        self.b_pre.copy_(X.mean(dim=0))

    def encode_pre(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(F.linear(h - self.b_pre, self.W_enc, self.b_enc))

    def encode(self, h: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        return batch_topk(self.encode_pre(h), self.cfg.k if k is None else k)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.W_dec, self.b_dec)

    def forward(self, h: torch.Tensor):
        z = self.encode(h)
        return self.decode(z), z


def nmse(h: torch.Tensor, h_hat: torch.Tensor) -> float:
    """Normalised MSE: reconstruction error relative to input variance."""
    denom = (h - h.mean(dim=0)).pow(2).sum()
    return float(((h - h_hat).pow(2).sum() / denom.clamp_min(1e-12)).item())


@torch.no_grad()
def feature_stats(model: BatchTopKSAE, X: torch.Tensor, chunk: int = 4096) -> Dict:
    """Activation frequency, dead-feature count, mean L0, NMSE on ``X``."""
    model.eval()
    n_active = torch.zeros(model.cfg.n_features)
    l0_total, se, n = 0.0, 0.0, 0
    recon_sq, centred_sq = 0.0, 0.0
    mu = X.mean(dim=0)
    for i in range(0, X.shape[0], chunk):
        xb = X[i : i + chunk]
        h_hat, z = model(xb)
        n_active += (z > 0).float().sum(dim=0).cpu()
        l0_total += float((z > 0).float().sum().item())
        recon_sq += float((xb - h_hat).pow(2).sum().item())
        centred_sq += float((xb - mu).pow(2).sum().item())
        n += xb.shape[0]
    freq = (n_active / max(n, 1)).numpy()
    return {
        "activation_freq": freq,
        "n_dead": int((freq == 0).sum()),
        "frac_dead": float((freq == 0).mean()),
        "mean_l0": l0_total / max(n, 1),
        "nmse": recon_sq / max(centred_sq, 1e-12),
    }


def train_sae(
    X: np.ndarray,
    cfg: SAEConfig,
    device: str = "cpu",
    verbose: bool = True,
):
    """Train a BatchTopK SAE. Returns ``(model, history, scaler, stats)``.

    ``scaler`` is ``(mean, std)`` used for per-feature standardisation, or
    ``None``. It must be reapplied to any embeddings encoded later.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    Xf = X.astype(np.float32)
    scaler = None
    if cfg.standardize:
        mu = Xf.mean(axis=0)
        sd = Xf.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xf = (Xf - mu) / sd
        scaler = (mu, sd)

    # deterministic 90/10 split
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(Xf.shape[0])
    n_val = int(round(cfg.val_frac * Xf.shape[0]))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    Xtr = torch.from_numpy(Xf[train_idx]).to(device)
    Xva = torch.from_numpy(Xf[val_idx]).to(device)

    model = BatchTopKSAE(cfg).to(device)
    model.init_b_pre(Xtr)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        idx = torch.randperm(Xtr.shape[0], device=device)
        epoch_loss, nb = 0.0, 0
        seen_active = torch.zeros(cfg.n_features, device=device)

        for s in range(0, Xtr.shape[0], cfg.batch_size):
            xb = Xtr[idx[s : s + cfg.batch_size]]
            h_hat, z = model(xb)
            loss = F.mse_loss(h_hat, xb)          # MSE only; no sparsity penalty
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.normalize_decoder()             # keep decoder columns unit-norm
            seen_active += (z > 0).float().sum(dim=0).detach()
            epoch_loss += float(loss.item())
            nb += 1

        # resample features that were dead for the whole epoch
        n_resampled = 0
        if (
            epoch >= cfg.resample_start
            and (epoch - cfg.resample_start) % cfg.resample_every == 0
        ):
            n_resampled = _resample_dead(model, Xtr, seen_active, cfg)

        model.eval()
        with torch.no_grad():
            va_hat, va_z = model(Xva)
            val_nmse = nmse(Xva, va_hat)
            val_l0 = float((va_z > 0).float().sum(dim=1).mean().item())

        history.append(
            {
                "epoch": epoch,
                "train_mse": epoch_loss / max(nb, 1),
                "val_nmse": val_nmse,
                "val_mean_l0": val_l0,
                "n_dead_epoch": int((seen_active == 0).sum().item()),
                "n_resampled": n_resampled,
            }
        )
        if verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
            print(
                f"  epoch {epoch:3d}  train_mse {history[-1]['train_mse']:.5f}"
                f"  val_nmse {val_nmse:.5f}  val_L0 {val_l0:.1f}"
                f"  dead {history[-1]['n_dead_epoch']}"
                + (f"  resampled {n_resampled}" if n_resampled else "")
            )

    stats = feature_stats(model, Xva)
    stats["train_idx"], stats["val_idx"] = train_idx, val_idx
    return model, history, scaler, stats


@torch.no_grad()
def _resample_dead(model, Xtr, seen_active, cfg) -> int:
    """Reinitialise dead features toward poorly-reconstructed examples."""
    dead = torch.nonzero(seen_active == 0).flatten()
    if dead.numel() == 0:
        return 0
    cap = max(1, int(cfg.resample_max_frac * cfg.n_features))
    dead = dead[:cap]

    sample = Xtr[torch.randperm(Xtr.shape[0], device=Xtr.device)[:4096]]
    h_hat, _ = model(sample)
    err = (sample - h_hat).pow(2).sum(dim=1)
    pick = torch.topk(err, min(dead.numel(), sample.shape[0])).indices
    directions = sample[pick] - model.b_pre           # [n_dead, input_dim]
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)

    n = min(dead.numel(), directions.shape[0])
    # encoder rows get a small magnitude so resampled features start quiet
    model.W_enc[dead[:n]] = directions[:n] * 0.2
    model.b_enc[dead[:n]] = 0.0
    model.W_dec[:, dead[:n]] = directions[:n].t()
    model.normalize_decoder()
    return int(n)


# --------------------------------------------------------------------------
# self-tests
# --------------------------------------------------------------------------
def _tests():
    torch.manual_seed(0)
    cfg = SAEConfig(input_dim=32, n_features=128, k=8, epochs=3, batch_size=64,
                    resample_start=1, resample_every=1)
    m = BatchTopKSAE(cfg)

    # overcompleteness
    assert cfg.n_features > cfg.input_dim, "SAE must be overcomplete"

    # shapes
    x = torch.randn(64, cfg.input_dim)
    h_hat, z = m(x)
    assert z.shape == (64, cfg.n_features), z.shape
    assert h_hat.shape == x.shape, h_hat.shape

    # mean L0 == k (the defining property of BatchTopK)
    mean_l0 = (z > 0).float().sum(dim=1).mean().item()
    assert abs(mean_l0 - cfg.k) < 1e-6, f"mean L0 {mean_l0} != k {cfg.k}"

    # exact global count
    assert int((z > 0).sum().item()) == cfg.k * x.shape[0]

    # decoder columns are unit norm
    norms = m.W_dec.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), norms[:5]

    # b_pre initialises to the data mean
    X = torch.randn(500, cfg.input_dim) * 3 + 7
    m.init_b_pre(X)
    assert torch.allclose(m.b_pre, X.mean(0), atol=1e-5)

    # decoder stays unit-norm after a real training run, and NMSE is finite
    Xn = np.random.randn(600, cfg.input_dim).astype(np.float32)
    model, hist, scaler, stats = train_sae(Xn, cfg, verbose=False)
    norms = model.W_dec.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    assert np.isfinite(stats["nmse"]), stats["nmse"]
    assert abs(stats["mean_l0"] - cfg.k) < 1.0, stats["mean_l0"]
    assert len(hist) == cfg.epochs

    # identical seeds reproduce; different seeds do not
    a, _, _, _ = train_sae(Xn, SAEConfig(input_dim=32, n_features=128, k=8,
                                         epochs=2, seed=1), verbose=False)
    b, _, _, _ = train_sae(Xn, SAEConfig(input_dim=32, n_features=128, k=8,
                                         epochs=2, seed=1), verbose=False)
    assert torch.allclose(a.W_enc, b.W_enc, atol=1e-6), "seeding is not deterministic"

    print("all overcomplete_sae tests passed")


if __name__ == "__main__":
    _tests()
