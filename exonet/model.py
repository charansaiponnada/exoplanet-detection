"""PHANTOM and the baseline networks.

PHANTOM (Physics-informed Harmonic-ATtention Object Model) has three parts:

``ViewEncoder``
    A residual 1-D convolutional trunk that turns each phase-folded view into a
    single token.  Views of the same length share an architecture but not
    weights, because a centroid view and a flux view mean different things.

``HarmonicAttention``
    A transformer encoder over the view tokens.  Each token carries a *period
    hypothesis* embedding (P/2, P, 2P) in addition to its view-type embedding,
    and an explicit harmonic-contrast head compares the pooled evidence under
    the three hypotheses.  This is what lets the network reason about whether
    the reported period is the true period or a harmonic of it - the dominant
    failure mode of classical transit vetting.

``TransitDecoder``
    A differentiable forward model of a transit.  A five-parameter physical
    bottleneck (depth, half-duration, phase offset, limb-darkening, ingress
    softness) is rendered onto the local-view grid and compared with the
    observed view.  The renderer is analytic and differentiable, so the physical
    parameters are learned by reconstruction rather than supervised regression,
    and the reconstruction residual becomes a classification feature: a genuine
    transit is well described by the model, a stellar eclipse or a systematic is
    not.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Which views exist, their length, and which period hypothesis they test.
# hypothesis: 0 -> P/2, 1 -> P, 2 -> 2P
VIEW_TABLE = {
    "global":      (2001, 1),
    "local":       (201,  1),
    "odd":         (201,  1),
    "even":        (201,  1),
    "secondary":   (201,  1),
    "half":        (201,  0),
    "double":      (2001, 2),
    "cent_global": (2001, 1),
    "cent_local":  (201,  1),
}
N_HYPOTHESES = 3


class ResBlock1d(nn.Module):
    """Pre-activation residual block with GroupNorm (batch-size independent)."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.norm1 = nn.GroupNorm(min(8, c_in), c_in)
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=pad)
        self.norm2 = nn.GroupNorm(min(8, c_out), c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.relu(self.norm1(x)))
        h = self.conv2(F.relu(self.norm2(h)))
        return h + self.skip(x)


class ViewEncoder(nn.Module):
    """Encode one 1-D view into a token of dimension ``d_model``."""

    def __init__(self, length: int, d_model: int = 256, width: int = 32):
        super().__init__()
        # Long views need more downsampling stages than short ones.
        n_stages = 5 if length > 1000 else 3
        chans = [1] + [width * (2 ** min(i, 3)) for i in range(n_stages)]
        blocks = []
        for i in range(n_stages):
            blocks.append(ResBlock1d(chans[i], chans[i + 1]))
            blocks.append(nn.MaxPool1d(kernel_size=5, stride=2, padding=2))
        self.trunk = nn.Sequential(*blocks)
        self.norm = nn.GroupNorm(8, chans[-1])
        # Mean and max pooling capture, respectively, the average level and the
        # depth of the deepest excursion - both matter for transit shape.
        self.proj = nn.Linear(2 * chans[-1], d_model)

    def forward(self, x):  # x: (B, L)
        h = self.trunk(x.unsqueeze(1))
        h = F.relu(self.norm(h))
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return self.proj(pooled)


class TransitDecoder(nn.Module):
    """Differentiable analytic transit renderer with a physical bottleneck.

    The five parameters are constrained to physically meaningful ranges by
    bounded activations, so the bottleneck can be read directly as a transit fit.
    """

    N_PARAMS = 5

    def __init__(self, d_in: int, n_bins: int = 201, n_durations: float = 4.0):
        super().__init__()
        self.n_bins = n_bins
        self.head = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, self.N_PARAMS)
        )
        # Local-view grid in units of transit durations from mid-transit.
        grid = torch.linspace(-n_durations, n_durations, n_bins)
        self.register_buffer("grid", grid)

    def decode_params(self, h):
        """Map network output to bounded physical parameters."""
        raw = self.head(h)
        depth = F.softplus(raw[:, 0]) + 1e-3          # > 0, view is normalised so ~1
        half_dur = 0.05 + 1.95 * torch.sigmoid(raw[:, 1])  # in duration units
        offset = 0.5 * torch.tanh(raw[:, 2])          # mid-transit phase error
        limb = torch.sigmoid(raw[:, 3])               # 0 = box, 1 = strong curvature
        soft = 0.02 + 0.98 * torch.sigmoid(raw[:, 4])  # ingress softness: U vs V
        return depth, half_dur, offset, limb, soft

    def render(self, depth, half_dur, offset, limb, soft):
        """Render the model light curve on the local-view grid.

        ``soft`` interpolates continuously between a box/U-shaped profile (small
        ``soft``, flat-bottomed, characteristic of a planet crossing the stellar
        disc) and a V-shaped profile (large ``soft``, characteristic of a grazing
        or blended eclipsing binary).
        """
        x = self.grid.unsqueeze(0)                      # (1, n_bins)
        z = (x - offset.unsqueeze(1)).abs() / half_dur.unsqueeze(1)
        # Smooth in/out-of-transit indicator; the width of the ramp is `soft`.
        edge = torch.sigmoid((1.0 - z) / soft.unsqueeze(1))
        # Limb darkening: the star is brighter at its centre, so a transit is
        # deeper near mid-transit.  sqrt(1 - z^2) is the standard chord profile.
        core = torch.sqrt(torch.clamp(1.0 - z ** 2, min=0.0) + 1e-6)
        shape = 1.0 - limb.unsqueeze(1) * (1.0 - core)
        return -depth.unsqueeze(1) * edge * shape

    def forward(self, h, target):
        depth, half_dur, offset, limb, soft = self.decode_params(h)
        model = self.render(depth, half_dur, offset, limb, soft)
        residual = target - model
        # Residual summary used as a classification feature.
        feats = torch.stack(
            [
                residual.pow(2).mean(dim=1).sqrt(),      # reconstruction RMS
                residual.abs().amax(dim=1),              # worst-case mismatch
                depth, half_dur, offset, limb, soft,
            ],
            dim=1,
        )
        params = torch.stack([depth, half_dur, offset, limb, soft], dim=1)
        return model, residual, feats, params


class PHANTOM(nn.Module):
    """Physics-informed harmonic-attention network for transit vetting."""

    def __init__(
        self,
        n_scalars: int,
        view_names: list[str] | None = None,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        n_classes: int = 2,
        use_decoder: bool = True,
        use_harmonic: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.view_names = view_names or list(VIEW_TABLE.keys())
        self.use_decoder = use_decoder
        self.use_harmonic = use_harmonic
        self.d_model = d_model

        # Keys are prefixed because some view names ("half", "double") collide
        # with nn.Module method names and cannot be used as submodule attributes.
        self.encoders = nn.ModuleDict(
            {f"enc_{n}": ViewEncoder(VIEW_TABLE[n][0], d_model) for n in self.view_names}
        )
        self.view_emb = nn.Embedding(len(self.view_names), d_model)
        self.hyp_emb = nn.Embedding(N_HYPOTHESES, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.attn = nn.TransformerEncoder(layer, n_layers)

        # Harmonic contrast: compare pooled evidence under P/2, P and 2P.
        if use_harmonic:
            self.harmonic_mlp = nn.Sequential(
                nn.Linear(4 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout)
            )

        if use_decoder:
            self.decoder = TransitDecoder(d_model)
            n_dec_feats = 7
        else:
            n_dec_feats = 0

        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalars, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.ReLU(),
        )

        fused = d_model + (d_model if use_harmonic else 0) + 128 + n_dec_feats
        self.head = nn.Sequential(
            nn.Linear(fused, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout),
        )
        self.logit = nn.Linear(512, 1 if n_classes == 2 else n_classes)
        # Transit parameters are *not* regressed against catalogue values: that
        # would only teach the network to reproduce the DV fit it is meant to be
        # compared against.  They are read directly off the differentiable
        # decoder's physical bottleneck, which is trained by reconstruction
        # alone, and their uncertainty is taken from the ensemble spread.

        self.register_buffer(
            "_hyp_index",
            torch.tensor([VIEW_TABLE[n][1] for n in self.view_names], dtype=torch.long),
            persistent=False,
        )

    def forward(self, views: dict[str, torch.Tensor], scalars: torch.Tensor):
        B = scalars.shape[0]
        dev = scalars.device
        tokens = []
        for i, name in enumerate(self.view_names):
            tok = self.encoders[f"enc_{name}"](views[name])
            tok = tok + self.view_emb.weight[i] + self.hyp_emb.weight[self._hyp_index[i]]
            tokens.append(tok)
        tok = torch.stack(tokens, dim=1)                      # (B, V, D)
        seq = torch.cat([self.cls_token.expand(B, -1, -1), tok], dim=1)
        enc = self.attn(seq)
        cls, view_out = enc[:, 0], enc[:, 1:]

        parts = [cls]

        if self.use_harmonic:
            hyp = self._hyp_index.to(dev)
            pooled = []
            for h in range(N_HYPOTHESES):
                sel = (hyp == h)
                if sel.any():
                    pooled.append(view_out[:, sel].mean(dim=1))
                else:
                    pooled.append(torch.zeros(B, self.d_model, device=dev))
            h_half, h_p, h_double = pooled
            # Contrasts state explicitly whether the evidence prefers P over its
            # harmonics, rather than leaving it implicit in the attention.
            contrast = torch.cat(
                [h_p, h_p - h_half, h_p - h_double, (h_p - h_double).abs()], dim=-1
            )
            parts.append(self.harmonic_mlp(contrast))

        dec_out = None
        if self.use_decoder:
            idx = self.view_names.index("local")
            model, residual, feats, params = self.decoder(view_out[:, idx], views["local"])
            parts.append(feats)
            dec_out = {"model": model, "residual": residual, "params": params}

        parts.append(self.scalar_mlp(scalars))
        h = self.head(torch.cat(parts, dim=-1))

        out = {"logit": self.logit(h).squeeze(-1)}
        if dec_out is not None:
            out.update(dec_out)
        return out


class AstroNet(nn.Module):
    """Faithful reimplementation of the Shallue & Vanderburg (2018) baseline.

    Two disjoint convolutional columns over the global and local views, each a
    stack of (conv, conv, maxpool) blocks, concatenated into four fully connected
    layers of width 512.  Reproduced here so that the comparison with PHANTOM is
    run on identical data, splits and training budget rather than against numbers
    quoted from a paper with a different preprocessing chain.
    """

    def __init__(self, n_scalars: int = 0, dropout: float = 0.0):
        super().__init__()
        self.n_scalars = n_scalars

        def column(n_blocks: int, kernel: int, pool: int):
            layers, c_in = [], 1
            for i in range(n_blocks):
                c_out = 16 * (2 ** i)
                layers += [
                    nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2), nn.ReLU(),
                    nn.Conv1d(c_out, c_out, kernel, padding=kernel // 2), nn.ReLU(),
                    nn.MaxPool1d(pool, stride=2, padding=pool // 2),
                ]
                c_in = c_out
            return nn.Sequential(*layers), c_in

        self.global_col, cg = column(5, 5, 5)
        self.local_col, cl = column(2, 5, 7)
        # Flattened widths after the pooling stack.
        with torch.no_grad():
            wg = self.global_col(torch.zeros(1, 1, 2001)).numel()
            wl = self.local_col(torch.zeros(1, 1, 201)).numel()
        self.fc = nn.Sequential(
            nn.Linear(wg + wl + n_scalars, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.ReLU(),
        )
        self.logit = nn.Linear(512, 1)

    def forward(self, views, scalars=None):
        g = self.global_col(views["global"].unsqueeze(1)).flatten(1)
        l = self.local_col(views["local"].unsqueeze(1)).flatten(1)
        parts = [g, l]
        if self.n_scalars and scalars is not None:
            parts.append(scalars)
        h = self.fc(torch.cat(parts, dim=-1))
        return {"logit": self.logit(h).squeeze(-1)}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
