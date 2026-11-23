"""
recurrent / convolutional classifiers and the model registry.

whats expected :

    logits, aux = model(x)          # x: (batch, T, C) float tensor
        logits : (batch,) for binary tasks, (batch, K) for multiclass
        aux    : dict, may contain
                   "updates"    (batch,) number of RNN state updates actually performed per sequence
                                (baseline = T; Skip RNN will report fewer -> this is the headline metric)
                   "extra_loss" scalar tensor added to the training loss (Skip RNN budget penalty)
                   "macs"       dict part -> (batch,) multiply-accumulates per sequence (FLOPs ~ 2 x MACs)
                   "mask"       (batch, T) 1 where the state was updated (skip models)
    model.stream(x)                 # x: (T, C), one sequence step by step (batch 1), used for latency
"""
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_rnn(rnn, cell, chrono_T=None):
    for name, p in rnn.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p)
        elif "weight_hh" in name:
            for chunk in p.data.chunk(3 if cell == "gru" else 4, 0):
                nn.init.orthogonal_(chunk)
        elif "bias" in name:
            nn.init.zeros_(p)
    if cell == "lstm" and chrono_T is None:
        for name, p in rnn.named_parameters():
            if "bias_ih" in name:
                n = p.size(0) // 4
                p.data[n:2 * n] = 1.0
    if chrono_T is not None:
        for name, p in rnn.named_parameters():
            if "bias_ih" in name:
                n = p.size(0) // (3 if cell == "gru" else 4)
                b = torch.log(torch.empty(n).uniform_(1, chrono_T - 1))
                p.data[n:2 * n] = b                  # GRU: update gate z | LSTM: forget gate f
                if cell == "lstm":
                    p.data[0:n] = -b                 # LSTM: input gate i


def cell_macs(cell, n_in, hidden):
    return (3 if cell == "gru" else 4) * hidden * (n_in + hidden)


class AttnPool(nn.Module):
    # softmax attention over the computed hidden states, e_t = w.h_t; skipped steps are masked out

    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Linear(hidden, 1)
        self.macs = 2 * hidden + 1

    def forward(self, hs, mask=None):
        e = self.score(hs).squeeze(-1)
        a = torch.exp(e - e.max(1, keepdim=True)[0].detach())
        if mask is not None:
            a = a * mask
        a = a / a.sum(1, keepdim=True)
        return (a.unsqueeze(-1) * hs).sum(1)

    def step(self, st, h):
        # online softmax for streaming, st = (running max, normaliser, weighted sum)
        e = self.score(h).item()
        if st is None:
            return e, 1.0, h
        m, s, v = st
        if e > m:
            a = math.exp(m - e)
            return e, s * a + 1.0, v * a + h
        b = math.exp(e - m)
        return m, s + b, v + b * h


class RNNClassifier(nn.Module):
    # Stacked GRU/LSTM; classifies from the output at the LAST time step (readout="attn": + attention pool)
    
    def __init__(self, cell="gru", input_size=1, hidden=64, layers=1, num_outputs=1, dropout=0.0, chrono_T=None,
                 readout="last", **kw):
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(input_size, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.pool = AttnPool(hidden) if readout == "attn" else None
        self.fc = nn.Linear(hidden * (2 if self.pool else 1), num_outputs)
        self.num_outputs = num_outputs
        self.cell, self.input_size, self.hidden, self.layers = cell, input_size, hidden, layers
        self._init_weights(cell, chrono_T)

    def _init_weights(self, cell, chrono_T=None):
        init_rnn(self.rnn, cell, chrono_T)

    def forward(self, x):
        out, _ = self.rnn(x)
        z = out[:, -1]
        if self.pool is not None:
            z = torch.cat([z, self.pool(out)], -1)
        logits = self.fc(self.drop(z))
        if self.num_outputs == 1:
            logits = logits.squeeze(-1)
        updates = torch.full((x.size(0),), float(x.size(1)), device=x.device)   # a vanilla RNN updates every step
        step = sum(cell_macs(self.cell, self.input_size if l == 0 else self.hidden, self.hidden)
                   for l in range(self.layers))
        macs = {"cell": updates * step, "head": torch.full_like(updates, self.fc.in_features * self.fc.out_features)}
        if self.pool is not None:
            macs["attn"] = updates * self.pool.macs
        return logits, {"updates": updates, "macs": macs}

    @torch.no_grad()
    def stream(self, x):
        h = [x.new_zeros(1, self.hidden) for _ in range(self.layers)]
        c = [x.new_zeros(1, self.hidden) for _ in range(self.layers)]
        w = [[getattr(self.rnn, f"{n}_l{l}") for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
             for l in range(self.layers)]
        st = None
        for t in range(x.size(0)):
            inp = x[t:t + 1]
            for l in range(self.layers):
                if self.cell == "gru":
                    h[l] = torch.gru_cell(inp, h[l], *w[l])
                else:
                    h[l], c[l] = torch.lstm_cell(inp, (h[l], c[l]), *w[l])
                inp = h[l]
            if self.pool is not None:
                st = self.pool.step(st, inp)
        if self.pool is not None:
            inp = torch.cat([inp, st[2] / st[1]], -1)
        return self.fc(inp)


class SkipRNN(nn.Module):
    """Skip RNN (Campos et al., ICLR 2018), one layer. Gate variants:
        skip   : paper, sigmoid(linear(state)); state = h for GRU, c for LSTM
        phys   : small MLP on [proj(state), physiological features], recomputed on an update OR a new detected beat
        mlp    : same gate network as phys without features / beat events (gate capacity control)
        random : update with prob 1 - p_skip, nothing learned (control)
    max_skip forces an update after that many skipped steps."""

    def __init__(self, cell="gru", mode="skip", input_size=1, hidden=64, num_outputs=1, dropout=0.0, chrono_T=None,
                 layers=1, lam=1e-4, p_skip=0.5, max_skip=None, n_feats=0, feat_keep=None, gate_hidden=8,
                 det_macs=0, feat_macs=0, readout="last", **kw):
        super().__init__()
        assert layers == 1, "skip models are single layer"
        self.cell, self.mode, self.hidden, self.input_size = cell, mode, hidden, input_size
        self.lam, self.p_skip, self.max_skip = lam, p_skip, max_skip
        self.n_feats = n_feats if mode in ("phys", "mlp") else 0
        self.det_macs, self.feat_macs = det_macs, feat_macs
        self.rnn = (nn.GRUCell if cell == "gru" else nn.LSTMCell)(input_size, hidden)
        init_rnn(self.rnn, cell, chrono_T)
        self.gate, self.gate_macs = None, 0
        if mode in ("phys", "mlp"):
            self.proj = nn.Linear(hidden, gate_hidden)
            self.gate = nn.Sequential(nn.Linear(gate_hidden + self.n_feats, gate_hidden), nn.ReLU(),
                                      nn.Linear(gate_hidden, 1))
            nn.init.constant_(self.gate[-1].bias, 1.0)
            self.gate_macs = hidden * gate_hidden + (gate_hidden + self.n_feats) * gate_hidden + gate_hidden
        elif mode == "skip":
            self.gate = nn.Linear(hidden, 1)
            nn.init.constant_(self.gate.bias, 1.0)           # b_p = 1 as in the paper
            self.gate_macs = hidden
        keep = torch.ones(self.n_feats)
        if feat_keep is not None:
            keep.zero_()
            keep[list(feat_keep)] = 1
        self.register_buffer("feat_keep", keep)
        self.drop = nn.Dropout(dropout)
        self.pool = AttnPool(hidden) if readout == "attn" else None
        self.fc = nn.Linear(hidden * (2 if self.pool else 1), num_outputs)
        self.num_outputs = num_outputs

    def _gate(self, s, f=None):
        if self.mode == "skip":
            return torch.sigmoid(self.gate(s))
        return torch.sigmoid(self.gate(torch.cat([self.proj(s), f], -1)))

    def forward(self, x):
        B, T, _ = x.shape
        H, gru, phys = self.hidden, self.cell == "gru", self.mode == "phys"
        sig = x[..., :self.input_size]
        if phys:
            feats = x[..., self.input_size:] * self.feat_keep
            ev = x[..., -1:]
        zero_f = x.new_zeros(B, self.n_feats)
        gi = F.linear(sig, self.rnn.weight_ih, self.rnn.bias_ih).unbind(1)      # unbind, slicing per step makes backward O(T^2)
        h, c = x.new_zeros(B, H), x.new_zeros(B, H)
        u_t = x.new_ones(B, 1)                  # u~_1 = 1, first step always updates
        du, since, n_gate = x.new_zeros(B, 1), x.new_zeros(B, 1), x.new_zeros(B, 1)
        us, hs = [], []
        for t in range(T):
            if self.mode == "random":
                u = (torch.rand(B, 1, device=x.device) >= self.p_skip).float() if t else u_t
            else:
                u = u_t + (torch.round(u_t) - u_t).detach()          # straight-through estimator
            if self.max_skip:
                forced = (since >= self.max_skip).float()
                u = u + forced * (1 - u)
            gh = F.linear(h, self.rnn.weight_hh, self.rnn.bias_hh)
            if gru:
                ir, iz, i_n = gi[t].chunk(3, 1)
                hr, hz, h_n = gh.chunk(3, 1)
                r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
                h = u * ((1 - z) * torch.tanh(i_n + r * h_n) + z * h) + (1 - u) * h
                s = h
            else:
                i, f, g, o = (gi[t] + gh).chunk(4, 1)
                c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                h = u * torch.sigmoid(o) * torch.tanh(c_new) + (1 - u) * h
                c = u * c_new + (1 - u) * c
                s = c
            if self.pool is not None:
                hs.append(h)
            if self.mode != "random":
                if phys:
                    # features change every step (dt), so the gate output is held until an update or a beat
                    refresh = u + ev[:, t] * (1 - u)
                    du = refresh * self._gate(s, feats[:, t]) + (1 - refresh) * du
                    n_gate = n_gate + refresh.detach()
                else:
                    # state is unchanged when skipping, so this equals evaluating the gate only on updates
                    du = self._gate(s, zero_f)
                    n_gate = n_gate + u.detach()
                u_t = u * du + (1 - u) * (u_t + torch.min(du, 1 - u_t))
            since = (since + 1) * (1 - u.detach())
            us.append(u)
        U = torch.cat(us, 1)
        z = h
        if self.pool is not None:
            z = torch.cat([h, self.pool(torch.stack(hs, 1), U)], -1)
        logits = self.fc(self.drop(z))
        if self.num_outputs == 1:
            logits = logits.squeeze(-1)
        upd = U.detach().sum(1)
        macs = {"cell": upd * cell_macs(self.cell, self.input_size, H), "gate": n_gate.squeeze(1) * self.gate_macs,
                "head": torch.full_like(upd, self.fc.in_features * self.fc.out_features)}
        if self.pool is not None:
            macs["attn"] = upd * self.pool.macs
        if phys:
            macs["detector"] = torch.full_like(upd, T * self.det_macs)
            macs["features"] = ev.squeeze(-1).sum(1) * self.feat_macs
        aux = {"updates": upd, "macs": macs, "mask": U.detach()}
        if self.mode != "random":
            aux["extra_loss"] = self.lam * U.sum(1).mean()          # budget loss, lambda * sum_t u_t
        return logits, aux

    @torch.no_grad()
    def stream(self, x):
        gru, phys = self.cell == "gru", self.mode == "phys"
        sig = x[:, :self.input_size]
        if phys:
            feats = x[:, self.input_size:] * self.feat_keep
            ev = x[:, -1].tolist()
        zero_f = x.new_zeros(1, self.n_feats)
        w = (self.rnn.weight_ih, self.rnn.weight_hh, self.rnn.bias_ih, self.rnn.bias_hh)
        h, c = x.new_zeros(1, self.hidden), x.new_zeros(1, self.hidden)
        u_t, du, since, st = 1.0, 0.0, 0, None
        for t in range(x.size(0)):
            if self.mode == "random":
                u = t == 0 or random.random() >= self.p_skip
            else:
                u = u_t > 0.5                   # same as torch.round (0.5 -> 0)
            if self.max_skip and since >= self.max_skip:
                u = True
            if u:
                if gru:
                    h = torch.gru_cell(sig[t:t + 1], h, *w)
                else:
                    h, c = torch.lstm_cell(sig[t:t + 1], (h, c), *w)
                since = 0
                if self.pool is not None:
                    st = self.pool.step(st, h)
            else:
                since += 1
            if self.mode == "random":
                continue
            if u or (phys and ev[t]):
                du = self._gate(h if gru else c, feats[t:t + 1] if phys else zero_f).item()
            u_t = du if u else u_t + min(du, 1 - u_t)
        if self.pool is not None:
            h = torch.cat([h, st[2] / st[1]], -1)
        return self.fc(h)


class CNN1D(nn.Module):
    # 4 x (conv k=5, BN, ReLU, maxpool 2) -> global average pool -> linear

    def __init__(self, input_size=1, num_outputs=1, dropout=0.0, channels=(16, 32, 32, 32), k=5, **kw):
        super().__init__()
        layers, c_in = [], input_size
        for c in channels:
            layers += [nn.Conv1d(c_in, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2)]
            c_in = c
        self.net = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, num_outputs)
        self.input_size, self.num_outputs = input_size, num_outputs

    def forward(self, x):
        z, macs = x[..., :self.input_size].transpose(1, 2), 0
        for m in self.net:
            z = m(z)
            if isinstance(m, nn.Conv1d):
                macs += z.size(-1) * m.out_channels * m.in_channels * m.kernel_size[0]
        logits = self.fc(self.drop(z.mean(-1)))
        if self.num_outputs == 1:
            logits = logits.squeeze(-1)
        B = x.size(0)
        return logits, {"macs": {"conv": x.new_full((B,), float(macs)),
                                 "head": x.new_full((B,), float(self.fc.in_features * self.fc.out_features))}}


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, dropout):
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.Conv1d(c_in, c_out, k, dilation=d)
        self.c2 = nn.Conv1d(c_out, c_out, k, dilation=d)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.drop(torch.relu(self.c1(F.pad(x, (self.pad, 0)))))
        y = self.drop(torch.relu(self.c2(F.pad(y, (self.pad, 0)))))
        return torch.relu(y + (x if self.down is None else self.down(x)))


class TCN(nn.Module):
    # causal dilated TCN (Bai et al. 2018), dilations 1..2^(levels-1), classifies from the last step

    def __init__(self, input_size=1, num_outputs=1, dropout=0.0, tcn_channels=16, levels=9, k=3, **kw):
        super().__init__()
        self.blocks = nn.Sequential(*[TCNBlock(input_size if i == 0 else tcn_channels, tcn_channels, k, 2 ** i, dropout)
                                      for i in range(levels)])
        self.fc = nn.Linear(tcn_channels, num_outputs)
        self.input_size, self.num_outputs = input_size, num_outputs
        self.receptive_field = 1 + 2 * (k - 1) * (2 ** levels - 1)

    def forward(self, x):
        z = self.blocks(x[..., :self.input_size].transpose(1, 2))
        logits = self.fc(z[:, :, -1])
        if self.num_outputs == 1:
            logits = logits.squeeze(-1)
        per_step = sum(m.out_channels * m.in_channels * m.kernel_size[0] for m in self.blocks.modules()
                       if isinstance(m, nn.Conv1d))
        B = x.size(0)
        return logits, {"macs": {"conv": x.new_full((B,), float(per_step * x.size(1))),
                                 "head": x.new_full((B,), float(self.fc.in_features * self.fc.out_features))}}


def _gru(**kw):  return RNNClassifier(cell="gru", **kw)
def _lstm(**kw): return RNNClassifier(cell="lstm", **kw)


MODELS = {"gru": _gru, "lstm": _lstm,
          "skip_gru": lambda **kw: SkipRNN("gru", "skip", **kw),
          "skip_lstm": lambda **kw: SkipRNN("lstm", "skip", **kw),
          "phys_skip_gru": lambda **kw: SkipRNN("gru", "phys", **kw),
          "phys_skip_lstm": lambda **kw: SkipRNN("lstm", "phys", **kw),
          "mlp_skip_gru": lambda **kw: SkipRNN("gru", "mlp", **kw),
          "mlp_skip_lstm": lambda **kw: SkipRNN("lstm", "mlp", **kw),
          "rand_gru": lambda **kw: SkipRNN("gru", "random", **kw),
          "rand_lstm": lambda **kw: SkipRNN("lstm", "random", **kw),
          "cnn": lambda **kw: CNN1D(**kw),
          "tcn": lambda **kw: TCN(**kw)}


def build_model(name, input_size, num_outputs, hidden=64, layers=1, dropout=0.0, chrono_T=None, **kw):
    # extra kw (lam, p_skip, max_skip, n_feats, ...) are only used by the skip models
    return MODELS[name](input_size=input_size, hidden=hidden, layers=layers,
                        num_outputs=num_outputs, dropout=dropout, chrono_T=chrono_T, **kw)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
