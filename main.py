import os
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from utils import DFT_matrix, vec_by_col, print_metrics, reim_to_complex, batch_nmse, rank_error

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Simulation environment variables
N_r = 16
N_RIS = 32
M_r = 8
M_RIS = 28

N_train = 60000
N_val_frac = 0.10
N_test = 10000

EPOCHS_PINN = 30
EPOCHS_MLP = 30
EPOCHS_UNF = 30
LR = 1e-3
BATCH_SIZE = 64
VAL_BATCH = 2048
GRAD_CLIP = 5.0
HIDDEN = 256

L1_PATHS = 1
L2_PATHS = 1

W1_np = DFT_matrix(N_r)[0:M_r, :]
Om_np = DFT_matrix(N_RIS)[:, 0:M_RIS]
W1 = torch.from_numpy(W1_np).to(torch.complex64)
Omega1 = torch.from_numpy(Om_np).to(torch.complex64)
W1_pinv = torch.pinverse(W1)
Om_pinv = torch.pinverse(Omega1)

# Build dataset
@torch.no_grad()
def data_generate_YH(N, snr_dB, L_BR=1, L_RM=1):
    Y = torch.zeros(N, M_r, M_RIS, dtype=torch.complex64)
    H = torch.zeros(N, N_r, N_RIS, dtype=torch.complex64)
    snr_lin = (10.0 ** (snr_dB.view(-1) / 10.0)).to(torch.float32)

    for i in range(N):
        f_RM = torch.rand(L_RM)
        A_RM = torch.vander(torch.exp(1j * math.pi * f_RM), N_RIS, increasing=True).to(torch.complex64)
        rho_RM = torch.randn(L_RM, dtype=torch.complex64)
        H_RM = torch.matmul(torch.transpose(A_RM, 0, 1), rho_RM)

        f1_BR = torch.rand(L_BR)
        f2_BR = torch.rand(L_BR)
        A1_BR = torch.vander(torch.exp(1j * math.pi * f1_BR), N_r, increasing=True).to(torch.complex64)
        A2_BR = torch.vander(torch.exp(1j * math.pi * f2_BR), N_RIS, increasing=True).to(torch.complex64)
        rho_BR = torch.randn(L_BR, dtype=torch.complex64)
        H_BR = torch.matmul(torch.matmul(torch.transpose(A1_BR, 0, 1), torch.diag(rho_BR)), A2_BR)

        H[i, :, :] = torch.matmul(H_BR, torch.diag(H_RM))

        Y[i, :, :] = (torch.matmul(W1, torch.matmul(H[i, :, :], Omega1)) +
                      (1.0 / np.sqrt(snr_lin[i].item())) *
                      torch.matmul(W1, torch.randn(N_r, M_RIS, dtype=torch.complex64)))

    return Y, H


class YHDataset(Data.Dataset):
    def __init__(self, Y, H):
        self.Y = Y
        self.H = H
        y_vec = vec_by_col(Y)
        self.y_reim = torch.view_as_real(y_vec).reshape(Y.size(0), -1).to(torch.float32)
        h_vec = vec_by_col(H)
        self.h_reim = torch.view_as_real(h_vec).reshape(H.size(0), -1).to(torch.float32)

    def __len__(self):
        return self.Y.size(0)

    def __getitem__(self, idx):
        return self.y_reim[idx], self.Y[idx], self.H[idx], self.h_reim[idx]


# Enhanced features: Y & physics-based preprocessing
class PINNFeatureDataset(Data.Dataset):
    def __init__(self, Y, H, mean_std=None):
        self.Y = Y
        self.H = H

        # Feature 1: vec(Y)
        y_vec = vec_by_col(Y)
        y_reim = torch.view_as_real(y_vec).reshape(Y.size(0), -1).float()

        # Feature 2: vec(W^H Y Omega^*) - physics-based preprocessing
        WH = W1.conj().transpose(0, 1).unsqueeze(0)
        OMtH = Omega1.conj().transpose(0, 1).unsqueeze(0)
        PsiHy = (WH @ Y @ OMtH)
        psi_vec = vec_by_col(PsiHy)
        psi_reim = torch.view_as_real(psi_vec).reshape(Y.size(0), -1).float()

        feats = torch.cat([y_reim, psi_reim], dim=1)

        if mean_std is None:
            mu = feats.mean(dim=0)
            sig = feats.std(dim=0)
            sig = torch.clamp(sig, min=1e-3)
            self.mu, self.sig = mu, sig
            feats = (feats - mu) / sig
        else:
            mu, sig = mean_std
            sig = torch.clamp(sig, min=1e-3)
            self.mu, self.sig = mu, sig
            feats = (feats - mu) / sig

        self.x_feats = feats

    def get_mean_std(self):
        return self.mu, self.sig

    def __len__(self):
        return self.Y.size(0)

    def __getitem__(self, idx):
        return self.x_feats[idx], self.Y[idx], self.H[idx]
    

# Evaluation
@torch.no_grad()
def eval_full_metrics_model(predict_Hc, loader, L_rank=L1_PATHS):
    obs_list, nmse_list, rank_list = [], [], []
    for y_reim_b, Y_b, H_true_b, _ in loader:
        Y_b = Y_b.to(device)
        H_true_b = H_true_b.to(device)
        H_hat_b = predict_Hc(Y_b)
        Y_hat_b = (W1.to(device).unsqueeze(0) @ H_hat_b @ Omega1.to(device).unsqueeze(0))
        obs = torch.mean(torch.abs(Y_b - Y_hat_b) ** 2, dim=(1, 2))
        nmse_b = batch_nmse(H_hat_b, H_true_b)
        rank_b = rank_error(H_hat_b, L=L_rank)
        obs_list.append(obs.cpu())
        nmse_list.append(nmse_b.cpu())
        rank_list.append(rank_b.cpu())
    obs_all = torch.cat(obs_list)
    nmse_all = torch.cat(nmse_list)
    rank_all = torch.cat(rank_list)
    return {
        'obs_mse_mean': obs_all.mean().item(),
        'obs_mse_std': obs_all.std(unbiased=False).item(),
        'nmse_mean': nmse_all.mean().item(),
        'nmse_std': nmse_all.std(unbiased=False).item(),
        'rank_err_mean': rank_all.mean().item(),
        'rank_err_std': rank_all.std(unbiased=False).item(),
        'N': len(obs_all)
    }

@torch.no_grad()
def baseline_ls_predict(Y_b):
    Wp = W1_pinv.to(Y_b.device)
    Op = Om_pinv.to(Y_b.device)
    return (Wp.unsqueeze(0) @ Y_b @ Op.unsqueeze(0))

# Proposed DL model and baseline methods
class HybridPINN(nn.Module):
    """
    Learns H directly but uses physics-informed features.
    No bottleneck - can represent any H!
    """

    def __init__(self):
        super().__init__()
        # Input: physics-informed features (Y + W^H Y Omega^*)
        D_in = 2 * (M_r * M_RIS + N_r * N_RIS)
        D_out = 2 * N_r * N_RIS  # Direct H prediction

        self.net = nn.Sequential(
            nn.Linear(D_in, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, D_out)
        )

        # Good initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_feats):
        h_reim = self.net(x_feats)
        H = reim_to_complex(h_reim, rows=N_r, cols=N_RIS)
        return H


# ============================================
# Baselines
# ============================================
class SupMLP(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        D_in = 2 * (M_r * M_RIS)
        D_out = 2 * (N_r * N_RIS)
        self.net = nn.Sequential(
            nn.Linear(D_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, D_out)
        )

    def forward(self, y_reim):
        return self.net(y_reim)


class UnfoldedNet(nn.Module):
    """
    Model-based unfolding with learned linear layers (like paper's approach)
    Each iteration: h = σ(M @ [h_prev, data_term, reg_term] + b)
    """

    def __init__(self, T=10):
        super().__init__()
        self.T = T

        # Learnable transformation at each layer
        # Input: concatenate [h_real, h_imag, gradient_real, gradient_imag]
        hidden_size = 2 * N_r * N_RIS  # Real/imag of H
        self.layers = nn.ModuleList([
            nn.Linear(4 * N_r * N_RIS, 2 * N_r * N_RIS) for _ in range(T)
        ])

        # Initialize to approximate identity + gradient step
        for layer in self.layers:
            nn.init.eye_(layer.weight[:2 * N_r * N_RIS, :2 * N_r * N_RIS])  # Identity on h part
            nn.init.normal_(layer.weight[:, 2 * N_r * N_RIS:], 0, 0.01)  # Small gradient coupling
            nn.init.zeros_(layer.bias)

    def forward(self, Y_b):
        B = Y_b.size(0)

        # Initialize with LS solution
        Wp = W1_pinv.to(Y_b.device)
        Op = Om_pinv.to(Y_b.device)
        H = Wp.unsqueeze(0) @ Y_b @ Op.unsqueeze(0)

        W = W1.to(Y_b.device)
        Om = Omega1.to(Y_b.device)
        WH = W.conj().transpose(0, 1)
        OMH = Om.conj().transpose(0, 1)

        # Flatten to vector
        h = torch.cat([H.real.reshape(B, -1), H.imag.reshape(B, -1)], dim=1)

        for t in range(self.T):
            # Reconstruct H from h
            H_real = h[:, :N_r * N_RIS].reshape(B, N_r, N_RIS)
            H_imag = h[:, N_r * N_RIS:].reshape(B, N_r, N_RIS)
            H = torch.complex(H_real, H_imag)

            # Compute gradient (data fidelity term)
            Y_pred = W.unsqueeze(0) @ H @ Om.unsqueeze(0)
            R = Y_b - Y_pred
            grad = WH.unsqueeze(0) @ R @ OMH.unsqueeze(0)
            grad_flat = torch.cat([grad.real.reshape(B, -1), grad.imag.reshape(B, -1)], dim=1)

            # Concatenate current estimate and gradient
            features = torch.cat([h, grad_flat], dim=1)

            # Learnable update with ReLU activation (except last layer)
            if t < self.T - 1:
                h = torch.relu(self.layers[t](features))
            else:
                h = self.layers[t](features)  # No activation on last layer

        # Reconstruct final H
        H_real = h[:, :N_r * N_RIS].reshape(B, N_r, N_RIS)
        H_imag = h[:, N_r * N_RIS:].reshape(B, N_r, N_RIS)
        H = torch.complex(H_real, H_imag)

        return H
    
# Training functions
def train_hybrid_pinn(Y_tr, H_tr, snr_tag="20dB"):
    print(f"\n{'=' * 60}")
    print("Hybrid PINN: Direct H prediction with physics features")
    print(f"{'=' * 60}\n")

    full_ds = PINNFeatureDataset(Y_tr, H_tr, mean_std=None)
    mu, sig = full_ds.get_mean_std()

    n_val = int(len(Y_tr) * N_val_frac)
    perm = torch.randperm(len(Y_tr))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_ds = PINNFeatureDataset(Y_tr[tr_idx], H_tr[tr_idx], mean_std=(mu, sig))
    val_ds = PINNFeatureDataset(Y_tr[val_idx], H_tr[val_idx], mean_std=(mu, sig))
    tr_loader = Data.DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = Data.DataLoader(val_ds, batch_size=VAL_BATCH, shuffle=False)

    with torch.no_grad():
        x_dbg, Y_dbg, H_dbg = next(iter(tr_loader))
        print(f"[init] Feature stats: mean={x_dbg.mean().item():.4f}, std={x_dbg.std().item():.4f}")
        print(f"[init] Y magnitude: {Y_dbg.abs().mean().item():.4e}")
        print(f"[init] H magnitude: {H_dbg.abs().mean().item():.4e}\n")

    model = HybridPINN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PINN, eta_min=1e-5)

    best_val = float("inf")
    logs = {'train_loss': [], 'val_nmse_h': [], 'epoch_time': []}
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_w = os.path.join(ckpt_dir, f"pinn_best_{snr_tag}.pt")

    for ep in range(1, EPOCHS_PINN + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        for x_feats, Y_b, H_b in tr_loader:
            x_feats = x_feats.to(device)
            H_b = H_b.to(device)

            H_pred = model(x_feats)

            # Simple supervised loss
            loss = (F.mse_loss(H_pred.real, H_b.real) +
                    F.mse_loss(H_pred.imag, H_b.imag))

            # Optional: add rank regularization
            # rank_reg = torch.mean(torch.linalg.vector_norm(H_pred.reshape(H_pred.size(0), -1), ord=2, dim=-1))
            # loss = loss + 1e-6 * rank_reg

            opt.zero_grad()
            loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_nmse_h = 0.0
        count = 0

        with torch.no_grad():
            for x_feats, Y_b, H_b in val_loader:
                x_feats = x_feats.to(device)
                H_b = H_b.to(device)
                H_pred = model(x_feats)

                nmse_h_batch = batch_nmse(H_pred, H_b)
                val_nmse_h += torch.sum(nmse_h_batch).item()
                count += H_b.size(0)

        val_nmse_h /= max(count, 1)

        logs['train_loss'].append(epoch_loss / len(tr_loader))
        logs['val_nmse_h'].append(val_nmse_h)
        logs['epoch_time'].append(time.time() - t0)

        print(f"[PINN] Ep {ep:02d} | train_loss={epoch_loss / len(tr_loader):.6f} | "
              f"val_NMSE(H)={val_nmse_h:.6f}")

        if val_nmse_h < best_val:
            best_val = val_nmse_h
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'val_nmse_h': val_nmse_h,
                'mu': mu,
                'sig': sig
            }, ckpt_w)
            print(f"  ✓ Best model saved (NMSE={best_val:.6f})")

    checkpoint = torch.load(ckpt_w, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    mu = checkpoint['mu']
    sig = checkpoint['sig']
    model.eval()
    print(f"\nLoaded best model from epoch {checkpoint['epoch']} with NMSE(H)={checkpoint['val_nmse_h']:.6f}")

    def predict_Hc(Y_b):
        with torch.no_grad():
            y_vec = vec_by_col(Y_b)
            y_reim = torch.view_as_real(y_vec).reshape(Y_b.size(0), -1).float()
            WH = W1.conj().transpose(0, 1).unsqueeze(0)
            OMtH = Omega1.conj().transpose(0, 1).unsqueeze(0)
            PsiHy = (WH.to(Y_b.device) @ Y_b @ OMtH.to(Y_b.device))
            psi_vec = vec_by_col(PsiHy)
            psi_reim = torch.view_as_real(psi_vec).reshape(Y_b.size(0), -1).float()
            feats = torch.cat([y_reim, psi_reim], dim=1).to(device)
            feats = (feats - mu.to(device)) / sig.to(device)
            H = model(feats)
        return H

    return predict_Hc, logs


def train_mlp(Y_tr, H_tr, tag="mlp_20dB"):
    n_val = int(len(Y_tr) * N_val_frac)
    perm = torch.randperm(len(Y_tr))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_ds = YHDataset(Y_tr[tr_idx], H_tr[tr_idx])
    val_ds = YHDataset(Y_tr[val_idx], H_tr[val_idx])
    tr_loader = Data.DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = Data.DataLoader(val_ds, batch_size=VAL_BATCH, shuffle=False)

    model = SupMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val = float("inf")
    logs = {'train_loss': [], 'val_nmse': [], 'epoch_time': []}
    ckpt_w = os.path.join("checkpoints", f"mlp_best_{tag}.pt")

    for ep in range(1, EPOCHS_MLP + 1):
        model.train()
        t0 = time.time()
        run = 0.0
        for y_reim, _, H_true, h_reim in tr_loader:
            y_reim = y_reim.to(device)
            h_reim = h_reim.to(device)
            out = model(y_reim)
            loss = F.mse_loss(out, h_reim)
            opt.zero_grad()
            loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            run += loss.item()

        model.eval()
        nmse_acc = 0.0
        cnt = 0
        with torch.no_grad():
            for y_reim, _, H_true, _ in val_loader:
                y_reim = y_reim.to(device)
                H_true = H_true.to(device)
                out = model(y_reim)
                H_hat = reim_to_complex(out, rows=N_r, cols=N_RIS)
                nmse_b = batch_nmse(H_hat, H_true).mean().item()
                nmse_acc += nmse_b * H_true.size(0)
                cnt += H_true.size(0)
        val_nmse = nmse_acc / max(cnt, 1)

        logs['train_loss'].append(run / len(tr_loader))
        logs['val_nmse'].append(val_nmse)
        logs['epoch_time'].append(time.time() - t0)
        print(f"[MLP ] Epoch {ep:02d} | train={logs['train_loss'][-1]:.6f} | val_NMSE={val_nmse:.6f}")

        if val_nmse < best_val:
            best_val = val_nmse
            torch.save(model.state_dict(), ckpt_w)
            print(f"  ✓ saved best ({best_val:.6f})")

    model.load_state_dict(torch.load(ckpt_w, map_location=device))
    model.eval()

    def predict_Hc(Y_b):
        with torch.no_grad():
            B = Y_b.size(0)
            y_vec = vec_by_col(Y_b)
            y_reim = torch.view_as_real(y_vec).reshape(B, -1).float().to(device)
            out = model(y_reim)
            H_hat = reim_to_complex(out, rows=N_r, cols=N_RIS)
        return H_hat

    return predict_Hc, logs


def train_unfolded(Y_tr, H_tr, tag="unf_20dB"):
    n_val = int(len(Y_tr) * N_val_frac)
    perm = torch.randperm(len(Y_tr))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_ds = YHDataset(Y_tr[tr_idx], H_tr[tr_idx])
    val_ds = YHDataset(Y_tr[val_idx], H_tr[val_idx])
    tr_loader = Data.DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = Data.DataLoader(val_ds, batch_size=VAL_BATCH, shuffle=False)

    model = UnfoldedNet(T=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_UNF)

    best_val = float("inf")
    logs = {'train_loss': [], 'val_nmse': [], 'epoch_time': []}
    ckpt_w = os.path.join("checkpoints", f"unf_best_{tag}.pt")

    print(f"[UNF] Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    for ep in range(1, EPOCHS_UNF + 1):
        model.train()
        t0 = time.time()
        run = 0.0
        for _, Y_b, H_true, _ in tr_loader:
            Y_b = Y_b.to(device)
            H_true = H_true.to(device)
            H_hat = model(Y_b)

            loss = F.mse_loss(H_hat.real, H_true.real) + F.mse_loss(H_hat.imag, H_true.imag)

            opt.zero_grad()
            loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            run += loss.item()

        scheduler.step()

        model.eval()
        nmse_acc = 0.0
        cnt = 0
        with torch.no_grad():
            for _, Y_b, H_true, _ in val_loader:
                Y_b = Y_b.to(device)
                H_true = H_true.to(device)
                H_hat = model(Y_b)
                nmse_b = batch_nmse(H_hat, H_true).mean().item()
                nmse_acc += nmse_b * H_true.size(0)
                cnt += H_true.size(0)
        val_nmse = nmse_acc / max(cnt, 1)

        logs['train_loss'].append(run / len(tr_loader))
        logs['val_nmse'].append(val_nmse)
        logs['epoch_time'].append(time.time() - t0)
        print(f"[UNF ] Epoch {ep:02d} | train={logs['train_loss'][-1]:.6f} | val_NMSE={val_nmse:.6f}")

        if val_nmse < best_val:
            best_val = val_nmse
            torch.save(model.state_dict(), ckpt_w)
            print(f"  ✓ saved best ({best_val:.6f})")

    model.load_state_dict(torch.load(ckpt_w, map_location=device))
    model.eval()

    def predict_Hc(Y_b):
        with torch.no_grad():
            return model(Y_b.to(device))

    return predict_Hc, logs

# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Hybrid PINN: Physics Features + Direct Prediction")
    print("=" * 70)

    print("\nGenerating training data...")
    Y_tr, H_tr = data_generate_YH(N_train, 20.0 * torch.ones(N_train, 1), L_BR=1, L_RM=1)
    print(f"Data: Y mag={Y_tr.abs().mean():.4e}, H mag={H_tr.abs().mean():.4e}")

    predict_pinn, logs_pinn = train_hybrid_pinn(Y_tr, H_tr, snr_tag="20dB")

    print("\n[Train] MLP...")
    predict_mlp, logs_mlp = train_mlp(Y_tr, H_tr, tag="20dB")

    print("\n[Train] Unfolded...")
    predict_unf, logs_unf = train_unfolded(Y_tr, H_tr, tag="20dB")

    print("\nEvaluating on validation set...")
    Y_val, H_val = data_generate_YH(int(N_train * N_val_frac),
                                    20.0 * torch.ones(int(N_train * N_val_frac), 1),
                                    L_BR=1, L_RM=1)
    val_loader = Data.DataLoader(YHDataset(Y_val, H_val), batch_size=VAL_BATCH, shuffle=False)

    m_ls = eval_full_metrics_model(lambda Y: baseline_ls_predict(Y), val_loader)
    m_pinn = eval_full_metrics_model(lambda Y: predict_pinn(Y), val_loader)
    m_mlp = eval_full_metrics_model(lambda Y: predict_mlp(Y), val_loader)
    m_unf = eval_full_metrics_model(lambda Y: predict_unf(Y), val_loader)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print_metrics("LS  ", m_ls, L1_PATHS)
    print_metrics("PINN", m_pinn, L1_PATHS)
    print_metrics("MLP ", m_mlp, L1_PATHS)
    print_metrics("UNF ", m_unf, L1_PATHS)
    print("=" * 70)
