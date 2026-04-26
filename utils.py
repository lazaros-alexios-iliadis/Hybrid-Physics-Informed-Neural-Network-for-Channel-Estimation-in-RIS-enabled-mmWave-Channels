import pandas as pd
import numpy as np
import torch

def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(-2j * np.pi / N)
    W = np.power(omega, i * j) / np.sqrt(N)
    return W


def vec_by_col(X):
    return X.transpose(-1, -2).reshape(X.shape[:-2] + (-1,))


def print_metrics(tag, m, L1paths):
    print(f"[{tag}] N={m['N']} | ObsMSE={m['obs_mse_mean']:.4e}±{m['obs_mse_std']:.1e}"
          f" | NMSE(H)={m['nmse_mean']:.4e}±{m['nmse_std']:.1e}"
          f" | RankErr(L>{L1paths})={m['rank_err_mean']:.4e}±{m['rank_err_std']:.1e}")
    
    
def reim_to_complex(h_reim, rows, cols):
    B = h_reim.size(0)
    h_re, h_im = torch.chunk(h_reim, 2, dim=1)
    h_c = torch.complex(h_re, h_im)
    H = h_c.view(B, cols, rows).transpose(1, 2).contiguous()
    return H


def batch_nmse(H_hat, H_true):
    num = torch.sum(torch.abs(H_hat - H_true) ** 2, dim=(1, 2))
    den = torch.sum(torch.abs(H_true) ** 2, dim=(1, 2)) + 1e-12
    return (num / den)


def rank_error(H_hat, L):
    errs = []
    for Hi in H_hat:
        s = torch.linalg.svdvals(Hi)
        num = torch.sum(s[L:] ** 2)
        den = torch.sum(s ** 2) + 1e-12
        errs.append((num / den).real)
    return torch.stack(errs)