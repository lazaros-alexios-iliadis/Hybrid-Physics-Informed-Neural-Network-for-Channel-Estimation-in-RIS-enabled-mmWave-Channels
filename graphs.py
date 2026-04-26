import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Settings
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

# Color scheme
COLORS = {
    'PINN': '#2E86AB',  # Blue
    'MLP': '#A23B72',  # Purple
    'UNF': '#F18F01',  # Orange
    'LS': '#C73E1D',  # Red
}


def plot_training_curve(logs_pinn, save_path='fig1_training_curve.pdf'):
    """
    PINN Training and Validation Loss Curves
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))

    epochs = np.arange(1, len(logs_pinn['val_nmse_h']) + 1)

    marker_idx = np.arange(0, len(epochs), 5)

    ax.semilogy(epochs, logs_pinn['val_nmse_h'],
                color=COLORS['PINN'], linewidth=2,
                label='Validation NMSE(H)', marker='o',
                markevery=marker_idx, markersize=5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('NMSE(H)')
    ax.set_title('Hybrid PINN Training Convergence')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    ax.set_xlim([1, len(epochs)])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_method_comparison_bar(results, save_path='fig2_method_comparison.pdf'):
    """
    Bar chart comparing all methods
    Shows NMSE(H), ObsMSE, and RankErr
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    methods = ['LS', 'PINN', 'MLP', 'UNF']
    x_pos = np.arange(len(methods))
    width = 0.6

    nmse_means = [results[m]['nmse_mean'] for m in methods]
    nmse_stds = [results[m]['nmse_std'] for m in methods]
    bars1 = axes[0].bar(x_pos, nmse_means, width,
                        color=[COLORS[m] for m in methods],
                        yerr=nmse_stds, capsize=5, alpha=0.8,
                        edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('NMSE(H)')
    axes[0].set_title('(a) Channel Estimation Error')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(methods)
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[0].set_ylim([0, max(nmse_means) * 1.2])

    best_idx = np.argmin(nmse_means)
    bars1[best_idx].set_edgecolor('gold')
    bars1[best_idx].set_linewidth(2.5)

    obs_means = [results[m]['obs_mse_mean'] for m in methods]
    obs_stds = [results[m]['obs_mse_std'] for m in methods]
    bars2 = axes[1].bar(x_pos, obs_means, width,
                        color=[COLORS[m] for m in methods],
                        yerr=obs_stds, capsize=5, alpha=0.8,
                        edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Observation MSE')
    axes[1].set_title('(b) Observation Fitting')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(methods)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[1].set_yscale('log')

    rank_means = [results[m]['rank_err_mean'] for m in methods]
    rank_stds = [results[m]['rank_err_std'] for m in methods]
    bars3 = axes[2].bar(x_pos, rank_means, width,
                        color=[COLORS[m] for m in methods],
                        yerr=rank_stds, capsize=5, alpha=0.8,
                        edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel('Rank Error')
    axes[2].set_title('(c) Low-Rank Structure')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(methods)
    axes[2].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[2].set_ylim([0, max(rank_means) * 1.2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_nmse_vs_snr(snr_results, save_path='fig3_nmse_vs_snr.pdf'):
    """
    NMSE(H) vs SNR for all methods
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))

    snr_values = sorted(snr_results.keys())
    methods = ['LS', 'PINN', 'MLP', 'UNF']

    for method in methods:
        means = [snr_results[snr][method][0] for snr in snr_values]
        stds = [snr_results[snr][method][1] for snr in snr_values]

        ax.errorbar(snr_values, means, yerr=stds,
                    marker='o', markersize=6, linewidth=2,
                    label=method, color=COLORS[method],
                    capsize=4, capthick=1.5, alpha=0.9)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('NMSE(H)')
    ax.set_title('Channel Estimation vs Signal-to-Noise Ratio')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim([min(snr_values) - 1, max(snr_values) + 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_nmse_vs_paths(path_results, save_path='fig4_nmse_vs_paths.pdf'):
    """
    NMSE(H) vs Number of Paths (Channel Sparsity) for all methods
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))

    L_values = sorted(path_results.keys())
    methods = ['LS', 'PINN', 'MLP', 'UNF']

    for method in methods:
        means = [path_results[L][method][0] for L in L_values]
        stds = [path_results[L][method][1] for L in L_values]

        ax.errorbar(L_values, means, yerr=stds,
                    marker='s', markersize=7, linewidth=2,
                    label=method, color=COLORS[method],
                    capsize=4, capthick=1.5, alpha=0.9)

    ax.set_xlabel('Number of Paths (L)')
    ax.set_ylabel('NMSE(H)')
    ax.set_title('Channel Estimation vs Channel Sparsity')
    ax.set_yscale('log')
    ax.set_xticks(L_values)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_box_comparison(results_list, labels, save_path='fig6_box_plots.pdf'):
    """
    Box plots showing NMSE distribution across test samples
    Requires raw data (list of NMSE values per method)
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    positions = np.arange(1, len(labels) + 1)
    bp = ax.boxplot(results_list, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red',
                                   markeredgecolor='red', markersize=6))

    # Color boxes
    colors_list = [COLORS[label] for label in labels]
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels)
    ax.set_ylabel('NMSE(H)')
    ax.set_title('Channel Estimation Error Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# Main plotting function
def generate_all_figures(logs_pinn, results_val, results_paths=None, results_snr=None):
    """
    Generate all figures
    """
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION FIGURES FOR EuCAP 2026")
    print("=" * 60 + "\n")

    plot_training_curve(logs_pinn)

    plot_method_comparison_bar(results_val)

    if results_snr is not None:
        plot_nmse_vs_snr(results_snr)

    if results_paths is not None:
        plot_nmse_vs_paths(results_paths)

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print("\nFigures saved")
