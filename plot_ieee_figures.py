# ============================================================
# 脚本名: plot_ieee_figures_color.py (v4.0 Color Upgrade)
# 作用: 生成符合 IEEE 顶刊审美的彩色高清图表
# 改进: 移除灰度/纹理，使用专业的学术配色方案 (Viridis/Coolwarm)
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os
import urllib.request
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors


# --- 1. 字体加载 ---
def install_and_set_font():
    font_path = 'Times_New_Roman.ttf'
    if not os.path.exists(font_path):
        url = "https://github.com/michaelwecn/dotfiles/raw/master/.fonts/Times_New_Roman.ttf"
        try:
            urllib.request.urlretrieve(url, font_path)
        except:
            pass
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        return fm.FontProperties(fname=font_path).get_name()
    return 'serif'


target_font = install_and_set_font()

# --- 2. 顶刊现代风格配置 ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [target_font, 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (4, 3),  # 稍微加宽一点点
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,  # 网格淡一点
    'grid.linestyle': '--',
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

OUTPUT_DIR = 'ieee_figures_color'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_dual(filename):
    plt.savefig(f'{OUTPUT_DIR}/{filename}.pdf')
    plt.savefig(f'{OUTPUT_DIR}/{filename}.png')
    print(f"   Saved {filename} (Color)")


def load_logs(pattern):
    files = glob.glob(pattern)
    return pd.concat([pd.read_csv(f) for f in files]) if files else None


# --- 绘图逻辑 ---

def plot_fig1_vulnerability():
    print("🎨 Plotting Fig 1...")
    df = load_logs('logs/exp1/*.csv')
    if df is None: return
    plt.figure()
    # 对比色: 宝石绿 vs 砖红
    colors = {'Ideal': '#109648', 'Vulnerable': '#d62728'}
    sns.lineplot(data=df, x='round', y='accuracy', hue='scenario', style='scenario',
                 palette=colors, dashes={'Ideal': '', 'Vulnerable': (2, 1)}, lw=2.5)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    plt.legend(frameon=True, fancybox=False, edgecolor='0.8')
    save_dual('fig1_vulnerability')
    plt.close()


def plot_fig2_efficacy():
    print("🎨 Plotting Fig 2...")
    df = load_logs('logs/exp2/*.csv')
    if df is None: return
    plt.figure()
    # 方案色: 绿(上界), 红(下界), 蓝(Ours)
    palette = {'Ideal': '#2ca02c', 'Vulnerable': '#d62728', 'R-JORA': '#1f77b4'}
    sns.lineplot(data=df, x='round', y='accuracy', hue='scenario', palette=palette, lw=2)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    plt.legend(loc='lower right')
    save_dual('fig2_efficacy')
    plt.close()


def plot_fig3_baselines():
    print("🎨 Plotting Fig 3...")
    data = []
    for f in glob.glob('logs/exp3/*.csv'):
        df = pd.read_csv(f)
        final_acc = df['accuracy'].iloc[-5:].mean()
        name = os.path.basename(f)
        parts = name.split('_')
        method = parts[0]
        beta = float(parts[1].replace('beta', ''))
        data.append({'Method': method, 'Beta': beta, 'Accuracy': final_acc})

    if not data: return
    plt.figure(figsize=(5, 3.5))
    # 使用 Seaborn 的 "Paired" 或 "Set2" 配色，这种配色非常学术
    sns.barplot(data=pd.DataFrame(data), x='Beta', y='Accuracy', hue='Method',
                palette='Paired', edgecolor='black', linewidth=0.5)

    plt.ylim(0, 90)
    plt.xlabel("Malicious Ratio ($\\beta$)")
    plt.ylabel("Accuracy (%)")
    # 图例放上面，横向排列
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    save_dual('fig3_baselines')
    plt.close()


def plot_fig4_pru():
    print("🎨 Plotting Fig 4...")
    data = []
    files = glob.glob('logs/exp4/*.csv')
    if not files: return
    for f in files:
        df = pd.read_csv(f)
        if df['accuracy'].max() < 5.0: continue
        final_acc = df['accuracy'].iloc[-5:].mean()
        sigma = df['sigma_z'].iloc[0]
        method = df['scenario'].iloc[0].split('_')[0]
        data.append({'Sigma': sigma, 'Accuracy': final_acc, 'Method': method})

    plt.figure()
    sns.lineplot(data=pd.DataFrame(data), x='Sigma', y='Accuracy', hue='Method', style='Method',
                 markers=True, markersize=8, palette={'Vulnerable': '#d62728', 'R-JORA': '#1f77b4'}, lw=2)
    plt.xscale('log')
    plt.xlabel("DP Noise $\\sigma_z$ (Log Scale)")
    plt.ylabel("Accuracy (%)")

    # 区域背景色 (淡雅)
    plt.axvspan(0.001, 0.01, color='#fff59d', alpha=0.3)  # Yellow tint
    plt.text(0.0012, 20, "Privacy Risk", fontsize=9, color='#fbc02d', rotation=90)

    plt.axvspan(0.5, 1.0, color='#e0f2f1', alpha=0.5)  # Teal tint
    plt.text(0.6, 20, "Graph Collapse", fontsize=9, color='#00695c', rotation=90)

    save_dual('fig4_pru')
    plt.close()


def plot_fig5_ablation():
    print("🎨 Plotting Fig 5...")
    data = []
    for f in glob.glob('logs/exp5/*.csv'):
        df = pd.read_csv(f)
        data.append({'Config': df['scenario'].iloc[0], 'Accuracy': df['accuracy'].iloc[-5:].mean()})

    if not data: return
    plt.figure(figsize=(4.5, 3))
    order = ['Full', 'No-STGA', 'No-OptDP', 'No-ISAC']
    # 渐变蓝，表示完整度的缺失
    sns.barplot(data=pd.DataFrame(data), x='Config', y='Accuracy', order=order,
                palette="Blues_r", edgecolor='black', linewidth=0.8)
    plt.ylim(40, 75)
    plt.xlabel(None)
    plt.ylabel("Accuracy (%)")
    save_dual('fig5_ablation')
    plt.close()


# --- 高级可视化 ---

def plot_fig6_tsne():
    if not os.path.exists('viz_data'): return
    print("🎨 Plotting Fig 6: t-SNE...")
    try:
        updates = np.load('viz_data/updates_r10.npy')
        types = np.load('viz_data/client_types.npy')[:len(updates)]
        if updates.shape[0] < 5: return

        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        emb = tsne.fit_transform(updates)

        plt.figure(figsize=(4, 4))
        # 良性：蓝色圆点；恶意：红色三角
        plt.scatter(emb[types == 'Benign', 0], emb[types == 'Benign', 1],
                    c='#1f77b4', label='Benign', alpha=0.6, s=40, edgecolors='w')
        plt.scatter(emb[types == 'Malicious', 0], emb[types == 'Malicious', 1],
                    c='#d62728', label='Malicious', marker='^', s=60, edgecolors='k')

        plt.title("Feature Space (t-SNE)")
        plt.legend(loc='upper right')
        plt.xticks([]);
        plt.yticks([])  # 去掉刻度更干净
        save_dual('fig6_tsne')
        plt.close()
    except:
        pass


def plot_fig7_heatmap():
    if not os.path.exists('viz_data'): return
    print("🎨 Plotting Fig 7: Heatmap...")

    weights_hist = []
    files = sorted(glob.glob('viz_data/weights_r*.npy'), key=lambda x: int(x.split('_r')[1].split('.')[0]))
    for f in files:
        w = np.load(f)
        if len(w) > 20: w = w[:20]  # 展示前20个
        weights_hist.append(w)

    if not weights_hist: return
    data = np.stack(weights_hist).T

    plt.figure(figsize=(5, 3.5))
    # "Rocket" 或 "Mako" 是 Seaborn 非常现代的配色
    # 或者用 "RdYlBu_r" (红=低分/被杀, 蓝=高分/存活)
    sns.heatmap(data, cmap="RdYlBu", vmin=0, vmax=0.15,
                cbar_kws={'label': 'Trust Score'})

    plt.xlabel("Round")
    plt.ylabel("Client ID")
    plt.title("Dynamic Trust Scores")
    save_dual('fig7_heatmap')
    plt.close()


def plot_fig8_mask():
    if not os.path.exists('viz_data'): return
    print("🎨 Plotting Fig 8: Mask...")
    try:
        m0 = np.load('viz_data/mask_r0.npy')[:25]
        m1 = np.load('viz_data/mask_r1.npy')[:25]

        fig, axes = plt.subplots(2, 1, figsize=(5, 2.5), sharex=True)
        # 0=Invisible (White/Grey), 1=Visible (Dark Blue)
        sns.heatmap(m0.reshape(1, -1), ax=axes[0], cmap="Blues", cbar=False, linecolor='k', linewidths=0.5)
        axes[0].set_ylabel("Round $t$")
        axes[0].set_yticks([])

        sns.heatmap(m1.reshape(1, -1), ax=axes[1], cmap="Blues", cbar=False, linecolor='k', linewidths=0.5)
        axes[1].set_ylabel("Round $t+1$")
        axes[1].set_yticks([])

        plt.xlabel("Client Index (Dynamic Silo Effect)")
        plt.tight_layout()
        save_dual('fig8_mask')
        plt.close()
    except:
        pass


def plot_fig9_dist():
    if not os.path.exists('viz_data'): return
    print("🎨 Plotting Fig 9: Dist...")
    try:
        last_file = sorted(glob.glob('viz_data/weights_r*.npy'), key=lambda x: int(x.split('_r')[1].split('.')[0]))[-1]
        w = np.load(last_file)
        types = np.load('viz_data/client_types.npy')[:len(w)]

        plt.figure(figsize=(4, 3))
        # 堆叠直方图
        sns.histplot(x=w, hue=types, bins=15, multiple="stack",
                     palette={'Benign': '#1f77b4', 'Malicious': '#d62728'}, edgecolor='white')
        plt.yscale('log')
        plt.xlabel("Aggregated Weight")
        plt.ylabel("Count (Log Scale)")
        plt.title("Weight Suppression")
        save_dual('fig9_weight_dist')
        plt.close()
    except:
        pass


if __name__ == "__main__":
    plot_fig1_vulnerability()
    plot_fig2_efficacy()
    plot_fig3_baselines()
    plot_fig4_pru()
    plot_fig5_ablation()
    plot_fig6_tsne()
    plot_fig7_heatmap()
    plot_fig8_mask()
    plot_fig9_dist()
    print(f"\n🎉 Color figures saved in '{OUTPUT_DIR}/'")