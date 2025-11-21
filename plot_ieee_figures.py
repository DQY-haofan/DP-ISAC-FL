# ============================================================
# 脚本名: plot_ieee_figures.py (v3.0 Final)
# 作用: 生成 9 张 IEEE 顶刊风格图表 (PDF + PNG 双格式)
# 包含: 核心结果(1-5) + 高级可视化(6-9)
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


# --- 1. 字体自动修复 ---
def install_and_set_font():
    font_path = 'Times_New_Roman.ttf'
    if not os.path.exists(font_path):
        # print("📥 Downloading Times New Roman...")
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

# --- 2. IEEE 绘图风格配置 ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [target_font, 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (3.5, 2.6),  # 标准单栏图
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

OUTPUT_DIR = 'ieee_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_dual_format(filename):
    """同时保存 PDF 和 PNG"""
    plt.savefig(f'{OUTPUT_DIR}/{filename}.pdf')
    plt.savefig(f'{OUTPUT_DIR}/{filename}.png')
    print(f"   Saved {filename} (.pdf & .png)")


def load_logs(pattern):
    files = glob.glob(pattern)
    return pd.concat([pd.read_csv(f) for f in files]) if files else None


# --- A. 核心结果 (Performance) ---

def plot_fig1_vulnerability():
    print("📊 Plotting Fig 1: Vulnerability...")
    df = load_logs('logs/exp1/*.csv')
    if df is None: return
    plt.figure()
    sns.lineplot(data=df, x='round', y='accuracy', hue='scenario', style='scenario',
                 palette=['#006400', '#d62728'], markers=False)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    plt.legend(frameon=True, edgecolor='black', fancybox=False)
    save_dual_format('fig1_vulnerability')
    plt.close()


def plot_fig2_efficacy():
    print("📊 Plotting Fig 2: Efficacy...")
    df = load_logs('logs/exp2/*.csv')
    if df is None: return
    plt.figure()
    palette = {'Ideal': '#2ca02c', 'Vulnerable': '#d62728', 'R-JORA': '#1f77b4'}
    sns.lineplot(data=df, x='round', y='accuracy', hue='scenario', style='scenario', palette=palette)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    plt.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)
    save_dual_format('fig2_efficacy')
    plt.close()


def plot_fig3_baselines():
    print("📊 Plotting Fig 3: Baselines...")
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
    plt.figure(figsize=(4, 3))
    # 使用填充纹理
    ax = sns.barplot(data=pd.DataFrame(data), x='Beta', y='Accuracy', hue='Method',
                     palette='Spectral', edgecolor='black', linewidth=0.8)
    hatches = ['/', '\\', '.', 'x', '+']
    for i, bar in enumerate(ax.patches):
        bar.set_hatch(hatches[int(i / 3) % len(hatches)])

    plt.ylim(0, 90)
    plt.xlabel("Malicious Ratio ($\\beta$)")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=8, frameon=False)
    save_dual_format('fig3_baselines')
    plt.close()


def plot_fig4_pru():
    print("📊 Plotting Fig 4: PRU Trade-off...")
    data = []
    for f in glob.glob('logs/exp4/*.csv'):
        df = pd.read_csv(f)
        if df['accuracy'].max() < 5.0: continue
        final_acc = df['accuracy'].iloc[-5:].mean()
        sigma = df['sigma_z'].iloc[0]
        method = df['scenario'].iloc[0].split('_')[0]
        data.append({'Sigma': sigma, 'Accuracy': final_acc, 'Method': method})

    if not data: return
    plt.figure()
    sns.lineplot(data=pd.DataFrame(data), x='Sigma', y='Accuracy', hue='Method', style='Method',
                 markers=True, markersize=7, palette={'Vulnerable': '#d62728', 'R-JORA': '#1f77b4'})
    plt.xscale('log')
    plt.xlabel("DP Noise $\\sigma_z$ (Log Scale)")
    plt.ylabel("Accuracy (%)")
    # 标注区域
    plt.axvspan(0.001, 0.01, color='gray', alpha=0.1)
    plt.text(0.0015, 20, "Privacy Leak", fontsize=8, color='gray', rotation=90)
    plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1)
    plt.text(0.6, 20, "Graph Collapse", fontsize=8, color='gray', rotation=90)
    save_dual_format('fig4_pru')
    plt.close()


def plot_fig5_ablation():
    print("📊 Plotting Fig 5: Ablation...")
    data = []
    for f in glob.glob('logs/exp5/*.csv'):
        df = pd.read_csv(f)
        data.append({'Config': df['scenario'].iloc[0], 'Accuracy': df['accuracy'].iloc[-5:].mean()})

    if not data: return
    plt.figure(figsize=(4, 3))
    order = ['Full', 'No-STGA', 'No-OptDP', 'No-ISAC']
    sns.barplot(data=pd.DataFrame(data), x='Config', y='Accuracy', order=order, palette="Blues_d", edgecolor='black')
    plt.ylim(40, 75)
    plt.xlabel(None)
    plt.xticks(rotation=15)
    plt.ylabel("Accuracy (%)")
    save_dual_format('fig5_ablation')
    plt.close()


# --- B. 深度机理可视化 (Mechanism) ---

def plot_fig6_tsne():
    if not os.path.exists('viz_data'): return
    print("🎨 Plotting Fig 6: t-SNE Analysis...")
    try:
        updates = np.load('viz_data/updates_r10.npy')  # 使用第10轮
        types = np.load('viz_data/client_types.npy')[:len(updates)]  # 对齐

        if updates.shape[0] < 5: return

        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        emb = tsne.fit_transform(updates)

        plt.figure(figsize=(4, 4))
        # 绘制良性
        benign_mask = (types == 'Benign')
        plt.scatter(emb[benign_mask, 0], emb[benign_mask, 1], c='green', label='Benign', alpha=0.6, s=30)
        # 绘制恶意
        mal_mask = (types == 'Malicious')
        plt.scatter(emb[mal_mask, 0], emb[mal_mask, 1], c='red', label='Malicious', marker='x', s=50)

        plt.title("Update Distribution (t-SNE)")
        plt.legend()
        plt.xlabel("Dim 1");
        plt.ylabel("Dim 2")
        plt.grid(False)  # t-SNE 通常不加网格
        save_dual_format('fig6_tsne')
        plt.close()
    except Exception as e:
        print(f"Skip t-SNE: {e}")


def plot_fig7_heatmap():
    if not os.path.exists('viz_data'): return
    print("🎨 Plotting Fig 7: Trust Heatmap...")

    weights_list = []
    rounds = []
    files = sorted(glob.glob('viz_data/weights_r*.npy'), key=lambda x: int(x.split('_r')[1].split('.')[0]))

    for f in files:
        w = np.load(f)
        if len(w) > 15: w = w[:15]  # 只展示前15个客户端
        weights_list.append(w)
        rounds.append(int(f.split('_r')[1].split('.')[0]))

    if not weights_list: return
    data = np.stack(weights_list).T

    plt.figure(figsize=(5, 3))
    # 反转颜色: 深色代表低权重(被防御)，浅色代表高权重
    sns.heatmap(data, cmap="Greys_r", vmax=0.15, cbar_kws={'label': 'STGA Weight'})
    plt.xlabel("Communication Round")
    plt.ylabel("Client ID")
    plt.title("Defense Dynamics")
    save_dual_format('fig7_heatmap')
    plt.close()


def plot_fig8_mask_evolution():
    if not os.path.exists('viz_data'): return
    print("🎨 Plotting Fig 8: ISAC Mask Evolution...")
    # 对比第0轮和第1轮的掩码
    try:
        m0 = np.load('viz_data/mask_r0.npy')[:20]  # 取前20个客户端
        m1 = np.load('viz_data/mask_r1.npy')[:20]

        fig, axes = plt.subplots(1, 2, figsize=(6, 2))

        sns.heatmap(m0.reshape(1, -1), ax=axes[0], cmap="binary", cbar=False, yticklabels=[])
        axes[0].set_title("Mask (Round t)")
        axes[0].set_xlabel("Client Index")

        sns.heatmap(m1.reshape(1, -1), ax=axes[1], cmap="binary", cbar=False, yticklabels=[])
        axes[1].set_title("Mask (Round t+1)")
        axes[1].set_xlabel("Client Index")

        save_dual_format('fig8_mask_evolution')
        plt.close()
    except:
        pass


def plot_fig9_weight_dist():
    if not os.path.exists('viz_data'): return
    print("🎨 Plotting Fig 9: Weight Distribution...")
    # 取最后一轮的权重
    try:
        last_file = sorted(glob.glob('viz_data/weights_r*.npy'), key=lambda x: int(x.split('_r')[1].split('.')[0]))[-1]
        w = np.load(last_file)
        types = np.load('viz_data/client_types.npy')[:len(w)]

        df = pd.DataFrame({'Weight': w, 'Type': types})

        plt.figure(figsize=(4, 3))
        sns.histplot(data=df, x='Weight', hue='Type', bins=20, palette={'Benign': 'green', 'Malicious': 'red'},
                     multiple="stack")
        plt.yscale('log')  # 对数坐标看清低权重
        plt.title("Weight Distribution (Final Round)")
        save_dual_format('fig9_weight_dist')
        plt.close()
    except:
        pass


if __name__ == "__main__":
    # 1. Core Results
    plot_fig1_vulnerability()
    plot_fig2_efficacy()
    plot_fig3_baselines()
    plot_fig4_pru()
    plot_fig5_ablation()

    # 2. Deep Visualization
    plot_fig6_tsne()
    plot_fig7_heatmap()
    plot_fig8_mask_evolution()
    plot_fig9_weight_dist()

    print(f"\n🎉 All 9 Figures (PDF+PNG) saved in '{OUTPUT_DIR}/'")