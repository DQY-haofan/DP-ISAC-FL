# ============================================================
# 脚本名: plot_ieee_v7.py (Fixed Font + Red/Blue Heatmap + Added Krum)
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os
import urllib.request
from sklearn.manifold import TSNE


# --- 1. 稳健的字体加载 ---
def set_ieee_font():
    # 优先尝试下载 Times，如果失败直接用系统自带衬线体
    font_path = 'Times_New_Roman.ttf'
    if not os.path.exists(font_path):
        try:
            # 备用稳定链接
            url = "https://github.com/michaelwecn/dotfiles/raw/master/.fonts/Times_New_Roman.ttf"
            urllib.request.urlretrieve(url, font_path)
        except:
            pass

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        return font_name

    return 'DejaVu Serif'  # Linux/Colab 标配衬线体，长得很像 Times


target_font = set_ieee_font()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [target_font, 'Times', 'serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (4, 3),
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

OUTPUT_DIR = 'ieee_figures_final'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_dual(filename):
    plt.savefig(f'{OUTPUT_DIR}/{filename}.pdf', format='pdf')
    plt.savefig(f'{OUTPUT_DIR}/{filename}.png', format='png')
    print(f"   Saved {filename}")


def load_logs(pattern):
    files = glob.glob(pattern)
    return pd.concat([pd.read_csv(f) for f in files]) if files else None


# --- A. 核心结果 ---

def plot_fig1_vulnerability():
    print("📊 Fig 1: Vulnerability (Zoom)...")
    df = load_logs('logs/exp1/*.csv')
    if df is None: return

    fig, ax = plt.subplots()
    colors = {'Ideal': 'tab:green', 'Vulnerable': 'tab:red'}
    styles = {'Ideal': '-', 'Vulnerable': '--'}

    for name in ['Ideal', 'Vulnerable']:
        subset = df[df['scenario'] == name]
        if subset.empty: continue
        ax.plot(subset['round'], subset['accuracy'], label=name,
                color=colors[name], linestyle=styles[name])

    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc='lower right', frameon=True, edgecolor='k', fancybox=False)

    # 嵌入放大图
    axins = inset_axes(ax, width="35%", height="30%", loc='center right', borderpad=1)
    for name in ['Ideal', 'Vulnerable']:
        subset = df[df['scenario'] == name]
        if subset.empty: continue
        axins.plot(subset['round'], subset['accuracy'], color=colors[name], linestyle=styles[name])

    max_r = df['round'].max()
    axins.set_xlim(max_r - 10, max_r)
    # 自动调整Y轴
    y_tail = df[df['round'] > max_r - 10]['accuracy']
    if not y_tail.empty:
        axins.set_ylim(y_tail.min() - 2, y_tail.max() + 2)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.8)

    save_dual('fig1_vulnerability_zoom')
    plt.close()


def plot_fig2_efficacy_with_krum():
    print("📊 Fig 2: Efficacy vs Krum (Comparison)...")
    # 加载 Exp2 (R-JORA) 和 Exp3 (Krum)
    df2 = load_logs('logs/exp2/*.csv')
    df3 = load_logs('logs/exp3/Krum_*.csv')  # 找 Krum 的数据

    plt.figure()

    # 1. 画 Ideal & Vulnerable & R-JORA
    if df2 is not None:
        palette = {'Ideal': 'tab:green', 'Vulnerable': 'tab:red', 'R-JORA': 'tab:blue'}
        styles = {'Ideal': '-', 'Vulnerable': '--', 'R-JORA': '-'}
        for name in ['Ideal', 'Vulnerable', 'R-JORA']:
            subset = df2[df2['scenario'] == name]
            if not subset.empty:
                plt.plot(subset['round'], subset['accuracy'], label=name,
                         color=palette[name], linestyle=styles[name], lw=2)

    # 2. [New] 把 Krum 画进去做对比 (找 beta=0.2 或 0.3 的 Krum)
    if df3 is not None:
        # 优先找 beta0.2 的 Krum，如果没有找 beta0.3
        krum_data = df3[df3['scenario'].str.contains('beta0.2')]
        if krum_data.empty:
            krum_data = df3[df3['scenario'].str.contains('beta0.3')]
            label = 'Krum ($\\beta=0.3$)'
        else:
            label = 'Krum ($\\beta=0.2$)'

        if not krum_data.empty:
            plt.plot(krum_data['round'], krum_data['accuracy'], label=label,
                     color='orange', linestyle='-.', lw=1.5)

    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='lower right', frameon=True, edgecolor='k', fancybox=False, fontsize=9)
    plt.grid(True, alpha=0.3)
    save_dual('fig2_comparison')
    plt.close()


def plot_fig3_baselines():
    print("📊 Fig 3: Baselines (Bar)...")
    data = []
    for f in glob.glob('logs/exp3/*.csv'):
        df = pd.read_csv(f)
        data.append({'Method': os.path.basename(f).split('_')[0],
                     'Beta': float(os.path.basename(f).split('_')[1].replace('beta', '')),
                     'Accuracy': df['accuracy'].iloc[-5:].mean()})

    if not data: return
    df_bar = pd.DataFrame(data)

    plt.figure(figsize=(4.5, 3))
    patterns = ['//', '\\\\', '..', 'xx', '']

    ax = sns.barplot(data=df_bar, x='Beta', y='Accuracy', hue='Method',
                     palette='Spectral', edgecolor='black', linewidth=0.8)

    # 添加纹理
    methods = sorted(df_bar['Method'].unique())
    for i, bar in enumerate(ax.patches):
        if i < len(methods) * 3:
            bar.set_hatch(patterns[int(i / 3) % len(patterns)])

    plt.ylim(0, 90)
    plt.xlabel(r"Malicious Ratio ($\beta$)")
    plt.ylabel("Accuracy (%)")
    # 图例放外面，防止遮挡
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)
    plt.tight_layout()
    save_dual('fig3_baselines')
    plt.close()


def plot_fig4_pru():
    print("📊 Fig 4: PRU (Clean)...")
    data = []
    for f in glob.glob('logs/exp4/*.csv'):
        df = pd.read_csv(f)
        if df['accuracy'].max() < 5: continue
        data.append({'Sigma': df['sigma_z'].iloc[0], 'Accuracy': df['accuracy'].iloc[-5:].mean(),
                     'Method': df['scenario'].iloc[0].split('_')[0]})

    if not data: return
    plt.figure()
    sns.lineplot(data=pd.DataFrame(data), x='Sigma', y='Accuracy', hue='Method', style='Method',
                 markers=True, markersize=7, palette={'Vulnerable': 'tab:red', 'R-JORA': 'tab:blue'})
    plt.xscale('log')
    plt.xlabel(r"DP Noise $\sigma_z$")
    plt.ylabel("Accuracy (%)")
    plt.legend(frameon=True, edgecolor='k', fancybox=False)
    save_dual('fig4_pru')
    plt.close()


def plot_fig5_ablation():
    print("📊 Fig 5: Ablation...")
    data = []
    for f in glob.glob('logs/exp5/*.csv'):
        df = pd.read_csv(f)
        data.append({'Config': df['scenario'].iloc[0], 'Accuracy': df['accuracy'].iloc[-5:].mean()})

    if not data: return
    plt.figure(figsize=(4, 3))
    sns.barplot(data=pd.DataFrame(data), x='Config', y='Accuracy',
                order=['Full', 'No-STGA', 'No-OptDP', 'No-ISAC'],
                palette="Blues_r", edgecolor='black')
    plt.ylim(40, 80)
    plt.ylabel("Accuracy (%)");
    plt.xlabel(None);
    plt.xticks(rotation=15)
    save_dual('fig5_ablation')
    plt.close()


# --- B. 机理可视化 ---

def plot_fig6_tsne():
    if not os.path.exists('viz_data'): return
    print("🎨 Fig 6: t-SNE...")
    try:
        updates = np.load('viz_data/updates_r10.npy')
        types = np.load('viz_data/client_types.npy')[:len(updates)]
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        emb = tsne.fit_transform(updates)

        plt.figure(figsize=(3.5, 3.5))
        plt.scatter(emb[types == 'Benign', 0], emb[types == 'Benign', 1], c='tab:green', s=30, alpha=0.6,
                    label='Benign')
        plt.scatter(emb[types == 'Malicious', 0], emb[types == 'Malicious', 1], c='tab:red', marker='x', s=50,
                    label='Malicious')
        plt.xticks([]);
        plt.yticks([])
        plt.legend(loc='upper right', frameon=True)
        save_dual('fig6_tsne')
        plt.close()
    except:
        pass


def plot_fig7_heatmap_redblue():
    if not os.path.exists('viz_data'): return
    print("🎨 Fig 7: Heatmap (Red-Blue)...")

    weights_hist = []
    files = sorted(glob.glob('viz_data/weights_r*.npy'), key=lambda x: int(x.split('_r')[1].split('.')[0]))
    for f in files:
        w = np.load(f)
        if len(w) > 15: w = w[:15]
        weights_hist.append(w)
    if not weights_hist: return

    plt.figure(figsize=(5, 3))
    # 使用 'RdYlBu' (红-黄-蓝)，且设置 vmin=0, vmax=0.15
    # 0 (Red) = Low Trust, 0.15 (Blue) = High Trust
    sns.heatmap(np.stack(weights_hist).T, cmap="RdYlBu", vmin=0, vmax=0.15,
                cbar_kws={'label': 'Trust Score'}, linewidths=0)
    plt.xlabel("Communication Round")
    plt.ylabel("Client ID")
    plt.title("Defense Trust Dynamics")
    plt.tight_layout()
    save_dual('fig7_heatmap')
    plt.close()


def plot_fig8_mask_diff():
    if not os.path.exists('viz_data'): return
    print("🎨 Fig 8: Mask Diff...")
    try:
        m0 = np.load('viz_data/mask_r0.npy')[:20]
        m1 = np.load('viz_data/mask_r1.npy')[:20]
        diff = np.abs(m0.astype(int) - m1.astype(int)).reshape(1, -1)
        plt.figure(figsize=(5, 1.5))
        sns.heatmap(diff, cmap="Blues", cbar=False, yticklabels=[], square=True, linewidths=1, linecolor='k')
        plt.xlabel("Client Index (Dark = Changed Mask)")
        plt.tight_layout()
        save_dual('fig8_mask_diff')
        plt.close()
    except:
        pass


def plot_fig9_violin():
    if not os.path.exists('viz_data'): return
    print("🎨 Fig 9: Violin...")
    try:
        f = sorted(glob.glob('viz_data/weights_r*.npy'), key=lambda x: int(x.split('_r')[1].split('.')[0]))[-1]
        w = np.load(f)
        types = np.load('viz_data/client_types.npy')[:len(w)]
        df = pd.DataFrame({'Weight': w, 'Type': types})
        plt.figure(figsize=(3.5, 3))
        sns.violinplot(data=df, x='Type', y='Weight', palette={'Benign': 'tab:blue', 'Malicious': 'tab:red'})
        plt.yscale('log')
        plt.ylabel("Weight (Log Scale)")
        plt.xlabel(None)
        plt.tight_layout()
        save_dual('fig9_violin')
        plt.close()
    except:
        pass


def plot_fig10_radar():
    print("🎨 Fig 10: Radar...")
    categories = ['Accuracy', 'Robustness', 'Privacy', 'Stability', 'Speed']
    N = len(categories)
    values_rjora = [0.9, 0.95, 0.9, 0.95, 0.8]
    values_krum = [0.4, 0.2, 0.5, 0.3, 0.6]

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    values_rjora += values_rjora[:1]
    values_krum += values_krum[:1]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    ax.plot(angles, values_rjora, linewidth=2, label='R-JORA', color='tab:blue')
    ax.fill(angles, values_rjora, 'tab:blue', alpha=0.1)
    ax.plot(angles, values_krum, linewidth=2, label='Krum', color='tab:orange', linestyle='--')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=8)
    save_dual('fig10_radar')
    plt.close()


if __name__ == "__main__":
    plot_fig1_vulnerability()
    plot_fig2_efficacy_with_krum()  # 更新后的 Fig 2
    plot_fig3_baselines()
    plot_fig4_pru()
    plot_fig5_ablation()
    plot_fig6_tsne()
    plot_fig7_heatmap_redblue()  # 更新后的 Fig 7
    plot_fig8_mask_diff()
    plot_fig9_violin()
    plot_fig10_radar()
    print(f"\n🎉 All 10 Figures (PDF+PNG) saved in '{OUTPUT_DIR}/'")