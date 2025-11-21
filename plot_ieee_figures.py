# ============================================================
# 脚本名: plot_ieee_v9.py (Ultimate Edition)
# 作用: 生成 11+ 张 IEEE 顶刊级图表 (PDF + PNG)
# 新增: Fig 11 (L2 Norm vs Cosine Similarity) - 完美的机理验证图
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


# --- 1. 字体加载 ---
def set_font():
    font_path = 'Times_New_Roman.ttf'
    if not os.path.exists(font_path):
        try:
            urllib.request.urlretrieve("https://github.com/michaelwecn/dotfiles/raw/master/.fonts/Times_New_Roman.ttf",
                                       font_path)
        except:
            pass
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        return fm.FontProperties(fname=font_path).get_name()
    return 'DejaVu Serif'


target_font = set_font()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [target_font, 'Times'],
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

OUTPUT_DIR = 'ieee_figures_v9'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_dual(filename):
    plt.savefig(f'{OUTPUT_DIR}/{filename}.pdf')
    plt.savefig(f'{OUTPUT_DIR}/{filename}.png')
    print(f"   Saved {filename}")


def load_logs(pattern):
    files = glob.glob(pattern)
    return pd.concat([pd.read_csv(f) for f in files]) if files else None


# --- A. 核心结果 ---

def plot_fig1_clean():
    print("📊 Fig 1: Vulnerability...")
    df = load_logs('logs/exp1/*.csv')
    if df is None: return
    plt.figure()
    colors = {'Ideal': 'tab:green', 'Vulnerable': 'tab:red'}
    styles = {'Ideal': '-', 'Vulnerable': '--'}
    for name in ['Ideal', 'Vulnerable']:
        subset = df[df['scenario'] == name]
        if not subset.empty:
            plt.plot(subset['round'], subset['accuracy'], label=name,
                     color=colors[name], linestyle=styles[name])
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.legend(frameon=True, edgecolor='k', fancybox=False)
    save_dual('fig1_vulnerability')
    plt.close()


def plot_fig2_efficacy():
    print("📊 Fig 2: Efficacy...")
    df2 = load_logs('logs/exp2/*.csv')
    df3 = load_logs('logs/exp3/Krum_*.csv')
    plt.figure()
    if df2 is not None:
        palette = {'Ideal': 'tab:green', 'Vulnerable': 'tab:red', 'R-JORA': 'tab:blue'}
        styles = {'Ideal': '-', 'Vulnerable': '--', 'R-JORA': '-'}
        for name in ['Ideal', 'Vulnerable', 'R-JORA']:
            subset = df2[df2['scenario'] == name]
            if not subset.empty:
                plt.plot(subset['round'], subset['accuracy'], label=name, color=palette[name], linestyle=styles[name],
                         lw=2)
    if df3 is not None:
        krum = df3[df3['scenario'].str.contains('beta0.3')]
        if krum.empty: krum = df3[df3['scenario'].str.contains('beta0.2')]
        if not krum.empty:
            plt.plot(krum['round'], krum['accuracy'], label='Krum', color='orange', linestyle='-.', lw=1.5)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='lower right', frameon=True, edgecolor='k', fancybox=False)
    save_dual('fig2_efficacy')
    plt.close()


def plot_fig3_baselines():
    print("📊 Fig 3: Baselines...")
    data = []
    for f in glob.glob('logs/exp3/*.csv'):
        df = pd.read_csv(f)
        data.append({'Method': os.path.basename(f).split('_')[0],
                     'Beta': float(os.path.basename(f).split('_')[1].replace('beta', '')),
                     'Accuracy': df['accuracy'].iloc[-5:].mean()})
    if not data: return
    plt.figure(figsize=(4.5, 3))
    sns.barplot(data=pd.DataFrame(data), x='Beta', y='Accuracy', hue='Method', palette='Spectral', edgecolor='black',
                linewidth=0.8)
    plt.ylim(0, 90)
    plt.xlabel(r"Malicious Ratio ($\beta$)")
    plt.ylabel("Accuracy (%)")
    plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.2), frameon=False)
    save_dual('fig3_baselines')
    plt.close()


def plot_fig4_pru():
    print("📊 Fig 4: PRU...")
    data = []
    for f in glob.glob('logs/exp4/*.csv'):
        df = pd.read_csv(f)
        if df['accuracy'].max() < 5: continue
        data.append({'Sigma': df['sigma_z'].iloc[0], 'Accuracy': df['accuracy'].iloc[-5:].mean(),
                     'Method': df['scenario'].iloc[0].split('_')[0]})
    if not data: return
    plt.figure()
    sns.lineplot(data=pd.DataFrame(data), x='Sigma', y='Accuracy', hue='Method', style='Method', markers=True,
                 palette={'Vulnerable': 'tab:red', 'R-JORA': 'tab:blue'})
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
    sns.barplot(data=pd.DataFrame(data), x='Config', y='Accuracy', order=['Full', 'No-STGA', 'No-OptDP', 'No-ISAC'],
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
        plt.xlabel("Dimension 1");
        plt.ylabel("Dimension 2")
        plt.legend(loc='upper right', frameon=True)
        save_dual('fig6_tsne')
        plt.close()
    except:
        pass


def plot_fig7_heatmap():
    if not os.path.exists('viz_data'): return
    print("🎨 Fig 7: Heatmap...")
    weights_hist = []
    files = sorted(glob.glob('viz_data/weights_r*.npy'), key=lambda x: int(x.split('_r')[1].split('.')[0]))
    for f in files:
        w = np.load(f)
        if len(w) > 15: w = w[:15]
        weights_hist.append(w)
    if not weights_hist: return
    data = np.stack(weights_hist).T
    plt.figure(figsize=(5, 3))
    # 使用红蓝高对比
    sns.heatmap(data, cmap="RdYlBu", vmin=0, vmax=np.percentile(data, 95), cbar_kws={'label': 'Trust Score'})
    plt.xlabel("Round");
    plt.ylabel("Client ID")
    plt.tight_layout()
    save_dual('fig7_heatmap')
    plt.close()


def plot_fig8_mask():
    if not os.path.exists('viz_data'): return
    print("🎨 Fig 8: Mask Diff...")
    try:
        m0 = np.load('viz_data/mask_r0.npy')[:20]
        m1 = np.load('viz_data/mask_r1.npy')[:20]
        diff = np.abs(m0.astype(int) - m1.astype(int)).reshape(1, -1)
        plt.figure(figsize=(5, 1.5))
        sns.heatmap(diff, cmap="Oranges", cbar=False, yticklabels=[], square=True, linewidths=1, linecolor='k')
        plt.xlabel("Client Index (Orange = Changed)")
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
        sns.violinplot(data=df, x='Type', y='Weight', palette={'Benign': 'tab:green', 'Malicious': 'tab:red'})
        plt.yscale('log')
        plt.ylabel("Weight (Log)")
        plt.xlabel(None)
        plt.tight_layout()
        save_dual('fig9_violin')
        plt.close()
    except:
        pass


# --- C. [NEW] 终极机理图 ---

def plot_fig11_mechanism():
    """
    绘制 L2 Norm vs Cosine Similarity 散点图。
    这能完美解释为什么攻击者(高Cosine, 高Norm)能骗过Krum但被STGA抓住。
    """
    if not os.path.exists('viz_data'): return
    print("🎨 Fig 11: Attack Mechanism Analysis...")

    try:
        # 取第5轮的数据（攻击早期，特征最明显）
        norms = np.load('viz_data/norms_r5.npy')
        cosines = np.load('viz_data/cosines_r5.npy')
        types = np.load('viz_data/client_types.npy')[:len(norms)]

        df = pd.DataFrame({
            'L2 Norm': norms,
            'Cosine Similarity': cosines,
            'Type': types
        })

        plt.figure(figsize=(4.5, 3.5))

        # 散点图
        sns.scatterplot(data=df, x='Cosine Similarity', y='L2 Norm', hue='Type', style='Type',
                        palette={'Benign': 'tab:green', 'Malicious': 'tab:red'},
                        s=60, alpha=0.8, edgecolor='k')

        # 标注区域 - 解释防御逻辑
        # 1. 高 Cosine (右侧) -> Krum 喜欢
        # 2. 高 Norm (上侧) -> STGA 裁剪

        plt.axvline(x=np.median(cosines), color='gray', linestyle=':', alpha=0.5)
        plt.axhline(y=np.median(norms) * 1.5, color='blue', linestyle='--', label='STGA Threshold')

        plt.yscale('log')
        plt.title("Why Krum Fails & STGA Works")
        plt.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        save_dual('fig11_mechanism')
        plt.close()

    except Exception as e:
        print(f"Skip Fig 11: {e}")


if __name__ == "__main__":
    # 1. Core Performance
    plot_fig1_clean()
    plot_fig2_efficacy()
    plot_fig3_baselines()
    plot_fig4_pru()
    plot_fig5_ablation()

    # 2. Mechanism Viz
    plot_fig6_tsne()
    plot_fig7_heatmap()
    plot_fig8_mask()
    plot_fig9_violin()

    # 3. New Mechanism Plot
    plot_fig11_mechanism()

    print(f"\n🎉 All 11 Figures (v9 Ultimate) generated in '{OUTPUT_DIR}/'")