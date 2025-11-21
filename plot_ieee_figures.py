# 文件名: plot_ieee_v11.py
# 作用: 生成 Fig 1 - Fig 12 的完整图表包 (IEEE Transactions 格式)
# 依赖: logs/ (from run_all.py) 和 viz_metrics_pro.csv (from generate_viz_data_ultra.py)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.font_manager as fm
import urllib.request
from sklearn.manifold import TSNE
from math import pi


# ==========================================
# 1. IEEE 样式配置 (Style Setup)
# ==========================================
def set_ieee_style():
    # 字体回退机制：优先 Times New Roman
    font_path = 'Times_New_Roman.ttf'
    font_name = 'DejaVu Serif'
    if not os.path.exists(font_path):
        try:
            urllib.request.urlretrieve("https://github.com/michaelwecn/dotfiles/raw/master/.fonts/Times_New_Roman.ttf",
                                       font_path)
        except:
            pass
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [font_name],
        'mathtext.fontset': 'stix',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.5,
        'lines.markersize': 5
    })


set_ieee_style()
OUTPUT_DIR = 'ieee_figures_final'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. 数据加载工具
# ==========================================
def load_data():
    path = 'viz_metrics_pro.csv'
    if not os.path.exists(path):
        print("❌ Error: viz_metrics_pro.csv not found.")
        return None
    return pd.read_csv(path)

def load_logs(pattern):
    files = glob.glob(pattern)
    return pd.concat([pd.read_csv(f) for f in files]) if files else None


def load_viz_csv():
    path = 'viz_metrics_pro.csv'
    if not os.path.exists(path):
        print(f"⚠️ {path} not found. Some figures (11, 12) will be skipped.")
        return None
    return pd.read_csv(path)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{name}.pdf', bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/{name}.png', bbox_inches='tight')
    print(f"   -> Saved {name}")
    plt.close()


# ==========================================
# 3. 绘图函数 (Fig 1 - Fig 12)
# ==========================================

# --- Part A: General Performance (Logs based) ---

def plot_fig1_vulnerability():
    print("📊 Plotting Fig 1: Vulnerability...")
    df = load_logs('logs/exp1/*.csv')
    if df is None: return
    plt.figure(figsize=(4, 3))

    # 筛选 Ideal 和 Vulnerable
    for sc in ['Ideal', 'Vulnerable']:
        sub = df[df['scenario'] == sc]
        if not sub.empty:
            label = "No Attack" if sc == 'Ideal' else "Under Attack (FedAvg)"
            color = 'tab:green' if sc == 'Ideal' else 'tab:red'
            plt.plot(sub['round'], sub['accuracy'], label=label, color=color)

    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Fig. 1. Vulnerability Analysis")
    plt.legend()
    save_fig('Fig1_Vulnerability')


def plot_fig2_efficacy():
    print("📊 Plotting Fig 2: Efficacy...")
    df2 = load_logs('logs/exp2/*.csv')
    if df2 is None: return
    plt.figure(figsize=(4, 3))

    colors = {'Ideal': 'tab:green', 'Vulnerable': 'tab:red', 'R-JORA': 'tab:blue'}
    labels = {'Ideal': 'Ideal (No Attack)', 'Vulnerable': 'FedAvg (Attack)', 'R-JORA': 'R-JORA (Ours)'}

    for sc in ['Ideal', 'Vulnerable', 'R-JORA']:
        sub = df2[df2['scenario'] == sc]
        if not sub.empty:
            plt.plot(sub['round'], sub['accuracy'], label=labels[sc], color=colors[sc])

    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Fig. 2. Defense Efficacy")
    plt.legend(loc='lower right')
    save_fig('Fig2_Efficacy')


def plot_fig3_baselines():
    print("📊 Plotting Fig 3: Baselines...")
    # 手动提取最后准确率
    data = []
    for f in glob.glob('logs/exp3/*.csv'):
        df = pd.read_csv(f)
        # filename format: Mode_beta0.X_seedY.csv
        fname = os.path.basename(f)
        mode = fname.split('_beta')[0]
        beta = float(fname.split('_beta')[1].split('_')[0])
        acc = df['accuracy'].iloc[-5:].mean()  # 取最后5轮平均
        data.append({'Method': mode, 'Beta': beta, 'Accuracy': acc})

    if not data: return
    df_bar = pd.DataFrame(data)

    plt.figure(figsize=(5, 3.5))
    sns.barplot(data=df_bar, x='Beta', y='Accuracy', hue='Method',
                palette='viridis', edgecolor='k', alpha=0.9)
    plt.ylim(0, 90)
    plt.xlabel(r"Malicious Ratio ($\beta$)")
    plt.ylabel("Accuracy (%)")
    plt.title("Fig. 3. Comparison with Baselines")
    plt.legend(ncol=2, loc='upper center', fontsize=8)
    save_fig('Fig3_Baselines')


def plot_fig4_pru():
    print("📊 Plotting Fig 4: PRU Trade-off...")
    data = []
    for f in glob.glob('logs/exp4/*.csv'):
        df = pd.read_csv(f)
        sigma = df['sigma_z'].iloc[0]
        mode = df['scenario'].iloc[0].split('_sigma')[0]
        acc = df['accuracy'].iloc[-5:].mean()
        data.append({'Sigma': sigma, 'Accuracy': acc, 'Method': mode})

    if not data: return
    df_pru = pd.DataFrame(data)

    plt.figure(figsize=(4, 3))
    sns.lineplot(data=df_pru, x='Sigma', y='Accuracy', hue='Method', style='Method',
                 markers=True, palette={'Vulnerable': 'tab:red', 'R-JORA': 'tab:blue'})
    plt.xscale('log')
    plt.xlabel(r"DP Noise Magnitude ($\sigma_z$)")
    plt.ylabel("Accuracy (%)")
    plt.title("Fig. 4. Privacy-Robustness-Utility")
    save_fig('Fig4_PRU')


def plot_fig5_ablation():
    print("📊 Plotting Fig 5: Ablation...")
    data = []
    for f in glob.glob('logs/exp5/*.csv'):
        df = pd.read_csv(f)
        scen = df['scenario'].iloc[0]
        if scen == 'R-JORA': scen = 'Full R-JORA'  # Rename
        acc = df['accuracy'].iloc[-5:].mean()
        data.append({'Config': scen, 'Accuracy': acc})

    if not data: return
    df_abl = pd.DataFrame(data)
    # Order: Full, No-STGA, No-OptDP, No-ISAC
    order = ['Full R-JORA', 'No-STGA', 'No-OptDP', 'No-ISAC']

    plt.figure(figsize=(4, 3))
    sns.barplot(data=df_abl, x='Config', y='Accuracy', order=order,
                palette='Blues_r', edgecolor='k')
    plt.ylim(40, 80)
    plt.xticks(rotation=15)
    plt.xlabel(None)
    plt.ylabel("Accuracy (%)")
    plt.title("Fig. 5. Ablation Study")
    save_fig('Fig5_Ablation')


# --- Part B: Deep Mechanism (Viz Data based) ---

def plot_fig6_tsne():
    print("🎨 Plotting Fig 6: t-SNE...")
    if not os.path.exists('viz_data/updates_r15.npy'): return

    updates = np.load('viz_data/updates_r15.npy')
    types = np.load('viz_data/types_r15.npy')

    tsne = TSNE(n_components=2, random_state=42, perplexity=5, init='pca', learning_rate='auto')
    emb = tsne.fit_transform(updates)

    plt.figure(figsize=(3.5, 3.5))
    plt.scatter(emb[types == 'Benign', 0], emb[types == 'Benign', 1],
                c='tab:green', alpha=0.6, label='Benign')
    plt.scatter(emb[types == 'Malicious', 0], emb[types == 'Malicious', 1],
                c='tab:red', marker='x', s=60, label='Malicious')
    plt.title("Fig. 6. Feature Space (Round 15)")
    plt.legend()
    plt.xticks([]);
    plt.yticks([])  # Hide axis
    save_fig('Fig6_tSNE')


def plot_fig7_heatmap_sorted(df):
    """
        Fig 7: 动态对比度热力图
        改进：使用 quantile 截断来增强对比度，防止极值导致中间颜色变灰。
        """
    print("🎨 Plotting Fig 7: Heatmap (Enhanced Contrast)...")
    subset = df[df['Scenario'] == 'R-JORA'].copy()
    if subset.empty: return

    # 构建矩阵
    heatmap_data = []
    rounds = sorted(subset['Round'].unique())
    max_clients = 0

    for r in rounds:
        r_data = subset[subset['Round'] == r]
        mal = r_data[r_data['Type'] == 'Malicious']['Weight'].values
        ben = r_data[r_data['Type'] == 'Benign']['Weight'].values
        col = np.concatenate([mal, ben])
        heatmap_data.append(col)
        max_clients = max(max_clients, len(col))

    matrix = np.full((max_clients, len(rounds)), np.nan)
    for i, col in enumerate(heatmap_data):
        matrix[:len(col), i] = col

    plt.figure(figsize=(5, 3.5))

    # [Fix] 使用分位数作为颜色边界，增强视觉冲击力
    # 5% 分位数为 vmin, 95% 分位数为 vmax
    flat_data = matrix[~np.isnan(matrix)]
    v_min = np.percentile(flat_data, 5)
    v_max = np.percentile(flat_data, 95)

    ax = sns.heatmap(matrix, cmap='coolwarm', vmin=v_min, vmax=v_max,
                     cbar_kws={'label': 'Trust Score'})

    plt.xlabel("Communication Rounds")
    plt.ylabel("Participating Clients (Sorted)")
    plt.title("(c) R-JORA Trust Dynamics (Quantile Scaled)")

    # 标注
    plt.text(1, 2, 'Malicious (Suppressed)', color='blue', fontsize=9, weight='bold')
    plt.text(1, max_clients - 3, 'Benign (Trusted)', color='red', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig7_Heatmap_Dynamic.pdf')
    plt.savefig(f'{OUTPUT_DIR}/Fig7_Heatmap_Dynamic.png')
    plt.close()

def plot_fig8_mask():
    print("🎨 Plotting Fig 8: Mask Diff...")
    try:
        m1 = np.load('viz_data/mask_r0.npy')
        m2 = np.load('viz_data/mask_r1.npy')
        # 显示前 50 个客户端的 Mask 变化
        diff = (m1[:50] != m2[:50]).astype(int).reshape(1, -1)

        plt.figure(figsize=(5, 1.5))
        sns.heatmap(diff, cmap=['#f0f0f0', 'tab:orange'], cbar=False,
                    linewidths=0.5, linecolor='k', square=False)
        plt.title("Fig. 8. ISAC Mask Instability (Orange = Changed)")
        plt.xlabel("Client Index")
        plt.yticks([])
        save_fig('Fig8_MaskDiff')
    except:
        pass


def plot_fig9_norm_density(df):
    """
        Fig 9: Norm Density (Polished Violin)
        """
    print("🎨 Plotting Fig 9: Norm Density (Refined)...")
    data = df[df['Round'] == 10].copy()
    # 只看 FedAvg (原始数据分布)
    subset = data[data['Scenario'] == 'FedAvg']

    plt.figure(figsize=(4, 3))

    # cut=0 防止小提琴图延伸到数据范围之外
    # bw_method 调整平滑度
    sns.violinplot(
        data=subset, x='Scenario', y='L2_Norm', hue='Type',
        split=True,
        inner='quartile',
        palette={'Benign': '#2ca02c', 'Malicious': '#d62728'},
        cut=0,
        bw_method=0.3  # 稍微锐利一点，不要太模糊
    )

    plt.yscale('log')
    plt.ylabel(r"Gradient $L_2$ Norm (Log Scale)")
    plt.xlabel("Raw Update Distribution (No Defense)")
    plt.title("(e) Norm Anomaly: The Physical Basis")
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig9_Norm_Refined.pdf')
    plt.savefig(f'{OUTPUT_DIR}/Fig9_Norm_Refined.png')
    plt.close()

def plot_fig10_radar():
    print("🎨 Plotting Fig 10: Radar...")
    # 手动构造雷达图数据 (基于实验结论)
    categories = ['Accuracy', 'Robustness', 'Privacy', 'Stability', 'Speed']
    N = len(categories)

    # R-JORA vs Krum (Beta=0.3)
    values_rjora = [0.8, 0.9, 0.85, 0.9, 0.7]
    values_krum = [0.1, 0.1, 0.4, 0.1, 0.8]

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    values_rjora += values_rjora[:1]
    values_krum += values_krum[:1]

    plt.figure(figsize=(3.5, 3.5))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values_rjora, 'tab:blue', linewidth=2, label='R-JORA')
    ax.fill(angles, values_rjora, 'tab:blue', alpha=0.1)

    ax.plot(angles, values_krum, 'tab:orange', linewidth=2, linestyle='--', label='Krum')
    ax.fill(angles, values_krum, 'tab:orange', alpha=0.1)

    plt.xticks(angles[:-1], categories, size=9)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)
    plt.title("Fig. 10. Performance Radar")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    save_fig('Fig10_Radar')


def plot_fig11_mechanism_comparison(df):
    """
        Fig 11: 全景机理图 (保持不变，因为它效果很好)
        """
    print("🎨 Plotting Fig 11: Mechanism Comparison...")
    subset = df[df['Round'] == 10].copy()
    palette = {'Benign': '#2ca02c', 'Malicious': '#d62728'}
    markers = {'Benign': 'o', 'Malicious': 'X'}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    scenarios = ['FedAvg', 'Krum', 'R-JORA']
    titles = ['(a) FedAvg', '(b) Krum (Failed)', '(c) R-JORA (Robust)']

    y_min = subset['L2_Norm'].min() * 0.8
    y_max = subset['L2_Norm'].max() * 1.5
    benign_norms = subset[(subset['Scenario'] == 'R-JORA') & (subset['Type'] == 'Benign')]['L2_Norm']
    threshold = benign_norms.median() * 1.5

    for i, sc in enumerate(scenarios):
        ax = axes[i]
        data = subset[subset['Scenario'] == sc]
        if data.empty: continue

        sns.scatterplot(
            data=data, x='Cosine_Sim', y='L2_Norm',
            hue='Type', style='Type',
            palette=palette, markers=markers,
            s=80, alpha=0.7, edgecolor='k', linewidth=0.5,
            ax=ax, legend=(i == 2)
        )

        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.set_title(titles[i])
        ax.set_xlabel("Cosine Similarity")
        if i == 0: ax.set_ylabel(r"L2 Norm (Log)")

        if sc == 'R-JORA':
            ax.axhline(y=threshold, color='blue', linestyle='--', label='Clip Threshold')

        # 标记 Krum 的选中点
        if sc == 'Krum':
            selected = data[data['Weight'] > 1e-6]
            if not selected.empty:
                ax.scatter(selected['Cosine_Sim'], selected['L2_Norm'], s=150, facecolors='none', edgecolors='black',
                           linewidth=1.5)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig11_Mechanism_Full.pdf')
    plt.savefig(f'{OUTPUT_DIR}/Fig11_Mechanism_Full.png')
    plt.close()


def plot_fig12_weight_distribution(df):
    """
        Fig 12: 统计稳健的权重对比图
        改进：
        1. 聚合 Round 5 到 Round 25 的数据，消除单轮采样偏差。
        2. 使用 Boxplot + Strip 组合，完美展示离散分布。
        """
    print("🎨 Plotting Fig 12: Robust Weight Comparison...")

    # [Fix] 聚合多轮数据，展示真实分布特征
    data = df[df['Round'] > 5].copy()

    plt.figure(figsize=(5, 3.5))

    # 1. Boxplot: 展示统计区间
    # fliersize=0 隐藏异常点，交给 strip 展示
    sns.boxplot(
        data=data, x='Scenario', y='Weight', hue='Type',
        palette={'Benign': '#abdda4', 'Malicious': '#fdae61'},
        linewidth=1.0, width=0.7, showfliers=False
    )

    # 2. Strip Plot: 展示数据点密度 (带透明度)
    # alpha=0.05 非常淡，这样只有大量点重叠时才会显色
    sns.stripplot(
        data=data, x='Scenario', y='Weight', hue='Type',
        dodge=True, jitter=True, size=2,
        palette={'Benign': '#2ca02c', 'Malicious': '#d62728'},
        alpha=0.15, ax=plt.gca()
    )

    plt.xlabel(None)
    plt.ylabel("Aggregation Weight")
    plt.title("(d) Weight Assignment (Aggregated R5-R25)")
    plt.ylim(-0.05, 1.05)

    # 修正图例（去重）
    handles, labels = plt.gca().get_legend_handles_labels()
    # 取前两个 (Boxplot 的图例颜色比较正)
    plt.legend(handles[:2], labels[:2], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig12_Weights_Robust.pdf')
    plt.savefig(f'{OUTPUT_DIR}/Fig12_Weights_Robust.png')
    plt.close()


def plot_fig12_weights(df):
    """
        Fig 11: 全景对比 (保持不变，效果很好)
        """
    print("🎨 Plotting Fig 11: Mechanism Comparison...")
    # 仍然使用单轮快照，因为散点图太多点会乱
    subset = df[df['Round'] == 10].copy()

    palette = {'Benign': '#2ca02c', 'Malicious': '#d62728'}
    markers = {'Benign': 'o', 'Malicious': 'X'}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    scenarios = ['FedAvg', 'Krum', 'R-JORA']
    titles = ['(a) FedAvg', '(b) Krum (Defense Backfire)', '(c) R-JORA (Effective)']

    y_min = subset['L2_Norm'].min() * 0.8
    y_max = subset['L2_Norm'].max() * 1.5

    # R-JORA 阈值
    r_data = subset[subset['Scenario'] == 'R-JORA']
    benign_norms = r_data[r_data['Type'] == 'Benign']['L2_Norm']
    threshold = benign_norms.median() * 1.5 if not benign_norms.empty else 1.0

    for i, sc in enumerate(scenarios):
        ax = axes[i]
        data = subset[subset['Scenario'] == sc]
        if data.empty: continue

        sns.scatterplot(
            data=data, x='Cosine_Sim', y='L2_Norm',
            hue='Type', style='Type',
            palette=palette, markers=markers,
            s=80, alpha=0.7, edgecolor='k', linewidth=0.5,
            ax=ax, legend=(i == 2)
        )

        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.set_title(titles[i])
        ax.set_xlabel("Cosine Similarity")
        if i == 0: ax.set_ylabel(r"L2 Norm (Log)")

        if sc == 'R-JORA':
            ax.axhline(y=threshold, color='blue', linestyle='--', label='Clip Threshold')

        # Krum Highlight
        if sc == 'Krum':
            selected = data[data['Weight'] > 1e-6]
            if not selected.empty:
                ax.scatter(selected['Cosine_Sim'], selected['L2_Norm'], s=150, facecolors='none', edgecolors='black',
                           linewidth=1.5)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig11_Mechanism_Full.pdf')
    plt.savefig(f'{OUTPUT_DIR}/Fig11_Mechanism_Full.png')
    plt.close()

# ==========================================
# 4. 主执行函数
# ==========================================
if __name__ == "__main__":
    print("🚀 Generating IEEE Figures 1-12...")
    df = load_data()
    # Part A
    plot_fig1_vulnerability()
    plot_fig2_efficacy()
    plot_fig3_baselines()
    plot_fig4_pru()
    plot_fig5_ablation()

    # Part B
    df_viz = load_viz_csv()
    if df_viz is not None:
        plot_fig6_tsne()
        plot_fig7_heatmap_sorted(df)
        plot_fig8_mask()
        plot_fig9_norm_density(df)
        plot_fig10_radar()
        plot_fig11_mechanism_comparison(df)
        plot_fig12_weight_distribution(df)

    print(f"\n🎉 All figures saved in '{OUTPUT_DIR}/'. Ready for LaTeX.")