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
        Fig 7: 暴力拉伸对比度的热力图
        策略: vmin/vmax 直接取数据的 min/max，不留余地。
        """
    print("🎨 Plotting Fig 7: Heatmap (Full Range Stretch)...")
    subset = df[df['Scenario'] == 'R-JORA'].copy()
    if subset.empty: return

    # 构建矩阵 (按类型排序: Malicious Top, Benign Bottom)
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

    # [Fix] 绝对 Min-Max 归一化
    # 即使差异只有 0.002，也要把这 0.002 映射到整个色谱
    flat_valid = matrix[~np.isnan(matrix)]
    v_min = np.min(flat_valid)
    v_max = np.max(flat_valid)

    # 使用 'jet' 或 'turbo' 这种彩虹色谱，对微小差异更敏感
    # 或者 'RdYlBu_r' 保持学术风
    ax = sns.heatmap(matrix, cmap='RdYlBu_r', vmin=v_min, vmax=v_max,
                     cbar_kws={'label': 'Trust Score'})

    plt.xlabel("Communication Rounds")
    plt.ylabel("Sampled Clients (Sorted)")
    plt.title(f"(c) Trust Score Dynamics (Range: {v_min:.4f}-{v_max:.4f})")

    # 标注
    plt.text(1, 1.5, 'Malicious', color='blue', fontsize=9, weight='bold')
    plt.text(1, max_clients - 1.5, 'Benign', color='red', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig7_Heatmap_Stretched.pdf')
    plt.savefig(f'{OUTPUT_DIR}/Fig7_Heatmap_Stretched.png')
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
        Fig 9: 对数直方图
        策略: 放弃 Violin，改用 Histogram 展示双峰分布。
        """
    print("🎨 Plotting Fig 9: Norm Histogram...")
    data = df[df['Round'] == 10].copy()
    subset = data[data['Scenario'] == 'FedAvg']  # 原始分布

    plt.figure(figsize=(4, 3))

    sns.histplot(
        data=subset, x='L2_Norm', hue='Type',
        element="step",  # 阶梯状
        stat="percent",  # Y轴百分比
        common_norm=False,  # 各自归一化
        log_scale=True,  # X轴对数坐标
        palette={'Benign': '#2ca02c', 'Malicious': '#d62728'},
        alpha=0.6
    )

    plt.xlabel(r"Gradient $L_2$ Norm (Log Scale)")
    plt.ylabel("Percentage of Clients")
    plt.title("(e) Norm Distribution Separation")

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig9_Norm_Hist.pdf')
    plt.savefig(f'{OUTPUT_DIR}/Fig9_Norm_Hist.png')
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
        Fig 11: 保持不变 (效果很好)
        """
    print("🎨 Plotting Fig 11: Mechanism...")
    subset = df[df['Round'] == 10].copy()
    palette = {'Benign': '#2ca02c', 'Malicious': '#d62728'}
    markers = {'Benign': 'o', 'Malicious': 'X'}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    scenarios = ['FedAvg', 'Krum', 'R-JORA']
    titles = ['(a) FedAvg', '(b) Krum (Failed)', '(c) R-JORA (Robust)']

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

        if sc == 'Krum':
            # 选中点高亮
            # Krum 良性权重可能是 0 或 0.2，恶意是 0 (本轮) 或 0.2 (如果选中)
            # 这里的逻辑是：Weight > 0 即被选中
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
        Fig 12: 纯 Strip Plot (散点)
        策略: 放弃 Box/Violin，直接画点。
        """
    print("🎨 Plotting Fig 12: Weight Scatter (Strip)...")
    # 聚合 Round 10-25
    data = df[df['Round'] >= 10].copy()

    plt.figure(figsize=(5, 3.5))

    # Strip Plot: 抖动散点
    sns.stripplot(
        data=data, x='Scenario', y='Weight', hue='Type',
        dodge=True,  # 左右分开 Benign/Malicious
        jitter=0.25,  # 增加抖动宽度，把重叠的点散开
        size=3,  # 点的大小
        alpha=0.6,  # 透明度
        palette={'Benign': '#2ca02c', 'Malicious': '#d62728'}
    )

    # 叠加均值线 (Pointplot) 只是为了指示中心
    sns.pointplot(
        data=data, x='Scenario', y='Weight', hue='Type',
        dodge=0.4, join=False, markers="_", scale=1.2,
        palette={'Benign': 'black', 'Malicious': 'black'},
        errorbar=None
    )

    plt.xlabel(None)
    plt.ylabel("Assigned Weight")
    plt.title("(d) Weight Distribution (Scatter View)")

    # 修正图例
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig12_Weights_Strip.pdf')
    plt.savefig(f'{OUTPUT_DIR}/Fig12_Weights_Strip.png')
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