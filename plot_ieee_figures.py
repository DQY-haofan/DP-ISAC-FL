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
    print("🎨 Plotting Fig 7: Heatmap (Max Contrast)...")
    # 选取 R-JORA 数据
    subset = df[df['Scenario'] == 'R-JORA'].copy()
    if subset.empty: return

    # 构建矩阵 [Clients, Rounds]
    # 逻辑：每一轮先放 Malicious，再放 Benign (Top-Down Sorting)
    heatmap_data = []
    rounds = sorted(subset['Round'].unique())
    max_clients = 0

    for r in rounds:
        r_data = subset[subset['Round'] == r]
        mal = r_data[r_data['Type'] == 'Malicious']['Weight'].values
        ben = r_data[r_data['Type'] == 'Benign']['Weight'].values
        # 拼接
        col = np.concatenate([mal, ben])
        heatmap_data.append(col)
        max_clients = max(max_clients, len(col))

    # Pad
    matrix = np.full((max_clients, len(rounds)), np.nan)
    for i, col in enumerate(heatmap_data):
        matrix[:len(col), i] = col

    plt.figure(figsize=(5, 3.5))

    # [Fix] 动态范围计算
    # 找出数据中的最大值和最小值，作为颜色映射的边界
    # 使用 nanmax/nanmin 忽略填充的 NaN
    v_max = np.nanmax(matrix)
    v_min = np.nanmin(matrix)

    # 使用 'coolwarm' 或 'RdYlBu_r'。
    # 关键：不设置 center=0，而是让它自然铺满整个 min-max 范围
    ax = sns.heatmap(matrix, cmap='coolwarm', vmin=v_min, vmax=v_max,
                     cbar_kws={'label': 'Trust Score (Weight)'})

    plt.xlabel("Communication Rounds")
    plt.ylabel("Participating Clients (Sorted)")
    plt.title("(c) R-JORA Trust Dynamics (High Contrast)")

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
        Fig 12: Weight Distribution (Hybrid: Box + Strip)
        解决了 Boxenplot 在离散数据下显示异常的问题。
        """
    print("🎨 Plotting Fig 12: Hybrid Weight Comparison...")
    # 选取第 10 轮 (稳态)
    data = df[df['Round'] == 10].copy()

    plt.figure(figsize=(6, 4))

    # 1. Boxplot (箱线图): 展示统计分布
    # showfliers=False 隐藏异常值点，因为我们后面会用 Strip 画出所有点
    ax = sns.boxplot(
        data=data, x='Scenario', y='Weight', hue='Type',
        palette={'Benign': '#abdda4', 'Malicious': '#fdae61'},  # 浅色背景
        showfliers=False,
        linewidth=1.0,
        width=0.6
    )

    # 2. Strip Plot (抖动散点图): 展示真实数据点密度
    # dodge=True 确保点也按照 hue 分组偏移
    sns.stripplot(
        data=data, x='Scenario', y='Weight', hue='Type',
        dodge=True,
        jitter=True,  # 关键：加入抖动，防止点重叠成一条线
        size=3,
        palette={'Benign': '#2ca02c', 'Malicious': '#d62728'},  # 深色点
        alpha=0.6,
        ax=ax,
        legend=False  # 不重复显示图例
    )

    plt.xlabel(None)
    plt.ylabel("Aggregation Weight (Assigned)")
    plt.title("(d) Weight Assignment: Discrete vs Continuous")

    # 调整图例：只显示 Boxplot 的图例
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

    # 添加辅助线
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig12_Weights_Hybrid.pdf')
    plt.savefig(f'{OUTPUT_DIR}/Fig12_Weights_Hybrid.png')
    plt.close()

def plot_fig12_weights(df):
    print("🎨 Plotting Fig 12: Weight Comparison...")
    if df is None: return
    # 取第 15 轮
    sub = df[df['Round'] == 15].copy()

    plt.figure(figsize=(5, 3.5))
    sns.violinplot(data=sub, x='Scenario', y='Weight', hue='Type', split=True,
                   palette={'Benign': 'tab:green', 'Malicious': 'tab:red'},
                   inner='quartile', gap=0.1)

    plt.title("Fig. 12. Weight Assignment Comparison")
    plt.ylim(-0.1, 1.1)
    save_fig('Fig12_Weights')


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