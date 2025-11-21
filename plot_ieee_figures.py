# ============================================================
# 脚本名: plot_ieee_figures.py (v2.0 Fixed)
# 作用: 自动修复字体问题 + 生成 IEEE 顶刊风格图表
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os
import urllib.request


# --- 1. 核心修复：自动下载并注册 Times New Roman 字体 ---
def install_and_set_font():
    font_path = 'Times_New_Roman.ttf'
    # 检查是否已下载，没有则从 GitHub 镜像下载
    if not os.path.exists(font_path):
        print("📥 Downloading Times New Roman font for IEEE style...")
        url = "https://github.com/michaelwecn/dotfiles/raw/master/.fonts/Times_New_Roman.ttf"
        try:
            urllib.request.urlretrieve(url, font_path)
        except Exception as e:
            print(f"⚠️ Font download failed: {e}. Using fallback.")

    # 动态添加字体 (无需重启 runtime)
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        # 确认字体名称
        prop = fm.FontProperties(fname=font_path)
        font_name = prop.get_name()  # 通常是 'Times New Roman'
        print(f"✅ Font '{font_name}' registered successfully!")
        return font_name
    return 'serif'  # 回退方案


# 执行字体安装
target_font = install_and_set_font()

# --- 2. IEEE 顶刊绘图风格配置 ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [target_font, 'Times', 'DejaVu Serif', 'serif'],  # 优先使用 Times
    'mathtext.fontset': 'stix',  # 让数学公式 ($...$) 看起来像 LaTeX
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (3.5, 2.6),  # IEEE 标准单栏宽度 (3.5 inch)
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'savefig.dpi': 600,  # 顶刊要求的高 DPI
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'axes.axisbelow': True  # 网格线在数据下方
})

OUTPUT_DIR = 'ieee_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_logs(pattern):
    files = glob.glob(pattern)
    if not files: return None
    return pd.concat([pd.read_csv(f) for f in files])


# --- 绘图函数 (增强版) ---

def plot_exp1_vulnerability():
    print("Plotting Fig 1: Vulnerability (IEEE Style)...")
    df = load_logs('logs/exp1/*.csv')
    if df is None: return

    plt.figure()
    # 使用不同线型和标记，方便黑白打印识别
    sns.lineplot(data=df, x='round', y='accuracy', hue='scenario', style='scenario',
                 palette=['#006400', '#8B0000'],  # 深绿, 深红
                 dashes={'Ideal': (None, None), 'Vulnerable': (2, 2)},
                 markers=False)

    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    # 移除标题 (顶刊通常在 Caption 中写标题，图上不写，或者写得很小)
    # plt.title("VGAE Attack Impact")
    plt.legend(frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig1_vulnerability.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig1_vulnerability.png')
    plt.close()


def plot_exp2_efficacy():
    print("Plotting Fig 2: Efficacy (IEEE Style)...")
    df = load_logs('logs/exp2/*.csv')
    if df is None: return

    plt.figure()
    # 专业的学术配色
    palette = {'Ideal': '#2ca02c', 'Vulnerable': '#d62728', 'R-JORA': '#1f77b4'}
    styles = {'Ideal': '', 'Vulnerable': (2, 2), 'R-JORA': (1, 1)}

    sns.lineplot(data=df, x='round', y='accuracy', hue='scenario', style='scenario',
                 palette=palette, dashes=styles)

    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    # 将图例放在右下角，避免遮挡曲线
    plt.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig2_efficacy.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig2_efficacy.png')
    plt.close()


def plot_exp3_baselines():
    print("Plotting Fig 3: Baselines (IEEE Style)...")
    data = []
    files = glob.glob('logs/exp3/*.csv')
    if not files: return

    for f in files:
        df = pd.read_csv(f)
        final_acc = df['accuracy'].iloc[-5:].mean()
        name = os.path.basename(f)
        parts = name.split('_')
        method = parts[0]
        beta = float(parts[1].replace('beta', ''))
        data.append({'Method': method, 'Beta': beta, 'Accuracy': final_acc})

    df_bar = pd.DataFrame(data)

    plt.figure(figsize=(4, 3))
    # 使用填充纹理 (Hatching) 区分柱状图，这在黑白打印时非常有用
    # 注意: Seaborn 对 hatch 支持一般，这里用原生 matplotlib 微调
    ax = sns.barplot(data=df_bar, x='Beta', y='Accuracy', hue='Method',
                     palette='Spectral', edgecolor='black', linewidth=0.8)

    # 为每个柱子添加纹理
    hatches = ['/', '\\', 'x', '.', '+']
    for i, bar in enumerate(ax.patches):
        # 简单的纹理循环
        hatch = hatches[int(i / 3) % len(hatches)]
        bar.set_hatch(hatch)

    plt.ylim(0, 85)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Malicious Ratio ($\\beta$)")  # LaTeX 格式
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3,
               fontsize=8, frameon=False, handletextpad=0.2)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_baselines.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig3_baselines.png')
    plt.close()


def plot_exp4_pru():
    print("Plotting Fig 4: PRU Trade-off (IEEE Style)...")
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

    df_line = pd.DataFrame(data)

    plt.figure()
    # 使用带标记的折线图
    sns.lineplot(data=df_line, x='Sigma', y='Accuracy', hue='Method', style='Method',
                 markers=True, dashes=False, markersize=6,
                 palette={'Vulnerable': '#d62728', 'R-JORA': '#1f77b4'})

    plt.xscale('log')
    plt.xlabel("DP Noise $\\sigma_z$ (Log Scale)")
    plt.ylabel("Accuracy (%)")

    # 添加语义区域标注 (IEEE 风格)
    plt.axvline(x=0.01, color='gray', linestyle=':', linewidth=1)
    plt.text(0.0015, 15, "Privacy Leak", fontsize=8, color='gray')

    plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1)
    plt.text(0.6, 15, "Graph Collapse", fontsize=8, color='gray')

    plt.legend(loc='best', frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_pru.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig4_pru.png')
    plt.close()


def plot_exp5_ablation():
    print("Plotting Fig 5: Ablation (IEEE Style)...")
    data = []
    files = glob.glob('logs/exp5/*.csv')
    if not files: return

    for f in files:
        df = pd.read_csv(f)
        final_acc = df['accuracy'].iloc[-5:].mean()
        scen = df['scenario'].iloc[0]
        data.append({'Configuration': scen, 'Accuracy': final_acc})

    df_ab = pd.DataFrame(data)

    plt.figure(figsize=(4, 3))
    order = ['Full', 'No-STGA', 'No-OptDP', 'No-ISAC']
    # 使用单色渐变，显得更稳重
    ax = sns.barplot(data=df_ab, x='Configuration', y='Accuracy', order=order,
                     palette="Blues_d", edgecolor='black')

    plt.ylabel("Accuracy (%)")
    plt.xlabel(None)
    plt.ylim(40, 75)
    plt.xticks(rotation=15)  # 稍微倾斜标签
    plt.grid(axis='x')  # 仅横向网格
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig5_ablation.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig5_ablation.png')
    plt.close()


# --- 高级可视化 (需 viz_data) ---
def plot_trust_heatmap():
    if not os.path.exists('viz_data'): return
    print("Plotting Fig 7: Heatmap (IEEE Style)...")

    weights_hist = []
    rounds = []
    files = sorted(glob.glob('viz_data/weights_r*.npy'),
                   key=lambda x: int(x.split('_r')[1].replace('.npy', '')))

    for f in files:
        w = np.load(f)
        # 取前15个客户端 (假设前3个是恶意，或者混杂)
        # 注意：这里仅作演示，具体ID取决于 run_harvest 的记录
        if len(w) >= 15:
            weights_hist.append(w[:15])
            r = int(f.split('_r')[1].replace('.npy', ''))
            rounds.append(r)

    if not weights_hist: return
    data = np.stack(weights_hist).T  # (15, Rounds)

    plt.figure(figsize=(5, 3))
    # 使用 viridis 或 cividis (对色盲友好)
    sns.heatmap(data, cmap="Greys", vmax=0.2, cbar_kws={'label': 'Trust Score'})

    plt.xlabel("Communication Rounds")
    plt.ylabel("Client Index")
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig7_heatmap.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig7_heatmap.png')
    plt.close()


# --- 执行 ---
if __name__ == "__main__":
    plot_exp1_vulnerability()
    plot_exp2_efficacy()
    plot_exp3_baselines()
    plot_exp4_pru()
    plot_exp5_ablation()
    plot_trust_heatmap()
    print(f"🎉 IEEE-style figures generated in '{OUTPUT_DIR}'")