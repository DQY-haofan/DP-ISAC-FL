# 文件名: plotter.py
# 作用: 自动绘图工具。
# 版本: Final

import matplotlib.pyplot as plt
import pandas as pd
import glob, os, seaborn as sns


class ResultPlotter:
    def __init__(self):
        os.makedirs('figures', exist_ok=True)
        sns.set_style("whitegrid")

    def _ld(self, p):
        files = glob.glob(p)
        return pd.concat([pd.read_csv(f) for f in files]) if files else None

    def plot_all(self):
        print("Plotting...")
        # Exp 1
        df = self._ld('logs/exp1/*.csv')
        if df is not None:
            plt.figure();
            sns.lineplot(data=df, x='round', y='accuracy', hue='scenario');
            plt.savefig('figures/exp1.png');
            plt.close()

        # Exp 2
        df = self._ld('logs/exp2/*.csv')
        if df is not None:
            plt.figure();
            sns.lineplot(data=df, x='round', y='accuracy', hue='scenario');
            plt.savefig('figures/exp2.png');
            plt.close()

        # Exp 3 (Bar)
        d3 = []
        for f in glob.glob('logs/exp3/*.csv'):
            d = pd.read_csv(f)
            d3.append({'beta': d['beta'].iloc[0], 'mode': d['scenario'].iloc[0].split('_beta')[0],
                       'acc': d['accuracy'].iloc[-1]})
        if d3:
            plt.figure();
            sns.barplot(data=pd.DataFrame(d3), x='beta', y='acc', hue='mode');
            plt.savefig('figures/exp3.png');
            plt.close()

        # Exp 4 (Line)
        d4 = []
        for f in glob.glob('logs/exp4/*.csv'):
            d = pd.read_csv(f)
            d4.append({'sigma': d['sigma_z'].iloc[0], 'mode': d['scenario'].iloc[0].split('_sigma')[0],
                       'acc': d['accuracy'].iloc[-1]})
        if d4:
            plt.figure();
            sns.lineplot(data=pd.DataFrame(d4), x='sigma', y='acc', hue='mode', marker='o');
            plt.xscale('log');
            plt.savefig('figures/exp4.png');
            plt.close()

        # Exp 5 (Bar)
        d5 = []
        for f in glob.glob('logs/exp5/*.csv'):
            d = pd.read_csv(f)
            d5.append({'conf': d['scenario'].iloc[0], 'acc': d['accuracy'].iloc[-1]})
        if d5:
            plt.figure(figsize=(10, 6));
            sns.barplot(data=pd.DataFrame(d5), x='conf', y='acc');
            plt.savefig('figures/exp5.png');
            plt.close()