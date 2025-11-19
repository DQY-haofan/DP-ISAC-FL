# 文件名: run_all.py
# 作用: 主程序入口。执行所有实验并绘图。
# 版本: Final

import argparse
from runner import SimulationRunner
from plotter import ResultPlotter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all')
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()

    runner = SimulationRunner('config.yaml')
    print(f"=== Starting Experiments (Seeds={args.seeds}) ===")

    if args.exp in ['1', 'all']: runner.run_exp1_vulnerability(args.seeds)
    if args.exp in ['2', 'all']: runner.run_exp2_efficacy(args.seeds)
    if args.exp in ['3', 'all']: runner.run_exp3_baselines(args.seeds)
    if args.exp in ['4', 'all']: runner.run_exp4_pru_tradeoff()
    if args.exp in ['5', 'all']: runner.run_exp5_ablation(args.seeds)

    print("=== All Done. Generating Plots... ===")
    plotter = ResultPlotter()
    plotter.plot_all()
    print("Done! Check 'figures/' directory.")

if __name__ == "__main__": main()