# 文件名: logger.py
# 作用: CSV 日志记录。
# 版本: Final

import csv, os


class ExperimentLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.head = ['round', 'accuracy', 'loss', 'stga_trust_benign', 'stga_trust_mal', 'isac_instability', 'scenario',
                     'seed', 'beta', 'sigma_z']
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f: csv.writer(f).writerow(self.head)

    def log_round(self, r, s, m):
        row = [r, s.get('accuracy'), s.get('loss'), s.get('stga_trust_benign'), s.get('stga_trust_mal'),
               s.get('isac_instability'), m.get('scenario'), m.get('seed'), m.get('beta'), m.get('sigma_z')]
        with open(self.path, 'a', newline='') as f: csv.writer(f).writerow(row)