# 文件名: optimal_dp.py
# 作用: R-JORA 子模块 2 - 最优 DP 噪声搜索。
# 版本: Final

import math

class OptimalDPSearcher:
    def __init__(self, config):
        self.conf = config['r_jora']
        self.C_dp = self.conf['const_c_dp']
        self.C_attack = self.conf['const_c_attack']
        self.sigma_eaves_sq = config['attack']['eaves_sigma'] ** 2
        self.search_min = self.conf['dp_search_min']
        self.search_max = self.conf['dp_search_max']

    def error_pru(self, x):
        """Objective Function: Error = C_DP * x + C_Attack / (x + sigma_e)^2"""
        denom = (x + self.sigma_eaves_sq) ** 2 + 1e-9
        return self.C_dp * x + self.C_attack / denom

    def find_optimal_sigma2(self):
        if not self.conf['enabled'] or not self.conf['enable_optimal_dp']:
            return None

        a, b = self.search_min, self.search_max
        tol = 1e-4
        invphi = (math.sqrt(5) - 1) / 2
        invphi2 = (3 - math.sqrt(5)) / 2

        c = a + invphi2 * (b - a)
        d = a + invphi * (b - a)
        fc = self.error_pru(c)
        fd = self.error_pru(d)

        while (b - a) > tol:
            if fc < fd:
                b, d, fd = d, c, fc
                c = a + invphi2 * (b - a)
                fc = self.error_pru(c)
            else:
                a, c, fc = c, d, fd
                d = a + invphi * (b - a)
                fd = self.error_pru(d)

        return (a + b) / 2