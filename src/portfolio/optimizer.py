"""投資組合優化

Mean-Variance Optimization，用 ML 預測報酬率作為期望值。
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Mean-Variance 組合優化器"""

    def __init__(
        self,
        risk_free_rate: float = 0.015,
        max_weight: float = 0.25,
    ):
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        stock_ids: list[str] | None = None,
        method: str = "max_sharpe",
    ) -> dict:
        """最佳化投資組合權重

        Args:
            expected_returns: 各股票預期報酬率 (n,)
            cov_matrix: 報酬率共變異數矩陣 (n, n)
            stock_ids: 股票代號（用於標記）
            method: "max_sharpe" | "min_variance" | "equal_weight"

        Returns:
            {"weights": ndarray, "expected_return": float, "risk": float, "sharpe": float}
        """
        n = len(expected_returns)
        if stock_ids is None:
            stock_ids = [f"stock_{i}" for i in range(n)]

        if method == "equal_weight":
            weights = np.ones(n) / n
        elif method == "min_variance":
            weights = self._min_variance(cov_matrix)
        elif method == "max_sharpe":
            weights = self._max_sharpe_cvxpy(expected_returns, cov_matrix)
        else:
            weights = np.ones(n) / n

        # 限制單一權重
        weights = np.clip(weights, 0, self.max_weight)
        weights /= weights.sum()

        # 計算組合指標
        port_return = weights @ expected_returns
        port_risk = np.sqrt(weights @ cov_matrix @ weights)
        sharpe = (port_return - self.risk_free_rate) / (port_risk + 1e-8)

        result = {
            "weights": dict(zip(stock_ids, weights.tolist())),
            "expected_return": float(port_return),
            "risk": float(port_risk),
            "sharpe_ratio": float(sharpe),
        }

        logger.info("組合優化結果: %s", result)
        return result

    def _min_variance(self, cov_matrix: np.ndarray) -> np.ndarray:
        """最小變異數組合（解析解）"""
        n = cov_matrix.shape[0]
        try:
            inv_cov = np.linalg.inv(cov_matrix + np.eye(n) * 1e-8)
        except np.linalg.LinAlgError:
            return np.ones(n) / n

        ones = np.ones(n)
        weights = inv_cov @ ones / (ones @ inv_cov @ ones)
        return np.maximum(weights, 0)

    def _max_sharpe_cvxpy(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """最大 Sharpe ratio 組合（cvxpy 求解，fallback 解析解）

        Constraints: long-only, sum(w)=1, w <= max_weight
        """
        n = len(expected_returns)

        try:
            import cvxpy as cp

            w = cp.Variable(n)
            ret = expected_returns @ w
            risk = cp.quad_form(w, cov_matrix)

            # 最大化 Sharpe ≈ 最小化 risk subject to ret >= target
            # 用 max ret - lambda * risk 近似
            objective = cp.Maximize(ret - 0.5 * risk)
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                w <= self.max_weight,
            ]

            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS, verbose=False)

            if prob.status == "optimal" and w.value is not None:
                weights = np.array(w.value).flatten()
                return np.maximum(weights, 0)

            logger.warning("CVXPY 未收斂 (status=%s)，fallback 到解析解", prob.status)
        except ImportError:
            logger.debug("cvxpy 未安裝，使用解析解")
        except Exception as e:
            logger.warning("CVXPY 求解失敗: %s，fallback 到解析解", e)

        # Fallback: 解析解
        return self._max_sharpe_analytic(expected_returns, cov_matrix)

    def _max_sharpe_analytic(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """最大 Sharpe ratio 組合（解析解）"""
        n = len(expected_returns)
        excess_returns = expected_returns - self.risk_free_rate

        try:
            inv_cov = np.linalg.inv(cov_matrix + np.eye(n) * 1e-8)
        except np.linalg.LinAlgError:
            return np.ones(n) / n

        weights = inv_cov @ excess_returns
        if weights.sum() != 0:
            weights /= weights.sum()
        else:
            weights = np.ones(n) / n

        return np.maximum(weights, 0)

    def efficient_frontier(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_points: int = 50,
    ) -> dict:
        """計算效率前緣（用 cvxpy 逐點求解）

        Returns:
            {"returns": list, "risks": list}
        """
        n = len(expected_returns)
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_returns = []
        frontier_risks = []

        try:
            import cvxpy as cp

            for target in target_returns:
                w = cp.Variable(n)
                risk = cp.quad_form(w, cov_matrix)
                constraints = [
                    cp.sum(w) == 1,
                    w >= 0,
                    w <= self.max_weight,
                    expected_returns @ w >= target,
                ]
                prob = cp.Problem(cp.Minimize(risk), constraints)
                prob.solve(solver=cp.SCS, verbose=False)

                if prob.status == "optimal" and w.value is not None:
                    weights = np.array(w.value).flatten()
                    port_risk = float(np.sqrt(weights @ cov_matrix @ weights))
                    frontier_returns.append(float(target))
                    frontier_risks.append(port_risk)
        except ImportError:
            logger.debug("cvxpy 未安裝，用近似效率前緣")
            for target in target_returns:
                weights = self._min_variance(cov_matrix)
                risk = float(np.sqrt(weights @ cov_matrix @ weights))
                frontier_returns.append(float(target))
                frontier_risks.append(risk)

        return {
            "returns": frontier_returns,
            "risks": frontier_risks,
        }
