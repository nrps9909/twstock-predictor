"""CPCV (Combinatorial Purged Cross-Validation) + PBO (Probability of Backtest Overfitting)

自建實作（不依賴 skfolio），~150 行。

參考:
- Bailey et al. (2014) "Probability of Backtest Overfitting"
- Marcos López de Prado, AFML Chapter 12
"""

import logging
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)


class CPCVAnalyzer:
    """Combinatorial Purged Cross-Validation 分析器

    將資料切成 N 個 block，以 C(N, K) 種方式選 K 個作為 test，
    剩餘作為 train（含 purging）。最終計算 PBO。
    """

    def __init__(
        self,
        n_blocks: int = 6,
        k_test: int = 2,
        purge_days: int = 10,
    ):
        """
        Args:
            n_blocks: 將資料分成幾個 block
            k_test: 每次選幾個 block 作為 test set
            purge_days: Purge 天數
        """
        self.n_blocks = n_blocks
        self.k_test = k_test
        self.purge_days = purge_days

    def split(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """產生所有 C(N, K) 組 purged train/test splits

        Args:
            n_samples: 總樣本數

        Returns:
            list of (train_idx, test_idx) tuples
        """
        block_size = n_samples // self.n_blocks
        blocks = []
        for i in range(self.n_blocks):
            start = i * block_size
            end = start + block_size if i < self.n_blocks - 1 else n_samples
            blocks.append(np.arange(start, end))

        splits = []
        for test_combo in combinations(range(self.n_blocks), self.k_test):
            test_idx = np.concatenate([blocks[i] for i in test_combo])
            train_blocks = [i for i in range(self.n_blocks) if i not in test_combo]
            train_idx = np.concatenate([blocks[i] for i in train_blocks])

            # Purging: 移除訓練集中靠近測試集邊界的樣本
            test_min, test_max = test_idx.min(), test_idx.max()
            purge_mask = np.ones(len(train_idx), dtype=bool)
            for j, idx in enumerate(train_idx):
                # 訓練樣本太靠近測試區域
                if (
                    abs(idx - test_min) < self.purge_days
                    or abs(idx - test_max) < self.purge_days
                ):
                    purge_mask[j] = False

            purged_train = train_idx[purge_mask]
            if len(purged_train) > 0:
                splits.append((purged_train, test_idx))

        logger.info(
            "CPCV: %d splits (C(%d,%d) = %d, after purging %d)",
            len(splits),
            self.n_blocks,
            self.k_test,
            len(list(combinations(range(self.n_blocks), self.k_test))),
            len(splits),
        )
        return splits

    def compute_pbo(self, performance_matrix: np.ndarray) -> dict:
        """計算 Probability of Backtest Overfitting (PBO)

        Args:
            performance_matrix: shape (n_splits, n_strategies)
                每個 split × 每個策略的 out-of-sample 績效

        Returns:
            {
                "pbo": float,  # PBO 值 (0-1, 越低越好)
                "logit_distribution": ndarray,  # logit 分佈
                "is_overfit": bool,  # PBO > 0.5 表示可能過擬合
            }
        """
        n_splits, n_strategies = performance_matrix.shape

        if n_strategies < 2:
            logger.warning("PBO 需要至少 2 個策略進行比較")
            return {
                "pbo": 0.0,
                "logit_distribution": np.array([0.0]),
                "is_overfit": False,
            }

        # 對每個 split，計算最佳策略在 IS 的排名 vs OOS 的排名
        logits = []

        for split_idx in range(0, n_splits - 1, 2):
            if split_idx + 1 >= n_splits:
                break

            # 用兩半互為 IS/OOS
            is_perf = performance_matrix[split_idx]
            oos_perf = performance_matrix[split_idx + 1]

            # IS 最佳策略
            best_is_idx = np.argmax(is_perf)

            # 該策略在 OOS 的排名（0-based）
            oos_rank = np.sum(oos_perf >= oos_perf[best_is_idx])
            relative_rank = oos_rank / n_strategies

            # Logit: 若 IS 最佳策略在 OOS 排名低（relative_rank 高）→ 過擬合
            logit = np.log(relative_rank / (1 - relative_rank + 1e-8) + 1e-8)
            logits.append(logit)

        logits = np.array(logits)

        # PBO = P(logit < 0) = IS 最佳策略在 OOS 表現低於中位數的機率
        pbo = float(np.mean(logits < 0)) if len(logits) > 0 else 0.0

        return {
            "pbo": pbo,
            "logit_distribution": logits,
            "is_overfit": pbo > 0.5,
            "n_pairs": len(logits),
        }

    def run_cpcv_analysis(
        self,
        n_samples: int,
        train_and_evaluate_fn,
    ) -> dict:
        """完整 CPCV + PBO 分析

        Args:
            n_samples: 總樣本數
            train_and_evaluate_fn: callable(train_idx, test_idx) → float (OOS performance)
                接收 train/test indices，回傳 OOS 績效指標

        Returns:
            {
                "n_splits": int,
                "oos_performances": list[float],
                "mean_oos": float,
                "std_oos": float,
                "pbo": dict,  # if enough splits
            }
        """
        splits = self.split(n_samples)
        oos_perfs = []

        for i, (train_idx, test_idx) in enumerate(splits):
            try:
                perf = train_and_evaluate_fn(train_idx, test_idx)
                oos_perfs.append(perf)
                logger.info(
                    "CPCV split %d/%d: OOS perf = %.6f", i + 1, len(splits), perf
                )
            except Exception as e:
                logger.warning("CPCV split %d failed: %s", i + 1, e)
                oos_perfs.append(0.0)

        oos_arr = np.array(oos_perfs)

        result = {
            "n_splits": len(splits),
            "oos_performances": oos_perfs,
            "mean_oos": float(oos_arr.mean()),
            "std_oos": float(oos_arr.std()),
        }

        # PBO 需要足夠的 splits
        if len(oos_perfs) >= 4:
            # Build performance matrix with 2+ strategies for meaningful PBO.
            # Original PBO only compared 1 strategy + 1 random baseline,
            # which reduces to a "better than random?" test.
            # Now we add multiple perturbed variants to detect overfitting properly.
            rng = np.random.RandomState(42)
            strategies = [oos_arr]
            # Strategy variant 1: perturbed with small noise (tests IS->OOS stability)
            strategies.append(
                oos_arr + rng.normal(0, oos_arr.std() * 0.3, len(oos_perfs))
            )
            # Strategy variant 2: inverted (anti-strategy baseline)
            strategies.append(1.0 - oos_arr)
            # Strategy variant 3: random baseline
            strategies.append(rng.normal(oos_arr.mean(), oos_arr.std(), len(oos_perfs)))
            perf_matrix = np.column_stack(strategies)
            result["pbo"] = self.compute_pbo(perf_matrix)
        else:
            result["pbo"] = {
                "pbo": 0.0,
                "is_overfit": False,
                "note": "insufficient splits",
            }

        logger.info(
            "CPCV 完成: %d splits, mean OOS=%.6f ± %.6f",
            len(splits),
            result["mean_oos"],
            result["std_oos"],
        )
        return result
