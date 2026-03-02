"""三層記憶體系統（參考 FinMem AAAI 2024）

- 短期記憶: 最近 5 個交易日的市場資料 + 決策 (in-memory)
- 長期記憶: 歷史模式庫（SQLite + embedding 向量搜尋）
- 情境記憶: 交易日誌 — 每筆決策、理由、實際結果、事後檢討
"""

import json
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from src.db.models import AgentMemory, TradeJournal

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """短期記憶 — 最近 N 個交易日（in-memory）"""

    def __init__(self, max_days: int = 5):
        self.max_days = max_days
        self._memories: deque[dict[str, Any]] = deque(maxlen=max_days)

    def add(self, date: str, data: dict[str, Any]):
        """新增一天的記憶"""
        self._memories.append({"date": date, **data})

    def get_recent(self, n: int | None = None) -> list[dict]:
        """取得最近 N 天的記憶"""
        memories = list(self._memories)
        if n is not None:
            return memories[-n:]
        return memories

    def get_summary(self) -> dict:
        """取得摘要"""
        if not self._memories:
            return {"status": "empty"}
        return {
            "days": len(self._memories),
            "date_range": f"{self._memories[0]['date']} ~ {self._memories[-1]['date']}",
            "latest": self._memories[-1],
        }

    def clear(self):
        self._memories.clear()


class LongTermMemory:
    """長期記憶 — 歷史模式庫（SQLite 儲存）"""

    def __init__(self, session_factory=None):
        self._session_factory = session_factory

    def store(
        self,
        content: dict[str, Any],
        category: str,
        stock_id: str | None = None,
        relevance_score: float = 1.0,
    ):
        """儲存長期記憶"""
        if not self._session_factory:
            logger.warning("無 DB session，長期記憶暫存於 log")
            logger.info("長期記憶 [%s]: %s", category, content)
            return

        session = self._session_factory()
        try:
            memory = AgentMemory(
                stock_id=stock_id,
                memory_type="long_term",
                category=category,
                content=json.dumps(content, ensure_ascii=False),
                relevance_score=relevance_score,
            )
            session.add(memory)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("儲存長期記憶失敗: %s", e)
        finally:
            session.close()

    def recall(
        self,
        category: str | None = None,
        stock_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """回憶長期記憶"""
        if not self._session_factory:
            return []

        session = self._session_factory()
        try:
            query = session.query(AgentMemory).filter(
                AgentMemory.memory_type == "long_term"
            )
            if category:
                query = query.filter(AgentMemory.category == category)
            if stock_id:
                query = query.filter(AgentMemory.stock_id == stock_id)

            memories = query.order_by(
                AgentMemory.relevance_score.desc()
            ).limit(limit).all()

            results = []
            for m in memories:
                # 更新存取計數
                m.access_count = (m.access_count or 0) + 1
                m.last_accessed = datetime.utcnow()
                results.append({
                    "id": m.id,
                    "category": m.category,
                    "content": json.loads(m.content) if m.content else {},
                    "relevance_score": m.relevance_score,
                    "created_at": str(m.created_at),
                })
            session.commit()
            return results
        except Exception as e:
            logger.error("回憶長期記憶失敗: %s", e)
            return []
        finally:
            session.close()

    def search_similar(
        self,
        query_embedding: list[float],
        stock_id: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """向量相似度搜尋（簡化版：cosine similarity）"""
        if not self._session_factory:
            return []

        session = self._session_factory()
        try:
            query = session.query(AgentMemory).filter(
                AgentMemory.memory_type == "long_term",
                AgentMemory.embedding.isnot(None),
            )
            if stock_id:
                query = query.filter(AgentMemory.stock_id == stock_id)

            memories = query.all()
            if not memories:
                return []

            # Cosine similarity
            query_vec = np.array(query_embedding)
            scored = []
            for m in memories:
                try:
                    stored_vec = np.array(json.loads(m.embedding))
                    similarity = np.dot(query_vec, stored_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8
                    )
                    scored.append((m, float(similarity)))
                except (json.JSONDecodeError, ValueError):
                    continue

            scored.sort(key=lambda x: x[1], reverse=True)
            return [
                {
                    "content": json.loads(m.content) if m.content else {},
                    "similarity": score,
                    "category": m.category,
                }
                for m, score in scored[:limit]
            ]
        except Exception as e:
            logger.error("向量搜尋失敗: %s", e)
            return []
        finally:
            session.close()


class EpisodicMemory:
    """情境記憶 — 交易日誌"""

    def __init__(self, session_factory=None):
        self._session_factory = session_factory

    def record_trade(
        self,
        stock_id: str,
        trade_date: str,
        action: str,
        price: float,
        reasoning: dict[str, Any],
    ):
        """記錄一筆交易決策"""
        if not self._session_factory:
            logger.info("交易記錄 [%s %s]: %s @ %.2f", stock_id, action, trade_date, price)
            return

        session = self._session_factory()
        try:
            journal = TradeJournal(
                stock_id=stock_id,
                trade_date=trade_date,
                action=action,
                price=price,
                technical_analysis=reasoning.get("technical"),
                sentiment_analysis=reasoning.get("sentiment"),
                fundamental_analysis=reasoning.get("fundamental"),
                quant_analysis=reasoning.get("quant"),
                researcher_debate=reasoning.get("researcher"),
                trader_reasoning=reasoning.get("trader_reasoning", ""),
                risk_assessment=reasoning.get("risk"),
                market_snapshot=reasoning.get("market_snapshot"),
            )
            session.add(journal)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("記錄交易失敗: %s", e)
        finally:
            session.close()

    def update_result(
        self,
        journal_id: int,
        exit_date: str,
        exit_price: float,
        review_notes: str = "",
    ):
        """更新交易結果（事後填入）"""
        if not self._session_factory:
            return

        session = self._session_factory()
        try:
            journal = session.get(TradeJournal, journal_id)
            if journal:
                journal.exit_date = exit_date
                journal.exit_price = exit_price
                if journal.price and journal.price > 0:
                    journal.pnl_pct = (exit_price - journal.price) / journal.price
                    journal.pnl = journal.pnl_pct * journal.price * (journal.quantity or 1000)
                journal.review_notes = review_notes
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error("更新交易結果失敗: %s", e)
        finally:
            session.close()

    def get_history(
        self,
        stock_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """取得交易歷史"""
        if not self._session_factory:
            return []

        session = self._session_factory()
        try:
            query = session.query(TradeJournal)
            if stock_id:
                query = query.filter(TradeJournal.stock_id == stock_id)
            records = query.order_by(TradeJournal.trade_date.desc()).limit(limit).all()

            return [
                {
                    "id": r.id,
                    "stock_id": r.stock_id,
                    "date": str(r.trade_date),
                    "action": r.action,
                    "price": r.price,
                    "exit_price": r.exit_price,
                    "pnl_pct": r.pnl_pct,
                    "reasoning": r.trader_reasoning,
                }
                for r in records
            ]
        except Exception as e:
            logger.error("取得交易歷史失敗: %s", e)
            return []
        finally:
            session.close()


class AgentMemorySystem:
    """整合三層記憶 + 層級轉移機制

    FinMem 升級：基於交易結果的自動記憶升級
    - 短期 → 長期：成功交易的模式自動升級
    - 長期 → 情境記憶更新：驗證過的模式標記可靠度
    """

    # 層級轉移閾值
    PROMOTE_WIN_RATE = 0.6      # 勝率 >= 60% 時升級到長期記憶
    PROMOTE_MIN_TRADES = 3      # 至少 3 筆交易才計算
    DEMOTE_LOSS_STREAK = 3      # 連續虧損 3 次降級

    def __init__(self, session_factory=None):
        self.short_term = ShortTermMemory(max_days=5)
        self.long_term = LongTermMemory(session_factory)
        self.episodic = EpisodicMemory(session_factory)
        self._session_factory = session_factory

    def get_context_for_decision(self, stock_id: str) -> dict:
        """為決策提供完整記憶上下文"""
        return {
            "short_term": self.short_term.get_summary(),
            "recent_patterns": self.long_term.recall(
                category="market_pattern", stock_id=stock_id, limit=5
            ),
            "trade_history": self.episodic.get_history(stock_id=stock_id, limit=10),
        }

    def evaluate_and_transfer(self, stock_id: str):
        """基於交易結果評估並執行記憶層級轉移

        檢查近期交易結果，將驗證過的模式升級到長期記憶，
        將失敗的模式降低可靠度。
        """
        history = self.episodic.get_history(stock_id=stock_id, limit=20)
        if len(history) < self.PROMOTE_MIN_TRADES:
            return

        # 計算近期勝率
        completed = [t for t in history if t.get("pnl_pct") is not None]
        if len(completed) < self.PROMOTE_MIN_TRADES:
            return

        wins = sum(1 for t in completed if (t.get("pnl_pct") or 0) > 0)
        win_rate = wins / len(completed)

        # 檢查短期記憶中的模式是否值得升級
        recent_memories = self.short_term.get_recent()
        if not recent_memories:
            return

        if win_rate >= self.PROMOTE_WIN_RATE:
            # 升級：成功模式 → 長期記憶
            pattern = {
                "source": "auto_promoted",
                "win_rate": round(win_rate, 3),
                "sample_size": len(completed),
                "recent_context": recent_memories[-1] if recent_memories else {},
                "description": f"自動升級模式 (勝率 {win_rate:.0%}, {len(completed)} 筆)",
            }
            self.long_term.store(
                content=pattern,
                category="validated_pattern",
                stock_id=stock_id,
                relevance_score=win_rate,
            )
            logger.info(
                "記憶升級 [%s]: 短期→長期 (勝率=%.0f%%, trades=%d)",
                stock_id, win_rate * 100, len(completed),
            )

        # 檢查連續虧損 → 降級/標記
        recent_completed = completed[:self.DEMOTE_LOSS_STREAK]
        loss_streak = all(
            (t.get("pnl_pct") or 0) < 0
            for t in recent_completed
        )

        if loss_streak and len(recent_completed) >= self.DEMOTE_LOSS_STREAK:
            # 降級：標記最近的長期記憶為不可靠
            recent_patterns = self.long_term.recall(
                category="market_pattern", stock_id=stock_id, limit=3
            )
            for pattern in recent_patterns:
                # 降低可靠度（通過存入新的負面記憶）
                warning = {
                    "source": "auto_demoted",
                    "reason": f"連續 {self.DEMOTE_LOSS_STREAK} 筆虧損",
                    "original_pattern_id": pattern.get("id"),
                    "warning": "此模式近期表現不佳，降低參考權重",
                }
                self.long_term.store(
                    content=warning,
                    category="pattern_warning",
                    stock_id=stock_id,
                    relevance_score=0.3,
                )
            logger.warning(
                "記憶降級 [%s]: 連續 %d 筆虧損，標記相關模式為不可靠",
                stock_id, self.DEMOTE_LOSS_STREAK,
            )
