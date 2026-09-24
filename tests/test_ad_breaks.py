"""广告插播点核心逻辑单元测试（纯标准库）。"""

import itertools
import random
import unittest
from types import SimpleNamespace

from app.ad_breaks import select_ad_breaks
from app.errors import DomainError


def W(start, end, text="词", speaker="A"):
    return SimpleNamespace(start_ms=start, end_ms=end, text=text, speaker=speaker)


def words_with_gaps(gaps, word_duration=500):
    """按给定相邻词间隔构造词序列：gap[i] 为词 i 与词 i+1 之间的静音毫秒。"""
    words = []
    t = 0
    for gap in [0, *gaps]:
        t += gap
        words.append(W(t, t + word_duration))
        t += word_duration
    return words


def brute_force(words, break_count, min_silence_ms, min_spacing_ms):
    """独立生成候选并枚举所有组合，返回 (候选元组, 总静音) 或 None。

    候选元组每项为 (时间, 静音长)；同分时按时间序列字典序最小裁决。
    """
    cands = []
    for i in range(len(words) - 1):
        gap = words[i + 1].start_ms - words[i].end_ms
        if gap >= min_silence_ms:
            cands.append(((words[i].end_ms + words[i + 1].start_ms) // 2, gap))
    best = None
    for combo in itertools.combinations(range(len(cands)), break_count):
        times = [cands[i][0] for i in combo]
        if all(b - a >= min_spacing_ms for a, b in zip(times, times[1:])):
            total = sum(cands[i][1] for i in combo)
            # 总静音最大；同分时间序列字典序最小
            key = (-total, times)
            if best is None or key < best[0]:
                best = (key, tuple(cands[i] for i in combo), total)
    if best is None:
        return None
    return best[1], best[2]


class CandidateTests(unittest.TestCase):
    def test_gap_filter_and_midpoint(self):
        # 间隔 300 / 801 / 800：阈值 800 保留后两者；奇数间隔取整数中点（向下取整）
        words = words_with_gaps([300, 801, 800])
        breaks, total = select_ad_breaks(words, 2, 800, 0)
        # 静音 [1300,2101] -> 中点 1700；[2601,3401] -> 中点 3001
        self.assertEqual([b["time_ms"] for b in breaks], [1700, 3001])
        self.assertEqual([b["silence_ms"] for b in breaks], [801, 800])
        self.assertEqual(total, 1601)

    def test_odd_silence_integer_midpoint(self):
        # 前词 end=100，后词 start=601，静音 501（奇数），中点 floor(701/2)=350
        words = [W(0, 100), W(601, 900)]
        breaks, _ = select_ad_breaks(words, 1, 1, 0)
        self.assertEqual(breaks[0]["time_ms"], 350)
        self.assertEqual(breaks[0]["silence_start_ms"], 100)
        self.assertEqual(breaks[0]["silence_end_ms"], 601)
        self.assertEqual(breaks[0]["silence_ms"], 501)
        self.assertEqual(breaks[0]["after_word"], 0)

    def test_touching_words_not_candidates(self):
        # 端点相接（静音 0）永远不是候选（min_silence_ms 为正整数）
        words = [W(0, 500), W(500, 1000)]
        with self.assertRaises(DomainError) as ctx:
            select_ad_breaks(words, 1, 1, 0)
        self.assertEqual(ctx.exception.code, "infeasible_ad_breaks")
        self.assertEqual(ctx.exception.extra["candidate_count"], 0)

    def test_single_word_infeasible(self):
        with self.assertRaises(DomainError) as ctx:
            select_ad_breaks([W(0, 1000)], 1, 1, 0)
        self.assertEqual(ctx.exception.code, "infeasible_ad_breaks")


class SelectionTests(unittest.TestCase):
    def test_max_total_silence(self):
        # 静音 300/800/400，选 2、无间距约束 -> 800+400=1200 最大
        words = words_with_gaps([300, 800, 400])
        breaks, total = select_ad_breaks(words, 2, 300, 0)
        self.assertEqual([b["silence_ms"] for b in breaks], [800, 400])
        self.assertEqual([b["time_ms"] for b in breaks], [1700, 2800])
        self.assertEqual(total, 1200)
        self.assertTrue(
            all(b["time_ms"] >= 0 for b in breaks)
        )

    def test_spacing_changes_selection(self):
        # 候选时间 700/1600/2500，静音均为 400
        words = words_with_gaps([400, 400, 400])
        # 间距 >= 901：只有 (700,2500) 可行
        breaks, _ = select_ad_breaks(words, 2, 400, 901)
        self.assertEqual([b["time_ms"] for b in breaks], [700, 2500])
        # 间距 >= 1801：无解
        with self.assertRaises(DomainError) as ctx:
            select_ad_breaks(words, 2, 400, 1801)
        self.assertEqual(ctx.exception.code, "infeasible_ad_breaks")

    def test_tie_prefers_lexicographically_smallest_times(self):
        # 三个等长（400ms）静音，选 2：任意组合同分，取最早的两个
        words = words_with_gaps([400, 400, 400])
        breaks, total = select_ad_breaks(words, 2, 400, 0)
        self.assertEqual([b["time_ms"] for b in breaks], [700, 1600])
        self.assertEqual(total, 800)

    def test_breaks_sorted_and_indexed(self):
        # 最优组合不含首个候选时，返回仍须按时间排列并重新编号
        words = words_with_gaps([300, 800, 400])
        breaks, _ = select_ad_breaks(words, 2, 300, 0)
        times = [b["time_ms"] for b in breaks]
        self.assertEqual(times, sorted(times))
        self.assertEqual([b["index"] for b in breaks], [0, 1])
        self.assertEqual([b["after_word"] for b in breaks], [1, 2])

    def test_break_count_five(self):
        words = words_with_gaps([500] * 6)
        breaks, total = select_ad_breaks(words, 5, 1, 0)
        self.assertEqual(len(breaks), 5)
        self.assertEqual(total, 2500)

    def test_more_breaks_than_candidates(self):
        words = words_with_gaps([500, 0, 500])  # 仅 2 个候选
        with self.assertRaises(DomainError) as ctx:
            select_ad_breaks(words, 3, 1, 0)
        self.assertEqual(ctx.exception.code, "infeasible_ad_breaks")
        self.assertEqual(ctx.exception.extra["candidate_count"], 2)


class ReusedValidationTests(unittest.TestCase):
    def test_empty_words(self):
        with self.assertRaises(DomainError) as ctx:
            select_ad_breaks([], 1, 1, 0)
        self.assertEqual(ctx.exception.code, "empty_words")

    def test_overlap_rejected(self):
        words = [W(0, 2000), W(1500, 3000)]
        with self.assertRaises(DomainError) as ctx:
            select_ad_breaks(words, 1, 1, 0)
        self.assertEqual(ctx.exception.code, "overlapping_time_range")

    def test_out_of_order_rejected(self):
        words = [W(2000, 3000), W(1000, 2000)]
        with self.assertRaises(DomainError) as ctx:
            select_ad_breaks(words, 1, 1, 0)
        self.assertEqual(ctx.exception.code, "time_out_of_order")

    def test_word_end_before_start(self):
        with self.assertRaises(DomainError) as ctx:
            select_ad_breaks([W(2000, 1000)], 1, 1, 0)
        self.assertEqual(ctx.exception.code, "invalid_time_range")


class DifferentialTests(unittest.TestCase):
    def test_dp_matches_brute_force(self):
        random.seed(4002)
        feasible_trials = 0
        for trial in range(600):
            n = random.randint(1, 8)
            words = []
            t = 0
            for _ in range(n):
                t += random.choice([0, 0, 200, 350, 800])  # 词间隔（静音）
                dur = random.choice([100, 400, 900])
                words.append(W(t, t + dur))
                t += dur
            k = random.randint(1, 5)
            min_silence = random.choice([1, 200, 350])
            min_spacing = random.choice([0, 500, 1200])

            expected = brute_force(words, k, min_silence, min_spacing)
            if expected is None:
                with self.assertRaises(DomainError) as ctx:
                    select_ad_breaks(words, k, min_silence, min_spacing)
                self.assertEqual(ctx.exception.code, "infeasible_ad_breaks")
            else:
                exp_cands, exp_total = expected
                breaks, total = select_ad_breaks(words, k, min_silence, min_spacing)
                self.assertEqual(total, exp_total, f"trial {trial} total mismatch")
                self.assertEqual(len(breaks), k)
                self.assertEqual(
                    [b["time_ms"] for b in breaks],
                    [c[0] for c in exp_cands],
                    f"trial {trial} tie-break mismatch",
                )
                # 间距约束与静音和核对
                for a, b in zip(breaks, breaks[1:]):
                    self.assertGreaterEqual(
                        b["time_ms"] - a["time_ms"], min_spacing
                    )
                self.assertEqual(
                    sum(b["silence_ms"] for b in breaks), exp_total
                )
                feasible_trials += 1
        self.assertGreater(feasible_trials, 100)


if __name__ == "__main__":
    unittest.main()
