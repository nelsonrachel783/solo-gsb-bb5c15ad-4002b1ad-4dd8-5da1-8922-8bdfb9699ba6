"""广告插播点选取核心逻辑（纯标准库，不依赖 FastAPI，便于单测）。

输入按时间排列的词序列，只在相邻词之间、静音长度不少于 min_silence_ms 的
区间生成候选，候选时间取静音区间的整数中点（floor((start+end)/2)）。

需恰选 break_count 个候选，任意两个相邻选中时间至少相隔 min_spacing_ms，
在此约束下使选中静音总毫秒数最大；同分取时间序列字典序最小者。

算法（后向 DP + 贪心重建），O(m log m + K·m) 时间、O(K·m) 空间，
其中 m 为候选数（<= n-1）、K=break_count（1..5）：

* 候选按时间严格升序；next_ok[j] 用二分求出“与候选 j 间距达标的最早后续
  候选”，间距约束因此成为候选下标上的单调转移；
* g[k][j] = 在候选 j..m-1 中（下标）恰选 k 个、且选中候选彼此满足间距
  约束时的最大静音总毫秒数；None 表示不可达。
  转移（k>=1）：g[k][j] = max(不选 j:  g[k][j+1],
                              选 j:    w[j] + g[k-1][next_ok[j]])；
* 最优值为 g[K][0]；不可达即无解（DomainError -> 422）；
* 贪心重建时间序列字典序最小的最优解：从左到右扫描，在每个位置比较“选”与
  “不选”两支的最优值，相等时优先“选”——选中的当前下标（即当前最小可能
  时间）尽可能小，归纳可得全局字典序最小。
"""

from bisect import bisect_left

from .errors import DomainError
from .segmentation import validate_words


def _fail(code, message, **extra):
    raise DomainError(code, message, **extra)


def _build_candidates(words):
    """枚举相邻词之间满足静音阈值的候选：(时间, 静音起点, 静音终点, 静音长)。"""
    candidates = []
    for idx in range(len(words) - 1):
        silence_start = words[idx].end_ms
        silence_end = words[idx + 1].start_ms
        silence = silence_end - silence_start  # validate_words 已保证非负
        candidates.append(
            {
                "after_word": idx,
                "time_ms": (silence_start + silence_end) // 2,
                "silence_start_ms": silence_start,
                "silence_end_ms": silence_end,
                "silence_ms": silence,
            }
        )
    return candidates


def select_ad_breaks(words, break_count, min_silence_ms, min_spacing_ms):
    """选取总静音最大的 break_count 个插播点；同分取时间序列字典序最小。

    参数 words 中每个对象需具备属性：start_ms / end_ms / text / speaker。

    返回 (breaks, total_silence_ms)：breaks 为按时间排列的插播点字典列表；
    无可行方案时抛出 DomainError(code="infeasible_ad_breaks")。
    """
    # 沿用 /segments 的词序与时间校验（非空、词内/词间时间合法）
    validate_words(words)

    pool = [
        c
        for c in _build_candidates(words)
        if c["silence_ms"] >= min_silence_ms
    ]
    m = len(pool)

    if m < break_count:
        _fail(
            "infeasible_ad_breaks",
            "不存在满足全部约束的插播方案",
            candidate_count=m,
        )

    times = [c["time_ms"] for c in pool]
    weights = [c["silence_ms"] for c in pool]

    # next_ok[j]：与候选 j 的时间至少相隔 min_spacing_ms 的最早候选下标
    next_ok = [
        bisect_left(times, times[j] + min_spacing_ms, lo=j + 1)
        for j in range(m)
    ]

    # ---- 后向 DP：g[k][j] = 后缀候选 [j, m) 中选 k 个的最大静音和 ----
    g = [[None] * (m + 1) for _ in range(break_count + 1)]
    g[0] = [0] * (m + 1)  # 选 0 个：任何后缀都可行，和为 0
    for k in range(1, break_count + 1):
        row = g[k]
        prev = g[k - 1]
        for j in range(m - 1, -1, -1):
            best = row[j + 1]  # 不选候选 j
            tail = prev[next_ok[j]]  # 选候选 j
            if tail is not None:
                value = weights[j] + tail
                if best is None or value > best:
                    best = value
            row[j] = best

    if g[break_count][0] is None:
        _fail(
            "infeasible_ad_breaks",
            "不存在满足全部约束的插播方案",
            candidate_count=m,
        )
    optimum = g[break_count][0]

    # ---- 贪心重建字典序最小的最优下标序列 ----
    # remaining：从当前后缀出发仍需取得的静音和（初始为全局最优值）。
    chosen = []
    j = 0
    k = break_count
    remaining = optimum
    while k > 0:
        # “选 j”可达时与“不选 j”的最优值比较；等值优先选 j，
        # 使当前时间尽可能小 -> 时间序列字典序最小。
        # 注意必须 take == remaining：前缀选择可能已使当前后缀所需和低于
        # 该后缀自身的最优值，此时选 j 无法补回全局最优，只能跳过。
        tail = g[k - 1][next_ok[j]]
        take = None if tail is None else weights[j] + tail
        if take == remaining:  # None != int：不可达的“选 j”自然被跳过
            chosen.append(j)
            remaining -= weights[j]
            j = next_ok[j]
            k -= 1
        else:
            j += 1

    breaks = []
    total = 0
    for order, idx in enumerate(chosen):
        c = pool[idx]
        total += c["silence_ms"]
        breaks.append(
            {
                "index": order,
                "time_ms": c["time_ms"],
                "silence_start_ms": c["silence_start_ms"],
                "silence_end_ms": c["silence_end_ms"],
                "silence_ms": c["silence_ms"],
                "after_word": c["after_word"],
            }
        )
    return breaks, total
