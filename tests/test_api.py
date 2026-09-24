"""HTTP 接口层测试。

依赖 fastapi + httpx（TestClient 基于 httpx）。
未安装时自动跳过；核心算法测试见 test_segmentation.py（纯标准库即可运行）。
"""

import unittest

try:
    from fastapi.testclient import TestClient
    from app.main import app
    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False


@unittest.skipUnless(_HAS_DEPS, "fastapi/httpx 未安装，跳过接口层测试")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_basic_segmentation(self):
        resp = self.client.post(
            "/segments",
            json={
                "words": [
                    {"start_ms": 0, "end_ms": 2000, "text": "第一句。", "speaker": "A"},
                    {"start_ms": 2000, "end_ms": 4200, "text": "第二句！", "speaker": "B"},
                    {"start_ms": 4200, "end_ms": 6200, "text": "第三句？", "speaker": "B"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_score"], 0)
        self.assertEqual(len(data["segments"]), 2)  # B 的两句合段更优（末段零分）
        self.assertIn("1\n00:00:00,000 --> 00:00:02,000", data["srt"])

    def test_overlap_rejected(self):
        resp = self.client.post(
            "/segments",
            json={
                "words": [
                    {"start_ms": 0, "end_ms": 2000, "text": "甲", "speaker": "A"},
                    {"start_ms": 1500, "end_ms": 3000, "text": "乙", "speaker": "A"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "overlapping_time_range")

    def test_out_of_order_rejected(self):
        resp = self.client.post(
            "/segments",
            json={
                "words": [
                    {"start_ms": 2000, "end_ms": 3000, "text": "甲", "speaker": "A"},
                    {"start_ms": 1000, "end_ms": 1500, "text": "乙", "speaker": "A"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "time_out_of_order")

    def test_infeasible_returns_earliest_prefix(self):
        resp = self.client.post(
            "/segments",
            json={
                "words": [
                    {"start_ms": 0, "end_ms": 800, "text": "短", "speaker": "A"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 422)
        err = resp.json()["error"]
        self.assertEqual(err["code"], "infeasible_segmentation")
        self.assertEqual(err["last_word_index"], 0)

    def test_invalid_schema(self):
        resp = self.client.post("/segments", json={"words": []})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")


@unittest.skipUnless(_HAS_DEPS, "fastapi/httpx 未安装，跳过接口层测试")
class AdBreakApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _words(self):
        # 静音：300 / 800 / 400
        return [
            {"start_ms": 0, "end_ms": 500, "text": "词", "speaker": "A"},
            {"start_ms": 800, "end_ms": 1300, "text": "词", "speaker": "A"},
            {"start_ms": 2100, "end_ms": 2600, "text": "词", "speaker": "A"},
            {"start_ms": 3000, "end_ms": 3500, "text": "词", "speaker": "A"},
        ]

    def test_basic_ad_breaks(self):
        resp = self.client.post(
            "/ad-breaks",
            json={
                "words": self._words(),
                "break_count": 2,
                "min_silence_ms": 300,
                "min_spacing_ms": 0,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_silence_ms"], 1200)
        self.assertEqual(len(data["breaks"]), 2)
        first, second = data["breaks"]
        # 800ms 与 400ms 两个静音最优，按时间排列
        self.assertEqual(first["index"], 0)
        self.assertEqual(first["time_ms"], 1700)
        self.assertEqual(first["silence_start_ms"], 1300)
        self.assertEqual(first["silence_end_ms"], 2100)
        self.assertEqual(first["silence_ms"], 800)
        self.assertEqual(first["after_word"], 1)
        self.assertEqual(second["time_ms"], 2800)
        self.assertEqual(second["silence_ms"], 400)

    def test_infeasible_returns_422(self):
        # 仅两个候选却要 3 个插播点
        resp = self.client.post(
            "/ad-breaks",
            json={
                "words": self._words(),
                "break_count": 3,
                "min_silence_ms": 350,
                "min_spacing_ms": 0,
            },
        )
        self.assertEqual(resp.status_code, 422)
        err = resp.json()["error"]
        self.assertEqual(err["code"], "infeasible_ad_breaks")
        self.assertEqual(err["candidate_count"], 2)

    def test_invalid_params_rejected(self):
        resp = self.client.post(
            "/ad-breaks",
            json={
                "words": self._words(),
                "break_count": 6,
                "min_silence_ms": 1,
                "min_spacing_ms": 0,
            },
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")

        resp = self.client.post(
            "/ad-breaks",
            json={
                "words": self._words(),
                "break_count": 1,
                "min_silence_ms": 0,
                "min_spacing_ms": 0,
            },
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")

        resp = self.client.post(
            "/ad-breaks",
            json={
                "words": self._words(),
                "break_count": 1,
                "min_silence_ms": 1,
                "min_spacing_ms": -1,
            },
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_words_validations_reused(self):
        resp = self.client.post(
            "/ad-breaks",
            json={
                "words": [
                    {"start_ms": 0, "end_ms": 2000, "text": "甲", "speaker": "A"},
                    {"start_ms": 1500, "end_ms": 3000, "text": "乙", "speaker": "A"},
                ],
                "break_count": 1,
                "min_silence_ms": 1,
                "min_spacing_ms": 0,
            },
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "overlapping_time_range")

    def test_segments_still_works(self):
        resp = self.client.post(
            "/segments",
            json={
                "words": [
                    {"start_ms": 0, "end_ms": 2000, "text": "第一句。", "speaker": "A"},
                    {"start_ms": 2000, "end_ms": 4200, "text": "第二句！", "speaker": "B"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["total_score"], 0)


if __name__ == "__main__":
    unittest.main()
