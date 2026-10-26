"""dataclean —— 自检测试。

检查数据集里的重复、缺失字段、标签分布偏移与近重复样本，给出可执行的处理建议。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataclean  # noqa: E402


class TestDataClean(unittest.TestCase):
    def test_exact_duplicate_detected(self):
        rows = [{"content": "abc"}, {"content": "abc"}, {"content": "xyz"}]
        r = dataclean.analyze(rows, "content", 0.85)
        self.assertEqual(r["summary"]["exact_duplicate_groups"], 1)

    def test_near_duplicate_is_separate_from_exact(self):
        rows = [{"content": "时间戳解析方法说明"},
                {"content": "时间戳解析方法說明"},
                {"content": "完全不同的另一段内容"}]
        r = dataclean.analyze(rows, "content", 0.5)
        self.assertEqual(r["summary"]["exact_duplicate_groups"], 0)
        self.assertGreaterEqual(r["summary"]["near_duplicate_pairs"], 1)

    def test_missing_field_detected(self):
        rows = [{"content": "a"}, {"content": ""}, {"content": "c"}]
        r = dataclean.analyze(rows, "content", 0.85)
        self.assertEqual(r["summary"]["missing"], 1)

    def test_inconsistent_fields_detected(self):
        rows = [{"content": "a", "label": "x"}, {"content": "b"}]
        r = dataclean.analyze(rows, "content", 0.85)
        self.assertTrue(r["inconsistent_fields"])

    def test_clean_data_reports_no_issue(self):
        rows = [{"content": "".join(chr(0x4e00 + i * 7 + k % 11) for k in range(24))
                 + f"#{i}"} for i in range(10)]
        r = dataclean.analyze(rows, "content", 0.85)
        self.assertEqual(r["issues"], [])

    def test_does_not_modify_records(self):
        """防呆：体检不改数据，也不替用户决定删哪一条。"""
        rows = [{"content": "abc"}, {"content": "abc"}]
        before = [dict(r) for r in rows]
        dataclean.analyze(rows, "content", 0.85)
        self.assertEqual(rows, before)
