"""dataclean —— 给训练与评测数据做体检。

设计原则：
1. **按「会怎样影响训练」排序问题**，不按数量排。100 条轻微重复
   不如 1 条标签整体错位严重。
2. **真重复与近重复要分开**。两者的处理方式完全不同。
3. **不给修复动作，只给问题定位**。删哪一条是人的判断。

仅依赖 Python 标准库。

用法:
    python dataclean.py --help
    python dataclean.py --file data.jsonl
    python dataclean.py --file data.jsonl --text-field content --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# 近重复检测
# ---------------------------------------------------------------------------

# 中文按 2-gram、英文按词的集合，用 Jaccard 相似度判近重复
TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]+")


def shingles(text: str) -> set[str]:
    """把文本切成可比较的片段集合。"""
    text = (text or "").lower()
    out: set[str] = set()
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        if re.fullmatch(r"[A-Za-z0-9]+", tok):
            out.add(tok)
        else:
            for i in range(len(tok) - 1):
                out.add(tok[i:i + 2])
            if len(tok) == 1:
                out.add(tok)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def load_records(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as e:
        raise ValueError(f"读不了 {path}：{e}") from e

    # 支持 JSONL 与 JSON 数组
    stripped = raw.strip()
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败：{e}") from e
        if not isinstance(data, list):
            raise ValueError("JSON 顶层应该是一个数组")
        return [x for x in data if isinstance(x, dict)]

    records: list[dict] = []
    for lineno, line in enumerate(stripped.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"第 {lineno} 行 JSON 解析失败：{e}") from e
        if isinstance(obj, dict):
            records.append(obj)
    return records


def field_of(rec: dict, field: str) -> str:
    """取记录里的文本字段。field 支持 a.b.c 路径。"""
    cur: object = rec
    for part in field.split("."):
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(part, "")
    if cur is None:
        return ""
    if isinstance(cur, str):
        return cur
    return json.dumps(cur, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# 体检
# ---------------------------------------------------------------------------

def analyze(records: list[dict], text_field: str,
            near_threshold: float) -> dict:
    """体检一份数据集。"""
    total = len(records)
    if total == 0:
        return {"total": 0, "issues": [], "summary": {}}

    texts = [field_of(r, text_field) for r in records]

    # --- 缺失 ---
    missing: list[int] = [i for i, t in enumerate(texts) if not t.strip()]

    # --- 完全重复 ---
    counter = Counter(texts)
    exact_dupes = {t: c for t, c in counter.items() if c > 1 and t.strip()}

    # --- 近重复 ---
    # 只对「长度接近」的做两两比较，避免 O(n^2) 全量比较
    near_dupes: list[dict] = []
    shingle_cache = [shingles(t) for t in texts]
    order = sorted(range(total), key=lambda i: len(texts[i]))
    for ai in range(len(order)):
        i = order[ai]
        for bi in range(ai + 1, len(order)):
            j = order[bi]
            if not texts[i].strip() or not texts[j].strip():
                continue
            # 长度差太大不可能是近重复
            if len(texts[j]) > len(texts[i]) * 1.6 and len(texts[i]) > 0:
                continue
            sim = jaccard(shingle_cache[i], shingle_cache[j])
            if sim >= near_threshold:
                near_dupes.append({
                    "a": i, "b": j, "similarity": round(sim, 4),
                    "a_preview": texts[i][:60],
                    "b_preview": texts[j][:60],
                })
    # 完全重复的不再重复报近重复
    exact_pairs = set()
    seen: dict[str, int] = {}
    for i, t in enumerate(texts):
        if t in seen and t.strip():
            exact_pairs.add((seen[t], i))
        else:
            seen[t] = i
    near_dupes = [d for d in near_dupes
                  if (d["a"], d["b"]) not in exact_pairs
                  and (d["b"], d["a"]) not in exact_pairs]

    # --- 字段一致性 ---
    field_shapes = Counter()
    for rec in records:
        field_shapes[tuple(sorted(rec.keys()))] += 1
    inconsistent_fields = []
    if len(field_shapes) > 1:
        main_keys = field_shapes.most_common(1)[0][0]
        for shape, cnt in field_shapes.most_common():
            if shape != main_keys:
                missing_keys = sorted(set(main_keys) - set(shape))
                extra_keys = sorted(set(shape) - set(main_keys))
                inconsistent_fields.append({
                    "shape": list(shape), "count": cnt,
                    "missing": missing_keys, "extra": extra_keys,
                })

    # --- 标签分布偏移 ---
    # 找数值型字段，看分布是否高度集中（长尾数据里常见的坑）
    label_issues: list[dict] = []
    numeric_fields: dict[str, list[float]] = {}
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                numeric_fields.setdefault(k, []).append(float(v))
            elif isinstance(v, str) and k.lower() in ("label", "labels", "category",
                                                     "class", "tag", "tags",
                                                     "intent", "sentiment"):
                numeric_fields.setdefault(f"_cat_{k}", []).append(0.0)
    for name, vals in numeric_fields.items():
        if len(vals) < 10:
            continue
        vals_sorted = sorted(vals)
        median = vals_sorted[len(vals_sorted) // 2]
        if median == 0:
            continue
        zeros = sum(1 for v in vals if v == 0)
        if zeros / len(vals) > 0.6:
            label_issues.append({
                "field": name,
                "kind": "类别极度不均衡",
                "detail": f"{zeros}/{len(vals)} 个样本取值相同",
            })

    # --- 分类字段的类别分布 ---
    cat_fields: dict[str, Counter] = {}
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, str) and len(v) < 64:
                cat_fields.setdefault(k, Counter())[v] += 1
    for name, c in cat_fields.items():
        if len(c) < 2 or len(c) > 50:
            continue
        top, top_n = c.most_common(1)[0]
        ratio = top_n / total
        if ratio > 0.85 and len(c) > 1:
            label_issues.append({
                "field": name,
                "kind": "类别集中",
                "detail": f"「{top[:30]}」占 {ratio * 100:.1f}%（共 {len(c)} 类）",
            })

    # --- 汇总与排序 ---
    issues: list[dict] = []

    if inconsistent_fields:
        issues.append({
            "kind": "字段结构不一致",
            "severity": 3,
            "count": len(inconsistent_fields),
            "detail": f"{len(inconsistent_fields)} 种字段组合，主要组合之外还有其它形状",
        })

    if label_issues:
        for li in label_issues:
            issues.append({
                "kind": li["kind"],
                "severity": 3,
                "count": 1,
                "detail": f"{li['field']}：{li['detail']}",
            })

    if missing:
        issues.append({
            "kind": "字段缺失",
            "severity": 2,
            "count": len(missing),
            "detail": f"{len(missing)}/{total} 条记录的「{text_field}」为空",
        })

    if exact_dupes:
        dup_count = sum(c - 1 for c in exact_dupes.values())
        issues.append({
            "kind": "完全重复",
            "severity": 2,
            "count": dup_count,
            "detail": f"{len(exact_dupes)} 组重复，共 {dup_count} 条冗余",
        })

    if near_dupes:
        issues.append({
            "kind": "近重复",
            "severity": 1,
            "count": len(near_dupes),
            "detail": f"{len(near_dupes)} 对相似度 ≥ {near_threshold} 的样本",
        })

    # 按严重度排序：越影响训练越靠前
    issues.sort(key=lambda x: (-x["severity"], -x["count"]))

    return {
        "total": total,
        "text_field": text_field,
        "issues": issues,
        "missing_indexes": missing,
        "exact_duplicate_groups": [
            {"text": t[:80], "count": c} for t, c in list(exact_dupes.items())[:20]
        ],
        "near_duplicates": near_dupes[:40],
        "inconsistent_fields": inconsistent_fields,
        "label_issues": label_issues,
        "summary": {
            "issue_kinds": len(issues),
            "missing": len(missing),
            "exact_duplicate_groups": len(exact_dupes),
            "near_duplicate_pairs": len(near_dupes),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def render(result: dict) -> str:
    out: list[str] = []
    out.append("数据集体检报告")
    out.append("=" * 60)
    out.append(f"样本数：{result['total']}　文本字段：{result['text_field']}")
    out.append("")

    if not result["issues"]:
        out.append("未发现明显问题。")
        return "\n".join(out)

    out.append("问题（按对训练的影响排序）：")
    for i, iss in enumerate(result["issues"], 1):
        out.append(f"  {i}. [{iss['kind']}]  {iss['detail']}")
    out.append("")

    if result["inconsistent_fields"]:
        out.append("字段结构不一致的样本形状：")
        for f in result["inconsistent_fields"][:5]:
            miss = "、".join(f["missing"]) or "无"
            extra = "、".join(f["extra"]) or "无"
            out.append(f"  · {f['count']} 条：缺 {miss}；多 {extra}")
        out.append("")

    if result["exact_duplicate_groups"]:
        out.append("完全重复样例：")
        for g in result["exact_duplicate_groups"][:5]:
            out.append(f"  · ×{g['count']}  {g['text']!r}")
        out.append("")

    if result["near_duplicates"]:
        out.append("近重复样例：")
        for d in result["near_duplicates"][:5]:
            out.append(f"  · 第 {d['a']} 与第 {d['b']} 条，相似度 {d['similarity']}")
            out.append(f"    A: {d['a_preview']}")
            out.append(f"    B: {d['b_preview']}")
        out.append("")

    out.append("注：报告只定位问题，不替你决定删哪一条。")
    return "\n".join(out).rstrip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="dataclean",
        description="给训练与评测数据做体检（仅依赖标准库）",
        epilog=(
            "输入支持 JSONL 或 JSON 数组：\n"
            "  {\"content\": \"...\", \"label\": \"positive\"}\n"
            "  python dataclean.py --file data.jsonl --text-field content\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--file", required=True, help="数据集文件")
    ap.add_argument("--text-field", default="content", help="用于重复检测的文本字段")
    ap.add_argument("--near-threshold", type=float, default=0.85,
                    help="近重复的 Jaccard 相似度阈值")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    try:
        records = load_records(args.file)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    if not records:
        print("数据是空的。", file=sys.stderr)
        return 2

    result = analyze(records, args.text_field, args.near_threshold)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(render(result))
    return 1 if result["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
