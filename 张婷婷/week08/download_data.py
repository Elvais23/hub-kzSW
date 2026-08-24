

import json
import os
import random
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SRC_DIR, "..", "data"))

random.seed(42)


# ── 工具函数 ──────────────────────────────────────────────────────────────

def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_row(row):
    """统一字段名：sentence1 / sentence2 / label（0/1 整数）"""
    s1 = (row.get("sentence1") or row.get("text1") or row.get("query") or "")
    s2 = (row.get("sentence2") or row.get("text2") or row.get("candidate") or "")
    # score 字段在 C-MTEB / FinanceMTEB 中就是 0/1
    label = int(row.get("label", row.get("score", 0)))
    return {"sentence1": str(s1), "sentence2": str(s2), "label": label}


def print_stats(rows, split, path):
    pos = sum(1 for r in rows if r["label"] == 1)
    neg = len(rows) - pos
    print(f"  {split:12s}: {len(rows):>7,} 条  正样本 {pos:>6,}  负样本 {neg:>6,}  -> {path}")


def preview(path, n=3):
    print(f"\n  [预览 前 {n} 条]")
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            r = json.loads(line)
            tag = "✓ 相似" if r["label"] == 1 else "✗ 不相似"
            print(f"    [{tag}] {r['sentence1']!r}  ||  {r['sentence2']!r}")


# ── BQ Corpus ─────────────────────────────────────────────────────────────

def download_bq():
    print(f"\n{'='*55}")
    print("下载 BQ Corpus（FinanceMTEB/bq_corpus）...")
    print("  注：原始仓库无 train/val split，按 8:1:1 切分")
    out_dir = os.path.join(DATA_DIR, "bq_corpus")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("FinanceMTEB/bq_corpus", split="test")
    all_rows = [normalize_row(r) for r in ds]
    random.shuffle(all_rows)

    n = len(all_rows)
    n_val = n // 10
    n_test = n // 10
    splits = {
        "train": all_rows[n_val + n_test:],
        "validation": all_rows[:n_val],
        "test": all_rows[n_val: n_val + n_test],
    }
    for split, rows in splits.items():
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        save_jsonl(rows, out_path)
        print_stats(rows, split, out_path)

    preview(os.path.join(out_dir, "train.jsonl"))


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    print(f"HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
    print(f"数据保存目录: {DATA_DIR}")

    download_bq()

    print("\n\n全部完成。")


if __name__ == "__main__":
    main()
