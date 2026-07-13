"""
copy_data.py — 从原 RAG 项目复制向量索引文件到本项目

使用方式：
  python scripts/copy_data.py --src ../rag_annual_report/vectorstore

功能：
  1. 验证源目录存在且包含必需文件
  2. 显示文件大小，确认后复制
  3. 校验复制完整性（文件大小比对）
"""

import argparse
import shutil
import sys
from pathlib import Path

REQUIRED_FILES = ["faiss_index.bin", "faiss_meta.json"]
TARGET_DIR = Path(__file__).parent.parent / "vectorstore"


def copy_data(src_dir: str, yes: bool = False) -> None:
    src = Path(src_dir).resolve()

    if not src.exists():
        print(f"错误：源目录不存在：{src}")
        sys.exit(1)

    missing = [f for f in REQUIRED_FILES if not (src / f).exists()]
    if missing:
        print(f"错误：源目录缺少文件：{missing}")
        print("请检查路径是否指向 rag_annual_report/vectorstore/")
        sys.exit(1)

    print(f"源目录：{src}")
    print(f"目标目录：{TARGET_DIR}\n")
    total_mb = 0.0
    for fname in REQUIRED_FILES:
        size_mb = (src / fname).stat().st_size / 1024 / 1024
        total_mb += size_mb
        print(f"  {fname:<25} {size_mb:.1f} MB")
    print(f"\n  合计：{total_mb:.1f} MB")

    if not yes:
        confirm = input("\n确认复制以上文件到 vectorstore/ ？[y/N] ").strip().lower()
        if confirm != "y":
            print("已取消")
            sys.exit(0)

    TARGET_DIR.mkdir(exist_ok=True)
    for fname in REQUIRED_FILES:
        src_file = src / fname
        dst_file = TARGET_DIR / fname
        print(f"复制 {fname}...", end=" ", flush=True)
        shutil.copy2(src_file, dst_file)

        if src_file.stat().st_size != dst_file.stat().st_size:
            print("失败（大小不一致）")
            sys.exit(1)
        print("完成")

    print("\n数据迁移完成，可以运行各方式的脚本了。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 RAG 项目复制向量索引文件")
    parser.add_argument("--src", required=True, help="原 RAG 项目的 vectorstore 目录路径")
    parser.add_argument("-y", action="store_true", help="跳过确认直接复制")
    args = parser.parse_args()
    copy_data(args.src, yes=args.y)
