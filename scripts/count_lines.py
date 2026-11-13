"""Count committed project source lines without dependencies or generated data."""
import argparse
from collections import defaultdict
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".py", ".js", ".html", ".css", ".ps1", ".sh"}


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--output", help="write the Markdown report to this file")
    args = parser.parse_args()
    revision = git("rev-parse", "--verify", "--end-of-options", args.ref + "^{commit}").decode().strip()
    paths = [name for name in git("ls-tree", "-r", "--name-only", "-z", revision).decode().split("\0") if name]
    rows = []
    grouped = defaultdict(lambda: [0, 0, 0])
    for name in paths:
        if Path(name).suffix.lower() not in SOURCE_SUFFIXES:
            continue
        lines = git("show", revision + ":" + name).decode("utf-8-sig").splitlines()
        nonblank = sum(bool(line.strip()) for line in lines)
        category = "主程序 src" if name.startswith("src/") else "工具脚本 scripts" if name.startswith("scripts/") else "测试 tests" if name.startswith("tests/") else "启动脚本"
        values = grouped[category]
        for index, amount in enumerate((1, len(lines), nonblank)):
            values[index] += amount
        rows.append((name, len(lines), nonblank))
    totals = [sum(value[index] for value in grouped.values()) for index in range(3)]
    result = ["# GitHub 代码行数统计", "", f"统计修订：`{revision}`", "",
              "口径：只统计 Git 已提交的 Python、JavaScript、HTML、CSS、PowerShell 与 Shell 源码。",
              "总行数包含注释和空行；非空行数仍包含注释，不等同于纯代码或语句数量。",
              "不包含依赖库、虚拟环境、锁文件、文档、配置、实验 JSON、SVG、图片或模型权重。", "",
              "| 范围 | 文件数 | 总行数 | 非空行数 |", "|---|---:|---:|---:|"]
    for category, values in grouped.items():
        result.append(f"| {category} | {values[0]} | {values[1]} | {values[2]} |")
    result.append(f"| **合计** | **{totals[0]}** | **{totals[1]}** | **{totals[2]}** |")
    result += ["", "## 每个源码文件", "", "| 文件 | 总行数 | 非空行数 |", "|---|---:|---:|"]
    result += [f"| `{name}` | {total} | {nonblank} |" for name, total, nonblank in rows]
    result += ["", "复核命令：在仓库根目录运行 `python scripts/count_lines.py`。", ""]
    document = "\n".join(result)
    if args.output:
        Path(args.output).write_text(document, encoding="utf-8")
    print(document)


if __name__ == "__main__":
    main()
