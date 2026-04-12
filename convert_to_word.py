"""
将 thesis 文件夹下的所有 Markdown 文件批量转换为 Word (.docx)。
自动移除 LaTeX \tag{} 编号以避免 Pandoc 转换错误。
"""

import os
import re
import subprocess
import tempfile

PANDOC_PATH = r"C:\Users\syxgn\AppData\Local\Pandoc\pandoc.exe"
THESIS_DIR = r"g:\02_Projects\GP_2026\GP_howling_suppression\thesis"
OUTPUT_DIR = r"g:\02_Projects\GP_2026\GP_howling_suppression\thesis_word"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 获取所有 md 文件（排除 _word_formulas 版本）
md_files = [
    f for f in os.listdir(THESIS_DIR)
    if f.endswith(".md") and "word_formulas" not in f
]
md_files.sort()

print(f"找到 {len(md_files)} 个 Markdown 文件待转换：")
for f in md_files:
    print(f"  - {f}")
print()

for md_file in md_files:
    md_path = os.path.join(THESIS_DIR, md_file)
    docx_name = md_file.replace(".md", ".docx")
    docx_path = os.path.join(OUTPUT_DIR, docx_name)

    # 读取 markdown 内容
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 预处理：移除 \tag{x.y} 编号
    # 匹配 \tag{...} 各种格式
    content = re.sub(r"\\tag\{[^}]*\}", "", content)

    # 写入临时文件
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", encoding="utf-8", delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 调用 Pandoc 转换
        result = subprocess.run(
            [PANDOC_PATH, tmp_path, "-o", docx_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if result.returncode == 0:
            print(f"[OK] {md_file} -> {docx_name}")
        else:
            print(f"[FAIL] {md_file} failed: {result.stderr}")

        # show warnings if any
        if result.stderr and "WARNING" in result.stderr:
            for line in result.stderr.split("\n"):
                if "WARNING" in line:
                    print(f"   [WARN] {line.strip()}")

    finally:
        # 清理临时文件
        os.unlink(tmp_path)

print(f"\n转换完成！Word 文件保存在：{OUTPUT_DIR}")

# ========== 合并为单个 Word 文档 ==========
print("\n--- 正在合并为单个文档 ---")

merged_output = os.path.join(OUTPUT_DIR, "thesis_complete.docx")

# 收集所有预处理后的临时文件路径
tmp_paths = []
for md_file in md_files:
    md_path = os.path.join(THESIS_DIR, md_file)
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    # 移除 \tag{} 编号
    content = re.sub(r"\\tag\{[^}]*\}", "", content)
    # 添加分隔注释
    content = "\n\n---\n\n" + content

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", encoding="utf-8", delete=False
    ) as tmp:
        tmp.write(content)
        tmp_paths.append(tmp.name)

try:
    result = subprocess.run(
        [PANDOC_PATH] + tmp_paths + ["-o", merged_output],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode == 0:
        print(f"[OK] Merged -> thesis_complete.docx")
    else:
        print(f"[FAIL] Merge failed: {result.stderr}")
finally:
    for p in tmp_paths:
        os.unlink(p)

print(f"\nAll done! Files in: {OUTPUT_DIR}")
