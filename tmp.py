from pathlib import Path

folder_path = Path("src")
output_file = Path("ts.txt")

with output_file.open("w", encoding="utf-8") as out:
    for py_file in folder_path.rglob("*.ts"):
        out.write(f"\n\n{'=' * 80}\n")
        out.write(f"File: {py_file.relative_to(folder_path)}\n")
        out.write(f"{'=' * 80}\n\n")

        with py_file.open("r", encoding="utf-8") as f:
            out.write(f.read())