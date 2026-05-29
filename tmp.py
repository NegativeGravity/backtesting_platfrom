from pathlib import Path

folder_path = Path("src")
output_file = Path("combined_code.txt")  # Output file name

extensions = {".py", ".ts", ".tsx", ".yml"}

with output_file.open("w", encoding="utf-8") as out:
    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in extensions:
            out.write(f"\n\n{'=' * 80}\n")
            out.write(f"File: {file_path.relative_to(folder_path)}\n")
            out.write(f"{'=' * 80}\n\n")

            try:
                with file_path.open("r", encoding="utf-8") as f:
                    out.write(f.read())
            except Exception as e:
                out.write(f"// Error reading file: {e}\n")