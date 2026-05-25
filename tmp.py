from pathlib import Path

folder_path = Path("src")
output_file = Path("combined_code.txt")  # Output file name

# List of extensions we want to capture
extensions = {".py", ".ts", ".tsx", "yml"}

with output_file.open("w", encoding="utf-8") as out:
    # rglob("*") iterates through all files and directories recursively
    for file_path in folder_path.rglob("*"):
        # Check if it's a file and its extension is in our allowed set
        if file_path.is_file() and file_path.suffix in extensions:
            out.write(f"\n\n{'=' * 80}\n")
            out.write(f"File: {file_path.relative_to(folder_path)}\n")
            out.write(f"{'=' * 80}\n\n")

            try:
                with file_path.open("r", encoding="utf-8") as f:
                    out.write(f.read())
            except Exception as e:
                # Handle potential encoding issues or unreadable files gracefully
                out.write(f"// Error reading file: {e}\n")