#!/usr/bin/env python3
"""Generate split_comparison.pdf from TikZ source."""

import subprocess
from pathlib import Path

def main():
    script_dir = Path(__file__).parent.resolve()
    tex_file = script_dir / "split_comparison.tex"
    pdf_file = script_dir / "split_comparison.pdf"

    if not tex_file.exists():
        print(f"Error: {tex_file} not found")
        return 1

    print(f"Compiling {tex_file}...")

    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(script_dir), str(tex_file)],
        capture_output=True,
        text=True,
        cwd=script_dir
    )

    if result.returncode != 0:
        print("pdflatex failed:")
        print(result.stdout)
        print(result.stderr)
        return 1

    # Clean up auxiliary files
    for ext in [".aux", ".log"]:
        aux_file = script_dir / f"split_comparison{ext}"
        if aux_file.exists():
            aux_file.unlink()

    if pdf_file.exists():
        print(f"Successfully generated: {pdf_file}")
        return 0
    else:
        print("Error: PDF was not generated")
        return 1

if __name__ == "__main__":
    exit(main())
