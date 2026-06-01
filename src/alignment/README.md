# Alignment Tool

Note: the tool was in the end not used for generation of validation corpora due to time constraints, it can still be used for such task if time permits.

## How to use the alignment tool?

 Usage

  cd <path-to-repo>/sami_ocr
  nix develop

  # Start alignment
  python src/alignment/manual_aligner.py \
      --sme benchmark_data/source/muitalus-sme-Giellatekno.txt \
      --eng benchmark_data/source/muitalus-eng.txt \
      --output benchmark_data/corpus/aligned.jsonl

  Key commands in the TUI:
  - j/k - Navigate SME (North Sami) up/down
  - l/h - Navigate ENG (English) up/down
  - a or Enter - Approve current pair
  - d/D - Delete SME/ENG sentence
  - m/M - Merge with next sentence
  - w - Save session (resumes automatically on next run)
  - ? - Full help screen