#!/usr/bin/env python3
"""
Parallel Corpus Alignment Tool for North Sami ↔ English sentence pairs.
TUI-based manual alignment using curses.
"""

import argparse
import curses
import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AlignmentSession:
    """Holds the state of an alignment session."""
    sme_file: str
    eng_file: str
    sme_cursor: int = 0
    eng_cursor: int = 0
    aligned_pairs: list = field(default_factory=list)
    deleted_sme: set = field(default_factory=set)
    deleted_eng: set = field(default_factory=set)
    merged: dict = field(default_factory=dict)  # "sme:idx" or "eng:idx" -> merged text
    splits: dict = field(default_factory=dict)  # "sme:idx" or "eng:idx" -> [part1, part2]

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "sme_file": self.sme_file,
            "eng_file": self.eng_file,
            "sme_cursor": self.sme_cursor,
            "eng_cursor": self.eng_cursor,
            "aligned_pairs": self.aligned_pairs,
            "deleted_sme": list(self.deleted_sme),
            "deleted_eng": list(self.deleted_eng),
            "merged": self.merged,
            "splits": self.splits,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AlignmentSession":
        """Deserialize from dict."""
        return cls(
            sme_file=data["sme_file"],
            eng_file=data["eng_file"],
            sme_cursor=data.get("sme_cursor", 0),
            eng_cursor=data.get("eng_cursor", 0),
            aligned_pairs=data.get("aligned_pairs", []),
            deleted_sme=set(data.get("deleted_sme", [])),
            deleted_eng=set(data.get("deleted_eng", [])),
            merged=data.get("merged", {}),
            splits=data.get("splits", {}),
        )


class LazyFileLoader:
    """Lazy-loads file lines in chunks with LRU caching."""

    CHUNK_SIZE = 100
    MAX_CHUNKS = 5

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self._total_lines: Optional[int] = None
        self._cache: dict[int, list[str]] = {}
        self._access_order: list[int] = []

    @property
    def total_lines(self) -> int:
        if self._total_lines is None:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self._total_lines = sum(1 for _ in f)
        return self._total_lines

    def _load_chunk(self, chunk_idx: int) -> list[str]:
        """Load a chunk of lines from the file."""
        if chunk_idx in self._cache:
            # Move to end of access order
            self._access_order.remove(chunk_idx)
            self._access_order.append(chunk_idx)
            return self._cache[chunk_idx]

        # Evict oldest if at capacity
        while len(self._cache) >= self.MAX_CHUNKS:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        start = chunk_idx * self.CHUNK_SIZE
        lines = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= start + self.CHUNK_SIZE:
                    break
                if i >= start:
                    lines.append(line.rstrip("\n\r"))

        self._cache[chunk_idx] = lines
        self._access_order.append(chunk_idx)
        return lines

    def get_line(self, idx: int) -> Optional[str]:
        """Get a specific line by index."""
        if idx < 0 or idx >= self.total_lines:
            return None
        chunk_idx = idx // self.CHUNK_SIZE
        chunk = self._load_chunk(chunk_idx)
        offset = idx % self.CHUNK_SIZE
        return chunk[offset] if offset < len(chunk) else None

    def get_lines(self, start: int, count: int) -> list[tuple[int, str]]:
        """Get multiple lines with their indices."""
        result = []
        for i in range(start, min(start + count, self.total_lines)):
            line = self.get_line(i)
            if line is not None:
                result.append((i, line))
        return result


# ============================================================================
# TUI Application
# ============================================================================

class AlignerTUI:
    """Curses-based TUI for manual alignment."""

    # Color pairs
    COLOR_HEADER = 1
    COLOR_CURRENT = 2
    COLOR_DELETED = 3
    COLOR_STATUS = 4
    COLOR_HELP = 5

    def __init__(self, session: AlignmentSession, output_path: str, state_path: str):
        self.session = session
        self.output_path = Path(output_path)
        self.state_path = Path(state_path)
        self.sme_loader = LazyFileLoader(session.sme_file)
        self.eng_loader = LazyFileLoader(session.eng_file)
        self.message = ""
        self.show_help = False

    def run(self):
        """Main entry point."""
        curses.wrapper(self._main)

    def _main(self, stdscr):
        """Curses main loop."""
        self.stdscr = stdscr
        self._init_colors()
        curses.curs_set(0)  # Hide cursor

        while True:
            self._draw()
            key = stdscr.getch()
            if not self._handle_key(key):
                break

    def _init_colors(self):
        """Initialize color pairs."""
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(self.COLOR_HEADER, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.init_pair(self.COLOR_CURRENT, curses.COLOR_BLACK, curses.COLOR_YELLOW)
        curses.init_pair(self.COLOR_DELETED, curses.COLOR_RED, -1)
        curses.init_pair(self.COLOR_STATUS, curses.COLOR_BLACK, curses.COLOR_GREEN)
        curses.init_pair(self.COLOR_HELP, curses.COLOR_CYAN, -1)

    def _draw(self):
        """Draw the entire screen."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        if self.show_help:
            self._draw_help(h, w)
        else:
            self._draw_header(w)
            self._draw_columns(h, w)
            self._draw_status(h, w)

        self.stdscr.refresh()

    def _draw_header(self, w: int):
        """Draw the header bar."""
        sme_total = self.sme_loader.total_lines
        eng_total = self.eng_loader.total_lines
        aligned = len(self.session.aligned_pairs)

        header = f" Aligner | SME: {self.session.sme_cursor + 1}/{sme_total} | ENG: {self.session.eng_cursor + 1}/{eng_total} | Aligned: {aligned} | [?]help "
        header = header.ljust(w)[:w]
        self.stdscr.attron(curses.color_pair(self.COLOR_HEADER))
        self.stdscr.addstr(0, 0, header)
        self.stdscr.attroff(curses.color_pair(self.COLOR_HEADER))

    def _draw_columns(self, h: int, w: int):
        """Draw the two-column sentence view."""
        col_width = w // 2 - 1
        content_height = h - 4  # Leave room for header and status

        # Column headers
        sme_header = " NORTH SAMI".ljust(col_width)[:col_width]
        eng_header = " ENGLISH".ljust(col_width)[:col_width]
        self.stdscr.addstr(1, 0, sme_header, curses.A_BOLD)
        self.stdscr.addstr(1, col_width + 1, "|", curses.A_DIM)
        self.stdscr.addstr(1, col_width + 2, eng_header, curses.A_BOLD)

        # Separator
        self.stdscr.addstr(2, 0, "-" * col_width + "+" + "-" * col_width)

        # Get visible lines (context around cursor)
        context = content_height // 2
        sme_start = max(0, self.session.sme_cursor - context)
        eng_start = max(0, self.session.eng_cursor - context)

        sme_lines = self._get_visible_lines("sme", sme_start, content_height)
        eng_lines = self._get_visible_lines("eng", eng_start, content_height)

        # Draw lines
        for row in range(content_height):
            y = row + 3

            # SME column
            if row < len(sme_lines):
                idx, text, is_deleted = sme_lines[row]
                self._draw_line(y, 0, col_width, idx, text, is_deleted,
                                idx == self.session.sme_cursor)

            # Separator
            try:
                self.stdscr.addstr(y, col_width, "|", curses.A_DIM)
            except curses.error:
                pass

            # ENG column
            if row < len(eng_lines):
                idx, text, is_deleted = eng_lines[row]
                self._draw_line(y, col_width + 1, col_width, idx, text, is_deleted,
                                idx == self.session.eng_cursor)

    def _get_visible_lines(self, side: str, start: int, count: int) -> list[tuple[int, str, bool]]:
        """Get visible lines for a side, accounting for deletes/merges/splits."""
        loader = self.sme_loader if side == "sme" else self.eng_loader
        deleted = self.session.deleted_sme if side == "sme" else self.session.deleted_eng

        result = []
        idx = start
        while len(result) < count and idx < loader.total_lines:
            key = f"{side}:{idx}"
            is_deleted = idx in deleted

            # Check for splits
            if key in self.session.splits:
                parts = self.session.splits[key]
                for i, part in enumerate(parts):
                    if len(result) < count:
                        result.append((idx, f"[{idx}:{i}] {part}", is_deleted))
            # Check for merges
            elif key in self.session.merged:
                result.append((idx, self.session.merged[key], is_deleted))
            else:
                line = loader.get_line(idx)
                if line is not None:
                    result.append((idx, line, is_deleted))
            idx += 1

        return result

    def _draw_line(self, y: int, x: int, width: int, idx: int, text: str, is_deleted: bool, is_current: bool):
        """Draw a single line with proper formatting."""
        prefix = ">" if is_current else " "
        display = f"{prefix}[{idx:4d}] {text}"
        display = display[:width - 1].ljust(width - 1)

        attr = curses.A_NORMAL
        if is_current:
            attr = curses.color_pair(self.COLOR_CURRENT)
        elif is_deleted:
            attr = curses.color_pair(self.COLOR_DELETED) | curses.A_DIM

        try:
            self.stdscr.addstr(y, x, display, attr)
        except curses.error:
            pass

    def _draw_status(self, h: int, w: int):
        """Draw the status bar."""
        shortcuts = "[a]pprove [d/D]del [m/M]merge [s/S]split [x/X]swap [c/C]copy [j/k/l/h]nav [g]oto [w]save [e]xport [q]uit"
        status_line = shortcuts[:w - 1].ljust(w - 1)

        self.stdscr.attron(curses.color_pair(self.COLOR_STATUS))
        self.stdscr.addstr(h - 2, 0, status_line)
        self.stdscr.attroff(curses.color_pair(self.COLOR_STATUS))

        # Message line
        if self.message:
            self.stdscr.addstr(h - 1, 0, self.message[:w - 1])
            self.message = ""

    def _draw_help(self, h: int, w: int):
        """Draw the help screen."""
        help_text = [
            "PARALLEL CORPUS ALIGNER - HELP",
            "",
            "Navigation:",
            "  j/k     - Move SME cursor down/up",
            "  l/h     - Move ENG cursor down/up",
            "  g       - Go to specific line number",
            "",
            "Alignment Operations:",
            "  a/Enter - Approve current pair and advance",
            "  d/D     - Delete SME/ENG sentence",
            "  m/M     - Merge SME/ENG with next sentence",
            "  s/S     - Split SME/ENG sentence",
            "  x/X     - Swap SME/ENG with next sentence",
            "",
            "Clipboard:",
            "  c/C     - Copy SME/ENG to clipboard",
            "",
            "File Operations:",
            "  w       - Save session state",
            "  e       - Export aligned pairs",
            "  q       - Quit (prompts to save)",
            "",
            "Press any key to return..."
        ]

        self.stdscr.attron(curses.color_pair(self.COLOR_HELP))
        for i, line in enumerate(help_text):
            if i < h:
                self.stdscr.addstr(i, 0, line[:w - 1])
        self.stdscr.attroff(curses.color_pair(self.COLOR_HELP))

    def _handle_key(self, key: int) -> bool:
        """Handle keypress. Returns False to quit."""
        if self.show_help:
            self.show_help = False
            return True

        # Navigation
        if key == ord('j'):
            self._move_cursor("sme", 1)
        elif key == ord('k'):
            self._move_cursor("sme", -1)
        elif key == ord('l'):
            self._move_cursor("eng", 1)
        elif key == ord('h'):
            self._move_cursor("eng", -1)
        elif key == ord('g'):
            self._goto_line()

        # Alignment operations
        elif key in (ord('a'), ord('\n'), curses.KEY_ENTER):
            self._approve_pair()
        elif key == ord('d'):
            self._delete_sentence("sme")
        elif key == ord('D'):
            self._delete_sentence("eng")
        elif key == ord('m'):
            self._merge_with_next("sme")
        elif key == ord('M'):
            self._merge_with_next("eng")
        elif key == ord('s'):
            self._split_sentence("sme")
        elif key == ord('S'):
            self._split_sentence("eng")
        elif key == ord('x'):
            self._swap_with_next("sme")
        elif key == ord('X'):
            self._swap_with_next("eng")

        # Clipboard
        elif key == ord('c'):
            self._copy_to_clipboard("sme")
        elif key == ord('C'):
            self._copy_to_clipboard("eng")

        # File operations
        elif key == ord('w'):
            self._save_session()
        elif key == ord('e'):
            self._export_aligned()
        elif key == ord('q'):
            return self._quit_prompt()
        elif key == ord('?'):
            self.show_help = True

        return True

    # ========================================================================
    # Operations
    # ========================================================================

    def _move_cursor(self, side: str, delta: int):
        """Move cursor, skipping deleted sentences."""
        if side == "sme":
            new_pos = self.session.sme_cursor + delta
            max_pos = self.sme_loader.total_lines - 1
            self.session.sme_cursor = max(0, min(new_pos, max_pos))
        else:
            new_pos = self.session.eng_cursor + delta
            max_pos = self.eng_loader.total_lines - 1
            self.session.eng_cursor = max(0, min(new_pos, max_pos))

    def _goto_line(self):
        """Prompt for line number and jump."""
        curses.echo()
        curses.curs_set(1)
        h, w = self.stdscr.getmaxyx()
        self.stdscr.addstr(h - 1, 0, "Go to (sme,eng or just sme): ".ljust(w - 1))
        self.stdscr.refresh()

        try:
            inp = self.stdscr.getstr(h - 1, 30, 20).decode("utf-8").strip()
            if "," in inp:
                sme, eng = inp.split(",")
                self.session.sme_cursor = max(0, min(int(sme) - 1, self.sme_loader.total_lines - 1))
                self.session.eng_cursor = max(0, min(int(eng) - 1, self.eng_loader.total_lines - 1))
            else:
                line = int(inp) - 1
                self.session.sme_cursor = max(0, min(line, self.sme_loader.total_lines - 1))
                self.session.eng_cursor = max(0, min(line, self.eng_loader.total_lines - 1))
            self.message = f"Jumped to line {inp}"
        except (ValueError, IndexError):
            self.message = "Invalid input"
        finally:
            curses.noecho()
            curses.curs_set(0)

    def _get_current_text(self, side: str) -> str:
        """Get the current text for a side, accounting for modifications."""
        cursor = self.session.sme_cursor if side == "sme" else self.session.eng_cursor
        loader = self.sme_loader if side == "sme" else self.eng_loader
        key = f"{side}:{cursor}"

        if key in self.session.merged:
            return self.session.merged[key]
        elif key in self.session.splits:
            return " | ".join(self.session.splits[key])
        else:
            return loader.get_line(cursor) or ""

    def _approve_pair(self):
        """Approve the current pair and advance both cursors."""
        sme_text = self._get_current_text("sme")
        eng_text = self._get_current_text("eng")

        if not sme_text or not eng_text:
            self.message = "Cannot approve: empty text"
            return

        self.session.aligned_pairs.append({
            "sme": sme_text,
            "eng": eng_text,
            "sme_idx": self.session.sme_cursor,
            "eng_idx": self.session.eng_cursor,
        })

        self._advance_cursor("sme")
        self._advance_cursor("eng")
        self.message = f"Approved pair #{len(self.session.aligned_pairs)}"

    def _advance_cursor(self, side: str):
        """Advance cursor to next non-deleted sentence."""
        deleted = self.session.deleted_sme if side == "sme" else self.session.deleted_eng
        loader = self.sme_loader if side == "sme" else self.eng_loader

        if side == "sme":
            self.session.sme_cursor += 1
            while self.session.sme_cursor in deleted and self.session.sme_cursor < loader.total_lines:
                self.session.sme_cursor += 1
        else:
            self.session.eng_cursor += 1
            while self.session.eng_cursor in deleted and self.session.eng_cursor < loader.total_lines:
                self.session.eng_cursor += 1

    def _delete_sentence(self, side: str):
        """Mark the current sentence as deleted."""
        cursor = self.session.sme_cursor if side == "sme" else self.session.eng_cursor
        deleted = self.session.deleted_sme if side == "sme" else self.session.deleted_eng

        deleted.add(cursor)
        self._advance_cursor(side)
        self.message = f"Deleted {side.upper()} line {cursor + 1}"

    def _merge_with_next(self, side: str):
        """Merge current sentence with the next one."""
        cursor = self.session.sme_cursor if side == "sme" else self.session.eng_cursor
        loader = self.sme_loader if side == "sme" else self.eng_loader
        deleted = self.session.deleted_sme if side == "sme" else self.session.deleted_eng

        if cursor >= loader.total_lines - 1:
            self.message = "No next sentence to merge"
            return

        current = self._get_current_text(side)
        next_idx = cursor + 1
        while next_idx in deleted and next_idx < loader.total_lines:
            next_idx += 1

        if next_idx >= loader.total_lines:
            self.message = "No next sentence to merge"
            return

        next_key = f"{side}:{next_idx}"
        if next_key in self.session.merged:
            next_text = self.session.merged[next_key]
        else:
            next_text = loader.get_line(next_idx) or ""

        merged = f"{current} {next_text}"
        self.session.merged[f"{side}:{cursor}"] = merged
        deleted.add(next_idx)
        self.message = f"Merged {side.upper()} lines {cursor + 1} and {next_idx + 1}"

    def _split_sentence(self, side: str):
        """Split the current sentence at a specified position."""
        cursor = self.session.sme_cursor if side == "sme" else self.session.eng_cursor
        text = self._get_current_text(side)

        curses.echo()
        curses.curs_set(1)
        h, w = self.stdscr.getmaxyx()

        # Show the text and ask for split position
        self.stdscr.addstr(h - 1, 0, f"Split at char position (0-{len(text)}): ".ljust(w - 1))
        self.stdscr.refresh()

        try:
            pos = int(self.stdscr.getstr(h - 1, 35, 10).decode("utf-8").strip())
            if 0 < pos < len(text):
                part1 = text[:pos].strip()
                part2 = text[pos:].strip()
                self.session.splits[f"{side}:{cursor}"] = [part1, part2]
                self.message = f"Split {side.upper()} line {cursor + 1} at position {pos}"
            else:
                self.message = "Invalid split position"
        except ValueError:
            self.message = "Invalid input"
        finally:
            curses.noecho()
            curses.curs_set(0)

    def _swap_with_next(self, side: str):
        """Swap current sentence with the next one (reorder in merged dict)."""
        cursor = self.session.sme_cursor if side == "sme" else self.session.eng_cursor
        loader = self.sme_loader if side == "sme" else self.eng_loader

        if cursor >= loader.total_lines - 1:
            self.message = "No next sentence to swap"
            return

        current = self._get_current_text(side)
        next_idx = cursor + 1
        next_key = f"{side}:{next_idx}"

        if next_key in self.session.merged:
            next_text = self.session.merged[next_key]
        else:
            next_text = loader.get_line(next_idx) or ""

        # Store swapped versions in merged dict
        self.session.merged[f"{side}:{cursor}"] = next_text
        self.session.merged[next_key] = current
        self.message = f"Swapped {side.upper()} lines {cursor + 1} and {next_idx + 1}"

    def _copy_to_clipboard(self, side: str):
        """Copy current sentence to clipboard."""
        text = self._get_current_text(side)

        # Try different clipboard commands
        for cmd in [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"], ["pbcopy"]]:
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                proc.communicate(input=text.encode("utf-8"))
                if proc.returncode == 0:
                    self.message = f"Copied {side.upper()} to clipboard"
                    return
            except FileNotFoundError:
                continue

        self.message = "No clipboard tool found (xclip/xsel/pbcopy)"

    # ========================================================================
    # I/O
    # ========================================================================

    def _save_session(self):
        """Save session state to JSON."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.session.to_dict(), f, indent=2, ensure_ascii=False)
        self.message = f"Session saved to {self.state_path}"

    def _export_aligned(self):
        """Export aligned pairs to JSONL and TSV."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # JSONL
        with open(self.output_path, "w", encoding="utf-8") as f:
            for pair in self.session.aligned_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        # TSV
        tsv_path = self.output_path.with_suffix(".tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("sme\teng\n")
            for pair in self.session.aligned_pairs:
                sme = pair["sme"].replace("\t", " ")
                eng = pair["eng"].replace("\t", " ")
                f.write(f"{sme}\t{eng}\n")

        self.message = f"Exported {len(self.session.aligned_pairs)} pairs to {self.output_path}"

    def _quit_prompt(self) -> bool:
        """Prompt to save before quitting. Returns False to quit."""
        if not self.session.aligned_pairs:
            return False

        curses.echo()
        curses.curs_set(1)
        h, w = self.stdscr.getmaxyx()
        self.stdscr.addstr(h - 1, 0, "Save before quitting? [y/n/c(ancel)]: ".ljust(w - 1))
        self.stdscr.refresh()

        try:
            choice = self.stdscr.getstr(h - 1, 38, 1).decode("utf-8").strip().lower()
            if choice == "y":
                self._save_session()
                self._export_aligned()
                return False
            elif choice == "n":
                return False
            else:
                self.message = "Quit cancelled"
                return True
        finally:
            curses.noecho()
            curses.curs_set(0)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parallel corpus alignment tool for North Sami ↔ English"
    )
    parser.add_argument("--sme", required=True, help="Path to North Sami text file")
    parser.add_argument("--eng", required=True, help="Path to English text file")
    parser.add_argument("--output", default="aligned.jsonl", help="Output JSONL path")
    parser.add_argument("--state", help="Session state JSON (for resume)")

    args = parser.parse_args()

    # Resolve paths
    sme_path = Path(args.sme).resolve()
    eng_path = Path(args.eng).resolve()
    output_path = Path(args.output).resolve()
    state_path = Path(args.state) if args.state else output_path.with_suffix(".state.json")

    # Validate input files
    if not sme_path.exists():
        print(f"Error: SME file not found: {sme_path}", file=sys.stderr)
        sys.exit(1)
    if not eng_path.exists():
        print(f"Error: ENG file not found: {eng_path}", file=sys.stderr)
        sys.exit(1)

    # Load or create session
    if state_path.exists():
        print(f"Resuming session from {state_path}")
        with open(state_path, "r", encoding="utf-8") as f:
            session = AlignmentSession.from_dict(json.load(f))
    else:
        session = AlignmentSession(sme_file=str(sme_path), eng_file=str(eng_path))

    # Run TUI
    tui = AlignerTUI(session, str(output_path), str(state_path))
    tui.run()


if __name__ == "__main__":
    main()
