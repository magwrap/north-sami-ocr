"""Conjunction-based sentence splitter for North Sámi text.

Splits long sentences at conjunction boundaries to improve OCR accuracy
on width-constrained input images. Based on the finding that OCR accuracy
degrades beyond ~80 characters due to 800px width compression.

North Sámi Conjunctions:
- Coordinate: ja, muhto, dahje, dehe, vai, sihke
- Subordinate (causal): go, dasgo, dannego, dainnago, daningo
- Subordinate (temporal): goas, ovdal go, maŋŋil go, dassážii go, dan botta go, dalle go
- Subordinate (conditional): jos, jus, beare
- Subordinate (purpose): vai, amas
- Subordinate (other): ahte, nugo, dego, vaikko
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SplitResult:
    """Result of splitting a sentence."""
    segments: List[str]
    split_points: List[Tuple[int, str]]  # (position, conjunction)
    original_length: int


# North Sámi conjunctions organized by type
# Priority order matters - we check multi-word conjunctions first
MULTI_WORD_CONJUNCTIONS = [
    "ovdal go",     # before (when)
    "maŋŋil go",    # after (when)
    "dassážii go",  # until
    "dan botta go", # while
    "dalle go",     # when (then)
    "nu ahte",      # so that
]

# Single-word conjunctions to split BEFORE
# Ordered by frequency/importance in North Sámi texts
COORDINATE_CONJUNCTIONS = [
    "ja",      # and
    "muhto",   # but
    "dahje",   # or
    "dehe",    # or (alternative)
    "vai",     # or / in order to
    "sihke",   # both...and
]

SUBORDINATE_CONJUNCTIONS = [
    # Causal
    "go",       # when/because (most common subordinator)
    "dasgo",    # because
    "dannego",  # because
    "dainnago", # because (variant)
    "daningo",  # because (variant)
    # Conditional
    "jos",      # if
    "jus",      # if (variant)
    "beare",    # just/only if
    # Purpose
    "amas",     # so that not
    # Other
    "ahte",     # that (complementizer)
    "nugo",     # like/as
    "dego",     # like/as
    "vaikko",   # although/even if
    "goas",     # when (temporal)
]

# All single-word conjunctions (coordinate first for split priority)
ALL_SINGLE_CONJUNCTIONS = COORDINATE_CONJUNCTIONS + SUBORDINATE_CONJUNCTIONS


def find_conjunction_positions(text: str) -> List[Tuple[int, str, str]]:
    """
    Find all conjunction positions in text.

    Returns:
        List of (start_position, end_position, conjunction) tuples,
        sorted by position.
    """
    positions = []
    text_lower = text.lower()

    # Check multi-word conjunctions first (greedy matching)
    for conj in MULTI_WORD_CONJUNCTIONS:
        pattern = r'\b' + re.escape(conj) + r'\b'
        for match in re.finditer(pattern, text_lower):
            positions.append((match.start(), match.end(), conj))

    # Check single-word conjunctions
    for conj in ALL_SINGLE_CONJUNCTIONS:
        pattern = r'\b' + re.escape(conj) + r'\b'
        for match in re.finditer(pattern, text_lower):
            # Skip if this position is already covered by a multi-word conjunction
            start, end = match.start(), match.end()
            if any(p[0] <= start < p[1] for p in positions):
                continue
            positions.append((start, end, conj))

    # Sort by position
    positions.sort(key=lambda x: x[0])
    return positions


def split_at_conjunctions(
    text: str,
    min_segment_length: int = 20,
    max_segment_length: int = 80
) -> SplitResult:
    """
    Split text at conjunction boundaries, preserving word integrity.

    Splits BEFORE conjunctions to keep the conjunction with its clause.
    Enforces minimum segment length to avoid over-splitting.

    Args:
        text: Input text to split
        min_segment_length: Minimum characters per segment (default: 20)
        max_segment_length: Target max length that triggers splitting (default: 80)

    Returns:
        SplitResult with segments and split point information

    Example:
        >>> result = split_at_conjunctions("Mon lean ipmirdan, ahte dat lea buorre.")
        >>> print(result.segments)
        ['Mon lean ipmirdan,', 'ahte dat lea buorre.']
    """
    original_length = len(text)

    # Don't split short texts
    if original_length <= max_segment_length:
        return SplitResult(
            segments=[text.strip()],
            split_points=[],
            original_length=original_length
        )

    # Find all conjunction positions
    conj_positions = find_conjunction_positions(text)

    if not conj_positions:
        # No conjunctions found, return as-is
        return SplitResult(
            segments=[text.strip()],
            split_points=[],
            original_length=original_length
        )

    # Greedy splitting: try to keep segments under max_segment_length
    segments = []
    split_points = []
    current_start = 0

    for pos, end_pos, conj in conj_positions:
        # Calculate segment length if we split here
        potential_segment = text[current_start:pos].strip()

        # Skip if this would create a segment that's too short
        if len(potential_segment) < min_segment_length:
            continue

        # Skip if we're at the very beginning
        if pos == 0:
            continue

        # Check if current accumulated text is getting long
        current_length = pos - current_start

        if current_length >= max_segment_length:
            # Split here
            segments.append(potential_segment)
            split_points.append((pos, conj))
            current_start = pos

    # Add the final segment
    final_segment = text[current_start:].strip()
    if final_segment:
        segments.append(final_segment)

    # If no splits were made, return original
    if len(segments) <= 1:
        return SplitResult(
            segments=[text.strip()],
            split_points=[],
            original_length=original_length
        )

    return SplitResult(
        segments=segments,
        split_points=split_points,
        original_length=original_length
    )


def split_aggressive(
    text: str,
    min_segment_length: int = 20
) -> SplitResult:
    """
    More aggressive splitting - split at EVERY valid conjunction.

    Unlike split_at_conjunctions which only splits when needed,
    this splits at every conjunction position (respecting min_segment_length).

    Useful for comparing different splitting strategies.
    """
    original_length = len(text)
    conj_positions = find_conjunction_positions(text)

    if not conj_positions:
        return SplitResult(
            segments=[text.strip()],
            split_points=[],
            original_length=original_length
        )

    segments = []
    split_points = []
    current_start = 0

    for pos, end_pos, conj in conj_positions:
        potential_segment = text[current_start:pos].strip()

        # Only split if both resulting segments would be long enough
        remaining = text[pos:].strip()

        if len(potential_segment) >= min_segment_length and len(remaining) >= min_segment_length:
            segments.append(potential_segment)
            split_points.append((pos, conj))
            current_start = pos

    # Add final segment
    final_segment = text[current_start:].strip()
    if final_segment:
        segments.append(final_segment)

    if len(segments) <= 1:
        return SplitResult(
            segments=[text.strip()],
            split_points=[],
            original_length=original_length
        )

    return SplitResult(
        segments=segments,
        split_points=split_points,
        original_length=original_length
    )


def rejoin_segments(segments: List[str], separator: str = " ") -> str:
    """
    Rejoin split segments into a single string.

    Used after OCR to reconstruct the full prediction from segment predictions.
    """
    return separator.join(seg.strip() for seg in segments if seg.strip())


def analyze_text_conjunctions(text: str) -> dict:
    """
    Analyze conjunction usage in a text.

    Returns statistics about which conjunctions appear and where.
    Useful for understanding the splitting potential of a text.
    """
    positions = find_conjunction_positions(text)

    conj_counts = {}
    for _, _, conj in positions:
        conj_counts[conj] = conj_counts.get(conj, 0) + 1

    return {
        "text_length": len(text),
        "total_conjunctions": len(positions),
        "conjunction_counts": conj_counts,
        "positions": positions,
        "could_split": len(text) > 80 and len(positions) > 0
    }


# Example usage and testing
if __name__ == "__main__":
    # Test cases from the ground truth
    test_sentences = [
        # Short sentence - should not split
        "Mon lean okta sápmelaš.",

        # Medium sentence with 'ja' - may split
        "Mon lean okta sápmelaš, guhte lean bargan visot sámi bargguid ja mon dovddan visot sámi dili.",

        # Long sentence with multiple conjunctions (from entry 3)
        "Ja mon lean ipmirdan, ahte Ruoŧa hállehus háliida min veahkehit nu olu go sáhttá, muhto sii eai oaččo riekta čielgasa, jur got dat lea min eallin ja dilli, deinnago sápmelaš ii sáhte jur juste čilget nu got lea.",

        # Sentence with 'go' subordinate clause
        "Ja dasa lea dát sivva: go sápmelaš boahtá moskkus gámmirii, de son ii ipmir ii báljo maidege, go ii biegga beasa bossut njuni vuostá.",

        # Sentence with 'jos' conditional
        "Jos áldduid šaddá višahit garrasit, de šaddet hilbadat, ja nubbi dat, ahte go áldduid šaddá višahit sakka, de reitojit.",
    ]

    print("=" * 70)
    print("NORTH SÁMI CONJUNCTION SPLITTER - TEST")
    print("=" * 70)

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n[Test {i}] Original ({len(sentence)} chars):")
        print(f"  {sentence[:80]}{'...' if len(sentence) > 80 else ''}")

        # Analyze conjunctions
        analysis = analyze_text_conjunctions(sentence)
        print(f"  Conjunctions found: {analysis['conjunction_counts']}")

        # Try splitting
        result = split_at_conjunctions(sentence)

        if len(result.segments) > 1:
            print(f"  Split into {len(result.segments)} segments:")
            for j, seg in enumerate(result.segments, 1):
                print(f"    {j}. ({len(seg)} chars) {seg[:60]}{'...' if len(seg) > 60 else ''}")

            # Verify rejoin
            rejoined = rejoin_segments(result.segments)
            print(f"  Rejoined matches: {rejoined == sentence}")
        else:
            print(f"  No split needed (length <= 80 or no valid split points)")

    print("\n" + "=" * 70)
