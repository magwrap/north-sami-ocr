from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTChar
from collections import Counter
import re, unicodedata, nltk
nltk.download('punkt')

HEADER_FONT_RATIO   = 1.15
FOOTNOTE_FONT_RATIO = 0.85

def get_body_font_size(path: str) -> float:
    sizes = Counter()
    for page in extract_pages(path):
        for element in page:
            if not isinstance(element, LTTextBox):
                continue
            for line in element:
                if not isinstance(line, LTTextLine):
                    continue
                for char in line:
                    if isinstance(char, LTChar) and char.get_text().strip():
                        sizes[round(char.size, 1)] += 1
    return sizes.most_common(1)[0][0]

def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\xa0", " ").replace("\u200b", " ")
    s = "".join(c for c in s if unicodedata.category(c)[0] != "C")
    s = re.sub(r"[^\S\n]+", " ", s)
    return s.strip()

def extract_structured_lines(path: str, body_size: float) -> list[dict]:
    lines_out = []
    for page in extract_pages(path):
        for element in page:
            if not isinstance(element, LTTextBox):
                continue
            for line in element:
                if not isinstance(line, LTTextLine):
                    continue
                line_text = ""
                sizes = []
                for char in line:
                    if isinstance(char, LTChar):
                        line_text += char.get_text()
                        if char.get_text().strip():
                            sizes.append(char.size)
                line_text = clean_text(line_text)
                if not line_text or len(line_text) < 3 or not sizes:
                    continue
                avg_size = sum(sizes) / len(sizes)
                if avg_size >= body_size * HEADER_FONT_RATIO:
                    kind = "header"
                elif avg_size <= body_size * FOOTNOTE_FONT_RATIO:
                    kind = "footnote"
                else:
                    kind = "body"
                lines_out.append({"text": line_text, "kind": kind})
    return lines_out

def is_page_number(text: str) -> bool:
    """Check if text is a standalone page number like '134 .'"""
    return bool(re.match(r'^\d+\s*\.?\s*$', text.strip()))

def is_likely_header_fragment(text: str) -> bool:
    """Check if text is a broken header fragment (part of a multi-line header)."""
    # Very short all-caps lines are likely header fragments
    words = text.split()
    return (
        len(text) < 40 and
        text.isupper() and
        len(words) <= 4
    )

def is_footnote_marker(text: str) -> bool:
    """Check if text starts with a footnote marker (number followed by explanatory text)."""
    # Footnotes typically start with a number followed by explanatory content
    # E.g., "13 Turi refers here to..."
    match = re.match(r'^\d{1,3}\s+[A-Z]', text)
    if match:
        # Additional check: footnotes often contain phrases like "refers to", "i.e.", "see", etc.
        footnote_indicators = ['refers to', 'refers here', 'i.e.', 'i . e .', 'see', 'See', 'cf.', 'cf .']
        return any(indicator in text for indicator in footnote_indicators)
    return False

def build_sentences(structured_lines: list[dict]) -> list[str]:
    """
    Build sentences from structured lines.
    Strategy: Skip page numbers, footnotes, and header fragments, then use NLTK tokenization.
    Headers without ending punctuation get a period added to ensure proper sentence separation.
    """
    all_text = []

    for item in structured_lines:
        text = item["text"]
        kind = item["kind"]

        # Skip footnotes detected by font size
        if kind == "footnote":
            continue

        # Skip page numbers
        if is_page_number(text):
            continue

        # Skip footnotes detected by content markers (e.g., "13 Turi refers here to...")
        if is_footnote_marker(text):
            continue

        # Skip likely header fragments (very short all-caps lines)
        # These cause issues when they're parts of multi-line headers
        if is_likely_header_fragment(text):
            continue

        # # For headers, ensure they end with punctuation
        # # This prevents them from being concatenated with the following sentence
        # if kind == "header":
        #     # Check if header ends with sentence-ending punctuation
        #     if not text.endswith(('.', '!', '?', ':', ';')):
        #         text = text + " ."

        # Accumulate all other text
        all_text.append(text)

    # Join all text and use NLTK to tokenize into sentences
    combined = " ".join(all_text)
    sentences = []
    for sent in nltk.sent_tokenize(combined, language="english"):
        if sent.strip():
            # Additional check: remove any sentences that slipped through with footnote markers
            if not is_footnote_marker(sent):
                sentences.append(sent.strip())

    return sentences

# --- debug helper to tune ratios ---
def debug_font_sizes(path: str, max_lines: int = 50):
    print("\n--- Font size sample ---")
    count = 0
    for page in extract_pages(path):
        for element in page:
            if not isinstance(element, LTTextBox):
                continue
            for line in element:
                if not isinstance(line, LTTextLine):
                    continue
                sizes = [c.size for c in line if isinstance(c, LTChar) and c.get_text().strip()]
                text = clean_text(line.get_text())
                if sizes and text:
                    print(f"  size={sum(sizes)/len(sizes):.1f}  text='{text[:60]}'")
                    count += 1
                    if count >= max_lines:
                        return

def format_sentence(sent: str) -> str:
    """Format sentence to match Giellatekno format: spaces before and after punctuation."""
    # Remove existing spaces around punctuation first
    s = re.sub(r'\s*([,.;:!?()\[\]«»"""\'–—])\s*', r'\1', sent)
    # Add single space before and after punctuation
    s = re.sub(r'([,.;:!?()\[\]«»"""\'–—])', r' \1 ', s)
    # Normalize multiple spaces to single space
    s = re.sub(r' +', ' ', s)
    # Strip leading/trailing whitespace
    return s.strip()

def find_and_split_content_start(sentences: list[str], start_marker: str = "I am a Sámi who has done all sorts") -> tuple[int, list[str]]:
    """
    Find where actual content starts and split the sentence if the marker is embedded.
    Returns (index, modified_sentences) where the sentence containing the marker is split.
    Uses a longer marker to avoid matching table of contents entries.
    """
    for i, sent in enumerate(sentences):
        if start_marker in sent:
            # Check if this looks like the actual content start (not ToC)
            # ToC lines have "#" symbols
            if "#" in sent:
                continue  # Skip ToC entries

            # Find where to split - look for the actual sentence start
            # The actual first sentence is "I am a Sámi ."
            marker_pos = sent.find("I am a Sámi")
            if marker_pos == -1:
                continue

            # Check if it's at the start or embedded
            if marker_pos == 0 or marker_pos < 10:
                # Marker is at/near start, just return this index
                return i, sentences
            else:
                # Marker is embedded, need to split the sentence
                before = sent[:marker_pos].strip()
                after = sent[marker_pos:].strip()

                # Create new sentence list with the split
                new_sentences = sentences[:i]
                if before:  # Only add before part if it's not empty
                    new_sentences.append(before)

                # Re-tokenize the 'after' part to get proper sentences
                after_sentences = nltk.sent_tokenize(after, language="english")
                new_sentences.extend([s.strip() for s in after_sentences if s.strip()])
                new_sentences.extend(sentences[i+1:])

                # Return the index where "I am a Sámi ." now starts
                return len(sentences[:i]) + (1 if before else 0), new_sentences

    return 0, sentences  # Fallback to beginning if not found

def find_content_end(sentences: list[str], end_marker: str = "May they come in the end into God") -> int:
    """Find the index of the last sentence of actual content."""
    for i, sent in enumerate(sentences):
        if end_marker in sent or ('embrace' in sent and 'mother' in sent and 'God' in sent and 'protection' in sent):
            return i
    return len(sentences) - 1  # Fallback to end if not found

def align_with_reference(sentences: list[str], expected_count: int = 3214) -> list[str]:
    """
    Extract content between start and end markers.
    The reference corpus may have different sentence counts than the translation.
    """
    start_idx, modified_sentences = find_and_split_content_start(sentences)
    end_idx = find_content_end(modified_sentences)

    if start_idx == 0:
        print("Warning: Could not find content start marker 'I am a Sámi .'. Using full text.")
    else:
        print(f"Found content start at index {start_idx} (skipping {start_idx} front matter lines)")

    if end_idx == len(modified_sentences) - 1:
        print("Warning: Could not find content end marker. Using end of document.")
    else:
        print(f"Found content end at index {end_idx}")

    # Extract from start to end (inclusive)
    aligned = modified_sentences[start_idx:end_idx + 1]

    print(f"Extracted {len(aligned)} sentences (Sami reference has {expected_count})")

    if len(aligned) != expected_count:
        print(f"Note: English has {len(aligned)} sentences vs {expected_count} in Sami (difference: {len(aligned) - expected_count})")

    return aligned

def detect_and_split_embedded_headers(text: str) -> list[str]:
    """
    Detect and split sentences that have embedded title-case headers.

    Patterns to detect:
    1. "Title Case Header And then body text..." -> ["Title Case Header .", "And then body text..."]
    2. "Title Case Header Title case repeated..." -> ["Title Case Header .", "Title case repeated..."]
    """
    # Pattern 1: Title Case sequence ending with "And then" or "And when" (body text start)
    # Title case has Most Words Capitalized
    import re

    # Look for sequences of title-cased words followed by conjunctions starting body text
    # Match: sequence of capitalized words, then " And " or " And then " or " And when "
    pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){3,})\s+(And\s+(?:then|when|if|while|so|they|the|it|this|there)\s+)'

    match = re.match(pattern, text)
    if match:
        header = match.group(1).strip()
        rest = text[len(header):].strip()
        return [header + " .", rest]

    # Pattern 2: Detect duplicate text (header repeated as body)
    # "Title Case Text Title case text continues..."
    # Look for a title-case phrase that's repeated in sentence case
    words = text.split()
    if len(words) >= 8:  # Need enough words to have a meaningful header
        # Try to find where title case ends
        title_case_count = 0
        for i, word in enumerate(words):
            # Skip common words that can appear in headers
            if word in ['The', 'A', 'An', 'And', 'Of', 'To', 'In', 'On', 'For', 'With', 'At']:
                title_case_count += 1
                continue
            # Check if word is title cased
            if word and word[0].isupper() and len(word) > 1 and word[1:].islower():
                title_case_count += 1
            else:
                # End of title case sequence
                if title_case_count >= 4:  # At least 4 title-cased words
                    # Check if the next few words start with same text but in lower case
                    header_text = ' '.join(words[:i])
                    remaining_text = ' '.join(words[i:])

                    # Check if remaining text starts with similar words (case-insensitive)
                    header_words_lower = [w.lower() for w in words[:min(i, 5)]]
                    remaining_words_lower = [w.lower() for w in words[i:i+5]]

                    # If first few words match (duplicate header), split
                    if len(remaining_words_lower) >= 3:
                        matches = sum(1 for hw, rw in zip(header_words_lower[:3], remaining_words_lower[:3]) if hw == rw)
                        if matches >= 2:  # At least 2 words match
                            return [header_text + " .", remaining_text]
                break

    return [text]


def post_process_embedded_headers(sentences: list[str]) -> list[str]:
    """
    Post-process sentences to detect and split embedded title-case headers.
    """
    result = []
    for sent in sentences:
        split_sents = detect_and_split_embedded_headers(sent)
        result.extend(split_sents)
    return result


def post_process_first_sentence(sentences: list[str]) -> list[str]:
    """
    Post-process to handle the special case where 'I am a Sámi' appears twice.
    In the Sami version, this is two sentences, but in English it may be combined.
    """
    if not sentences or not sentences[0].startswith("I am a Sámi I am a Sámi"):
        return sentences

    # Split the duplicate "I am a Sámi I am a Sámi who..." into two sentences
    first = sentences[0]
    if first.startswith("I am a Sámi I am a Sámi who"):
        # Insert "I am a Sámi ." as the first sentence
        # Keep the rest as the second sentence
        rest = first[len("I am a Sámi "):]  # Remove the duplicate prefix
        return ["I am a Sámi ."] + [rest] + sentences[1:]

    return sentences

if __name__ == "__main__":
    PATH = "benchmark_data/source/muitalus-eng.pdf"

    # debug_font_sizes(PATH)  # ← uncomment first to tune ratios

    body_size = get_body_font_size(PATH)
    print(f"Detected body font size: {body_size}")

    structured = extract_structured_lines(PATH, body_size)
    sentences = build_sentences(structured)
    print(f"Total sentences extracted: {len(sentences)}")

    # Post-process to split embedded title-case headers
    sentences = post_process_embedded_headers(sentences)
    print(f"After splitting embedded headers: {len(sentences)}")

    # Align with Sami reference corpus (3214 lines)
    aligned_sentences = align_with_reference(sentences, expected_count=3214)

    # Post-process to fix the first sentence duplication
    aligned_sentences = post_process_first_sentence(aligned_sentences)

    # Format and write output
    with open("benchmark_data/source/muitalus-eng.txt", "w", encoding="utf-8") as f:
        for sent in aligned_sentences:
            f.write(format_sentence(sent) + "\n")

    print(f"\nWritten {len(aligned_sentences)} sentences to muitalus-eng.txt")

    # Verify first and last lines against ground truth
    if aligned_sentences:
        print("\nVerification:")
        print(f"First line: {format_sentence(aligned_sentences[0])}")
        print(f"Last line:  {format_sentence(aligned_sentences[-1])}")