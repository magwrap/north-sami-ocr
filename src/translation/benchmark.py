#!/usr/bin/env python3
"""
Translation Benchmark with Conjunction Splitting

Evaluates Northern Sami → English translation quality across three modes:
1. Baseline: Full sentences translated as-is
2. Simple Split: Sentences split on 7 common conjunctions
3. Full Split: Sentences split on 25+ conjunctions (including multi-word)

This allows empirical measurement of whether sentence splitting improves
or degrades translation quality.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from translation.conjunctions import split_by_conjunctions, get_conjunction_stats
from translation.sme_eng import translate_to_english

# Import metrics from benchmark_data
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from benchmark_data.evaluation.metrics import (
    compute_bleu,
    compute_chrf,
    compute_ter,
    bootstrap_confidence_interval
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Northern Sami → English translation with conjunction splitting"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="test_data/account_of_sami/ground_truth.json",
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/translation/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cache/--no-cache",
        dest="cache",
        default=True,
        help="Enable/disable API response caching"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-sample details"
    )
    return parser.parse_args()


def load_ground_truth(json_path: str) -> List[Dict]:
    """
    Load and filter ground truth data.

    Args:
        json_path: Path to ground_truth.json

    Returns:
        List of {id, source, reference} dictionaries
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for entry_id, entry in data.items():
        # Skip if entry is not a dict
        if not isinstance(entry, dict):
            continue

        # Skip empty entries (like entry 0)
        if not entry.get('original', '').strip() or not entry.get('translation', '').strip():
            continue

        samples.append({
            'id': entry_id,
            'source': entry['original'].strip(),
            'reference': entry['translation'].strip()
        })

    # Sort by ID for deterministic order
    samples.sort(key=lambda x: int(x['id']))

    return samples


def load_cache(cache_path: Path) -> Dict[str, str]:
    """
    Load cached translations from JSONL file.

    Args:
        cache_path: Path to cache file

    Returns:
        Dictionary mapping text -> translation
    """
    if not cache_path.exists():
        return {}

    cache = {}
    with open(cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                cache[entry['text']] = entry['translation']

    return cache


def save_cache_entry(cache_path: Path, text: str, translation: str):
    """
    Append a translation to the cache file.

    Args:
        cache_path: Path to cache file
        text: Source text
        translation: Translation result
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, 'a', encoding='utf-8') as f:
        entry = {
            'text': text,
            'translation': translation,
            'timestamp': datetime.now().isoformat()
        }
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def translate_with_cache(
    text: str,
    cache: Dict[str, str],
    cache_path: Path,
    use_cache: bool
) -> Tuple[str, bool]:
    """
    Translate text with caching and retry logic.

    Args:
        text: Text to translate
        cache: In-memory cache dictionary
        cache_path: Path to cache file
        use_cache: Whether to use caching

    Returns:
        Tuple of (translation, was_cached)
    """
    # Check cache first
    if use_cache and text in cache:
        return cache[text], True

    # Translate with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        translation = translate_to_english(text)

        # Check if translation was successful (not an error message)
        if not translation.startswith('[Translation'):
            # Success - update cache
            if use_cache:
                cache[text] = translation
                save_cache_entry(cache_path, text, translation)
            return translation, False

        # Error - retry with exponential backoff
        if attempt < max_retries - 1:
            wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
            time.sleep(wait_time)
        else:
            # Final attempt failed
            print(f"  WARNING: Translation failed after {max_retries} attempts: {text[:50]}...", file=sys.stderr)
            return translation, False

    return translation, False


def translate_segments(
    segments: List[Tuple[str, Optional[str]]],
    cache: Dict[str, str],
    cache_path: Path,
    use_cache: bool
) -> Tuple[str, List[Dict], int]:
    """
    Translate multiple segments and concatenate.

    Args:
        segments: List of (segment_text, conjunction) tuples
        cache: In-memory cache
        cache_path: Cache file path
        use_cache: Whether to use caching

    Returns:
        Tuple of (concatenated_translation, segment_details, api_calls)
    """
    translations = []
    segment_details = []
    api_calls = 0

    for seg_text, conj in segments:
        if not seg_text.strip():
            continue

        translation, was_cached = translate_with_cache(
            seg_text, cache, cache_path, use_cache
        )

        if not was_cached:
            api_calls += 1
            # Small delay to avoid rate limiting
            time.sleep(0.1)

        translations.append(translation)
        segment_details.append({
            'text': seg_text,
            'translation': translation,
            'conjunction': conj
        })

    # Concatenate with single space
    concatenated = ' '.join(translations)

    return concatenated, segment_details, api_calls


def process_mode(
    data: List[Dict],
    mode: str,
    cache: Dict[str, str],
    cache_path: Path,
    use_cache: bool,
    verbose: bool = False
) -> Tuple[List[Dict], int, int]:
    """
    Process all samples for one translation mode.

    Args:
        data: List of samples with id, source, reference
        mode: 'baseline', 'simple_split', or 'full_split'
        cache: In-memory cache
        cache_path: Cache file path
        use_cache: Whether to use caching
        verbose: Show per-sample details

    Returns:
        Tuple of (per_sample_results, api_calls, cache_hits)
    """
    per_sample_results = []
    total_api_calls = 0
    total_cache_hits = 0

    for i, sample in enumerate(data):
        source = sample['source']
        reference = sample['reference']

        if mode == 'baseline':
            # Translate full sentence
            translation, was_cached = translate_with_cache(
                source, cache, cache_path, use_cache
            )

            if was_cached:
                total_cache_hits += 1
            else:
                total_api_calls += 1
                time.sleep(0.1)  # Rate limiting

            segments_info = [{
                'text': source,
                'translation': translation,
                'conjunction': None
            }]
            num_segments = 1

        else:
            # Split and translate segments
            split_mode = 'simple' if mode == 'simple_split' else 'full'
            segments = split_by_conjunctions(source, split_mode)

            translation, segments_info, api_calls = translate_segments(
                segments, cache, cache_path, use_cache
            )

            total_api_calls += api_calls
            total_cache_hits += len(segments) - api_calls
            num_segments = len(segments)

        per_sample_results.append({
            'id': sample['id'],
            'source': source,
            'reference': reference,
            'prediction': translation,
            'segments': segments_info,
            'num_segments': num_segments
        })

        if verbose and (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(data)}")

    return per_sample_results, total_api_calls, total_cache_hits


def compute_metrics(per_sample_results: List[Dict]) -> Dict:
    """
    Compute aggregate translation metrics.

    Args:
        per_sample_results: List of result dictionaries

    Returns:
        Dictionary with BLEU, chrF, TER scores
    """
    references = [r['reference'] for r in per_sample_results]
    predictions = [r['prediction'] for r in per_sample_results]

    # Compute metrics
    bleu_result = compute_bleu(references, predictions)
    chrf_result = compute_chrf(references, predictions)
    ter_result = compute_ter(references, predictions)

    return {
        'bleu': float(bleu_result['bleu']),
        'chrf': float(chrf_result['chrf']),
        'ter': float(ter_result['ter'] / 100.0)  # TER is 0-100, normalize to 0-1
    }


def compute_split_statistics(per_sample_results: List[Dict]) -> Dict:
    """
    Compute statistics about sentence splitting.

    Args:
        per_sample_results: List of result dictionaries

    Returns:
        Dictionary with splitting statistics
    """
    total_segments = sum(r['num_segments'] for r in per_sample_results)
    samples_split = sum(1 for r in per_sample_results if r['num_segments'] > 1)
    samples_unchanged = sum(1 for r in per_sample_results if r['num_segments'] == 1)

    return {
        'avg_segments_per_sample': float(total_segments / len(per_sample_results)),
        'samples_split': int(samples_split),
        'samples_unchanged': int(samples_unchanged),
        'total_segments': int(total_segments)
    }


def compare_modes(baseline_results: Dict, split_results: Dict) -> Dict:
    """
    Generate comparative analysis between modes.

    Args:
        baseline_results: Results from baseline mode
        split_results: Results from split mode

    Returns:
        Comparison dictionary with deltas and confidence intervals
    """
    # Compute deltas
    deltas = {}
    for metric in ['bleu', 'chrf', 'ter']:
        baseline_val = baseline_results['aggregate_metrics'][metric]
        split_val = split_results['aggregate_metrics'][metric]
        deltas[metric] = split_val - baseline_val

    # Compute bootstrap confidence intervals for BLEU
    baseline_samples = baseline_results['per_sample_results']
    split_samples = split_results['per_sample_results']

    baseline_refs = [s['reference'] for s in baseline_samples]
    baseline_preds = [s['prediction'] for s in baseline_samples]
    split_refs = [s['reference'] for s in split_samples]
    split_preds = [s['prediction'] for s in split_samples]

    # Bootstrap CI for baseline BLEU
    baseline_ci = bootstrap_confidence_interval(
        baseline_refs, baseline_preds, compute_bleu, n_samples=1000
    )

    # Bootstrap CI for split BLEU
    split_ci = bootstrap_confidence_interval(
        split_refs, split_preds, compute_bleu, n_samples=1000
    )

    # Determine statistical significance (non-overlapping CIs)
    significant = (baseline_ci['ci_upper'] < split_ci['ci_lower'] or
                   split_ci['ci_upper'] < baseline_ci['ci_lower'])

    return {
        'deltas': {k: float(v) for k, v in deltas.items()},
        'bleu_ci_lower': float(split_ci['ci_lower'] - baseline_ci['mean']),
        'bleu_ci_upper': float(split_ci['ci_upper'] - baseline_ci['mean']),
        'bleu_confidence': 0.95,
        'significant': bool(significant)
    }


def get_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    args = parse_args()

    print("=== Northern Sami → English Translation Benchmark ===")
    print(f"Dataset: {args.data_path}")
    print(f"Cache: {'enabled' if args.cache else 'disabled'}")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{timestamp}_translation_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Load data
    print("[1/4] Loading data...")
    data = load_ground_truth(args.data_path)
    print(f"  Loaded {len(data)} samples")
    print()

    # Get conjunction stats
    simple_stats = get_conjunction_stats('simple')
    full_stats = get_conjunction_stats('full')

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'data_path': args.data_path,
        'data_hash': get_file_hash(args.data_path),
        'total_samples': len(data),
        'modes': ['baseline', 'simple_split', 'full_split'],
        'cache_enabled': args.cache,
        'conjunction_placement': 'right',
        'conjunctions': {
            'simple': simple_stats['conjunctions'],
            'full': full_stats['conjunctions']
        }
    }

    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Process each mode
    modes_config = [
        ('baseline', 'baseline_cache.jsonl'),
        ('simple_split', 'simple_split_cache.jsonl'),
        ('full_split', 'full_split_cache.jsonl')
    ]

    all_results = {}

    for mode, cache_filename in modes_config:
        mode_display = mode.replace('_', ' ').title()
        print(f"[{'2' if mode == 'baseline' else '3'}/4] Translating ({mode_display})...")

        # Load cache
        cache_path = cache_dir / cache_filename
        cache = load_cache(cache_path) if args.cache else {}

        # Process mode
        per_sample_results, api_calls, cache_hits = process_mode(
            data, mode, cache, cache_path, args.cache, args.verbose
        )

        # Compute metrics
        aggregate_metrics = compute_metrics(per_sample_results)
        aggregate_metrics['total_samples'] = len(data)
        aggregate_metrics['api_calls'] = api_calls
        aggregate_metrics['cache_hits'] = cache_hits

        # Add split statistics if applicable
        if mode != 'baseline':
            split_stats = compute_split_statistics(per_sample_results)
            aggregate_metrics.update(split_stats)

        all_results[mode] = {
            'mode': mode,
            'aggregate_metrics': aggregate_metrics,
            'per_sample_results': per_sample_results
        }

        # Save mode results
        with open(output_dir / f'results_{mode}.json', 'w', encoding='utf-8') as f:
            json.dump(all_results[mode], f, indent=2, ensure_ascii=False)

        # Print progress
        print(f"  Progress: {len(data)}/{len(data)} (100%)")
        if args.cache:
            print(f"  Cache hits: {cache_hits}/{cache_hits + api_calls} ({cache_hits/(cache_hits + api_calls)*100:.1f}%)")
        print(f"  API calls: {api_calls}")
        print()

    # Generate comparison
    print("[4/4] Computing metrics and comparison...")
    print()

    # Compare modes
    baseline_vs_simple = compare_modes(all_results['baseline'], all_results['simple_split'])
    baseline_vs_full = compare_modes(all_results['baseline'], all_results['full_split'])

    comparison = {
        'modes': ['baseline', 'simple_split', 'full_split'],
        'metrics_comparison': {},
        'split_statistics': {},
        'statistical_significance': {
            'baseline_vs_simple': baseline_vs_simple,
            'baseline_vs_full': baseline_vs_full
        }
    }

    # Build metrics comparison
    for metric in ['bleu', 'chrf', 'ter']:
        comparison['metrics_comparison'][metric] = {
            'baseline': all_results['baseline']['aggregate_metrics'][metric],
            'simple_split': all_results['simple_split']['aggregate_metrics'][metric],
            'full_split': all_results['full_split']['aggregate_metrics'][metric],
            'delta_simple': baseline_vs_simple['deltas'][metric],
            'delta_full': baseline_vs_full['deltas'][metric]
        }

    # Add split statistics
    for mode in ['simple_split', 'full_split']:
        stats = compute_split_statistics(all_results[mode]['per_sample_results'])
        comparison['split_statistics'][mode] = stats

    # Save comparison
    with open(output_dir / 'comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # Save CSV summary
    with open(output_dir / 'comparison.csv', 'w', encoding='utf-8') as f:
        f.write('mode,bleu,chrf,ter,avg_segments,samples_split,total_samples\n')
        for mode in ['baseline', 'simple_split', 'full_split']:
            m = all_results[mode]['aggregate_metrics']
            avg_segs = m.get('avg_segments_per_sample', 1.0)
            samples_split = m.get('samples_split', 0)
            f.write(f"{mode},{m['bleu']:.2f},{m['chrf']:.2f},{m['ter']:.3f},{avg_segs:.1f},{samples_split},{m['total_samples']}\n")

    # Print results
    print("=== RESULTS ===")
    print()
    print("Baseline (no splitting):")
    m = all_results['baseline']['aggregate_metrics']
    print(f"  BLEU:  {m['bleu']:.2f}")
    print(f"  chrF:  {m['chrf']:.2f}")
    print(f"  TER:   {m['ter']:.3f} ({m['ter']*100:.1f}% edit rate)")
    print()

    print("Simple Split (7 conjunctions):")
    m = all_results['simple_split']['aggregate_metrics']
    delta = baseline_vs_simple['deltas']
    print(f"  BLEU:  {m['bleu']:.2f}  (Δ {delta['bleu']:+.1f})")
    print(f"  chrF:  {m['chrf']:.2f}  (Δ {delta['chrf']:+.1f})")
    print(f"  TER:   {m['ter']:.3f}  (Δ {delta['ter']:+.3f})")
    print(f"  Avg segments: {m['avg_segments_per_sample']:.1f}  ({m['samples_split']}/{m['total_samples']} samples split)")
    print()

    print("Full Split (25+ conjunctions):")
    m = all_results['full_split']['aggregate_metrics']
    delta = baseline_vs_full['deltas']
    print(f"  BLEU:  {m['bleu']:.2f}  (Δ {delta['bleu']:+.1f})")
    print(f"  chrF:  {m['chrf']:.2f}  (Δ {delta['chrf']:+.1f})")
    print(f"  TER:   {m['ter']:.3f}  (Δ {delta['ter']:+.3f})")
    print(f"  Avg segments: {m['avg_segments_per_sample']:.1f}  ({m['samples_split']}/{m['total_samples']} samples split)")
    print()

    print("Statistical Significance (95% CI):")
    sig_simple = "✓ significant" if baseline_vs_simple['significant'] else "✗ not significant"
    print(f"  Baseline vs Simple: BLEU Δ = {baseline_vs_simple['deltas']['bleu']:+.1f} [{baseline_vs_simple['bleu_ci_lower']:+.1f}, {baseline_vs_simple['bleu_ci_upper']:+.1f}] {sig_simple}")

    sig_full = "✓ significant" if baseline_vs_full['significant'] else "✗ not significant"
    print(f"  Baseline vs Full:   BLEU Δ = {baseline_vs_full['deltas']['bleu']:+.1f} [{baseline_vs_full['bleu_ci_lower']:+.1f}, {baseline_vs_full['bleu_ci_upper']:+.1f}] {sig_full}")
    print()

    print(f"Results saved to: {output_dir}")
    print()
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
