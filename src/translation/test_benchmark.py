#!/usr/bin/env python3
"""Quick validation test for benchmark logic without API calls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from translation.conjunctions import split_by_conjunctions
from translation.benchmark import load_ground_truth, compute_split_statistics


def test_data_loading():
    """Test that ground truth loads correctly."""
    print("Testing data loading...")
    data = load_ground_truth("test_data/account_of_sami/ground_truth.json")
    print(f"  ✓ Loaded {len(data)} samples")

    # Check first few samples
    for i, sample in enumerate(data[:3]):
        print(f"  Sample {sample['id']}: {len(sample['source'])} chars -> {len(sample['reference'])} chars")

    # Verify no empty entries
    empty = [s for s in data if not s['source'].strip() or not s['reference'].strip()]
    assert len(empty) == 0, f"Found {len(empty)} empty samples"
    print("  ✓ No empty samples")
    print()


def test_splitting_logic():
    """Test conjunction splitting on real data."""
    print("Testing splitting logic...")
    data = load_ground_truth("test_data/account_of_sami/ground_truth.json")

    # Test on a few samples
    test_samples = data[:5]

    for mode in ['simple', 'full']:
        print(f"\n  {mode.upper()} mode:")
        for sample in test_samples:
            segments = split_by_conjunctions(sample['source'], mode)
            if len(segments) > 1:
                print(f"    Sample {sample['id']}: {len(segments)} segments")
                for i, (seg, conj) in enumerate(segments):
                    conj_str = f"[{conj}]" if conj else ""
                    print(f"      {i+1}. {seg[:50]}... {conj_str}")
    print()


def test_statistics_computation():
    """Test split statistics computation."""
    print("Testing statistics computation...")

    # Create mock results
    mock_results = [
        {'id': '1', 'num_segments': 1, 'reference': 'ref1', 'prediction': 'pred1'},
        {'id': '2', 'num_segments': 2, 'reference': 'ref2', 'prediction': 'pred2'},
        {'id': '3', 'num_segments': 3, 'reference': 'ref3', 'prediction': 'pred3'},
        {'id': '4', 'num_segments': 1, 'reference': 'ref4', 'prediction': 'pred4'},
    ]

    stats = compute_split_statistics(mock_results)
    print(f"  Total segments: {stats['total_segments']}")
    print(f"  Avg segments/sample: {stats['avg_segments_per_sample']:.2f}")
    print(f"  Samples split: {stats['samples_split']}")
    print(f"  Samples unchanged: {stats['samples_unchanged']}")

    assert stats['total_segments'] == 7
    assert stats['samples_split'] == 2
    assert stats['samples_unchanged'] == 2
    print("  ✓ Statistics correct")
    print()


def test_full_pipeline_structure():
    """Test the full pipeline structure without API calls."""
    print("Testing full pipeline structure...")
    data = load_ground_truth("test_data/account_of_sami/ground_truth.json")

    # Count how many samples would be split
    simple_splits = 0
    full_splits = 0

    for sample in data:
        simple_segs = split_by_conjunctions(sample['source'], 'simple')
        full_segs = split_by_conjunctions(sample['source'], 'full')

        if len(simple_segs) > 1:
            simple_splits += 1
        if len(full_segs) > 1:
            full_splits += 1

    print(f"  Total samples: {len(data)}")
    print(f"  Simple mode would split: {simple_splits}/{len(data)} ({simple_splits/len(data)*100:.1f}%)")
    print(f"  Full mode would split: {full_splits}/{len(data)} ({full_splits/len(data)*100:.1f}%)")
    print("  ✓ Pipeline structure valid")
    print()


if __name__ == "__main__":
    print("=== Benchmark Validation Tests ===\n")

    try:
        test_data_loading()
        test_splitting_logic()
        test_statistics_computation()
        test_full_pipeline_structure()

        print("=== All Tests Passed ✓ ===")
        print("\nReady to run full benchmark with:")
        print("  python src/translation/benchmark.py")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
