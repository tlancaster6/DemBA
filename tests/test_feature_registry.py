#!/usr/bin/env python
"""
Test script to verify the feature registry pattern implementation.
"""
import sys
import os
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')

def test_registry_structure():
    """Test that the feature registry is properly initialized."""
    print("Test 1: Feature registry structure")
    print("-" * 50)

    # Import after path setup
    from demba.feature_extraction import FeatureExtractor

    # Create a mock extractor (this will fail to load data but registry should initialize)
    try:
        fe = FeatureExtractor(
            video_path="dummy_video.mp4",
            pose_h5_path="dummy_pose.h5"
        )
    except Exception as e:
        print(f"Note: Constructor failed as expected (no real data): {type(e).__name__}")
        print("This is OK for registry testing\n")
        # Create minimal mock to test registry
        import pandas as pd
        from demba.feature_extraction import FeatureExtractor

        # Monkey-patch to bypass data loading
        class MockExtractor(FeatureExtractor):
            def __init__(self):
                self.pose_df = None
                self.quivering_annotation_df = None
                self._init_frame_feature_registry()

        fe = MockExtractor()

    # Check registry exists
    assert hasattr(fe, 'frame_feature_registry'), "Registry not initialized"
    print(f"✓ Registry initialized with {len(fe.frame_feature_registry)} features")

    # Check expected features
    expected_features = [
        'nfish_frame', 'nfish_pipe', 'min_dist_nose_to_stripe4',
        'mouthing_event_id', 'spawning_event_id', 'double_occupancy_event_id',
        'male_lead_quiver', 'male_circle_quiver', 'female_circle_quiver'
    ]

    for feat in expected_features:
        assert feat in fe.frame_feature_registry, f"Missing feature: {feat}"
        print(f"  - {feat}: ✓")

    print("\n✅ Test 1 PASSED\n")
    return fe


def test_registry_metadata(fe):
    """Test that each feature has required metadata."""
    print("Test 2: Feature metadata completeness")
    print("-" * 50)

    required_keys = ['method', 'depends_on', 'requires_pose', 'requires_annotations']

    for feat_name, spec in fe.frame_feature_registry.items():
        for key in required_keys:
            assert key in spec, f"Feature '{feat_name}' missing key: {key}"

        # Check types
        assert callable(spec['method']), f"Feature '{feat_name}' method not callable"
        assert isinstance(spec['depends_on'], list), f"Feature '{feat_name}' depends_on not a list"
        assert isinstance(spec['requires_pose'], bool), f"Feature '{feat_name}' requires_pose not bool"
        assert isinstance(spec['requires_annotations'], bool), f"Feature '{feat_name}' requires_annotations not bool"

        print(f"  - {feat_name}: ✓")

    print("\n✅ Test 2 PASSED\n")


def test_dependency_graph(fe):
    """Test that the dependency graph is valid (no cycles, valid references)."""
    print("Test 3: Dependency graph validity")
    print("-" * 50)

    all_features = set(fe.frame_feature_registry.keys())

    # Check all dependencies are valid
    for feat_name, spec in fe.frame_feature_registry.items():
        for dep in spec['depends_on']:
            assert dep in all_features, f"Feature '{feat_name}' depends on unknown feature: {dep}"
            print(f"  - {feat_name} → {dep}: ✓")

    # Test topological sort (will raise on cycles)
    try:
        execution_order = fe._topological_sort(list(all_features))
        print(f"\n  Topological sort successful: {len(execution_order)} features")
        print(f"  Execution order: {' → '.join(execution_order)}")
    except ValueError as e:
        print(f"  ❌ Topological sort failed: {e}")
        raise

    print("\n✅ Test 3 PASSED\n")


def test_selective_extraction(fe):
    """Test that selective feature extraction works."""
    print("Test 4: Selective feature extraction")
    print("-" * 50)

    # Test extracting just the dependency chain for mouthing
    features_to_extract = ['mouthing_event_id']

    try:
        execution_order = fe._topological_sort(features_to_extract)
        print(f"  Requesting: {features_to_extract}")
        print(f"  Execution order: {execution_order}")

        # Should include dependencies
        assert 'min_dist_nose_to_stripe4' in execution_order, "Missing dependency"
        assert execution_order.index('min_dist_nose_to_stripe4') < execution_order.index('mouthing_event_id'), \
            "Dependency not ordered before dependent"

        print("  ✓ Dependencies resolved correctly")
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        raise

    # Test extracting spawning (should pull in mouthing and distances)
    features_to_extract = ['spawning_event_id']
    execution_order = fe._topological_sort(features_to_extract)
    print(f"\n  Requesting: {features_to_extract}")
    print(f"  Execution order: {execution_order}")

    assert len(execution_order) == 3, "Should resolve full chain"
    assert execution_order == ['min_dist_nose_to_stripe4', 'mouthing_event_id', 'spawning_event_id'], \
        "Incorrect dependency chain"
    print("  ✓ Full dependency chain resolved")

    print("\n✅ Test 4 PASSED\n")


def test_cycle_detection(fe):
    """Test that circular dependencies are detected."""
    print("Test 5: Circular dependency detection")
    print("-" * 50)

    # Temporarily create a circular dependency
    original_deps = fe.frame_feature_registry['mouthing_event_id']['depends_on'].copy()

    try:
        # Create cycle: mouthing → spawning → mouthing
        fe.frame_feature_registry['mouthing_event_id']['depends_on'] = ['spawning_event_id']

        try:
            fe._topological_sort(['mouthing_event_id', 'spawning_event_id'])
            print("  ❌ Should have detected cycle!")
            assert False, "Cycle detection failed"
        except ValueError as e:
            print(f"  ✓ Cycle correctly detected: {e}")
    finally:
        # Restore original
        fe.frame_feature_registry['mouthing_event_id']['depends_on'] = original_deps

    print("\n✅ Test 5 PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("Feature Registry Pattern Test Suite")
    print("=" * 50 + "\n")

    try:
        fe = test_registry_structure()
        test_registry_metadata(fe)
        test_dependency_graph(fe)
        test_selective_extraction(fe)
        test_cycle_detection(fe)

        print("=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50 + "\n")
        return 0

    except Exception as e:
        print("\n" + "=" * 50)
        print(f"❌ TEST SUITE FAILED: {e}")
        print("=" * 50 + "\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
