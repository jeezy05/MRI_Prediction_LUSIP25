"""
Simple test suite for frame indexing - no pytest required.
Run with: python simple_test_indexing.py
"""

from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from indexing.frame_indexer import FrameIndexer


def test_database_created():
    """Test that database is created."""
    db_path = "test_frames.db"
    indexer = FrameIndexer(db_path)
    
    assert Path(db_path).exists(), "❌ Database file not created"
    
    indexer.close()
    Path(db_path).unlink()
    
    print("✅ Database created successfully")


def test_frames_indexed():
    """Test that frames are indexed from CSV."""
    db_path = "test_frames.db"
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    
    indexer = FrameIndexer(db_path)
    count = indexer.index_frames_from_csv(str(telemetry_csv))
    
    assert count == 31, f"❌ Expected 31 frames indexed, got {count}"
    
    indexer.close()
    Path(db_path).unlink()
    
    print("✅ All 31 frames indexed successfully")


def test_query_by_object():
    """Test querying frames by object type."""
    db_path = "test_frames.db"
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    
    indexer = FrameIndexer(db_path)
    indexer.index_frames_from_csv(str(telemetry_csv))
    
    results = indexer.query_frames_by_object("vehicle")
    assert len(results) > 0, "❌ No vehicle frames found"
    
    indexer.close()
    Path(db_path).unlink()
    
    print(f"✅ Query by object works: {len(results)} vehicle frames found")


def test_query_by_time():
    """Test querying frames by time range."""
    db_path = "test_frames.db"
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    
    indexer = FrameIndexer(db_path)
    indexer.index_frames_from_csv(str(telemetry_csv))
    
    results = indexer.query_frames_by_time("10:00:00", "10:00:10")
    assert len(results) > 0, "❌ No frames found in time range"
    
    indexer.close()
    Path(db_path).unlink()
    
    print(f"✅ Query by time works: {len(results)} frames in time range")


def test_query_by_location():
    """Test querying frames by location."""
    db_path = "test_frames.db"
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    
    indexer = FrameIndexer(db_path)
    indexer.index_frames_from_csv(str(telemetry_csv))
    
    results = indexer.query_frames_by_location("Main Gate")
    assert len(results) > 0, "❌ No frames found at Main Gate"
    assert all(r["location"] == "Main Gate" for r in results), "❌ Location mismatch"
    
    indexer.close()
    Path(db_path).unlink()
    
    print(f"✅ Query by location works: {len(results)} frames at Main Gate")


def test_query_people_frames():
    """Test querying frames with people."""
    db_path = "test_frames.db"
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    
    indexer = FrameIndexer(db_path)
    indexer.index_frames_from_csv(str(telemetry_csv))
    
    results = indexer.query_frames_with_people()
    assert len(results) > 0, "❌ No frames with people found"
    
    indexer.close()
    Path(db_path).unlink()
    
    print(f"✅ Query people frames works: {len(results)} frames with people")


def test_combined_query():
    """Test combined query (object + time)."""
    db_path = "test_frames.db"
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    
    indexer = FrameIndexer(db_path)
    indexer.index_frames_from_csv(str(telemetry_csv))
    
    results = indexer.query_frames_by_object_and_time("vehicle", "10:00:00", "10:00:15")
    assert len(results) > 0, "❌ No vehicle frames in time range"
    
    indexer.close()
    Path(db_path).unlink()
    
    print(f"✅ Combined query works: {len(results)} vehicle frames in time range")


def test_frame_details():
    """Test getting frame details."""
    db_path = "test_frames.db"
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    
    indexer = FrameIndexer(db_path)
    indexer.index_frames_from_csv(str(telemetry_csv))
    
    details = indexer.get_frame_details("frame_0000")
    assert details is not None, "❌ Frame details not found"
    assert "frame_id" in details, "❌ Missing frame_id"
    assert "objects" in details, "❌ Missing objects"
    
    indexer.close()
    Path(db_path).unlink()
    
    print("✅ Frame details retrieval works")


def test_statistics():
    """Test getting indexing statistics."""
    db_path = "test_frames.db"
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    
    indexer = FrameIndexer(db_path)
    indexer.index_frames_from_csv(str(telemetry_csv))
    
    stats = indexer.get_statistics()
    assert stats["total_frames"] == 31, f"❌ Expected 31 frames, got {stats['total_frames']}"
    assert stats["unique_object_types"] > 0, "❌ No object types found"
    
    indexer.close()
    Path(db_path).unlink()
    
    print(f"✅ Statistics work: {stats['total_frames']} frames, {stats['unique_object_types']} object types")


def test_truck_logged():
    """Test that trucks are logged correctly."""
    db_path = "test_frames.db"
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    
    indexer = FrameIndexer(db_path)
    indexer.index_frames_from_csv(str(telemetry_csv))
    
    results = indexer.query_frames_by_object("vehicle")
    truck_frames = [r for r in results if "truck" in r["description"].lower()]
    
    assert len(truck_frames) > 0, "❌ No trucks found"
    
    indexer.close()
    Path(db_path).unlink()
    
    print(f"✅ Truck logging works: {len(truck_frames)} truck frames found")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FRAME INDEXING - SIMPLE TEST SUITE")
    print("="*70 + "\n")
    
    tests = [
        test_database_created,
        test_frames_indexed,
        test_query_by_object,
        test_query_by_time,
        test_query_by_location,
        test_query_people_frames,
        test_combined_query,
        test_frame_details,
        test_statistics,
        test_truck_logged
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"{e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")
