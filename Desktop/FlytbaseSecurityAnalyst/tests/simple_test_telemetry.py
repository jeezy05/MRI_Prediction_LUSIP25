"""
Simple test suite for telemetry generation - no pytest required.
Run with: python simple_test_telemetry.py
"""

import csv
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_simulation.generate_telemetry import process_frames_and_generate_csv


def test_csv_created():
    """Test that CSV file is created."""
    csv_path = "test_telemetry.csv"
    frames_dir = Path(__file__).parent.parent / "frames"
    process_frames_and_generate_csv(str(frames_dir), csv_path)
    
    assert Path(csv_path).exists(), "❌ CSV file not created"
    print("✅ CSV file created successfully")
    
    Path(csv_path).unlink()


def test_correct_headers():
    """Test CSV has correct headers."""
    csv_path = "test_telemetry.csv"
    frames_dir = Path(__file__).parent.parent / "frames"
    process_frames_and_generate_csv(str(frames_dir), csv_path)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        expected = {"Frame ID", "Frame Description", "Location", "Time", "Altitude (meters)"}
        actual = set(reader.fieldnames)
        assert expected == actual, f"❌ Headers mismatch. Expected {expected}, got {actual}"
    
    print("✅ CSV headers are correct")
    Path(csv_path).unlink()


def test_frame_count():
    """Test all 31 frames are processed."""
    csv_path = "test_telemetry.csv"
    frames_dir = Path(__file__).parent.parent / "frames"
    process_frames_and_generate_csv(str(frames_dir), csv_path)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 31, f"❌ Expected 31 frames, got {len(rows)}"
    
    print("✅ All 31 frames processed")
    Path(csv_path).unlink()


def test_frame_ids_sequential():
    """Test frame IDs are sequential."""
    csv_path = "test_telemetry.csv"
    frames_dir = Path(__file__).parent.parent / "frames"
    process_frames_and_generate_csv(str(frames_dir), csv_path)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            expected = f"frame_{idx:04d}"
            assert row["Frame ID"] == expected, f"❌ Frame ID mismatch at {idx}"
    
    print("✅ Frame IDs are sequential")
    Path(csv_path).unlink()


def test_valid_locations():
    """Test locations are from predefined set."""
    csv_path = "test_telemetry.csv"
    frames_dir = Path(__file__).parent.parent / "frames"
    process_frames_and_generate_csv(str(frames_dir), csv_path)
    
    valid = {"Main Gate", "Garage", "North Perimeter", "South Perimeter", "Parking Lot"}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row["Location"] in valid, f"❌ Invalid location: {row['Location']}"
    
    print("✅ All locations are valid")
    Path(csv_path).unlink()


def test_valid_altitudes():
    """Test altitudes are from predefined set."""
    csv_path = "test_telemetry.csv"
    frames_dir = Path(__file__).parent.parent / "frames"
    process_frames_and_generate_csv(str(frames_dir), csv_path)
    
    valid = {50, 75, 100, 125, 150}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            altitude = int(row["Altitude (meters)"])
            assert altitude in valid, f"❌ Invalid altitude: {altitude}"
    
    print("✅ All altitudes are valid")
    Path(csv_path).unlink()


def test_sequential_times():
    """Test times increment by 2 seconds."""
    csv_path = "test_telemetry.csv"
    frames_dir = Path(__file__).parent.parent / "frames"
    process_frames_and_generate_csv(str(frames_dir), csv_path)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            expected_seconds = idx * 2
            hours = expected_seconds // 3600
            minutes = (expected_seconds % 3600) // 60
            seconds = expected_seconds % 60
            expected_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            assert row["Time"] == expected_time, f"❌ Time mismatch at frame {idx}"
    
    print("✅ Times are sequential (2-second intervals)")
    Path(csv_path).unlink()


def test_descriptions_not_empty():
    """Test all frames have descriptions."""
    csv_path = "test_telemetry.csv"
    frames_dir = Path(__file__).parent.parent / "frames"
    process_frames_and_generate_csv(str(frames_dir), csv_path)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row["Frame Description"].strip(), f"❌ Empty description for {row['Frame ID']}"
    
    print("✅ All frames have descriptions")
    Path(csv_path).unlink()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TELEMETRY GENERATION - SIMPLE TEST SUITE")
    print("="*70 + "\n")
    
    tests = [
        test_csv_created,
        test_correct_headers,
        test_frame_count,
        test_frame_ids_sequential,
        test_valid_locations,
        test_valid_altitudes,
        test_sequential_times,
        test_descriptions_not_empty
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
