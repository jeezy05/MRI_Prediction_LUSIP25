"""
Simple test suite for alert system - no pytest required.
Run with: python simple_test_alerts.py
"""

from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alert_system.delivery_alert_agent import SecurityAnalystAgent


def test_agent_loads_data():
    """Test that agent loads telemetry data."""
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    agent = SecurityAnalystAgent(str(telemetry_csv), use_ai_rules=False)
    
    success = agent.load_telemetry_data()
    assert success, "❌ Failed to load telemetry data"
    assert len(agent.frames_data) == 31, f"❌ Expected 31 frames, got {len(agent.frames_data)}"
    
    print("✅ Agent loads telemetry data correctly")


def test_extract_objects():
    """Test object extraction."""
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    agent = SecurityAnalystAgent(str(telemetry_csv), use_ai_rules=False)
    
    agent.load_telemetry_data()
    agent.extract_objects_and_people()
    
    assert len(agent.detected_objects) > 0, "❌ No objects detected"
    assert len(agent.detected_people) > 0, "❌ No people detected"
    
    print(f"✅ Objects extracted: {len(agent.detected_objects)} objects, {len(agent.detected_people)} people")


def test_parse_time():
    """Test time parsing."""
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    agent = SecurityAnalystAgent(str(telemetry_csv), use_ai_rules=False)
    
    time_obj = agent.parse_time("10:00:00")
    assert time_obj is not None, "❌ Failed to parse time"
    assert time_obj.hour == 10, "❌ Hour mismatch"
    assert time_obj.minute == 0, "❌ Minute mismatch"
    
    print("✅ Time parsing works correctly")


def test_night_time_detection():
    """Test night time detection."""
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    agent = SecurityAnalystAgent(str(telemetry_csv), use_ai_rules=False)
    
    assert agent.is_night_time("23:30:00"), "❌ 23:30 should be night"
    assert agent.is_night_time("03:00:00"), "❌ 03:00 should be night"
    assert not agent.is_night_time("10:00:00"), "❌ 10:00 should not be night"
    assert not agent.is_night_time("06:00:00"), "❌ 06:00 should not be night"
    
    print("✅ Night time detection works correctly")


def test_process_and_alert():
    """Test alert generation."""
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    agent = SecurityAnalystAgent(str(telemetry_csv), use_ai_rules=False)
    
    result = agent.process_and_alert()
    
    assert result.get("success"), "❌ Processing failed"
    assert "alerts" in result, "❌ No alerts in result"
    assert len(result["alerts"]) > 0, "❌ No alerts generated"
    
    print(f"✅ Alerts generated: {len(result['alerts'])} alerts")


def test_alert_fields():
    """Test alert has required fields."""
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    agent = SecurityAnalystAgent(str(telemetry_csv), use_ai_rules=False)
    
    result = agent.process_and_alert()
    alerts = result["alerts"]
    
    required_fields = {"alert_type", "severity", "location", "time", "message", "evidence"}
    
    for alert in alerts[:3]:
        for field in required_fields:
            assert field in alert, f"❌ Alert missing field: {field}"
    
    print("✅ All alerts have required fields")


def test_severity_levels():
    """Test alert severity levels are valid."""
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    agent = SecurityAnalystAgent(str(telemetry_csv), use_ai_rules=False)
    
    result = agent.process_and_alert()
    alerts = result["alerts"]
    
    valid_severities = {"INFO", "WARNING", "HIGH"}
    
    for alert in alerts:
        assert alert["severity"] in valid_severities, f"❌ Invalid severity: {alert['severity']}"
    
    print("✅ All alerts have valid severity levels")


def test_truck_detected():
    """Test that trucks are detected."""
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    agent = SecurityAnalystAgent(str(telemetry_csv), use_ai_rules=False)
    
    result = agent.process_and_alert()
    
    vehicles = [obj for obj in result["detected_objects"] if obj.get("object_type") == "vehicle"]
    assert len(vehicles) > 0, "❌ No vehicles detected"
    
    print(f"✅ Trucks detected: {len(vehicles)} vehicle frames")


def test_people_detected():
    """Test that people are detected."""
    telemetry_csv = Path(__file__).parent.parent / "drone_telemetry.csv"
    agent = SecurityAnalystAgent(str(telemetry_csv), use_ai_rules=False)
    
    result = agent.process_and_alert()
    
    assert len(result["detected_people"]) > 0, "❌ No people detected"
    
    print(f"✅ People detected: {len(result['detected_people'])} person frames")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ALERT SYSTEM - SIMPLE TEST SUITE")
    print("="*70 + "\n")
    
    tests = [
        test_agent_loads_data,
        test_extract_objects,
        test_parse_time,
        test_night_time_detection,
        test_process_and_alert,
        test_alert_fields,
        test_severity_levels,
        test_truck_detected,
        test_people_detected
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
