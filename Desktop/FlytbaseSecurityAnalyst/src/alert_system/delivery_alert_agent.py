import csv
import json
from pathlib import Path
from datetime import datetime, time
from typing import List, Dict, Tuple
import requests

class SecurityAnalystAgent:
    """Agent to detect security/safety events and generate alerts from drone telemetry data."""
    
    def __init__(self, telemetry_csv_path: str, use_ai_rules: bool = True, ollama_url: str = "http://localhost:11434"):
        self.telemetry_csv_path = telemetry_csv_path
        self.frames_data = []
        self.alerts = []
        self.detected_objects = []
        self.detected_people = []
        self.use_ai_rules = use_ai_rules
        self.ollama_url = ollama_url
        self.ollama_model = "mistral"
        
        self.object_keywords = {
            "vehicle": ["truck", "car", "van", "fedex", "ups", "amazon", "delivery"],
            "person": ["man", "woman", "person", "people"],
            "package": ["box", "package", "parcel", "carrying", "holding"],
            "suspicious": ["loitering", "lurking", "suspicious", "climbing", "breaking"]
        }
        
        self.activity_keywords = {
            "delivery": ["carrying", "holding", "delivering", "package", "box"],
            "loitering": ["loitering", "standing", "waiting", "lingering"],
            "movement": ["walking", "running", "moving", "entering", "exiting"],
            "suspicious": ["climbing", "breaking", "lurking", "suspicious"]
        }
        
    def load_telemetry_data(self) -> bool:
        """Load telemetry data from CSV file."""
        try:
            with open(self.telemetry_csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                self.frames_data = list(reader)
            print(f"Loaded {len(self.frames_data)} frames from telemetry data")
            return True
        except Exception as e:
            print(f"Error loading telemetry data: {e}")
            return False
    
    def parse_time(self, time_str: str) -> datetime.time:
        """Parse time string to time object."""
        try:
            return datetime.strptime(time_str, "%H:%M:%S").time()
        except:
            return None
    
    def is_night_time(self, time_str: str) -> bool:
        """Check if time is night (11 PM to 6 AM)."""
        parsed_time = self.parse_time(time_str)
        if not parsed_time:
            return False
        night_start = time(23, 0, 0)
        night_end = time(6, 0, 0)
        return parsed_time >= night_start or parsed_time < night_end
    
    def extract_objects_and_people(self) -> None:
        """Extract all detected objects and people from frame descriptions."""
        for frame in self.frames_data:
            description = frame.get("Frame Description", "").lower()
            frame_id = frame.get("Frame ID", "")
            location = frame.get("Location", "")
            frame_time = frame.get("Time", "")
            
            frame_info = {
                "frame_id": frame_id,
                "description": description,
                "location": location,
                "time": frame_time
            }
            
            for obj_type, keywords in self.object_keywords.items():
                if any(keyword in description for keyword in keywords):
                    if obj_type == "person":
                        self.detected_people.append(frame_info)
                    else:
                        self.detected_objects.append({**frame_info, "object_type": obj_type})
    
    
    def analyze_frame_with_ai(self, frame: Dict) -> Dict:
        """Use Ollama LLM to intelligently analyze a frame and generate context-aware alerts."""
        try:
            description = frame.get("Frame Description", "")
            location = frame.get("Location", "")
            frame_time = frame.get("Time", "")
            frame_id = frame.get("Frame ID", "")
            
            prompt = f"""Analyze this security frame and respond with JSON only.
Frame: {frame_id} | {description} | {location} | {frame_time}

Respond with ONLY this JSON (no other text):
{{"event_detected": true/false, "event_type": "delivery/loitering/suspicious/movement/none", "severity": "INFO/WARNING/HIGH", "alert_message": "brief message", "confidence": 0.5, "reasoning": "why", "recommended_action": "action"}}"""
            
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            response_text = result.get("response", "").strip()
            
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    ai_analysis = json.loads(json_str)
                    return ai_analysis
            except (json.JSONDecodeError, ValueError):
                pass
            
            return {
                "event_detected": False,
                "event_type": "none",
                "severity": "INFO",
                "alert_message": "Unable to analyze frame",
                "confidence": 0.0,
                "reasoning": "Failed to parse AI response",
                "recommended_action": "Review frame manually"
            }
        except requests.exceptions.ConnectionError:
            print("Warning: Cannot connect to Ollama. Ensure Ollama is running on http://localhost:11434")
            return {
                "event_detected": False,
                "event_type": "none",
                "severity": "INFO",
                "alert_message": "Ollama unavailable",
                "confidence": 0.0,
                "reasoning": "Ollama service not accessible",
                "recommended_action": "Start Ollama service"
            }
        except Exception as e:
            return {
                "event_detected": False,
                "event_type": "none",
                "severity": "INFO",
                "alert_message": "Unable to analyze frame",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "recommended_action": "Review frame manually"
            }
    
    
    def process_and_alert(self) -> Dict:
        """Main processing function to detect events and generate alerts."""
        if not self.load_telemetry_data():
            return {"success": False, "error": "Failed to load telemetry data"}
        
        self.extract_objects_and_people()
        alerts = self.process_frames_with_ai_rules()
        
        self.alerts = alerts
        
        result = {
            "success": True,
            "total_frames_processed": len(self.frames_data),
            "alerts_generated": len(alerts),
            "detected_objects": self.detected_objects,
            "detected_people": self.detected_people,
            "alerts": alerts
        }
        
        return result
    
    def process_frames_with_ai_rules(self) -> List[Dict]:
        """Process frames using AI-powered alert rules."""
        alerts = []
        print("\nProcessing frames with AI-powered alert rules...")
        
        for idx, frame in enumerate(self.frames_data):
            print(f"Analyzing frame {idx + 1}/{len(self.frames_data)}...", end='\r')
            
            ai_analysis = self.analyze_frame_with_ai(frame)
            
            if ai_analysis.get("event_detected"):
                alert = {
                    "alert_type": ai_analysis.get("event_type", "UNKNOWN").upper(),
                    "severity": ai_analysis.get("severity", "INFO"),
                    "timestamp": datetime.now().isoformat(),
                    "location": frame.get("Location", "Unknown"),
                    "time": frame.get("Time", "Unknown"),
                    "message": ai_analysis.get("alert_message", ""),
                    "event_type": ai_analysis.get("event_type", "unknown"),
                    "confidence": ai_analysis.get("confidence", 0.0),
                    "reasoning": ai_analysis.get("reasoning", ""),
                    "recommended_action": ai_analysis.get("recommended_action", ""),
                    "evidence": {
                        "frame_id": frame.get("Frame ID"),
                        "frame_description": frame.get("Frame Description"),
                        "detected_at": frame.get("Time")
                    }
                }
                alerts.append(alert)
        
        print(f"\nCompleted analysis of {len(self.frames_data)} frames")
        return alerts
    
    def save_alerts(self, output_path: str) -> bool:
        """Save generated alerts to a JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.alerts, f, indent=2)
            print(f"Alerts saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving alerts: {e}")
            return False
    
    def print_alerts(self, alerts: List[Dict]) -> None:
        """Print alerts in a user-friendly format."""
        if not alerts:
            print("No alerts generated")
            return
        
        print("\n" + "="*70)
        print(f"SECURITY ANALYSIS REPORT - {len(alerts)} Alert(s) Generated")
        print("="*70)
        
        for idx, alert in enumerate(alerts, 1):
            severity_icon = "�" if alert.get('severity') == "HIGH" else "🟡" if alert.get('severity') == "WARNING" else "🟢"
            print(f"\n{severity_icon} ALERT #{idx}: {alert.get('alert_type')}")
            print(f"   Severity: {alert.get('severity')}")
            print(f"   Message: {alert.get('message')}")
            print(f"   Location: {alert.get('location')}")
            print(f"   Time: {alert.get('time')}")
            
            evidence = alert.get('evidence', {})
            print(f"   Evidence:")
            print(f"     - Frame: {evidence.get('frame_id')}")
            print(f"     - Description: {evidence.get('frame_description')}")
            print(f"     - Detected at: {evidence.get('detected_at')}")
        
        print("\n" + "="*70 + "\n")
    
    def print_detected_objects_and_people(self) -> None:
        """Print summary of detected objects and people."""
        print("\n" + "="*70)
        print("DETECTED OBJECTS AND PEOPLE SUMMARY")
        print("="*70)
        
        if self.detected_objects:
            print(f"\nObjects Detected ({len(self.detected_objects)} total):")
            object_types = {}
            for obj in self.detected_objects:
                obj_type = obj.get("object_type", "unknown")
                if obj_type not in object_types:
                    object_types[obj_type] = []
                object_types[obj_type].append(obj)
            
            for obj_type, objs in object_types.items():
                print(f"  - {obj_type.upper()}: {len(objs)} detections")
                for obj in objs[:2]:
                    print(f"    • {obj.get('frame_id')} at {obj.get('location')} ({obj.get('time')})")
                if len(objs) > 2:
                    print(f"    ... and {len(objs) - 2} more")
        
        if self.detected_people:
            print(f"\nPeople Detected ({len(self.detected_people)} total):")
            for person in self.detected_people[:3]:
                print(f"  - {person.get('frame_id')} at {person.get('location')} ({person.get('time')})")
                print(f"    Description: {person.get('description')}")
            if len(self.detected_people) > 3:
                print(f"  ... and {len(self.detected_people) - 3} more detections")
        
        print("="*70 + "\n")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    telemetry_csv = project_root / "drone_telemetry.csv"
    alerts_output = project_root / "security_alerts.json"
    
    agent = SecurityAnalystAgent(str(telemetry_csv), use_ai_rules=True)
    result = agent.process_and_alert()
    
    if result.get("success"):
        print(f"\n{'='*70}")
        print("SECURITY ANALYST AGENT - PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total frames processed: {result.get('total_frames_processed')}")
        print(f"Alerts generated: {result.get('alerts_generated')}")
        
        agent.print_detected_objects_and_people()
        
        alerts = result.get("alerts", [])
        if alerts:
            agent.print_alerts(alerts)
            agent.save_alerts(str(alerts_output))
        else:
            print("No security alerts generated")
    else:
        print(f"Error: {result.get('error')}")


if __name__ == "__main__":
    main()
