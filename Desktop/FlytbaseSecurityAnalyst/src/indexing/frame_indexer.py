import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import csv

class FrameIndexer:
    """Frame-by-frame indexing system for drone telemetry data."""
    
    def __init__(self, db_path: str = "frames.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.init_database()
    
    def init_database(self) -> None:
        """Initialize SQLite database with frame indexing schema."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                location TEXT NOT NULL,
                time TEXT NOT NULL,
                altitude INTEGER,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS detected_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id TEXT NOT NULL,
                object_type TEXT NOT NULL,
                object_name TEXT,
                confidence REAL DEFAULT 0.5,
                FOREIGN KEY (frame_id) REFERENCES frames(frame_id)
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS detected_people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id TEXT NOT NULL,
                person_description TEXT,
                activity TEXT,
                FOREIGN KEY (frame_id) REFERENCES frames(frame_id)
            )
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_frame_id ON frames(frame_id)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_time ON frames(time)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_location ON frames(location)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_object_type ON detected_objects(object_type)
        ''')
        
        self.conn.commit()
        print(f"Database initialized at {self.db_path}")
    
    def index_frames_from_csv(self, csv_path: str) -> int:
        """Index frames from telemetry CSV file."""
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                frames_count = 0
                
                for row in reader:
                    frame_id = row.get("Frame ID")
                    description = row.get("Frame Description", "")
                    location = row.get("Location", "")
                    time = row.get("Time", "")
                    altitude = row.get("Altitude (meters)", 0)
                    
                    try:
                        altitude = int(altitude) if altitude else 0
                    except ValueError:
                        altitude = 0
                    
                    self.cursor.execute('''
                        INSERT OR REPLACE INTO frames 
                        (frame_id, description, location, time, altitude)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (frame_id, description, location, time, altitude))
                    
                    self.extract_and_index_objects(frame_id, description)
                    frames_count += 1
                
                self.conn.commit()
                print(f"Indexed {frames_count} frames from {csv_path}")
                return frames_count
        except Exception as e:
            print(f"Error indexing frames: {e}")
            return 0
    
    def extract_and_index_objects(self, frame_id: str, description: str) -> None:
        """Extract and index objects from frame description."""
        description_lower = description.lower()
        
        object_keywords = {
            "vehicle": ["truck", "car", "van", "fedex", "ups", "amazon", "delivery"],
            "person": ["man", "woman", "person", "people"],
            "package": ["box", "package", "parcel"],
            "suspicious": ["loitering", "lurking", "suspicious", "climbing", "breaking"]
        }
        
        for obj_type, keywords in object_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    self.cursor.execute('''
                        INSERT OR IGNORE INTO detected_objects 
                        (frame_id, object_type, object_name)
                        VALUES (?, ?, ?)
                    ''', (frame_id, obj_type, keyword))
                    break
        
        if any(keyword in description_lower for keyword in ["man", "woman", "person"]):
            self.cursor.execute('''
                INSERT OR IGNORE INTO detected_people 
                (frame_id, person_description)
                VALUES (?, ?)
            ''', (frame_id, description))
        
        self.conn.commit()
    
    def query_frames_by_time(self, start_time: str, end_time: str) -> List[Dict]:
        """Query frames within a time range."""
        self.cursor.execute('''
            SELECT * FROM frames 
            WHERE time >= ? AND time <= ?
            ORDER BY time
        ''', (start_time, end_time))
        
        rows = self.cursor.fetchall()
        return self._rows_to_dicts(rows)
    
    def query_frames_by_object(self, object_type: str) -> List[Dict]:
        """Query all frames containing a specific object type."""
        self.cursor.execute('''
            SELECT DISTINCT f.* FROM frames f
            JOIN detected_objects o ON f.frame_id = o.frame_id
            WHERE o.object_type = ?
            ORDER BY f.time
        ''', (object_type,))
        
        rows = self.cursor.fetchall()
        return self._rows_to_dicts(rows)
    
    def query_frames_by_location(self, location: str) -> List[Dict]:
        """Query all frames from a specific location."""
        self.cursor.execute('''
            SELECT * FROM frames 
            WHERE location = ?
            ORDER BY time
        ''', (location,))
        
        rows = self.cursor.fetchall()
        return self._rows_to_dicts(rows)
    
    def query_frames_with_people(self) -> List[Dict]:
        """Query all frames containing people."""
        self.cursor.execute('''
            SELECT DISTINCT f.* FROM frames f
            JOIN detected_people p ON f.frame_id = p.frame_id
            ORDER BY f.time
        ''')
        
        rows = self.cursor.fetchall()
        return self._rows_to_dicts(rows)
    
    def query_frames_by_object_and_time(self, object_type: str, start_time: str, end_time: str) -> List[Dict]:
        """Query frames with specific object type within a time range."""
        self.cursor.execute('''
            SELECT DISTINCT f.* FROM frames f
            JOIN detected_objects o ON f.frame_id = o.frame_id
            WHERE o.object_type = ? AND f.time >= ? AND f.time <= ?
            ORDER BY f.time
        ''', (object_type, start_time, end_time))
        
        rows = self.cursor.fetchall()
        return self._rows_to_dicts(rows)
    
    def get_frame_details(self, frame_id: str) -> Dict:
        """Get detailed information about a specific frame."""
        self.cursor.execute('SELECT * FROM frames WHERE frame_id = ?', (frame_id,))
        frame_row = self.cursor.fetchone()
        
        if not frame_row:
            return {}
        
        frame_dict = self._row_to_dict(frame_row)
        
        self.cursor.execute('SELECT * FROM detected_objects WHERE frame_id = ?', (frame_id,))
        objects = self.cursor.fetchall()
        frame_dict["objects"] = [dict(zip([col[0] for col in self.cursor.description], obj)) for obj in objects]
        
        self.cursor.execute('SELECT * FROM detected_people WHERE frame_id = ?', (frame_id,))
        people = self.cursor.fetchall()
        frame_dict["people"] = [dict(zip([col[0] for col in self.cursor.description], person)) for person in people]
        
        return frame_dict
    
    def get_statistics(self) -> Dict:
        """Get indexing statistics."""
        self.cursor.execute('SELECT COUNT(*) FROM frames')
        total_frames = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(DISTINCT object_type) FROM detected_objects')
        unique_objects = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(DISTINCT frame_id) FROM detected_people')
        frames_with_people = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(DISTINCT location) FROM frames')
        unique_locations = self.cursor.fetchone()[0]
        
        return {
            "total_frames": total_frames,
            "unique_object_types": unique_objects,
            "frames_with_people": frames_with_people,
            "unique_locations": unique_locations
        }
    
    def export_query_results(self, results: List[Dict], output_path: str) -> bool:
        """Export query results to JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Results exported to {output_path}")
            return True
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False
    
    def _rows_to_dicts(self, rows: List) -> List[Dict]:
        """Convert database rows to dictionaries."""
        if not rows:
            return []
        
        columns = [col[0] for col in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def _row_to_dict(self, row) -> Dict:
        """Convert a single database row to dictionary."""
        columns = [col[0] for col in self.cursor.description]
        return dict(zip(columns, row))
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    db_path = project_root / "frames.db"
    telemetry_csv = project_root / "drone_telemetry.csv"
    
    with FrameIndexer(str(db_path)) as indexer:
        print("\n" + "="*70)
        print("FRAME INDEXING SYSTEM")
        print("="*70)
        
        indexer.index_frames_from_csv(str(telemetry_csv))
        
        stats = indexer.get_statistics()
        print(f"\nIndexing Statistics:")
        print(f"  - Total frames indexed: {stats['total_frames']}")
        print(f"  - Unique object types: {stats['unique_object_types']}")
        print(f"  - Frames with people: {stats['frames_with_people']}")
        print(f"  - Unique locations: {stats['unique_locations']}")
        
        print("\n" + "="*70)
        print("QUERY EXAMPLES")
        print("="*70)
        
        print("\n1. Query all frames with vehicles:")
        vehicle_frames = indexer.query_frames_by_object("vehicle")
        print(f"   Found {len(vehicle_frames)} frames with vehicles")
        for frame in vehicle_frames[:2]:
            print(f"   - {frame['frame_id']} at {frame['location']} ({frame['time']})")
        if len(vehicle_frames) > 2:
            print(f"   ... and {len(vehicle_frames) - 2} more")
        
        print("\n2. Query all frames with people:")
        people_frames = indexer.query_frames_with_people()
        print(f"   Found {len(people_frames)} frames with people")
        for frame in people_frames[:2]:
            print(f"   - {frame['frame_id']} at {frame['location']} ({frame['time']})")
        if len(people_frames) > 2:
            print(f"   ... and {len(people_frames) - 2} more")
        
        print("\n3. Query frames by location (Main Gate):")
        location_frames = indexer.query_frames_by_location("Main Gate")
        print(f"   Found {len(location_frames)} frames at Main Gate")
        
        print("\n4. Query frames by time range (10:00:00 - 10:00:10):")
        time_frames = indexer.query_frames_by_time("10:00:00", "10:00:10")
        print(f"   Found {len(time_frames)} frames in time range")
        for frame in time_frames[:3]:
            print(f"   - {frame['frame_id']} ({frame['time']}): {frame['description'][:50]}...")
        
        print("\n5. Query vehicles detected between 10:00:15 - 10:00:25:")
        vehicle_time_frames = indexer.query_frames_by_object_and_time("vehicle", "10:00:15", "10:00:25")
        print(f"   Found {len(vehicle_time_frames)} vehicle frames in time range")
        
        print("\n6. Get detailed information for a specific frame:")
        if vehicle_frames:
            frame_id = vehicle_frames[0]['frame_id']
            details = indexer.get_frame_details(frame_id)
            print(f"   Frame: {frame_id}")
            print(f"   Description: {details.get('description', 'N/A')}")
            print(f"   Location: {details.get('location', 'N/A')}")
            print(f"   Time: {details.get('time', 'N/A')}")
            print(f"   Altitude: {details.get('altitude', 'N/A')} meters")
            if details.get('objects'):
                print(f"   Objects detected: {len(details['objects'])}")
                for obj in details['objects']:
                    print(f"     - {obj.get('object_type')}: {obj.get('object_name')}")
        
        print("\n" + "="*70)
        indexer.export_query_results(vehicle_frames, str(project_root / "indexed_vehicles.json"))
        indexer.export_query_results(people_frames, str(project_root / "indexed_people.json"))
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
