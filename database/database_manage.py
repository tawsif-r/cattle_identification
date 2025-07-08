import sqlite3
from typing import List, Dict
import numpy as np

class CattleDatabase:
    """Database for storing cattle features and information"""
    
    def __init__(self, db_path: str = "cattle_database.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cattle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cattle_id TEXT UNIQUE,
                name TEXT,
                breed TEXT,
                age INTEGER,
                gender TEXT,
                owner TEXT,
                registration_date TEXT,
                face_features BLOB,
                muzzle_features BLOB,
                ear_tag_text TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def store_cattle(self, cattle_info: Dict[str, any]):
        """Store cattle information and features"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert numpy arrays to bytes
        face_features_bytes = cattle_info['face_features'].tobytes() if 'face_features' in cattle_info else None
        muzzle_features_bytes = cattle_info['muzzle_features'].tobytes() if 'muzzle_features' in cattle_info else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO cattle 
            (cattle_id, name, breed, age, gender, owner, registration_date, 
             face_features, muzzle_features, ear_tag_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            cattle_info['cattle_id'],
            cattle_info.get('name', ''),
            cattle_info.get('breed', ''),
            cattle_info.get('age', 0),
            cattle_info.get('gender', ''),
            cattle_info.get('owner', ''),
            cattle_info.get('registration_date', ''),
            face_features_bytes,
            muzzle_features_bytes,
            cattle_info.get('ear_tag_text', '')
        ))
        
        conn.commit()
        conn.close()
        
    def get_all_cattle_features(self) -> List[Dict[str, any]]:
        """Get all cattle features from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT cattle_id, face_features, muzzle_features, ear_tag_text 
            FROM cattle WHERE face_features IS NOT NULL OR muzzle_features IS NOT NULL
        ''')
        
        results = []
        for row in cursor.fetchall():
            cattle_id, face_features_bytes, muzzle_features_bytes, ear_tag_text = row
            
            cattle_data = {'cattle_id': cattle_id, 'ear_tag_text': ear_tag_text}
            
            # Convert bytes back to numpy arrays
            if face_features_bytes:
                cattle_data['face_features'] = np.frombuffer(face_features_bytes, dtype=np.float32)
            if muzzle_features_bytes:
                cattle_data['muzzle_features'] = np.frombuffer(muzzle_features_bytes, dtype=np.float32)
                
            results.append(cattle_data)
        
        conn.close()
        return results
    
    def search_cattle(self, query_features: Dict[str, any], threshold: float = 0.7) -> List[Dict[str, any]]:
        """Search for cattle based on features"""
        # This would implement the matching logic using distance metrics
        # For now, return empty list
        return []