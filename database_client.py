import sqlite3
import numpy as np
import json
import os
from typing import Dict
class DatabaseClient:
    def __init__(self, db_path:str="cow.db"):
        """
        Initialize the sqlite database if it doesnt exist already

        """
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """
        Create feature table if it doesnt exist
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS muzzle_features (
                        reference_number TEXT PRIMARY KEY,
                        features BLOB NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                                )
                """ 
                )
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error creating table: {e}")

    def save_features(self,reference_number, features):
        """
        Save extracted features to the database with a reference number.
        
        Args:
            reference_number (str): Unique identifier for the feature set
            features (np.ndarray): Feature vector to store
        """
        try:
            features_blob = features.tobytes()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT OR REPLACE INTO muzzle_features (reference_number, features) VALUES (?,?)""",(reference_number,features_blob)
                )
                conn.commit()
                print(f"Features saved for reference number: {reference_number}")
        except sqlite3.Error as e:
            print(f"Database error saving features: {e}")

    def match_features(self, reference_number, query_features, threshold=8.0):
        """
        Check if the query features match the stored features for the given reference number.
        
        Args:
            reference_number (str): Reference number to look up
            query_features (np.ndarray): Feature vector to compare against
            threshold (float): Maximum Euclidean distance for a match
            
        Returns:
            tuple: (bool, float) - (True if match within threshold, Euclidean distance)
                   or (False, None) if no match or reference number not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT features FROM muzzle_features WHERE reference_number = ?", (reference_number,))
                result = cursor.fetchone()
                
                if result is None:
                    print(f"No features found for reference number: {reference_number}")
                    return False, None
                
                stored_features_blob = result[0]
                # Convert blob back to numpy array
                stored_features = np.frombuffer(stored_features_blob, dtype=query_features.dtype)
                
                # Ensure shapes match
                if stored_features.shape != query_features.shape:
                    print(f"Feature shape mismatch for reference number: {reference_number}")
                    return False, None
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(query_features - stored_features)
                
                if distance <= threshold:
                    return True, distance
                else:
                    return False, distance
                
        except sqlite3.Error as e:
            print(f"Database error matching features: {e}")
            return False, None
