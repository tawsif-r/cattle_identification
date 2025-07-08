import numpy as np

# ===== Distance Metrics and Matching =====
class DistanceMetrics:
    """Distance calculation utilities for feature matching"""
    
    @staticmethod
    def euclidean_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate Euclidean distance between two feature vectors"""
        return np.linalg.norm(feat1 - feat2)
    
    @staticmethod
    def cosine_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate cosine distance between two feature vectors"""
        dot_product = np.dot(feat1, feat2)
        norm_feat1 = np.linalg.norm(feat1)
        norm_feat2 = np.linalg.norm(feat2)
        if norm_feat1 == 0 or norm_feat2 == 0:
            return 1.0
        cosine_sim = dot_product / (norm_feat1 * norm_feat2)
        return 1 - cosine_sim
    
    @staticmethod
    def levenshtein_distance(str1: str, str2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(str1) < len(str2):
            return DistanceMetrics.levenshtein_distance(str2, str1)
        
        if len(str2) == 0:
            return len(str1)
        
        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    

