from .distance_matrix import DistanceMetrics
from database.database_manage import CattleDatabase
from typing import Dict, List

# ===== Feature Matcher and Decision Making =====


class CattleMatcher:
    """Handles feature matching and decision making"""
    
    def __init__(self, database: CattleDatabase):
        self.database = database
        self.distance_metrics = DistanceMetrics()
        
    def match_features(self, query_features: Dict[str, any], top_k: int = 3) -> Dict[str, List[Dict[str, any]]]:
        """Match query features against database and return top-k matches for each modality"""
        
        # Get all cattle from database
        db_cattle = self.database.get_all_cattle_features()
        
        results = {
            'face_matches': [],
            'muzzle_matches': [],
            'ear_tag_matches': []
        }
        
        # Face matching
        if 'face_features' in query_features:
            face_distances = []
            for cattle in db_cattle:
                if 'face_features' in cattle:
                    distance = self.distance_metrics.euclidean_distance(
                        query_features['face_features'], 
                        cattle['face_features']
                    )
                    face_distances.append({
                        'cattle_id': cattle['cattle_id'],
                        'distance': distance,
                        'similarity': 1 / (1 + distance)
                    })
            
            # Sort and get top-k
            face_distances.sort(key=lambda x: x['distance'])
            results['face_matches'] = face_distances[:top_k]
        
        # Muzzle matching
        if 'muzzle_features' in query_features:
            muzzle_distances = []
            for cattle in db_cattle:
                if 'muzzle_features' in cattle:
                    distance = self.distance_metrics.euclidean_distance(
                        query_features['muzzle_features'], 
                        cattle['muzzle_features']
                    )
                    muzzle_distances.append({
                        'cattle_id': cattle['cattle_id'],
                        'distance': distance,
                        'similarity': 1 / (1 + distance)
                    })
            
            # Sort and get top-k
            muzzle_distances.sort(key=lambda x: x['distance'])
            results['muzzle_matches'] = muzzle_distances[:top_k]
        
        # Ear tag matching
        if 'ear_tag_text' in query_features and query_features['ear_tag_text']:
            ear_tag_distances = []
            for cattle in db_cattle:
                if cattle['ear_tag_text']:
                    distance = self.distance_metrics.levenshtein_distance(
                        query_features['ear_tag_text'], 
                        cattle['ear_tag_text']
                    )
                    # Normalize by max string length
                    max_len = max(len(query_features['ear_tag_text']), len(cattle['ear_tag_text']))
                    normalized_distance = distance / max_len if max_len > 0 else 0
                    
                    ear_tag_distances.append({
                        'cattle_id': cattle['cattle_id'],
                        'distance': normalized_distance,
                        'similarity': 1 - normalized_distance
                    })
            
            # Sort and get top-k
            ear_tag_distances.sort(key=lambda x: x['distance'])
            results['ear_tag_matches'] = ear_tag_distances[:top_k]
        
        return results

# ===== Voting Classifier =====
class VotingClassifier:
    """Implements the voting classifier from the diagram"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {'face': 0.4, 'muzzle': 0.4, 'ear_tag': 0.2}
        
    def predict(self, match_results: Dict[str, List[Dict[str, any]]], confidence_threshold: float = 0.5) -> Dict[str, any]:
        """Make final prediction using voting from all modalities"""
        
        # Collect all candidate cattle IDs
        all_candidates = set()
        for modality_matches in match_results.values():
            for match in modality_matches:
                all_candidates.add(match['cattle_id'])
        
        # Calculate weighted scores for each candidate
        candidate_scores = {}
        detailed_results = {}
        
        for cattle_id in all_candidates:
            total_score = 0
            total_weight = 0
            modality_scores = {}
            
            # Face score
            face_score = self._get_modality_score(cattle_id, match_results['face_matches'])
            if face_score is not None:
                total_score += face_score * self.weights['face']
                total_weight += self.weights['face']
                modality_scores['face'] = face_score
            
            # Muzzle score
            muzzle_score = self._get_modality_score(cattle_id, match_results['muzzle_matches'])
            if muzzle_score is not None:
                total_score += muzzle_score * self.weights['muzzle']
                total_weight += self.weights['muzzle']
                modality_scores['muzzle'] = muzzle_score
            
            # Ear tag score
            ear_tag_score = self._get_modality_score(cattle_id, match_results['ear_tag_matches'])
            if ear_tag_score is not None:
                total_score += ear_tag_score * self.weights['ear_tag']
                total_weight += self.weights['ear_tag']
                modality_scores['ear_tag'] = ear_tag_score
            
            # Calculate final weighted score
            if total_weight > 0:
                final_score = total_score / total_weight
                candidate_scores[cattle_id] = final_score
                detailed_results[cattle_id] = {
                    'final_score': final_score,
                    'modality_scores': modality_scores,
                    'num_modalities': len(modality_scores)
                }
        
        # Find best match
        if not candidate_scores:
            return {
                'predicted_id': None,
                'confidence': 0.0,
                'status': 'No matches found',
                'detailed_results': {}
            }
        
        best_cattle_id = max(candidate_scores.keys(), key=lambda x: candidate_scores[x])
        best_score = candidate_scores[best_cattle_id]
        
        # Check confidence threshold
        if best_score < confidence_threshold:
            status = 'Low confidence match'
        else:
            status = 'Match found'
        
        return {
            'predicted_id': best_cattle_id,
            'confidence': best_score,
            'status': status,
            'detailed_results': detailed_results,
            'all_candidates': dict(sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True))
        }
    
    def _get_modality_score(self, cattle_id: str, matches: List[Dict[str, any]]) -> float:
        """Get similarity score for a specific cattle ID in a modality"""
        for match in matches:
            if match['cattle_id'] == cattle_id:
                return match['similarity']
        return None