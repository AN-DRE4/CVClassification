from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

class CVVectorizer:
    def __init__(self, vector_cache_path: str = "memory/vector_cache.json"):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vector_cache_path = vector_cache_path
        self.vector_cache = self._load_vector_cache()
        self._fit_vectorizer()
    
    def _load_vector_cache(self) -> Dict:
        """Load vector cache from file"""
        if os.path.exists(self.vector_cache_path):
            with open(self.vector_cache_path, 'r') as f:
                return json.load(f)
        return {"vectors": {}, "texts": {}}
    
    def _save_vector_cache(self):
        """Save vector cache to file"""
        os.makedirs(os.path.dirname(self.vector_cache_path), exist_ok=True)
        with open(self.vector_cache_path, 'w') as f:
            json.dump(self.vector_cache, f, indent=2)
    
    def _fit_vectorizer(self):
        """Fit the vectorizer with existing texts"""
        if self.vector_cache["texts"]:
            texts = list(self.vector_cache["texts"].values())
            self.vectorizer.fit(texts)
    
    def _get_cv_text(self, cv_data: Dict) -> str:
        """Extract and combine relevant text from CV data"""
        cv_data = cv_data.get("extracted_info", {})
        education = cv_data.get("education", "")
        work_experience = cv_data.get("work_experience", "")
        skills = cv_data.get("extra_skills", "")
        
        # Combine all text with appropriate weights
        return f"{education} {work_experience} {skills}"
    
    def get_vector(self, cv_data: Dict, resume_id: str) -> np.ndarray:
        """Get or create vector for a CV"""
        if resume_id in self.vector_cache["vectors"]:
            return np.array(self.vector_cache["vectors"][resume_id])
        
        # Create new vector
        cv_text = self._get_cv_text(cv_data)
        vector = self.vectorizer.transform([cv_text]).toarray()[0]
        
        # Cache the vector and text
        self.vector_cache["vectors"][resume_id] = vector.tolist()
        print("DEBUG: length of vector: ", len(self.vector_cache["vectors"][resume_id]))
        self.vector_cache["texts"][resume_id] = cv_text
        self._save_vector_cache()
        
        return vector
    
    def find_similar_cvs(self, cv_data: Dict, resume_id: str, threshold: float = 0.7) -> List[Dict]:
        """Find similar CVs based on vector similarity"""
        if not self.vector_cache["vectors"]:
            # Fit the vectorizer on the current CV text if cache is empty
            cv_text = self._get_cv_text(cv_data)
            try:
                self.vectorizer.fit([cv_text])
                # Get vector for current CV
                current_vector = self.vectorizer.transform([cv_text]).toarray()[0]
                # Cache the vector and text
                self.vector_cache["vectors"][resume_id] = current_vector.tolist()
                self.vector_cache["texts"][resume_id] = cv_text
                self._save_vector_cache()
            except Exception as e:
                print(f"DEBUG: Error during vectorization: {str(e)}")
                print(f"DEBUG: Error type: {type(e)}")
                raise
            return []
        
        print("DEBUG: vector cache found")
        
        # Get vector for current CV
        current_vector = self.get_vector(cv_data, resume_id)

        print("DEBUG: current vector found")
        
        # Calculate similarities with all cached vectors
        similarities = []
        for cached_id, cached_vector in self.vector_cache["vectors"].items():
            if cached_id != resume_id:  # Don't compare with self
                similarity = cosine_similarity(
                    [current_vector],
                    [np.array(cached_vector)]
                )[0][0]
                similarities.append((cached_id, similarity))

        print("DEBUG: similarities calculated")
        
        # Sort by similarity and filter by threshold
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [id for id, sim in similarities if sim >= threshold]
    
    def clear_cache(self):
        """Clear the vector cache"""
        self.vector_cache = {"vectors": {}, "texts": {}}
        self._save_vector_cache() 