"""
Local Vector Database Example
Using sentence-transformers for embeddings and numpy for similarity search
No external services required!
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Tuple, Dict, Any

class LocalVectorDB:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the local vector database with a sentence transformer model."""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.vectors = []
        self.texts = []
        self.metadata = []
        
    def add_texts(self, texts: List[str], metadata: List[Dict[str, Any]] = None) -> List[str]:
        """Add texts to the vector database."""
        if metadata is None:
            metadata = [{"id": f"doc_{i}"} for i in range(len(texts))]
        
        print(f"Encoding {len(texts)} texts into embeddings...")
        embeddings = self.model.encode(texts)
        
        # Store the data
        start_idx = len(self.vectors)
        ids = []
        
        for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
            doc_id = f"doc_{start_idx + i}"
            ids.append(doc_id)
            
            self.vectors.append(embedding)
            self.texts.append(text)
            self.metadata.append({**meta, "id": doc_id, "text": text})
        
        print(f"Added {len(texts)} documents to the database")
        return ids
    
    def query(self, query_text: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Query the vector database for similar texts."""
        if not self.vectors:
            return []
        
        # Encode the query
        query_embedding = self.model.encode([query_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, np.array(self.vectors))[0]
        
        # Get top-k results above threshold
        results = []
        for i, score in enumerate(similarities):
            if score >= threshold:
                results.append({
                    "id": self.metadata[i]["id"],
                    "score": float(score),
                    "text": self.texts[i],
                    "metadata": self.metadata[i]
                })
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def save(self, filepath: str):
        """Save the vector database to a file."""
        data = {
            "vectors": [v.tolist() for v in self.vectors],
            "texts": self.texts,
            "metadata": self.metadata
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Saved vector database to {filepath}")
    
    def load(self, filepath: str):
        """Load the vector database from a file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vectors = [np.array(v) for v in data["vectors"]]
        self.texts = data["texts"]
        self.metadata = data["metadata"]
        print(f"Loaded vector database from {filepath} with {len(self.texts)} documents")


def main():
    """Demo of the local vector database."""
    print("🚀 Local Vector Database Demo\n")
    
    # Initialize the vector database
    db = LocalVectorDB()
    
    # Sample documents about renewable energy
    documents = [
        "Solar energy is a renewable resource that harnesses sunlight to generate electricity.",
        "Wind turbines convert kinetic energy from wind into electrical power efficiently.",
        "Nuclear energy provides powerful electricity generation but raises safety concerns.",
        "Hydroelectric power uses flowing water to generate clean renewable energy.",
        "Geothermal energy taps into Earth's internal heat for sustainable power generation.",
        "Fossil fuels like coal and oil are non-renewable and contribute to climate change.",
        "Battery storage technology is crucial for managing intermittent renewable energy sources.",
        "Smart grids help optimize the distribution of renewable energy across power networks.",
        "Electric vehicles reduce emissions when powered by renewable energy sources.",
        "Carbon capture technology can help mitigate emissions from fossil fuel power plants."
    ]
    
    # Add documents with metadata
    metadata = [{"category": "energy", "source": "knowledge_base"} for _ in documents]
    doc_ids = db.add_texts(documents, metadata)
    
    print(f"\n📚 Added {len(doc_ids)} documents to the database")
    
    # Test queries
    queries = [
        "How is renewable electricity produced?",
        "What are the environmental impacts of energy sources?",
        "How do we store renewable energy?",
        "What is the future of clean energy?"
    ]
    
    print("\n🔍 Testing similarity search:")
    print("=" * 50)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        results = db.query(query, top_k=3, threshold=0.2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. Score: {result['score']:.4f}")
                print(f"   Text: {result['text']}")
                print()
        else:
            print("No relevant results found.")
    
    # Save the database
    db.save("energy_vector_db.json")
    
    print("✅ Demo completed successfully!")


if __name__ == "__main__":
    main()