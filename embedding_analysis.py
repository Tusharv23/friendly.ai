"""
Demo: How does the model know relationships?
Let's inspect the actual embedding vectors to see semantic similarity
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def show_embedding_magic():
    print("🔍 Embedding Analysis: How does the model know relationships?\n")
    
    # Load the same model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Test sentences
    sentences = [
        "Wind turbines convert wind energy into electricity",
        "Renewable energy is environmentally friendly",
        "Solar panels harness sunlight for power",
        "Pizza is a delicious Italian food",
        "Clean energy sources reduce pollution",
        "Cats are furry domestic animals"
    ]
    
    print("📊 Computing embeddings for test sentences...")
    embeddings = model.encode(sentences)
    
    # Show actual embedding values (first 10 dimensions)
    print("\n🧮 Actual Embedding Values (first 10 dimensions):")
    print("=" * 70)
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        print(f"\n{i+1}. '{sentence[:50]}...'")
        print(f"   Vector: [{', '.join([f'{x:.3f}' for x in embedding[:10]])}...]")
    
    # Calculate all pairwise similarities
    print("\n🎯 Similarity Matrix:")
    print("=" * 50)
    
    similarity_matrix = cosine_similarity(embeddings)
    
    # Print header
    print("     ", end="")
    for i in range(len(sentences)):
        print(f"  {i+1}  ", end="")
    print()
    
    # Print similarity scores
    for i in range(len(sentences)):
        print(f"{i+1}.   ", end="")
        for j in range(len(sentences)):
            score = similarity_matrix[i][j]
            if i == j:
                print(f"1.00 ", end=" ")  # Self-similarity is always 1.0
            else:
                print(f"{score:.2f} ", end=" ")
        print(f" <- '{sentences[i][:30]}...'")
    
    print("\n🔍 Key Observations:")
    print("-" * 40)
    
    # Find highest cross-similarities (not self-similarity)
    max_similarity = 0
    max_pair = (0, 0)
    
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sim = similarity_matrix[i][j]
            if sim > max_similarity:
                max_similarity = sim
                max_pair = (i, j)
    
    print(f"Highest similarity ({max_similarity:.3f}):")
    print(f"  '{sentences[max_pair[0]]}'")
    print(f"  '{sentences[max_pair[1]]}'")
    
    # Show energy-related similarities
    print(f"\n💡 Energy-related similarities:")
    energy_indices = [0, 1, 2, 4]  # Wind, renewable, solar, clean energy
    
    for i in energy_indices:
        for j in energy_indices:
            if i < j:
                sim = similarity_matrix[i][j]
                print(f"  {sim:.3f}: '{sentences[i][:25]}...' ↔ '{sentences[j][:25]}...'")
    
    # Show non-energy similarities
    print(f"\n🍕 Non-energy similarities:")
    non_energy = [3, 5]  # Pizza, cats
    
    for i in range(len(sentences)):
        for j in non_energy:
            if i != j and i not in non_energy:
                sim = similarity_matrix[i][j]
                print(f"  {sim:.3f}: '{sentences[i][:25]}...' ↔ '{sentences[j][:25]}...'")

def show_concept_clustering():
    print("\n\n🧭 Concept Clustering Demo")
    print("=" * 40)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Different concept groups
    concepts = {
        "Energy": [
            "wind turbine electricity generation",
            "solar panel renewable power",
            "hydroelectric dam clean energy",
            "geothermal sustainable electricity"
        ],
        "Food": [
            "pizza Italian cuisine",
            "sushi Japanese food",
            "burger American meal",
            "pasta Mediterranean dish"
        ],
        "Animals": [
            "cat furry pet",
            "dog loyal companion",
            "bird flying creature",
            "fish swimming animal"
        ]
    }
    
    all_texts = []
    labels = []
    
    for category, texts in concepts.items():
        all_texts.extend(texts)
        labels.extend([category] * len(texts))
    
    embeddings = model.encode(all_texts)
    
    print("🎯 Cross-category similarity analysis:")
    print("-" * 40)
    
    # Calculate average within-category vs cross-category similarities
    within_category_sims = []
    cross_category_sims = []
    
    for i in range(len(all_texts)):
        for j in range(i+1, len(all_texts)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            
            if labels[i] == labels[j]:
                within_category_sims.append(sim)
                print(f"SAME ({labels[i]}): {sim:.3f} - '{all_texts[i]}' ↔ '{all_texts[j]}'")
            else:
                cross_category_sims.append(sim)
    
    print(f"\n📊 Statistical Analysis:")
    print(f"Average within-category similarity: {np.mean(within_category_sims):.3f}")
    print(f"Average cross-category similarity: {np.mean(cross_category_sims):.3f}")
    print(f"Difference: {np.mean(within_category_sims) - np.mean(cross_category_sims):.3f}")

if __name__ == "__main__":
    show_embedding_magic()
    show_concept_clustering()