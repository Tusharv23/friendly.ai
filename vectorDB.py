"""
Pinecone Vector Database Example (Fixed)
Note: This requires a valid Pinecone API key and internet connection
"""
import os
import ssl
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

try:
    # 1. Initialize Pinecone (use environment variable for security)
    api_key = os.getenv('PINECONE_API_KEY', 'pcsk_4KJv86_LT3pavBgdyHiANiuynDZLY2bdim37dxrm5Bq13fVSdnAdYyecbUKxuiVpXBvep5')
    pc = Pinecone(api_key=api_key)

    # 2. Create index
    index_name = "semantic-search"
    
    try:
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        print("Please check your API key and internet connection")
        exit(1)

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created index: {index_name}")
    else:
        print(f"Index {index_name} already exists")

    index = pc.Index(index_name)
    
except Exception as e:
    print(f"Failed to initialize Pinecone: {e}")
    print("\nTip: Try the local_vectorDB.py instead for offline vector search!")
    exit(1)

try:
    # 3. Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 4. Insert vectors
    texts = [
        "Solar energy is a renewable resource.",
        "Wind turbines convert wind energy into electricity.",
        "Nuclear energy is powerful but controversial.",
        "Hydroelectric power uses flowing water to generate electricity.",
        "Geothermal energy taps into Earth's internal heat."
    ]
    
    print("Generating embeddings...")
    embeddings = model.encode(texts)

    vectors = [(f"id-{i}", emb.tolist(), {"text": texts[i]}) for i, emb in enumerate(embeddings)]
    
    print("Upserting vectors to Pinecone...")
    index.upsert(vectors)
    print(f"Successfully added {len(vectors)} vectors to Pinecone")

    # 5. Query
    query = "How is renewable electricity produced?"
    print(f"\nQuerying: '{query}'")
    
    query_vec = model.encode([query]).tolist()
    results = index.query(vector=query_vec[0], top_k=3, include_metadata=True)

    # 6. Display results
    print("\nResults:")
    print("-" * 50)
    for i, match in enumerate(results['matches'], 1):
        print(f"{i}. Score: {match['score']:.4f}")
        print(f"   Text: {match['metadata']['text']}")
        print()
        
    print("✅ Pinecone vector database demo completed successfully!")
    
except Exception as e:
    print(f"Error during Pinecone operations: {e}")
    print("\nTip: Try the local_vectorDB.py instead for offline vector search!")
