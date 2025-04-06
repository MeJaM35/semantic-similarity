import os
import requests
from neo4j import GraphDatabase
import numpy as np

# -------------------------------
# Config
# -------------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your db pwd"
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_TOKEN = 'your token'

TOP_N = 3  # Show top 3 matching candidates

# -------------------------------
# Hugging Face API-based embedder
# -------------------------------
def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text}

    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    print(f"API response for input '{text}': {response.text}")  # Debugging

    if response.status_code == 200:
        output = response.json()
        # If output is a list of token embeddings (list of lists), average them
        if isinstance(output, list) and isinstance(output[0], list):
            return np.mean(output, axis=0)
        return np.array(output)
    else:
        raise Exception(f"Hugging Face API Error: {response.status_code} {response.text}")

def compute_similarity(jd_skills, candidate_skills):
    try:
        jd_embeddings = np.array([get_embedding(skill) for skill in jd_skills])
        candidate_embeddings = np.array([get_embedding(skill) for skill in candidate_skills])

        # Compute cosine similarity between each pair of skills
        similarity_matrix = np.dot(jd_embeddings, candidate_embeddings.T) / (
            np.linalg.norm(jd_embeddings, axis=1, keepdims=True) * np.linalg.norm(candidate_embeddings, axis=1)
        )
        max_similarities = np.max(similarity_matrix, axis=1)  # Best match per JD skill
        return float(np.mean(max_similarities))
    except Exception as e:
        print(f"⚠️ Similarity computation failed: {e}")
        return 0.0

# -------------------------------
# Sample data
# -------------------------------
candidates = [
    {"name": "Alice", "skills": ["Python", "Machine Learning", "Data Science"]},
    {"name": "Bob", "skills": ["JavaScript", "React", "Web Development"]},
    {"name": "Charlie", "skills": ["Python", "Data Engineering", "SQL"]},
    {"name": "Diana", "skills": ["Deep Learning", "Computer Vision", "PyTorch"]}
]

job_description = {
    "title": "AI Engineer",
    "required_skills": ["Machine Learning", "Deep Learning", "Python", "NLP"]
}

# -------------------------------
# Neo4j logic
# -------------------------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def populate_data(tx, candidates):
    tx.run("MATCH (n) DETACH DELETE n")
    for candidate in candidates:
        tx.run(
            """
            CREATE (c:Candidate {name: $name})
            WITH c
            UNWIND $skills AS skill
            MERGE (s:Skill {name: skill})
            MERGE (c)-[:HAS_SKILL]->(s)
            """,
            name=candidate["name"],
            skills=candidate["skills"]
        )

def fetch_candidates(tx):
    result = tx.run(
        """
        MATCH (c:Candidate)-[:HAS_SKILL]->(s:Skill)
        RETURN c.name AS name, collect(s.name) AS skills
        """
    )
    return [{"name": row["name"], "skills": row["skills"]} for row in result]

with driver.session() as session:
    session.execute_write(populate_data, candidates)
    all_candidates = session.execute_read(fetch_candidates)

# -------------------------------
# Match JD to candidates
# -------------------------------
results = []
print("🔍 Computing semantic similarity...")
for candidate in all_candidates:
    similarity = compute_similarity(job_description["required_skills"], candidate["skills"])
    results.append({"name": candidate["name"], "similarity_score": round(similarity, 3)})

results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)

# -------------------------------
# Output
# -------------------------------
print(f"\n📝 Job Title: {job_description['title']}")
print(f"🔍 Top {TOP_N} Matching Candidates:\n")

for res in results[:TOP_N]:
    print(f"{res['name']}: Similarity Score = {res['similarity_score']}")

driver.close()
