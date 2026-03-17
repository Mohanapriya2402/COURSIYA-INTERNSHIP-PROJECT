import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Job description
job_description = """
Looking for a Python developer with experience in machine learning,
data analysis, and NLP.
"""

# Sample resumes
resumes = [
    "Python developer with machine learning experience and data science knowledge.",
    "Web developer skilled in HTML CSS JavaScript.",
    "Data scientist with NLP machine learning Python experience."
]

# Generate embeddings
job_embedding = model.encode([job_description])
resume_embeddings = model.encode(resumes)

# Calculate similarity
scores = cosine_similarity(job_embedding, resume_embeddings)

# Rank candidates
ranking = sorted(list(enumerate(scores[0])), key=lambda x: x[1], reverse=True)

# Print results
print("Candidate Ranking:\n")

for rank, (index, score) in enumerate(ranking):
    print(f"Rank {rank+1}: Resume {index+1} - Score: {score:.2f}")