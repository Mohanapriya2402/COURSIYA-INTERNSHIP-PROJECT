from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load files
with open("resume.txt","r") as f:
    resume = f.read()

with open("job.txt","r") as f:
    job = f.read()

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([resume, job])

# Calculate similarity
similarity = cosine_similarity(vectors[0:1], vectors[1:2])

print("Match Score:", round(similarity[0][0]*100,2), "%")