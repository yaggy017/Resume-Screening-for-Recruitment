import os
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# Download NLTK resources (if not already)
nltk.download('stopwords')

# ------------------------------
# TEXT PREPROCESSING FUNCTION
# ------------------------------
def preprocess(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

# ------------------------------
# LOAD ALL RESUMES
# ------------------------------
def load_resumes(folder_path):
    resumes = []
    file_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                resumes.append(file.read())
                file_names.append(filename)
    return resumes, file_names

# ------------------------------
# MAIN FUNCTION
# ------------------------------
def main():
    # Load the Job Description
    with open("job_description.txt", "r", encoding='utf-8') as file:
        job_desc = file.read()

    # Load the Resumes
    resumes, file_names = load_resumes("resumes")  # Make sure 'resumes' folder exists

    # Preprocess everything
    documents = [preprocess(job_desc)] + [preprocess(resume) for resume in resumes]

    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)

    # Compute Cosine Similarity (JD vs Resumes)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Threshold to decide if selected
    threshold = 0.2  # You can tweak this!

    # Results DataFrame
    results = pd.DataFrame({
        "Resume": file_names,
        "Similarity Score": similarities,
        "Status": ["✅ Selected" if score >= threshold else "❌ Rejected" for score in similarities]
    }).sort_values(by="Similarity Score", ascending=False)

    # Print Output
    print("\n📄 Resume Screening Results:\n")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()
