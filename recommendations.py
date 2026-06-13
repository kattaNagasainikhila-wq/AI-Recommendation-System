import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("courses.csv")

name = input("Enter your name: ")
level=input("Enter level(Beginner/Intermediate/Advanced):")
interest = input("Enter your interests: ")
print(
    f"\nHello {name}"
)

print(
    f"Top Recommendations for {level} Learner\n"
)

documents = [interest] + data["Skills"].tolist()

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)
similarity = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
)

scores = similarity.flatten()

data["Score"] = scores


print("TF-IDF Created Successfully")
recommendations = data.sort_values(
    by="Score",
    ascending=False
)

print("\nTop 5 Recommendations:\n")

for index,row in recommendations.head(5).iterrows():

    percentage = round(
        row["Score"] * 100,
        2
    )

    print(
        f"{row['Course']} -> {percentage}% Match"
    )