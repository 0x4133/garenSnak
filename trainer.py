import json
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_knowledge_database(db_name):
    db_conn = sqlite3.connect(db_name)
    cursor = db_conn.cursor()
    cursor.execute("SELECT file_path, content FROM knowledge")
    knowledge_data = cursor.fetchall()
    db_conn.close()
    return knowledge_data

def train_model(knowledge_data):
    file_paths = [item[0] for item in knowledge_data]
    summaries = [item[1] for item in knowledge_data]

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(summaries)

    return file_paths, tfidf_matrix, vectorizer

def ask_question(question, file_paths, tfidf_matrix, vectorizer):
    # Vectorize the question
    question_vector = vectorizer.transform([question])

    # Calculate cosine similarity between the question and document summaries
    similarity_scores = cosine_similarity(question_vector, tfidf_matrix)

    # Get the index of the most similar document
    most_similar_index = similarity_scores.argmax()

    return file_paths[most_similar_index]

def main():
    db_name = 'knowledge.db'
    knowledge_data = load_knowledge_database(db_name)

    file_paths, tfidf_matrix, vectorizer = train_model(knowledge_data)

    while True:
        question = input("Ask a question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        most_relevant_file = ask_question(question, file_paths, tfidf_matrix, vectorizer)
        print("Most relevant file:", most_relevant_file)

        # Retrieve the full text content of the most relevant file from the JSON file
        with open('knowledge.json') as file:
            knowledge_json = json.load(file)
            file_content = knowledge_json.get(most_relevant_file, "")

        print("File content:")
        print(file_content)
        print()

if __name__ == '__main__':
    main()