import os
import json
import PyPDF2
from colorama import Fore, Style
from multiprocessing import Pool, Manager, freeze_support
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    preprocessed_text = " ".join(lemmatized_tokens)

    return preprocessed_text


def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        try:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return preprocess_text(text)
        except PyPDF2.errors.PdfReadError as e:
            print(Fore.RED + f"Error processing PDF file: {file_path}")
            print(f"Error message: {str(e)}" + Style.RESET_ALL)
            return ""


def extract_text_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return preprocess_text(text)
    except UnicodeDecodeError as e:
        print(Fore.RED + f"Error processing file: {file_path}")
        print(f"Error message: {str(e)}" + Style.RESET_ALL)
        return ""


def process_file(file_path, knowledge_database, db_name):
    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        # Create a separate SQLite connection for each worker process
        db_conn = sqlite3.connect(db_name)
        cursor = db_conn.cursor()

        # Check if the file has already been processed
        cursor.execute(
            "SELECT COUNT(*) FROM knowledge WHERE file_path = ?", (file_path,)
        )
        result = cursor.fetchone()

        if result[0] > 0:
            # File already processed, skip it
            db_conn.close()
            return

        if file_extension == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif file_extension in [".conf", ".txt"]:
            text = extract_text_from_file(file_path)
        else:
            db_conn.close()
            return

        knowledge_database[file_path] = text

        # Save the file path and text content to the database
        cursor.execute(
            "INSERT INTO knowledge (file_path, content) VALUES (?, ?)",
            (file_path, text),
        )
        db_conn.commit()
        db_conn.close()

    except Exception as e:
        print(Fore.RED + f"Error processing file: {file_path}")
        print(f"Error message: {str(e)}" + Style.RESET_ALL)


def create_knowledge_database(directory, num_agents, db_name, output_file):
    # Create the SQLite database table if it doesn't exist
    db_conn = sqlite3.connect(db_name)
    cursor = db_conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS knowledge (file_path TEXT PRIMARY KEY, content TEXT)"
    )
    db_conn.close()

    manager = Manager()
    knowledge_database = manager.dict()
    file_list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    total_files = len(file_list)
    processed_files = 0

    with Pool(processes=num_agents) as pool:
        results = []
        for file_path in file_list:
            result = pool.apply_async(
                process_file, args=(file_path, knowledge_database, db_name)
            )
            results.append(result)

            processed_files += 1
            progress = (processed_files / total_files) * 100
            print(
                Fore.BLUE
                + f"Processing files: {progress:.2f}% completed"
                + Style.RESET_ALL,
                end="\r",
            )

        # Wait for all the results to complete
        for result in results:
            result.get()

    # Save the final knowledge database to a JSON file
    with open(output_file, "w") as file:
        json.dump(dict(knowledge_database), file, indent=4)

    return knowledge_database


def load_knowledge_database(knowledge_file):
    with open(knowledge_file) as file:
        knowledge_json = json.load(file)
    return list(knowledge_json.items())


def train_model(knowledge_data):
    file_paths = [item[0] for item in knowledge_data]
    summaries = [item[1] for item in knowledge_data]

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    try:
        tfidf_matrix = vectorizer.fit_transform(summaries)
        if tfidf_matrix.shape[1] == 0:
            raise ValueError("Empty vocabulary")
    except ValueError as e:
        print(Fore.RED + "Error training the model: " + str(e))
        print("Skipping question-answering for this checkpoint." + Style.RESET_ALL)
        return None, None, None

    return file_paths, tfidf_matrix, vectorizer


def ask_question(question, file_paths, tfidf_matrix, vectorizer):
    # Preprocess the question
    preprocessed_question = preprocess_text(question)

    # Vectorize the question
    question_vector = vectorizer.transform([preprocessed_question])

    # Calculate cosine similarity between the question and document summaries
    similarity_scores = cosine_similarity(question_vector, tfidf_matrix)

    # Get the index of the most similar document
    most_similar_index = similarity_scores.argmax()

    return file_paths[most_similar_index]


def ask_questions(knowledge_file):
    knowledge_data = load_knowledge_database(knowledge_file)
    file_paths, tfidf_matrix, vectorizer = train_model(knowledge_data)

    if file_paths is None or tfidf_matrix is None or vectorizer is None:
        return

    while True:
        question = input(
            "Ask a question (or type 'quit' to continue processing files): "
        )
        if question.lower() == "quit":
            break

        most_relevant_file = ask_question(
            question, file_paths, tfidf_matrix, vectorizer
        )
        print("Most relevant file:", most_relevant_file)

        # Retrieve the full text content of the most relevant file from the JSON file
        with open(knowledge_file) as file:
            knowledge_json = json.load(file)
            file_content = knowledge_json.get(most_relevant_file, "")

        print("File content:")
        print(file_content)
        print()


if __name__ == "__main__":
    freeze_support()

    # Example usage
    directory = "/Users/"
    num_agents = 4  # Number of worker agents to use
    db_name = "knowledge.db"  # Name of the SQLite database file
    output_file = "knowledge.json"  # Name of the output JSON file
    knowledge_db = create_knowledge_database(
        directory, num_agents, db_name, output_file
    )

    print(Fore.GREEN + "\nKnowledge database saved to", output_file)

    while True:
        user_input = input(
            Fore.YELLOW
            + "Press 'p' to pause and ask questions, or any other key to stop: "
            + Style.RESET_ALL
        )
        if user_input.lower() == "p":
            ask_questions(output_file)
        else:
            break
