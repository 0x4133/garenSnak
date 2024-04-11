import os
import json
import PyPDF2
from colorama import Fore, Style
from multiprocessing import Pool, Manager, freeze_support
import sqlite3

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        try:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except PyPDF2.errors.PdfReadError as e:
            print(Fore.RED + f"Error processing PDF file: {file_path}")
            print(f"Error message: {str(e)}" + Style.RESET_ALL)
            return ""

def extract_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
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
        cursor.execute("SELECT COUNT(*) FROM knowledge WHERE file_path = ?", (file_path,))
        result = cursor.fetchone()

        if result[0] > 0:
            # File already processed, skip it
            db_conn.close()
            return

        if file_extension == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_extension in ['.conf', '.txt']:
            text = extract_text_from_file(file_path)
        else:
            db_conn.close()
            return

        knowledge_database[file_path] = text

        # Save the file path and text content to the database
        cursor.execute("INSERT INTO knowledge (file_path, content) VALUES (?, ?)", (file_path, text))
        db_conn.commit()
        db_conn.close()

    except Exception as e:
        print(Fore.RED + f"Error processing file: {file_path}")
        print(f"Error message: {str(e)}" + Style.RESET_ALL)

def create_knowledge_database(directory, num_agents, db_name, output_file):
    # Create the SQLite database table if it doesn't exist
    db_conn = sqlite3.connect(db_name)
    cursor = db_conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS knowledge (file_path TEXT PRIMARY KEY, content TEXT)")
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
            result = pool.apply_async(process_file, args=(file_path, knowledge_database, db_name))
            results.append(result)

        for result in results:
            result.get()
            processed_files += 1
            progress = (processed_files / total_files) * 100
            print(Fore.BLUE + f"Processing files: {progress:.2f}% completed" + Style.RESET_ALL, end='\r')

    # Save the knowledge database to a JSON file
    with open(output_file, 'w') as file:
        json.dump(dict(knowledge_database), file, indent=4)

    return knowledge_database

if __name__ == '__main__':
    freeze_support()

    # Example usage
    directory = '/Users/aaron/Downloads/'
    num_agents = 4  # Number of worker agents to use
    db_name = 'knowledge.db'  # Name of the SQLite database file
    output_file = 'knowledge.json'  # Name of the output JSON file
    knowledge_db = create_knowledge_database(directory, num_agents, db_name, output_file)

    print(Fore.GREEN + "\nKnowledge database saved to", output_file)