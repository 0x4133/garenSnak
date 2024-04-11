# Knowledge Database Creation and Question-Answering

This project demonstrates how to create a knowledge database from PDF and text-based files, and then use that database to train an AI model for question-answering. The script processes the files, extracts the text content, and stores it in a SQLite database and a JSON file. It then uses the extracted summaries to train a TF-IDF vectorizer and performs cosine similarity to find the most relevant document for a given question.

## Prerequisites

- Python 3.x
- Required libraries: `PyPDF2`, `colorama`, `scikit-learn`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/knowledge-database-qa.git
```

2. Install the required libraries:

```bash
pip install PyPDF2 colorama scikit-learn
```

## Usage

1. Place your PDF and text-based files (`.pdf`, `.conf`, `.txt`) in a directory of your choice.

2. Update the `directory` variable in the `main.py` script with the path to your directory containing the files.

3. Run the `main.py` script to create the knowledge database:

```bash
python main.py
```

The script will process the files, extract the text content, and store it in a SQLite database (`knowledge.db`) and a JSON file (`knowledge.json`).

4. Run the `qa.py` script to train the model and start the question-answering interface:

```bash
python qa.py
```

The script will load the knowledge database, train the TF-IDF vectorizer, and prompt you to ask questions. It will find the most relevant file based on the question and display the file path and its content.

5. Type 'quit' to exit the question-answering interface.

## Files

- `main.py`: Script to create the knowledge database from PDF and text-based files.
- `qa.py`: Script to train the model and perform question-answering based on the knowledge database.
- `knowledge.db`: SQLite database file containing the file paths and their corresponding text content.
- `knowledge.json`: JSON file containing the file paths and their corresponding text content.

## Customization

- You can modify the `directory` variable in `main.py` to specify the path to your directory containing the PDF and text-based files.
- Adjust the `num_agents` variable in `main.py` to change the number of worker agents used for parallel processing of files.
- Customize the `db_name` and `output_file` variables in `main.py` to change the names of the SQLite database and JSON output files.

## Limitations

- The script uses a simple TF-IDF vectorizer and cosine similarity for finding the most relevant document. More advanced techniques like semantic similarity, named entity recognition, or machine learning algorithms can be incorporated for improved accuracy and performance.
- The script assumes that the text content extracted from the files is suitable for training the model. Preprocessing steps like text cleaning, tokenization, and removing stop words may be necessary depending on the quality and format of the text data.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [PyPDF2](https://pypi.org/project/PyPDF2/) - Library for extracting text from PDF files.
- [scikit-learn](https://scikit-learn.org/) - Machine learning library for TF-IDF vectorization and cosine similarity calculation