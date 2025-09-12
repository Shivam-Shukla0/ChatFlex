# ChatFlex

This project is a chatbot application for the NYD Hackathon. It includes modules for data loading, vector storage, and a web interface.

## Features
- Chatbot functionality
- Data loader for text files
- Vector store for efficient retrieval
- Web interface using Flask
- **Fast large dataset uploads**: Uploads are streamed and processed in the background, so the app remains responsive even for files up to 10MB or more.

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`

## Usage Notes
- When uploading large datasets, you'll see a message that processing is happening in the background. You can continue using the app while your data is being indexed.

## Folder Structure
- `app.py`: Main application
- `chatbot.py`: Chatbot logic
- `data_loader.py`: Data loading utilities
- `vector_store.py`: Vector storage logic
- `requirements.txt`: Python dependencies
- `sample_data/`: Example data files
- `templates/`: HTML templates

## Contact for any issues with this
lucifer84670@gmail.com
msharma42005@gmail.com
