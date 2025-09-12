from flask import Flask, request, render_template, jsonify
import threading
from data_loader import loadDataset
from vector_store import createIndex, searchContext
from chatbot import generateResponse

app = Flask(__name__)

data = {"index": None, "chunks": None}


def process_dataset_stream(content, data):
    # Process file content in background thread
    chunks = loadDataset(content)
    index = createIndex(chunks)
    data["index"] = index
    data["chunks"] = chunks

@app.route('/', methods=['GET', 'POST'])
def home():
    global data
    if request.method == 'POST':
        if 'dataset' in request.files:
            dataset = request.files['dataset']
            # Read file content in main thread
            content = dataset.read().decode('utf-8')
            # Start background thread for processing
            thread = threading.Thread(target=process_dataset_stream, args=(content, data))
            thread.start()
            return render_template('index.html', message="⏳ Dataset is being processed in the background. You can continue using the app.")
        elif 'query' in request.form:
            query = request.form['query']
            context = searchContext(query, data)
            answer = generateResponse(query, context)
            return jsonify({"response": answer})
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
