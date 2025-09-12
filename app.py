from flask import Flask, request, render_template, jsonify
from data_loader import loadDataset
from vector_store import createIndex, searchContext
from chatbot import generateResponse

app = Flask(__name__)

data = {"index": None, "chunks": None}


@app.route('/', methods=['GET', 'POST'])
def home():
    global data
    if request.method == 'POST':
        if 'dataset' in request.files:
            dataset = request.files['dataset']
            content = dataset.read().decode('utf-8')
            chunks = loadDataset(content)
            index = createIndex(chunks)
            data = {"index": index, "chunks": chunks}
            return render_template('index.html', message="✅ Dataset Loaded!")
        elif 'query' in request.form:
            query = request.form['query']
            context = searchContext(query, data)
            answer = generateResponse(query, context)
            return jsonify({"response": answer})
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
