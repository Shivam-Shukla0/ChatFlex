# app.py - Flask web app (fast upload + background RAG build)
import os
import threading
from flask import Flask, request, render_template, jsonify
import logging
from rag_pipeline import RAGIndex
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

os.environ['USE_LOCAL_GENERATOR'] = 'false'
from generator import Generator

app = Flask(__name__)
gen = Generator()

DATASETS = {}
CURRENT = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        MAX_BYTES = int(os.getenv('MAX_UPLOAD_BYTES', 60 * 1024 * 1024))
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({'status': 'error', 'message': 'no file uploaded'}), 400
        name = request.form.get('datasetName') or file.filename
        chunk_size = int(os.getenv('CHUNK_SIZE', 300))
        if hasattr(file, 'content_length') and file.content_length is not None and file.content_length > MAX_BYTES:
            return jsonify({'status': 'error', 'message': f'file too large (limit {MAX_BYTES} bytes)'}), 413

        os.makedirs('uploads', exist_ok=True)
        safe_name = name.replace(' ', '_')
        import time
        timestamp = int(time.time())
        saved_path = os.path.join('uploads', f"{timestamp}_{safe_name}")
        file.stream.seek(0)
        with open(saved_path, 'wb') as out_f:
            while True:
                chunk = file.stream.read(8192)
                if not chunk:
                    break
                out_f.write(chunk)

        DATASETS[name] = {'status': 'building', 'index': None, 'error': None, 'light': None, 'chunks': None, 'file_path': saved_path}

        try:
            tmp = RAGIndex()
            with open(saved_path, 'rb') as rf:
                raw = rf.read()
            try:
                full_text = raw.decode('utf-8', errors='ignore')
            except Exception:
                full_text = raw.decode('latin-1', errors='ignore')
            chunks = tmp.chunk_text(full_text, chunk_size=chunk_size)
            DATASETS[name]['chunks'] = chunks
            if SKLEARN_AVAILABLE and len(chunks) > 0:
                vec = TfidfVectorizer().fit(chunks)
                mat = vec.transform(chunks)
                DATASETS[name]['light'] = {'vectorizer': vec, 'matrix': mat}
                DATASETS[name]['status'] = 'ready'
            else:
                DATASETS[name]['light'] = None
                DATASETS[name]['status'] = 'ready'
        except Exception:
            logger.exception('Failed to build light index')
            DATASETS[name]['status'] = 'error'
            DATASETS[name]['error'] = 'light index build failed'

        batch_size = int(os.getenv('EMBED_BATCH_SIZE', 128))
        def worker_stream_from_path(path, ds_name):
            r = RAGIndex()
            try:
                with open(path, 'rb') as pf:
                    r.build_from_fileobj(pf, chunk_size=chunk_size, batch_size=batch_size)
                DATASETS[ds_name]['index'] = r
                DATASETS[ds_name]['status'] = 'ready'
                DATASETS[ds_name]['error'] = None
                global CURRENT
                CURRENT = ds_name
                try:
                    os.makedirs('indexes', exist_ok=True)
                    r.save(os.path.join('indexes', f"{ds_name}.pkl"))
                except Exception:
                    logger.exception('Failed to save full index')
            except Exception:
                logger.exception(f"[worker-stream] Failed to build dataset '{ds_name}'")
                DATASETS[ds_name]['status'] = 'error'
                DATASETS[ds_name]['error'] = 'full index build failed'

        t = threading.Thread(target=worker_stream_from_path, args=(saved_path, name), daemon=True)
        t.start()

        return jsonify({'status': 'ok', 'message': f"Dataset '{name}' ready (lightweight)"}), 200

    except Exception as e:
        logger.exception('Upload failed')
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/list_datasets')
def list_datasets():
    ds = [{ 'name': name, 'status': v.get('status','unknown'), 'error': v.get('error') } for name, v in DATASETS.items()]
    return jsonify({'datasets': ds, 'current': CURRENT})


@app.route('/set_dataset', methods=['POST'])
def set_dataset():
    global CURRENT
    payload = request.json
    name = None
    if isinstance(payload, dict):
        name = payload.get('name')
        if isinstance(name, dict):
            name = name.get('name') or str(name)
    else:
        name = payload
    if not name:
        return jsonify({'status': 'error', 'msg': 'missing name'}), 400
    if name in DATASETS and DATASETS[name]['status'] == 'ready':
        CURRENT = name
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'error', 'msg': 'dataset not found or not ready'}), 404


def _load_saved_indexes():
    idx_dir = 'indexes'
    if not os.path.isdir(idx_dir):
        return
    for fname in os.listdir(idx_dir):
        if fname.endswith('.pkl'):
            path = os.path.join(idx_dir, fname)
            try:
                r = RAGIndex()
                r.load(path)
                name = os.path.splitext(fname)[0]
                DATASETS[name] = {'status': 'ready', 'index': r, 'error': None}
            except Exception:
                logger.exception(f"Failed to load saved index {path}")


_load_saved_indexes()


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query')
    top_k = int(os.getenv('TOP_K', 3))
    if CURRENT is None or CURRENT not in DATASETS or DATASETS[CURRENT]['status'] != 'ready':
        return jsonify({'error': 'no dataset loaded or dataset not ready'}), 400
    # prefer full index if available
    r = DATASETS[CURRENT].get('index')
    if r is not None:
        contexts = r.retrieve(query, top_k=top_k)
    else:
        # fallback to TF-IDF or substring search on chunks
        light = DATASETS[CURRENT].get('light')
        chunks = DATASETS[CURRENT].get('chunks') or []
        if light and SKLEARN_AVAILABLE:
            vec = light['vectorizer']
            mat = light['matrix']
            qv = vec.transform([query])
            import numpy as np
            sims = linear_kernel(qv, mat).flatten()
            idx = np.argsort(-sims)[:top_k]
            contexts = [chunks[i] for i in idx]
        else:
            # naive substring match
            matches = [c for c in chunks if query.lower() in c.lower()]
            contexts = matches[:top_k]
    context_text = "\n\n---\n\n".join(contexts)
    answer = gen.generate(query, context_text)
    return jsonify({'answer': answer, 'sources': contexts})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
# app.py - Flask web app for NYD2026_ChatFlex
import os
from flask import Flask, request, render_template, jsonify
import logging
from rag_pipeline import RAGIndex
from generator import Generator
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
gen = Generator()

DATASETS = {}  # name -> RAGIndex
CURRENT = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global DATASETS, CURRENT
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({'status': 'error', 'message': 'no file uploaded'}), 400

        name = request.form.get('datasetName') or file.filename
        chunk_size = int(os.getenv('CHUNK_SIZE', 300))

        # read file safely
        raw = file.read()
        if not raw:
            return jsonify({'status': 'error', 'message': 'uploaded file is empty'}), 400
        try:
            text = raw.decode('utf-8', errors='ignore')
        except Exception:
            # fallback: try latin-1
            text = raw.decode('latin-1', errors='ignore')

        r = RAGIndex()
        r.build([text], chunk_size=chunk_size)
        DATASETS[name] = r
        CURRENT = name
        return jsonify({'status': 'ok', 'message': f"Dataset '{name}' loaded", 'dataset': name})
    except Exception as e:
        # Log full exception for debugging
        logging.exception('Upload failed')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/list_datasets')
def list_datasets():
    return jsonify({'datasets': list(DATASETS.keys()), 'current': CURRENT})

@app.route('/set_dataset', methods=['POST'])
def set_dataset():
    global CURRENT
    name = request.json.get('name')
    if name in DATASETS:
        CURRENT = name
        return jsonify({'status':'ok'})
    return jsonify({'status':'error','msg':'dataset not found'}),404

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query')
    top_k = int(os.getenv('TOP_K', 3))
    if CURRENT is None or CURRENT not in DATASETS:
        return jsonify({'error':'no dataset loaded'}),400
    r = DATASETS[CURRENT]
    contexts = r.retrieve(query, top_k=top_k)
    context_text = "\n\n---\n\n".join(contexts)
    answer = gen.generate(query, context_text)
    return jsonify({'answer': answer, 'sources': contexts})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
