# app_debug.py - Simplified Flask app for debugging
import os
import threading
from flask import Flask, request, render_template, jsonify
import logging
from rag_pipeline import RAGIndex
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Disable local generator for debugging (avoid heavy model load on startup)
os.environ['USE_LOCAL_GENERATOR'] = 'false'

from generator import Generator

app = Flask(__name__)
# generator stays lazy; we won't rely on it for upload/index build
gen = Generator()

# DATASETS now tracks status and index object: { name: {status:'building'|'ready'|'error', 'index': RAGIndex|None, 'error': str|null} }
DATASETS = {}
CURRENT = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_index_background(name: str, text: str, chunk_size: int):
    """Background worker to build RAG index for a dataset and update DATASETS status."""
    try:
        logger.info(f"[worker] Starting build for dataset '{name}'")
        r = RAGIndex()
        r.build([text], chunk_size=chunk_size)
        DATASETS[name]['index'] = r
        DATASETS[name]['status'] = 'ready'
        DATASETS[name]['error'] = None
        global CURRENT
        CURRENT = name
        logger.info(f"[worker] Dataset '{name}' build complete")
        # persist index for faster reloads
        try:
            os.makedirs('indexes', exist_ok=True)
            r.save(os.path.join('indexes', f"{name}.pkl"))
            logger.info(f"[worker] Saved index for '{name}' to disk")
        except Exception:
            logger.exception(f"[worker] Failed to save index for '{name}'")
    except Exception as e:
        logger.exception(f"[worker] Failed to build dataset '{name}'")
        DATASETS[name]['status'] = 'error'
        DATASETS[name]['error'] = str(e)


@app.route('/')
def index():
    logger.info("Index page requested")
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Accept upload, save input, and start index build in background to avoid blocking the request.

    Returns 202 Accepted while index is being built.
    """
    try:
        # enforce file size limit (env or 60MB default)
        MAX_BYTES = int(os.getenv('MAX_UPLOAD_BYTES', 60 * 1024 * 1024))
        file = request.files.get('file')
        logger.info(f"Upload endpoint called; file: {file.filename if file else 'None'}")

        if not file or file.filename == '':
            logger.warning("No file uploaded")
            return jsonify({'status': 'error', 'message': 'no file uploaded'}), 400

        name = request.form.get('datasetName') or file.filename
        chunk_size = int(os.getenv('CHUNK_SIZE', 300))

        # check size quickly if file provides content_length
        if hasattr(file, 'content_length') and file.content_length is not None:
            if file.content_length > MAX_BYTES:
                return jsonify({'status': 'error', 'message': f'file too large (limit {MAX_BYTES} bytes)'}), 413

        # save uploaded file to disk so multiple workers can reopen it
        os.makedirs('uploads', exist_ok=True)
        safe_name = name.replace(' ', '_')
        import time
        timestamp = int(time.time())
        saved_path = os.path.join('uploads', f"{timestamp}_{safe_name}")
        file.stream.seek(0)
        with open(saved_path, 'wb') as out_f:
            # stream copy
            while True:
                chunk = file.stream.read(8192)
                if not chunk:
                    break
                out_f.write(chunk)

        # Build a lightweight TF-IDF index synchronously so queries can be answered immediately
        DATASETS[name] = {'status': 'building', 'index': None, 'error': None, 'light': None, 'chunks': None, 'file_path': saved_path}

        try:
            # chunk file into text chunks (reuse RAGIndex.chunk_text logic via temporary RAGIndex)
            tmp = RAGIndex()
            # read as text
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
                logger.info(f"Built lightweight TF-IDF index for '{name}' ({len(chunks)} chunks)")
            else:
                # fallback: store chunks and mark ready; substring search will be used
                DATASETS[name]['light'] = None
                DATASETS[name]['status'] = 'ready'
                logger.info(f"Built lightweight fallback index for '{name}' ({len(chunks)} chunks)")
        except Exception:
            logger.exception('Failed to build light index')
            DATASETS[name]['status'] = 'error'
            DATASETS[name]['error'] = 'light index build failed'

        # start background thread to build full embedding index (non-blocking)
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
                logger.info(f"[worker-stream] Dataset '{ds_name}' full RAG build complete")
                # persist index
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

        logger.info(f"Upload completed for dataset '{name}'; lightweight index ready, full index building in background")
        return jsonify({'status': 'ok', 'message': f"Dataset '{name}' ready (lightweight)"}), 200

    except Exception as e:
        logger.exception('Upload failed')
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/list_datasets')
def list_datasets():
    # return dataset names with status and current
    ds = [{ 'name': name, 'status': v.get('status','unknown'), 'error': v.get('error') } for name, v in DATASETS.items()]
    return jsonify({'datasets': ds, 'current': CURRENT})


@app.route('/set_dataset', methods=['POST'])
def set_dataset():
    global CURRENT
    payload = request.json
    # accept either {name: 'xyz'} or {'name': 'xyz'} or raw string
    name = None
    if isinstance(payload, dict):
        name = payload.get('name')
        # sometimes client may send nested objects; coerce
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


# On startup, try to load saved indexes from disk for faster availability
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
                logger.info(f"Loaded saved index for dataset '{name}'")
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
    r = DATASETS[CURRENT]['index']
    contexts = r.retrieve(query, top_k=top_k)
    context_text = "\n\n---\n\n".join(contexts)
    answer = gen.generate(query, context_text)
    return jsonify({'answer': answer, 'sources': contexts})


if __name__ == '__main__':
    logger.info("Starting Flask app (debug) with non-blocking uploads...")
    app.run(debug=True, port=5000)