#!/usr/bin/env python3
"""Debug script to test upload functionality"""

import os
import sys
import tempfile
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_upload():
    """Test the upload functionality directly"""
    try:
        # Use the debug app which has non-blocking upload behavior
        from app_debug import app
        
        # Create a test client
        app.config['TESTING'] = True
        client = app.test_client()
        
        # Create a test file
        test_content = "This is a test document for uploading. It contains some sample text to test the RAG pipeline."
        
    # Test the upload
        print("Testing file upload...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file_path = f.name
        
        try:
            with open(temp_file_path, 'rb') as test_file:
                response = client.post('/upload', 
                    data={
                        'file': (test_file, 'test_document.txt'),
                        'datasetName': 'test_dataset'
                    },
                    content_type='multipart/form-data'
                )
            
            print(f"Response status: {response.status_code}")
            print(f"Response data: {response.get_json()}")
            
            if response.status_code in (200, 202):
                print("✅ Upload accepted (background build started)")
            else:
                print("❌ Upload failed!")
            
            # poll for status
            from time import sleep, time
            start = time()
            while time() - start < 60:  # wait up to 60s
                sleep(1)
                list_res = client.get('/list_datasets')
                ds = list_res.get_json()
                print('Datasets:', ds)
                for d in ds.get('datasets', []):
                    if d.get('name') == 'test_dataset':
                        if d.get('status') == 'ready':
                            print('✅ Dataset build complete')
                            return
                        if d.get('status') == 'error':
                            print('❌ Dataset build failed:', d.get('error'))
                            return
            print('⏳ Dataset build did not finish within 60s')
                
        finally:
            os.unlink(temp_file_path)
            
    except Exception as e:
        print(f"❌ Error during upload test: {str(e)}")
        import traceback
        traceback.print_exc()


def test_large_upload(size_mb=10):
    """Test uploading a large synthetic file (size_mb) to exercise streaming build."""
    try:
        from app_debug import app
        app.config['TESTING'] = True
        client = app.test_client()

        print(f"Creating synthetic file of ~{size_mb} MB...")
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            chunk = (b"word " * 1024)  # ~5KB per write
            writes = int(size_mb * 1024 * 1024 / len(chunk))
            for _ in range(writes):
                f.write(chunk)
            temp_large = f.name

        try:
            with open(temp_large, 'rb') as lf:
                response = client.post('/upload', data={ 'file': (lf, 'large_test.txt'), 'datasetName': 'large_dataset' }, content_type='multipart/form-data')
            print('Large upload response:', response.status_code, response.get_json())
            if response.status_code in (200, 202):
                print('✅ Large upload accepted')
            else:
                print('❌ Large upload failed')

            # poll for ready status (longer timeout)
            import time
            start = time.time()
            while time.time() - start < 180:  # 3 minutes
                time.sleep(2)
                ds = client.get('/list_datasets').get_json()
                print('Datasets:', ds)
                for d in ds.get('datasets', []):
                    if d.get('name') == 'large_dataset':
                        if d.get('status') == 'ready':
                            print('✅ Large dataset build complete')
                            return
                        if d.get('status') == 'error':
                            print('❌ Large dataset build error:', d.get('error'))
                            return
            print('⏳ Large dataset build did not finish in 3 minutes')
        finally:
            import os
            os.unlink(temp_large)

    except Exception as e:
        print('❌ Error during large upload test:', e)
        import traceback; traceback.print_exc()

def test_dependencies():
    """Test if all required dependencies are available"""
    print("Testing dependencies...")
    
    try:
        import flask
        print("✅ Flask available")
    except ImportError as e:
        print(f"❌ Flask not available: {e}")
    
    try:
        import sentence_transformers
        print("✅ sentence-transformers available")
    except ImportError as e:
        print(f"❌ sentence-transformers not available: {e}")
    
    try:
        import faiss
        print("✅ faiss available")
    except ImportError as e:
        print(f"❌ faiss not available: {e}")
    
    try:
        import torch
        print("✅ torch available")
    except ImportError as e:
        print(f"❌ torch not available: {e}")

if __name__ == '__main__':
    print("=== Debugging Upload Feature ===")
    test_dependencies()
    print("\n" + "="*50 + "\n")
    test_upload()