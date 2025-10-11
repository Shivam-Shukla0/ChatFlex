# test_upload.py - quick test for /upload route
import os
os.environ['USE_LOCAL_GENERATOR'] = 'false'
from app import app

client = app.test_client()

data = {
    'datasetName': 'test_dataset',
}

# create a small in-memory file
from io import BytesIO
file_data = BytesIO(b"This is a sample document for testing upload.\nIt has multiple lines.\n")
file_data.name = 'sample_upload.txt'

resp = client.post('/upload', data={'file': (file_data, file_data.name), 'datasetName': 'test_dataset'}, content_type='multipart/form-data')
print('status_code:', resp.status_code)
try:
    print('json:', resp.get_json())
except Exception as e:
    print('raw data:', resp.data)
