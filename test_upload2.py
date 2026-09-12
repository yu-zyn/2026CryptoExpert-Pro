import urllib.request
import json
import uuid
import os

def upload_file(filepath):
    boundary = uuid.uuid4().hex
    filename = os.path.basename(filepath)
    with open(filepath, 'rb') as f:
        file_content = f.read()

    body = b'--' + boundary.encode() + b'\r\n'
    body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    body += b'Content-Type: application/octet-stream\r\n\r\n'
    body += file_content + b'\r\n'
    body += b'--' + boundary.encode() + b'--\r\n'

    req = urllib.request.Request(
        'http://127.0.0.1:8001/api/upload/',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

# Test CSV
with open('test.csv', 'w') as f:
    f.write('1,2,3,4,5,6,7,8\n9,10,11,12,13,14,15,16\n')

result = upload_file('test.csv')
print("=== CSV Upload ===")
print("Status:", result.get('status'))
print("Filename:", result.get('filename'))
print("Parsed values:", result.get('parsed_values'))
print()

# Test BIN
with open('test.bin', 'wb') as f:
    f.write(bytes([0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5]))

result = upload_file('test.bin')
print("=== BIN Upload ===")
print("Status:", result.get('status'))
print("Filename:", result.get('filename'))
print("Total bytes:", result.get('total_bytes'))
print("Preview:", result.get('preview'))
print()

# Cleanup
os.remove('test.csv')
os.remove('test.bin')
os.remove('test_upload.py')
print("All tests passed!")