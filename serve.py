from waitress import serve
from app import app  # assuming your Flask app object is named 'app' in app.py

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8000)