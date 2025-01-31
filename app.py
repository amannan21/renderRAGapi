from flask import Flask, request, jsonify
import requests
import os
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from openai import OpenAI
from pinecone import Pinecone
import numpy as np

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')


app = Flask(__name__)
# Setup rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["50 per hour"]
)
# Setup logging
logging.basicConfig(level=logging.INFO)
# Bearer token-based authentication
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized'}), 401
        token = auth_header.split(" ")[1]
        if token != os.getenv('ACCESS_TOKEN'):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated


@app.route('/api/get_results', methods=['POST'])
@require_auth
@limiter.limit("10 per minute")
def get_results():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Text must be a non-empty string'}), 400

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)  # Pass API key directly
        embedding = client.embeddings.create(
            input=[text], 
            model="text-embedding-3-small", 
            dimensions=1536
        ).data[0].embedding
    except Exception as e:
        logging.error(f"OpenAI Error: {str(e)}")
        return jsonify({'error': f'OpenAI API Error: {str(e)}'}), 500
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("alloratesting")
        
        response = index.query(
            vector=embedding,
            top_k=10,
            include_metadata=True
        )
        
        results = response.matches
        if not results:
            return jsonify({'ids': [], 'message': 'No matches found'}), 200
        
        ids = ', '.join([result.id for result in results])
        return jsonify({'ids': ids})
    
    except Exception as e:
        logging.error(f"Pinecone Error: {str(e)}")
        return jsonify({'error': f'Pinecone API Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
