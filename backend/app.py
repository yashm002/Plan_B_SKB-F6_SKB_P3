from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/submit-crop-data', methods=['POST'])
def submit_crop_data():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['crop_type', 'temperature', 'rainfall', 'soil_type', 'irrigation', 'location']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Process the data (placeholder for now)
        response_data = {
            'message': 'Crop data submitted successfully',
            'submitted_data': data,
            'status': 'success'
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Backend is running'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
