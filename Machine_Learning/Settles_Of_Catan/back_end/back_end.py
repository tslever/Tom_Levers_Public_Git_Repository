from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route("/", methods = ['POST'])
def main():
    JSON_object_from_POST_request_body = request.json
    response = jsonify(JSON_object_from_POST_request_body)
    return response

if __name__ == '__main__':
    app.run(debug = True)