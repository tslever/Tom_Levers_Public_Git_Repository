from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route("/", methods = ['POST'])
def main():
    try:
        JSON_object_from_POST_request_body = request.json
        if JSON_object_from_POST_request_body['action'] == 'Click me to get started.':
            JSON_object_representing_body_of_response = {"list_of_possible_actions": ["Player Red, place your first settlement."]}
        else:
            JSON_object_representing_body_of_response = {"list_of_possible_actions": ["TODO"]}
        response = jsonify(JSON_object_representing_body_of_response)
        return response
    except Exception as e:
        print(e)

if __name__ == '__main__':
    app.run(debug = True)