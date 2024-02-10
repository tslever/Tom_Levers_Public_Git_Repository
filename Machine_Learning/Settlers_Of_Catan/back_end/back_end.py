from flask import Flask, jsonify, request
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route("/", methods = ['POST'])
def main():
    try:
        JSON_object_from_POST_request_body = request.json
        if JSON_object_from_POST_request_body['action'] == 'Player clicked action displayer with child \"Click me to get started.\".':
            JSON_object_representing_body_of_response = {
                "action_completed": JSON_object_from_POST_request_body['action'],
                "action_to_complete": "Player Red, place your first settlement.",
                "list_of_possible_actions": ["Player Red, place your first settlement."]
            }
        elif 'Player clicked action displayer with child \"Player Red, place your first settlement.\".' == JSON_object_from_POST_request_body['action']:
            JSON_object_representing_body_of_response = {
                "action_completed": JSON_object_from_POST_request_body['action'],
                "action_to_complete": f"Player Red, place your first settlement.",
                "list_of_possible_actions": ["Player Red, place your first settlement."]
            }
        elif re.search(r'When action to complete was \"Player Red, place your first settlement.\", player clicked base board at \((-?\d+), (-?\d+)\) relative to grid\.', JSON_object_from_POST_request_body['action']):
            JSON_object_representing_body_of_response = {
                "action_completed": JSON_object_from_POST_request_body['action'],
                "action_to_complete": "Player Orange, place your first settlement.",
                "list_of_possible_actions": ["Player Orange, place your first settlement."]
            }
        elif 'Player clicked action displayer with child \"Player Orange, place your first settlement.\".' == JSON_object_from_POST_request_body['action']:
            JSON_object_representing_body_of_response = {
                "action_completed": JSON_object_from_POST_request_body['action'],
                "action_to_complete": f"Player Orange, place your first settlement.",
                "list_of_possible_actions": ["Player Orange, place your first settlement."]
            }
        elif re.search(r'When action to complete was \"Player Orange, place your first settlement.\", player clicked base board at \((-?\d+), (-?\d+)\) relative to grid\.', JSON_object_from_POST_request_body['action']):
            JSON_object_representing_body_of_response = {
                "action_completed": JSON_object_from_POST_request_body['action'],
                "action_to_complete": "Player Blue, place your first settlement.",
                "list_of_possible_actions": ["Player Blue, place your first settlement."]
            }
        elif 'Player clicked action displayer with child \"Player Blue, place your first settlement.\".' == JSON_object_from_POST_request_body['action']:
            JSON_object_representing_body_of_response = {
                "action_completed": JSON_object_from_POST_request_body['action'],
                "action_to_complete": f"Player Blue, place your first settlement.",
                "list_of_possible_actions": ["Player Blue, place your first settlement."]
            }
        elif re.search(r'When action to complete was \"Player Blue, place your first settlement.\", player clicked base board at \((-?\d+), (-?\d+)\) relative to grid\.', JSON_object_from_POST_request_body['action']):
            JSON_object_representing_body_of_response = {
                "action_completed": JSON_object_from_POST_request_body['action'],
                "action_to_complete": "Player White, place your first settlement.",
                "list_of_possible_actions": ["Player White, place your first settlement."]
            }
        elif 'Player clicked action displayer with child \"Player White, place your first settlement.\".' == JSON_object_from_POST_request_body['action']:
            JSON_object_representing_body_of_response = {
                "action_completed": JSON_object_from_POST_request_body['action'],
                "action_to_complete": f"Player White, place your first settlement.",
                "list_of_possible_actions": ["Player White, place your first settlement."]
            }
        elif re.search(r'When action to complete was \"Player White, place your first settlement.\", player clicked base board at \((-?\d+), (-?\d+)\) relative to grid\.', JSON_object_from_POST_request_body['action']):
            JSON_object_representing_body_of_response = {
                "action_completed": JSON_object_from_POST_request_body['action'],
                "action_to_complete": "TODO",
                "list_of_possible_actions": ["TODO"]
            }
        elif 'Player clicked action displayer with child \"TODO\".' == JSON_object_from_POST_request_body['action']:
            JSON_object_representing_body_of_response = {
                "action_completed": JSON_object_from_POST_request_body['action'],
                "action_to_complete": "TODO",
                "list_of_possible_actions": ["TODO"],
            }
        response = jsonify(JSON_object_representing_body_of_response)
        return response
    except Exception as e:
        print(e)

if __name__ == '__main__':
    app.run(debug = True)