from flask import Blueprint, request, jsonify
from services.chat_service import generate_response

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    response = generate_response(data["message"])

    return jsonify({
        "response": response
    })