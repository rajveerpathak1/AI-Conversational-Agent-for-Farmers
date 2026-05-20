from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

from chatbot.chain import ask_question

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_message = data["message"]

    response = ask_question(user_message)

    return jsonify({
        "response": response["answer"],
        "sources": response["sources"]
    })

if __name__ == "__main__":
    app.run(debug=True)