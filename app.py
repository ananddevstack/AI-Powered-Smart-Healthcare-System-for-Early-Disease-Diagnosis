from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # allow frontend requests

# -------------------------
# Health Check (IMPORTANT)
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# -------------------------
# Home Route
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend running successfully"}), 200


# -------------------------
# Signup API (Demo)
# -------------------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    # (Later you will save to DB here)
    return jsonify({
        "message": "Signup successful",
        "user": {
            "name": name,
            "email": email
        }
    }), 200


# -------------------------
# Login API (Demo)
# -------------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    # (Later you will validate from DB here)
    if email == "anand@gmail.com" and password == "123456":
        return jsonify({
            "message": "Login successful",
            "token": "dummy-jwt-token"
        }), 200

    return jsonify({"error": "Invalid credentials"}), 401


# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
