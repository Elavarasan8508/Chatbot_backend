import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
import re

# ---------------------- INITIALIZATION ---------------------- #
load_dotenv()

# ---------------------- LOGGING CONFIG ---------------------- #
def setup_logger():
    """Configure logging with both console and file output"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        'chatbot.log',
        maxBytes=5*1024*1024,  # 5MB per file
        backupCount=3,         # Keep 3 backup files
        encoding='utf-8'
    )
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

# ---------------------- FLASK APP SETUP ---------------------- #
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------------- RATE LIMITING ---------------------- #
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",
    default_limits=["200 per day", "50 per hour"]
)

# ---------------------- MONGODB CONNECTION ---------------------- #
def get_mongo_client():
    """Establish MongoDB connection with error handling"""
    MONGO_URI = os.getenv("MONGODB_URI")
    if not MONGO_URI:
        logger.error("MONGODB_URI not found in environment variables")
        return None

    try:
        client = MongoClient(
            MONGO_URI,
            tlsCAFile=certifi.where(),
            connectTimeoutMS=5000,
            socketTimeoutMS=30000,
            serverSelectionTimeoutMS=5000,
            retryWrites=True
        )
        # Test connection
        client.admin.command('ping')
        logger.info("MongoDB connection successful")
        return client
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        return None

# Initialize MongoDB connection with proper None checks
mongo_client = get_mongo_client()
db = mongo_client.get_database("MindfulMateDB") if mongo_client is not None else None
user_collection = db.get_collection("users") if db is not None else None

# ---------------------- GROQ CHAT CONFIGURATION ---------------------- #
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("GROQ_API_KEY environment variable is required")

llm = ChatGroq(
    temperature=0.5,
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192",
    max_tokens=1024
)

# ---------------------- PROMPT ENGINEERING ---------------------- #
system_prompt = SystemMessage(content=f"""
You are MindfulMate, a compassionate mental health support assistant. Your role is to:

1. Provide empathetic, non-judgmental support
2. Offer evidence-based mental health information
3. Suggest coping strategies and mindfulness techniques
4. Recognize when to recommend professional help
5. Maintain appropriate boundaries

Guidelines:
- Always respond with care and validation first
- Use simple, clear language
- Ask clarifying questions when needed
- Never make diagnoses
- For crisis situations, provide hotline numbers

Current date: {datetime.now().strftime("%Y-%m-%d")}
""")

prompt_template = ChatPromptTemplate.from_messages([system_prompt, ("human", "{input}")])

# ---------------------- CRISIS DETECTION ---------------------- #
CRISIS_RESOURCES = {
    "US": "National Suicide Prevention Lifeline: 1-800-273-8255",
    "UK": "Samaritans: 116 123",
    "International": "Find a helpline: https://www.befrienders.org"
}

def is_crisis_message(message):
    crisis_keywords = [
        r'\bkill myself\b', r'\bend it all\b', r'\bsuicide\b',
        r'\bwant to die\b', r'\bharm myself\b', r'\bno reason to live\b'
    ]
    return any(re.search(keyword, message, re.IGNORECASE) for keyword in crisis_keywords)

# ---------------------- DATABASE HELPERS ---------------------- #
def get_user_history(email):
    """Retrieve user conversation history"""
    if user_collection is None:
        logger.warning("MongoDB not available - using empty history")
        return []

    try:
        user_data = user_collection.find_one({"email": email})
        return user_data.get("history", []) if user_data else []
    except Exception as e:
        logger.error(f"Error fetching user history: {str(e)}")
        return []

def update_user_history(email, user_input, bot_response):
    """Update user conversation history"""
    if user_collection is None:
        logger.warning("MongoDB not available - skipping history update")
        return False

    try:
        new_entry = {
            "user_input": user_input,
            "bot_response": bot_response,
            "timestamp": datetime.now()
        }

        user_collection.update_one(
            {"email": email},
            {"$push": {"history": new_entry}},
            upsert=True
        )
        return True
    except Exception as e:
        logger.error(f"Error updating user history: {str(e)}")
        return False

# ---------------------- API ROUTES ---------------------- #
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    db_status = "connected" if mongo_client is not None and mongo_client.admin.command('ping') else "disconnected"
    return jsonify({
        "status": "healthy",
        "database": db_status,
        "timestamp": datetime.now().isoformat(),
        "service": "MindfulMate Mental Health Support"
    })

@app.route("/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    """Main chat endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        logger.info(f"Received request from {request.remote_addr}")

        # Validate input
        if not data or "message" not in data:
            return jsonify({"error": "Message is required"}), 400

        user_email = data.get("email", "").strip()
        user_input = data["message"].strip()

        if not user_input:
            return jsonify({"error": "Message cannot be empty"}), 400

        # Crisis detection
        if is_crisis_message(user_input):
            crisis_response = (
                "I hear that you're in tremendous pain right now. Please reach out to:\n\n"
                + "\n".join(CRISIS_RESOURCES.values()) +
                "\n\nYour life matters, and there are people who want to help."
            )
            return jsonify({"answer": crisis_response})

        # Get conversation history
        conversation_history = get_user_history(user_email) if user_email else []

        # Generate response
        formatted_prompt = prompt_template.format_messages(
            input=f"History:\n{conversation_history[-3:]}\n\nNew message: {user_input}"
        )
        
        response = llm.invoke(formatted_prompt)
        bot_response = response.content

        # Ensure proper punctuation
        if not any(bot_response.rstrip().endswith(punc) for punc in ('?', '!', '.')):
            bot_response += " Would you like to share more?"

        # Update history
        if user_email:
            update_user_history(user_email, user_input, bot_response)

        return jsonify({
            "answer": bot_response,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Our service is temporarily unavailable. Please try again later.",
            "status": "error"
        }), 500

# ---------------------- MAIN ---------------------- #
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))  # Changed default port to 5001
    debug_mode = os.getenv("FLASK_ENV", "production") == "development"
    
    if mongo_client is not None:
        try:
            logger.info(f"MongoDB server info: {mongo_client.server_info()}")
        except Exception as e:
            logger.error(f"Failed to get MongoDB server info: {str(e)}")
    else:
        logger.warning("Running without MongoDB connection")
    
    logger.info(f"Starting server on port {port} (debug: {debug_mode})")
    try:
        app.run(host="0.0.0.0", port=port, debug=debug_mode)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise