import streamlit as st
import openai
import requests
import json
import logging
import os

# Constants for model selection and temperature
MODEL = "openai/gpt-4o"
FOLLOWUPMODEL = "openai/gpt-4o"
TEMPERATURE = 0.9
MAX_TOKENS = 4096  # Maximum number of tokens for the conversation history
TOKEN_MARGIN = 512  # Margin to ensure we don't exceed token limit with the model response

# Ensure the log file is created in the current directory
log_file_path = os.path.join(os.getcwd(), "log.txt")

# Initialize logging
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    filemode='a'  # Append to the log file each run
)

# Log the start of the script
logging.info("Script started.")

# Load OpenRouter API key from secrets
api_key = st.secrets["openrouter"]["api_key"]

# Set the OpenAI API key
openai.api_key = api_key

# Define OpenRouter API endpoint and headers
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "ai4christians.com",  # Optional
    "X-Title": "chatbot prototype",  # Optional
    "Content-Type": "application/json"
}

def log_and_return_response(response):
    response.raise_for_status()
    response_data = response.json()
    logging.info(f"Response data: {response_data}")
    if response_data and "choices" in response_data and response_data["choices"]:
        return response_data["choices"][0]["message"]["content"]
    else:
        logging.error("No valid response received.")
        return "Hello! How can I help you today? 😊"

def make_request(payload, stream=False):
    try:
        response = requests.post(OPENROUTER_API_URL, headers=HEADERS, data=json.dumps(payload), stream=stream)
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making request: {e}")
        return None

def generate_initial_message():
    initial_payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": "Generate a welcome message for the user in ten words or less. You can use Markdown and emojis 😊."}],
        "temperature": TEMPERATURE,
        "stream": False
    }
    response = make_request(initial_payload)
    if response:
        return log_and_return_response(response)
    return "Hello! How can I help you today? 😊"

def generate_follow_up_questions(messages):
    truncated_messages = truncate_messages(messages)
    payload = {
        "model": MODEL,
        "messages": truncated_messages + [{"role": "system", "content": "Generate up to four follow-up questions based on the conversation from the tense that a user may want to ask next. You can use Markdown and emojis 😊. Return the questions in the following JSON format: {\"q1\": \"Question 1\", \"q2\": \"Question 2\", \"q3\": \"Question 3\", \"q4\": \"Question 4\"}"}],
        "temperature": TEMPERATURE,
        "stream": False
    }
    response = make_request(payload)
    if response:
        response_data = response.json()
        logging.info(f"Follow-up questions response data: {response_data}")
        if response_data and "choices" in response_data and response_data["choices"]:
            try:
                content = response_data["choices"][0]["message"]["content"]
                content = content.strip()
                if content.startswith("{") and content.endswith("}"):
                    questions_data = json.loads(content)
                    questions = [q for q in questions_data.values() if q.strip()]
                    if questions:  # Only return questions if there are valid ones
                        return questions
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"Error parsing follow-up questions: {e}")
    return []  # Return an empty list if there are no valid suggestions

def process_llm_response(messages):
    truncated_messages = truncate_messages(messages)
    payload = {
        "model": FOLLOWUPMODEL,
        "messages": truncated_messages + [{"role": "system", "content": "You can use Markdown and emojis 😊."}],
        "temperature": TEMPERATURE,
        "stream": True
    }
    logging.info(f"Payload for LLM: {payload}")

    response = make_request(payload, stream=True)
    full_response = ""
    if response:
        try:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    logging.debug(f"Received line: {decoded_line}")
                    if decoded_line == "data: [DONE]":
                        logging.info("Stream completed.")
                        break
                    if decoded_line.startswith("data: "):
                        try:
                            data = json.loads(decoded_line[len("data: "):])
                            if "choices" in data and data["choices"]:
                                choice = data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    full_response += choice["delta"]["content"]
                                    st.session_state.response_container.markdown(full_response)
                        except json.JSONDecodeError as e:
                            logging.error(f"JSON decoding error: {e}")
                            logging.error(f"Failed to decode line: {decoded_line}")
                            logging.error(f"Raw response content: {decoded_line}")
        except requests.exceptions.RequestException as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            full_response = "An error occurred while processing your request."
            logging.error(error_message)

    logging.info(f"Final response: {full_response}")
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Generate follow-up questions AFTER the response
    follow_up_questions = generate_follow_up_questions(st.session_state.messages)
    st.session_state.suggestions = follow_up_questions

def truncate_messages(messages):
    """Truncate the messages to fit within the MAX_TOKENS limit."""
    total_tokens = 0
    truncated_messages = []

    for message in reversed(messages):
        message_tokens = len(message['content'].split())  # Simple token estimation, for better accuracy use a tokenizer
        if total_tokens + message_tokens + TOKEN_MARGIN <= MAX_TOKENS:
            truncated_messages.insert(0, message)
            total_tokens += message_tokens
        else:
            break

    return truncated_messages

# Streamlit app
st.title("Streamlit Chatbot with OpenRouter and OpenAI")

# Initialize session state for message history and follow-up suggestions
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "response_container" not in st.session_state:
    st.session_state.response_container = None
if "follow_up_clicked" not in st.session_state:
    st.session_state.follow_up_clicked = False

# Check for a session start and display the welcome message
if "init" not in st.session_state:
    st.session_state["init"] = True
    welcome_message = generate_initial_message()
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process follow-up suggestions as buttons
if not st.session_state.follow_up_clicked and st.session_state.suggestions:
    st.write("You might want to ask:")
    cols = st.columns(len(st.session_state.suggestions))
    for i, suggestion in enumerate(st.session_state.suggestions):
        if cols[i].button(suggestion):
            st.session_state.messages.append({"role": "user", "content": suggestion})
            st.session_state.follow_up_clicked = True  # Set flag to hide buttons
            st.session_state.suggestions = []  # Clear suggestions immediately
            st.experimental_rerun()  # Refresh the UI to hide buttons

# Handle follow-up button click
if st.session_state.follow_up_clicked:
    logging.info("Follow-up button clicked. Generating response.")
    with st.chat_message("assistant"):
        st.session_state.response_container = st.empty()
    process_llm_response(st.session_state.messages)
    st.session_state.follow_up_clicked = False  # Reset flag after processing

# User input
if user_input := st.chat_input("Type your message here..."):
    # Add user's message to the session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    logging.info(f"User input: {user_input}")

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        st.session_state.response_container = st.empty()
    process_llm_response(st.session_state.messages)

# Display follow-up suggestions
if st.session_state.suggestions:
    st.write("You might want to ask:")
    cols = st.columns(len(st.session_state.suggestions))
    for i, suggestion in enumerate(st.session_state.suggestions):
        if cols[i].button(suggestion):
            st.session_state.messages.append({"role": "user", "content": suggestion})
            st.session_state.suggestions = []  # Clear suggestions after use
            st.experimental_rerun()  # Refresh the UI to remove buttons

logging.info("Script ended.")
