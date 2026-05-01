import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "False").lower() == "true"

print("hello from llm_client.py")

if LOCAL_MODEL:
    response = completion(
        model="ollama/mistral:latest",
        messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
        api_base="http://localhost:11434"
    )
else:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    print("Using Gemini API Key:", GEMINI_API_KEY)  # Debugging line to check if the API key is loaded
    response = completion(
        model="gemini/gemini-2.5-flash",  # Using Gemini Pro model
        messages=[{ "content": "respond in 20 words. who are you?","role": "user"}],
        api_key=GEMINI_API_KEY
    )
# Parse and print the response based on the model used
if LOCAL_MODEL:
    # Ollama response parsing (assuming a similar structure to OpenAI chat completion)
    message_content = response["choices"][0]["message"]["content"]
else:
    # Gemini response parsing
    message_content = response["choices"][0]["message"]["content"]

print("Response:", message_content)

