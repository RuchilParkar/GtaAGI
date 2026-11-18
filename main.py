import babyagi
import os


app = babyagi.create_app('/dashboard')

# Add keys to enable automated descriptions, function calling, and self-building functions.
gemini_key = os.environ.get('GEMINI_API_KEY')
if gemini_key:
    babyagi.add_key_wrapper('gemini_api_key', gemini_key)

openai_key = os.environ.get('OPENAI_API_KEY')
if openai_key:
    babyagi.add_key_wrapper('openai_api_key', openai_key)


@app.route('/')
def home():
    return f"Welcome to the main app. Visit <a href=\"/dashboard\">/dashboard</a> for GtaAGI dashboard."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
