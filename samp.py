from google import genai
from decouple import config

client = genai.Client(api_key=config("API_KEY"))
try:
    result = client.models.embed_content(
        model="gemini-embedding-exp-03-07", contents="Test embedding"
    )
    print("Embedding successful:", result)
except Exception as e:
    print("Error:", e)
