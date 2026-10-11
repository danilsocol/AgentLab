`import requests
import json

# First API call with reasoning
response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer sk-or-v1-a2cef299fce0b7f48d246a1a980",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "inclusionai/ling-3.0-flash-vl:free",
    "messages": [
        {
          "role": "user",
          "content": "How many r's are in the word 'strawberry'?"
        }
      ],
    "reasoning": {"enabled": True}
  })
)

print(response)

# Extract the assistant message with reasoning_details
response = response.json()
response = response['choices'][0]['message']


print(response)

