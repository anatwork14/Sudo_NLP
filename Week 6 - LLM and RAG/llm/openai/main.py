import os
import json
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage, SystemMessage
from azure.core.credentials import AzureKeyCredential

# Load environment variables from .env file
load_dotenv()

# Fetch from environment
endpoint = "https://models.github.ai/inference"
api_key = os.getenv("AZURE_OPENAI_KEY")
model = "gpt-4.1"

# Initialize client
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key),
)

# Make request
response = client.complete(
    model=model,
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Hãy giải thích nguyên lý hoạt động của động cơ điện bằng ngôn ngữ dễ hiểu cho học sinh cấp 2, sau đó viết lại bằng phong cách hài hước như một đoạn stand-up comedy, cuối cùng tóm tắt lại bằng 3 gạch đầu dòng chỉ gồm emoji."),
    ],
)

print(response.choices[0].message.content)
