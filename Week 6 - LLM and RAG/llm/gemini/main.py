from google import genai
from dotenv import load_dotenv
load_dotenv()


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hãy giải thích nguyên lý hoạt động của động cơ điện bằng ngôn ngữ dễ hiểu cho học sinh cấp 2, sau đó viết lại bằng phong cách hài hước như một đoạn stand-up comedy, cuối cùng tóm tắt lại bằng 3 gạch đầu dòng chỉ gồm emoji."
)
print(response.text)