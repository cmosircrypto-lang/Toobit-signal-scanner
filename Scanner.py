import requests
import os

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

response = requests.get(url, timeout=15)

print("Telegram status:", response.status_code)

data = response.json()

print("Telegram response:")

if data.get("ok"):
    for item in data.get("result", []):
        message = item.get("message", {})
        chat = message.get("chat", {})

        print(
            "CHAT_ID:",
            chat.get("id"),
            "| TYPE:",
            chat.get("type"),
            "| NAME:",
            chat.get("first_name", ""),
            "| USERNAME:",
            chat.get("username", "")
        )
else:
    print(data)