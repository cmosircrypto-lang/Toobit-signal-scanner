import requests
import os

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SAVED_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

response = requests.get(
    url,
    timeout=15
)

data = response.json()

print("Telegram status:", response.status_code)

if data.get("ok"):

    found = False

    for item in data.get("result", []):

        message = item.get("message", {})
        chat = message.get("chat", {})
        chat_id = str(chat.get("id", ""))

        if chat_id == str(SAVED_CHAT_ID):
            found = True

            print("CHAT ID CHECK: MATCH")
            print("Telegram chat type:", chat.get("type"))
            print("Telegram username:", chat.get("username"))

    if not found:
        print("CHAT ID CHECK: MISMATCH")
        print("The GitHub TELEGRAM_CHAT_ID does not match the chat that messaged the bot.")

else:

    print("Telegram API ERROR")
    print(data)