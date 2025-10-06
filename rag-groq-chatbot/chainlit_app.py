import chainlit as cl
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

@cl.on_chat_start
async def start():
    await cl.Message(content="👋 Hi! Upload PDFs using `/ingest` endpoint, then ask questions!").send()

@cl.on_message
async def on_message(message: str):
    res = requests.post(f"{API_URL}/chat", data={"question": message})
    if res.status_code == 200:
        await cl.Message(content=res.json()["answer"]).send()
    else:
        await cl.Message(content=f"Error: {res.text}").send()
