import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from ai_agent import AgronomistAgent
import uvicorn

app = FastAPI()
agent = AgronomistAgent()

class ChatQuery(BaseModel):
    message: str

# Ruta za posluživanje logotipa
@app.get("/logo.png")
async def get_logo():
    return FileResponse("logo.png")

@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    response = agent.process_request(query.message)
    iot_status     = agent._get_latest_iot_data()
    weather_status = agent._get_weather_data()
    return {"reply": response, "iot_summary": iot_status, "weather_summary": weather_status}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
    
@app.get("/iot-init")
async def iot_init():
    return {"iot_summary": agent._get_latest_iot_data(), "weather_summary": agent._get_weather_data()}

@app.post("/new-session")
async def new_session():
    agent.chat_history.clear()
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)