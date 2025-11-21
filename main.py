# main.py

import asyncio
import base64
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import httpx
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from fastapi import BackgroundTasks, Body, Depends, FastAPI, HTTPException, Request
from google.api_core import exceptions
from google.generativeai import types
import google.generativeai as genai
from pydantic_settings import BaseSettings


# --- 1. Configuration Management ---
# Using Pydantic's BaseSettings for robust configuration and validation
class Settings(BaseSettings):
    AES_SECRET_KEY: str
    AES_IV: str
    TELEGRAM_BOT_TOKEN: str | None = None
    TELEGRAM_CHAT_ID: str | None = None
    # GEMINI_API_KEYS should be a JSON string like '{"key1_value": "label1", ...}'
    GEMINI_API_KEYS: Dict[str, str]
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    RATE_LIMIT_COOLDOWN_MINUTES: int = 5

    class Config:
        env_file = ".env"  # Load from a .env file if present
        env_file_encoding = "utf-8"


settings = Settings()

# Validate key lengths
if len(settings.AES_SECRET_KEY.encode("utf-8")) != 32:
    raise ValueError("AES_SECRET_KEY must be 32 bytes for AES-256.")
if len(settings.AES_IV.encode("utf-8")) != 16:
    raise ValueError("AES_IV must be 16 bytes for AES-CBC.")


app = FastAPI()

# Global lock to prevent race conditions during API key configuration
gemini_lock = asyncio.Lock()


# --- 2. Dependencies ---
# Manages the state and rotation of Gemini API keys
class APIKeyManager:
    def __init__(self, api_keys: Dict[str, str]):
        if not api_keys:
            raise ValueError("GEMINI_API_KEYS cannot be empty.")
        self._api_keys = api_keys
        self._key_list = list(api_keys.keys())
        self._key_state = {
            key: {"is_rate_limited": False, "cooldown_until": None}
            for key in self._key_list
        }
        self._current_index = 0
        self._lock = asyncio.Lock()
        print(f"Initialized API key manager with {len(self._key_list)} keys.")

    async def get_next_available_key(self, background_tasks: BackgroundTasks) -> str:
        async with self._lock:
            for _ in range(len(self._key_list)):
                key = self._key_list[self._current_index]
                state = self._key_state[key]

                # Check if the key's cooldown has expired
                if state["is_rate_limited"] and datetime.now() > state.get(
                    "cooldown_until", datetime.min
                ):
                    state["is_rate_limited"] = False
                    state["cooldown_until"] = None
                    label = self._api_keys.get(key, "Unknown")
                    print(f"Key '{label}' cooldown expired. Resetting.")
                    background_tasks.add_task(
                        send_telegram_notification,
                        f"✅ API Key '{label}' (ending ...{key[-4:]}) cooldown expired and is now active.",
                    )

                if not state["is_rate_limited"]:
                    self._current_index = (self._current_index + 1) % len(
                        self._key_list
                    )
                    return key

                self._current_index = (self._current_index + 1) % len(self._key_list)

            # If all keys are rate-limited
            raise HTTPException(
                status_code=429, detail="All API keys are currently rate-limited."
            )

    async def mark_key_as_rate_limited(
        self, key: str, background_tasks: BackgroundTasks
    ):
        async with self._lock:
            cooldown_until = datetime.now() + timedelta(
                minutes=settings.RATE_LIMIT_COOLDOWN_MINUTES
            )
            self._key_state[key]["is_rate_limited"] = True
            self._key_state[key]["cooldown_until"] = cooldown_until
            label = self._api_keys.get(key, "Unknown")
            print(f"Key '{label}' marked as rate-limited until {cooldown_until}.")
            background_tasks.add_task(
                send_telegram_notification,
                f"RATE LIMIT: API Key '{label}' (ending ...{key[-4:]}) is now on cooldown.",
            )


# Initialize the key manager as a singleton
api_key_manager = APIKeyManager(settings.GEMINI_API_KEYS)


# Dependency to provide the key manager to the endpoint
def get_key_manager():
    return api_key_manager


# --- 3. Services (Core Logic) ---

# Using httpx for async requests
async def send_telegram_notification(message: str):
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": settings.TELEGRAM_CHAT_ID, "text": message}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            print("Telegram notification sent successfully.")
    except httpx.RequestError as e:
        print(f"Failed to send Telegram notification: {e}")


def decrypt_data(encrypted_data: str) -> List[Dict[str, str]]:
    try:
        encrypted_bytes = base64.b64decode(encrypted_data)
        cipher = AES.new(
            settings.AES_SECRET_KEY.encode("utf-8"),
            AES.MODE_CBC,
            settings.AES_IV.encode("utf-8"),
        )
        decrypted_bytes = unpad(cipher.decrypt(encrypted_bytes), AES.block_size)
        data = json.loads(decrypted_bytes.decode("utf-8"))
        if not isinstance(data, list) or not all(isinstance(m, dict) for m in data):
            raise ValueError("Decrypted data is not a list of message dictionaries.")
        return data
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Decryption failed: {e}")


class GeminiService:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _prepare_gemini_request(self, messages: List[Dict[str, str]]):
        gemini_contents = []
        system_instructions = []

        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            if role == "system":
                system_instructions.append(content)
            elif role in ("user", "model"):
                gemini_role = "user" if role == "user" else "model"
                gemini_contents.append(
                    {"role": gemini_role, "parts": [{"text": content}]}
                )

        if len(gemini_contents) > 1:
            merged_contents = [gemini_contents[0]]
            for i in range(1, len(gemini_contents)):
                if gemini_contents[i]["role"] == merged_contents[-1]["role"]:
                    merged_contents[-1]["parts"].extend(gemini_contents[i]["parts"])
                else:
                    merged_contents.append(gemini_contents[i])
            gemini_contents = merged_contents

        generation_config = types.GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=2048,
            response_mime_type="text/plain",
        )
        
        return gemini_contents, system_instructions, generation_config


    async def generate_content(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        gemini_contents, system_instructions, generation_config = self._prepare_gemini_request(messages)

        if not gemini_contents:
            raise HTTPException(status_code=400, detail="No user/model messages found.")

        try:
            async with gemini_lock:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(
                    model_name=settings.GEMINI_MODEL_NAME,
                    system_instruction=system_instructions if system_instructions else None
                )
                
                response = await model.generate_content_async(
                    contents=gemini_contents,
                    generation_config=generation_config,
                )
            
            return {
                "id": f"chatcmpl-gemini-{os.urandom(8).hex()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": settings.GEMINI_MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response.text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                },
            }
        except exceptions.ResourceExhausted as e:
            raise HTTPException(status_code=429, detail=f"Gemini API rate limit exceeded: {e}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")


# --- 4. API Endpoint ---
@app.post("/chat")
async def chat_proxy(
    request: Request,
    background_tasks: BackgroundTasks,
    key_manager: APIKeyManager = Depends(get_key_manager),
):
    content_type = request.headers.get("content-type", "")
    
    if "application/json" in content_type:
        try:
            data = await request.json()
            encrypted_data = data.get("encrypted_data")
        except json.JSONDecodeError:
             raise HTTPException(status_code=400, detail="Invalid JSON body")
    else:
        body = await request.body()
        encrypted_data = body.decode("utf-8")

    if not encrypted_data:
        raise HTTPException(status_code=400, detail="Missing encrypted_data in body")

    decrypted_messages = decrypt_data(encrypted_data)
    
    max_retries = len(key_manager._key_list)
    for attempt in range(max_retries):
        api_key = await key_manager.get_next_available_key(background_tasks)
        
        try:
            gemini_service = GeminiService(api_key=api_key)
            response = await gemini_service.generate_content(decrypted_messages)
            return response
        
        except HTTPException as e:
            if e.status_code == 429:
                await key_manager.mark_key_as_rate_limited(api_key, background_tasks)
                if attempt < max_retries - 1:
                    continue
            raise e
            
    raise HTTPException(
        status_code=503, detail="All API keys are currently unavailable. Please try again later."
    )