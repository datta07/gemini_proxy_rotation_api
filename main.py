import asyncio
import base64
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

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
    # Comma-separated list of models, e.g. "gemini-2.5-flash,gemini-2.5-flash-lite"
    GEMINI_MODEL_NAME: str
    RATE_LIMIT_COOLDOWN_MINUTES: int = 60

    @property
    def GEMINI_MODELS(self) -> List[str]:
        return [m.strip() for m in self.GEMINI_MODEL_NAME.split(",") if m.strip()]

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
# Manages the state and rotation of Gemini API keys and Models
class ResourceManager:
    def __init__(self, api_keys: Dict[str, str], models: List[str]):
        if not api_keys:
            raise ValueError("GEMINI_API_KEYS cannot be empty.")
        if not models:
            raise ValueError("GEMINI_MODELS cannot be empty.")
        
        self._api_keys = api_keys
        self._key_list = list(api_keys.keys())
        self._models = models
        
        # Track state for each (key, model) pair
        # Structure: {(key, model): {"is_rate_limited": bool, "cooldown_until": datetime}}
        self._resource_state: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
        # Initialize state
        for key in self._key_list:
            for model in self._models:
                self._resource_state[(key, model)] = {
                    "is_rate_limited": False, 
                    "cooldown_until": None
                }

        self._current_model_index = 0
        self._current_key_index = 0
        self._lock = asyncio.Lock()
        print(f"Initialized Resource Manager with {len(self._key_list)} keys and {len(self._models)} models.")

    async def get_next_available_resource(self, background_tasks: BackgroundTasks) -> Tuple[str, str]:
        async with self._lock:
            # We iterate through models (priority) and then keys
            # To ensure fair usage and rotation, we start from the current indices
            # but we need to loop through ALL possibilities before giving up.
            
            total_combinations = len(self._models) * len(self._key_list)
            
            # We want to iterate through models as the outer loop (priority)
            # But we want to persist the rotation state.
            
            # Simple strategy:
            # 1. Try current model with all keys (starting from current key index)
            # 2. If all keys for current model are exhausted, move to next model.
            
            start_model_idx = self._current_model_index
            
            for m_offset in range(len(self._models)):
                model_idx = (start_model_idx + m_offset) % len(self._models)
                model_name = self._models[model_idx]
                
                # For this model, try all keys
                start_key_idx = self._current_key_index if model_idx == self._current_model_index else 0
                
                for k_offset in range(len(self._key_list)):
                    key_idx = (start_key_idx + k_offset) % len(self._key_list)
                    key = self._key_list[key_idx]
                    
                    state = self._resource_state[(key, model_name)]
                    
                    # Check cooldown
                    if state["is_rate_limited"]:
                        if datetime.now() > state.get("cooldown_until", datetime.min):
                            # Cooldown expired
                            state["is_rate_limited"] = False
                            state["cooldown_until"] = None
                            label = self._api_keys.get(key, "Unknown")
                            print(f"Resource ({label}, {model_name}) cooldown expired.")
                            background_tasks.add_task(
                                send_telegram_notification,
                                f"✅ <b>Restored Resource</b>\n"
                                f"🔑 Key: <code>...{key[-4:]}</code> ({label})\n"
                                f"🤖 Model: <code>{model_name}</code>\n"
                                f"Resource is active again."
                            )
                        else:
                            # Still rate limited, skip
                            continue
                    
                    # Found a valid resource
                    self._current_model_index = model_idx
                    self._current_key_index = (key_idx + 1) % len(self._key_list) # Rotate key for next time
                    return key, model_name

            # If we reach here, EVERYTHING is rate limited
            raise HTTPException(
                status_code=429, 
                detail="All (Key, Model) combinations are currently rate-limited."
            )

    async def mark_resource_as_rate_limited(
        self, key: str, model_name: str, background_tasks: BackgroundTasks
    ):
        async with self._lock:
            cooldown_until = datetime.now() + timedelta(
                minutes=settings.RATE_LIMIT_COOLDOWN_MINUTES
            )
            self._resource_state[(key, model_name)]["is_rate_limited"] = True
            self._resource_state[(key, model_name)]["cooldown_until"] = cooldown_until
            
            label = self._api_keys.get(key, "Unknown")
            print(f"Resource ({label}, {model_name}) marked as rate-limited until {cooldown_until}.")
            
            background_tasks.add_task(
                send_telegram_notification,
                f"🚨 <b>Rate Limit Alert</b>\n"
                f"🔑 Key: <code>...{key[-4:]}</code> ({label})\n"
                f"🤖 Model: <code>{model_name}</code>\n"
                f"⏳ Cooldown: {settings.RATE_LIMIT_COOLDOWN_MINUTES} mins"
            )


# Initialize the resource manager as a singleton
resource_manager = ResourceManager(settings.GEMINI_API_KEYS, settings.GEMINI_MODELS)


# Dependency to provide the manager to the endpoint
def get_resource_manager():
    return resource_manager


# --- 3. Services (Core Logic) ---

# Using httpx for async requests
async def send_telegram_notification(message: str):
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID, 
        "text": message,
        "parse_mode": "HTML"
    }
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
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

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
                    model_name=self.model_name,
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
                "model": self.model_name,
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
    resource_manager: ResourceManager = Depends(get_resource_manager),
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
    
    # Retry logic: Try until we run out of options or succeed
    # The resource manager will raise 429 if ALL options are exhausted.
    # We can loop a sufficient number of times (e.g., total combinations * 2) to be safe,
    # or just use a while True with a break, relying on the manager to raise 429.
    
    # However, to prevent infinite loops in case of weird state, let's limit retries.
    max_retries = len(resource_manager._key_list) * len(resource_manager._models) + 5
    
    for attempt in range(max_retries):
        api_key, model_name = await resource_manager.get_next_available_resource(background_tasks)
        
        try:
            gemini_service = GeminiService(api_key=api_key, model_name=model_name)
            response = await gemini_service.generate_content(decrypted_messages)
            return response
        
        except HTTPException as e:
            if e.status_code == 429:
                await resource_manager.mark_resource_as_rate_limited(api_key, model_name, background_tasks)
                # Continue to next iteration to get a new resource
                continue
            raise e
            
    raise HTTPException(
        status_code=503, detail="Unable to process request after multiple attempts."
    )