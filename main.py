from fastapi import FastAPI, HTTPException, Body
import requests
from Crypto.Cipher import AES  # Use pycryptodome for AES encryption
from Crypto.Util.Padding import unpad
import base64
import json
import os
from typing import Dict, List, Any
from datetime import datetime, timedelta
import asyncio # For async locks

# Import Google GenAI specific types
from google import genai
from google.genai import types

app = FastAPI()

# --- AES Configuration ---
SECRET_KEY = os.environ.get('AES_SECRET_KEY')
IV = os.environ.get('AES_IV')

if not SECRET_KEY or not IV:
    raise ValueError("AES_SECRET_KEY and AES_IV environment variables must be set.")
if len(SECRET_KEY.encode('utf-8')) != 32:
    raise ValueError("AES_SECRET_KEY must be 32 bytes for AES-256.")
if len(IV.encode('utf-8')) != 16:
    raise ValueError("AES_IV must be 16 bytes for AES-CBC.")

SECRET_KEY_BYTES = SECRET_KEY.encode('utf-8')
IV_BYTES = IV.encode('utf-8')

# --- Telegram Configuration ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("WARNING: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Telegram notifications will be disabled.")

# --- Gemini API Configuration (ADAPTED FOR MULTIPLE KEYS) ---
# API_KEYS should be a JSON string like '{"key1_value": "label1", "key2_value": "label2"}'
API_KEYS_JSON = os.environ.get('GEMINI_API_KEYS')
if not API_KEYS_JSON:
    raise ValueError("GEMINI_API_KEYS environment variable must be set (as a JSON string).")

try:
    # API_KEYS will be a dict: {api_key_value: label}
    API_KEYS: Dict[str, str] = json.loads(API_KEYS_JSON)
    if not API_KEYS:
        raise ValueError("GEMINI_API_KEYS is empty.")
except json.JSONDecodeError:
    raise ValueError("GEMINI_API_KEYS environment variable is not a valid JSON string.")
except Exception as e:
    raise ValueError(f"Error parsing GEMINI_API_KEYS: {e}")

# The target model and base URL for all these Gemini keys
GEMINI_MODEL_NAME = os.environ.get('GEMINI_MODEL_NAME', 'gemini-2.5-flash-preview-05-20')

# --- In-Memory API Key State Management (NEW) ---
# This dictionary will hold the state of each API key
# Key: API Key value (string)
# Value: Dict with 'is_rate_limited' (bool), 'cooldown_until' (datetime or None)
API_KEY_STATE: Dict[str, Dict[str, Any]] = {}
RATE_LIMIT_COOLDOWN_MINUTES = int(os.environ.get('RATE_LIMIT_COOLDOWN_MINUTES', 5)) # Cooldown period

# To ensure fair rotation and thread safety
_api_key_list = list(API_KEYS.keys()) # Ordered list of keys for rotation
_current_key_index = 0 # Current index for round-robin
_api_key_lock = asyncio.Lock() # Lock for protecting shared state

# Initialize API_KEY_STATE for all keys
for key in _api_key_list:
    API_KEY_STATE[key] = {
        'is_rate_limited': False,
        'cooldown_until': None
    }
print(f"Initialized API key states for {len(API_KEYS)} keys.")

# --- GeminiOpenAIWrapper Integration ---
class GeminiOpenAIWrapper:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def _convert_openai_to_gemini_content(self, messages: List[Dict[str, str]]):
        """Converts OpenAI-style messages to Gemini-style content, filtering out system messages."""
        gemini_contents = []
        for message in messages:
            if message["role"] == "system":
                continue # System instructions are handled separately
            
            role = "user" if message["role"] == "user" else "model"
            parts = [types.Part.from_text(text=message["content"])]
            gemini_contents.append(types.Content(role=role, parts=parts))
        return gemini_contents

    def _extract_system_instruction(self, messages: List[Dict[str, str]]):
        """Extracts and combines all system instructions into a single string for Gemini."""
        system_instructions_text = []
        for message in messages:
            if message["role"] == "system":
                system_instructions_text.append(message["content"])
        
        if system_instructions_text:
            # Join all system instructions into one coherent string
            return [types.Part.from_text(text="\n".join(system_instructions_text))]
        return []

    async def chat_completions(self, payload: Dict[str, Any]):
        """
        Mimics OpenAI's chat completions API.

        Args:
            payload (dict): A dictionary representing the OpenAI chat completions request.
                            Expected keys: 'model', 'messages', 'temperature', 'top_p', 'max_tokens'.

        Returns:
            dict: A dictionary mimicking OpenAI's chat completions response format.
        """
        model_name = payload.get("model", GEMINI_MODEL_NAME) 
        messages = payload.get("messages", [])
        temperature = payload.get("temperature", 0.7) 
        top_p = payload.get("top_p", 0.95) 
        max_tokens = payload.get("max_tokens") 

        gemini_contents = self._convert_openai_to_gemini_content(messages)
        system_instructions = self._extract_system_instruction(messages)

        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            response_mime_type="text/plain",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        
        if max_tokens is not None:
            generate_content_config.max_output_tokens = max_tokens
        
        if system_instructions:
            generate_content_config.system_instruction = system_instructions
            
        # Ensure there's at least one content message if system_instruction is not the only input.
        # Gemini usually expects the conversation history in `contents`.
        if not gemini_contents and not system_instructions:
             raise ValueError("No valid user/model messages or system instructions found in payload.")
        
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model_name,
                contents=gemini_contents,
                config=generate_content_config
            )

            # Construct OpenAI-like response
            openai_response = {
                "id": f"chatcmpl-wrapper-{os.urandom(8).hex()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response.text},
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
            }
            return openai_response

        except genai.types.rpc.Code as rpc_code:
            error_message = f"Gemini RPC Error ({rpc_code.value}): {rpc_code.name}"
            print(f"RPC Error: {error_message}")
            if rpc_code == genai.types.rpc.Code.RESOURCE_EXHAUSTED:
                raise GeminiRateLimitError(error_message)
            elif rpc_code == genai.types.rpc.Code.UNAUTHENTICATED:
                raise GeminiAuthenticationError(error_message)
            elif rpc_code == genai.types.rpc.Code.INVALID_ARGUMENT:
                raise GeminiAPIError(status_code=400, message=error_message)
            else:
                raise GeminiAPIError(status_code=500, message=error_message)
        except Exception as e:
            raise GeminiAPIError(status_code=500, message=f"Gemini API call failed: {str(e)}")

# Define custom exceptions to mimic OpenAI's for consistent handling
class GeminiRateLimitError(Exception):
    pass

class GeminiAuthenticationError(Exception):
    pass

class GeminiAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")


# --- Helper Functions ---

def send_telegram_notification(message: str):
    """Sends a Telegram message."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    telegram_api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    try:
        response = requests.post(telegram_api_url, data=payload)
        response.raise_for_status()
        print(f"Telegram notification sent: {message}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Telegram notification: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while sending Telegram notification: {e}")

def decrypt_data(encrypted_data: str) -> List[Dict[str, str]]: # Changed return type
    """Decrypt the data using AES."""
    try:
        encrypted_bytes = base64.b64decode(encrypted_data)
        cipher = AES.new(SECRET_KEY_BYTES, AES.MODE_CBC, IV_BYTES)
        decrypted_bytes = unpad(cipher.decrypt(encrypted_bytes), AES.block_size)
        decrypted_data = json.loads(decrypted_bytes.decode("utf-8"))
        
        # Validate that the decrypted data is a list of messages
        if not isinstance(decrypted_data, list) or not all(isinstance(m, dict) for m in decrypted_data):
            raise ValueError("Decrypted data must be a list of message dictionaries.")
            
        return decrypted_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decryption failed: {str(e)}")

async def mark_key_rate_limited(api_key: str, error_detail: str):
    """Mark an API key as rate-limited with a cooldown period."""
    async with _api_key_lock:
        API_KEY_STATE[api_key]['is_rate_limited'] = True
        API_KEY_STATE[api_key]['cooldown_until'] = datetime.now() + timedelta(minutes=RATE_LIMIT_COOLDOWN_MINUTES)
        label = API_KEYS.get(api_key, 'Unknown')
        send_telegram_notification(
            f"API Key '{label}' (ending in ...{api_key[-4:]}) marked rate-limited. Cooldown until {API_KEY_STATE[api_key]['cooldown_until'].strftime('%H:%M:%S')}. Detail: {error_detail}"
        )
        print(f"Key {label} ({api_key[-4:]}) marked rate-limited until {API_KEY_STATE[api_key]['cooldown_until'].strftime('%Y-%m-%d %H:%M:%S')}")


async def get_next_available_key() -> str:
    """
    Get the next available API key from the pool.
    Handles rotation and considers keys currently in cooldown.
    """
    global _current_key_index
    num_keys = len(_api_key_list)
    if num_keys == 0:
        raise HTTPException(status_code=500, detail="No API keys configured.")

    async with _api_key_lock:
        start_index = _current_key_index
        attempts = 0

        while attempts < num_keys:
            key_candidate = _api_key_list[_current_key_index]
            key_state = API_KEY_STATE[key_candidate]

            # Check if cooldown has expired
            if key_state['is_rate_limited'] and key_state['cooldown_until'] and datetime.now() > key_state['cooldown_until']:
                key_state['is_rate_limited'] = False
                key_state['cooldown_until'] = None
                label = API_KEYS.get(key_candidate, 'Unknown')
                send_telegram_notification(
                    f"API Key '{label}' (ending in ...{key_candidate[-4:]}) cooldown expired and reset."
                )
                print(f"Key {label} ({key_candidate[-4:]}) cooldown expired.")

            # If not rate-limited (or reset), return this key
            if not key_state['is_rate_limited']:
                _current_key_index = (_current_key_index + 1) % num_keys # Move to next key for next request
                label = API_KEYS.get(key_candidate, 'Unknown')
                print(f"Using API Key '{label}' (ending in ...{key_candidate[-4:]}).")
                return key_candidate
            
            # If rate-limited, move to the next key for the next attempt
            _current_key_index = (_current_key_index + 1) % num_keys
            attempts += 1

        # If we've looped through all keys and none are available
        send_telegram_notification("All configured API keys are currently rate-limited or unavailable.")
        raise HTTPException(status_code=429, detail="API limit reached for all available keys. Please try again later.")

async def call_gemini_api(api_key: str, messages_list: List[Dict[str, str]]): # Changed payload type
    """
    Calls the Google Gemini API using the selected API key.
    Initializes a new GeminiOpenAIWrapper client for each call to ensure the correct API key is used.
    """
    try:
        gemini_wrapper = GeminiOpenAIWrapper(api_key=api_key)
        
        # Construct the full OpenAI-like payload for the wrapper's chat_completions method
        # This includes hardcoded model, temperature, top_p, max_tokens etc.
        wrapper_payload = {
            "model": GEMINI_MODEL_NAME, # Use the globally defined model
            "messages": messages_list,  # Pass the raw messages list
            "temperature": 0.5,         # Hardcoded as per previous example
            "top_p": 0.1,               # Hardcoded as per previous example
            "max_tokens": 1000          # Hardcoded as per previous example
        }

        response_dict = await gemini_wrapper.chat_completions(wrapper_payload)
        
        return json.dumps(response_dict) 

    except GeminiRateLimitError as e:
        await mark_key_rate_limited(api_key, f"Rate limit: {e.message}")
        raise HTTPException(status_code=429, detail=f"Rate limit for key ending ...{api_key[-4:]} exceeded. Retrying with another key.")
    except GeminiAuthenticationError as e:
        await mark_key_rate_limited(api_key, f"Authentication error: {e.message}")
        send_telegram_notification(f"CRITICAL: API Key ending ...{api_key[-4:]} failed authentication. Please check it.")
        raise HTTPException(status_code=401, detail=f"Authentication failed for key ending ...{api_key[-4:]}. Retrying with another key.")
    except GeminiAPIError as e:
        error_message = f"Gemini API Error (status {e.status_code}): {e.message}"
        if e.status_code >= 500:
             await mark_key_rate_limited(api_key, f"Server error: {e.message}")
             raise HTTPException(status_code=502, detail=f"Bad Gateway (upstream server error for key ending ...{api_key[-4:]}). Retrying with another key.")
        send_telegram_notification(f"Gemini API Error for key ending ...{api_key[-4:]}: {error_message}")
        raise HTTPException(status_code=e.status_code, detail=error_message)
    except Exception as e:
        send_telegram_notification(f"Unexpected error during Gemini API call for key ending ...{api_key[-4:]}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- FastAPI Endpoint ---

@app.post("/chat")
async def chat_proxy(encrypted_data: str = Body(...)):
    decrypted_messages: List[Dict[str, str]] # Changed type
    try:
        # Decrypt the incoming data, which is now expected to be just the list of messages
        decrypted_messages = decrypt_data(encrypted_data)

        max_retries = len(API_KEYS) + 1 
        for attempt in range(max_retries):
            try:
                current_api_key = await get_next_available_key()

                # Pass the decrypted list of messages directly to call_gemini_api
                response_json = await call_gemini_api(current_api_key, decrypted_messages)
                
                # send_telegram_notification(f"Successfully processed request using key ending ...{current_api_key[-4:]} for model {GEMINI_MODEL_NAME}.")
                return json.loads(response_json)

            except HTTPException as e:
                if e.status_code in [429, 401, 502] and attempt < max_retries - 1:
                    print(f"Attempt {attempt+1}/{max_retries} failed ({e.status_code}, {e.detail}). Trying next key...")
                    continue
                else:
                    raise e
            except Exception as e:
                print(f"Unexpected error in API call attempt {attempt+1}: {e}")
                raise HTTPException(status_code=500, detail=f"Internal server error during API call: {str(e)}")
        
        raise HTTPException(status_code=429, detail="All configured API keys exhausted after multiple retries.")

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unhandled error in /chat endpoint: {e}")
        send_telegram_notification(f"Unhandled error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")