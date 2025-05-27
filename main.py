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

# Import the OpenAI library (which works for Google Gemini as shown in your example)
from openai import OpenAI
from openai import APIError, RateLimitError, AuthenticationError # Specific OpenAI errors

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
GEMINI_BASE_URL = os.environ.get('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta')

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

# --- Helper Functions ---

def send_telegram_notification(message: str):
    """Sends a Telegram message."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        # print(f"Telegram notification skipped (config missing): {message}")
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

def decrypt_data(encrypted_data: str) -> dict:
    """Decrypt the data using AES."""
    try:
        encrypted_bytes = base64.b64decode(encrypted_data)
        cipher = AES.new(SECRET_KEY_BYTES, AES.MODE_CBC, IV_BYTES)
        decrypted_bytes = unpad(cipher.decrypt(encrypted_bytes), AES.block_size)
        decrypted_data = json.loads(decrypted_bytes.decode("utf-8"))
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

async def call_gemini_api(api_key: str, payload: List[Dict[str, str]]):
    """
    Calls the Google Gemini API using the selected API key.
    Initializes a new OpenAI client for each call to ensure the correct API key is used.
    """
    try:
        client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
        response = client.chat.completions.create(
            model=GEMINI_MODEL_NAME,
            messages=payload,
            temperature=0.5,
            top_p=0.1,
            max_tokens=1000,
        )
        return response.model_dump_json() # Return JSON string
    except RateLimitError as e:
        # This key hit a rate limit, mark it and let the caller retry
        await mark_key_rate_limited(api_key, f"Rate limit: {e.response}")
        # Re-raise as an HTTPException to trigger the retry logic in the caller
        raise HTTPException(status_code=429, detail=f"Rate limit for key ending ...{api_key[-4:]} exceeded. Retrying with another key.")
    except AuthenticationError as e:
        # This key is likely invalid or revoked. Mark it as rate-limited permanently (or with a very long cooldown)
        # Or, you might want to remove it from the active pool entirely.
        # For simplicity here, we'll mark it as rate-limited for a long time.
        # Consider a separate mechanism for managing "permanently bad" keys.
        await mark_key_rate_limited(api_key, f"Authentication error: {e.response}")
        send_telegram_notification(f"CRITICAL: API Key ending ...{api_key[-4:]} failed authentication. Please check it.")
        raise HTTPException(status_code=401, detail=f"Authentication failed for key ending ...{api_key[-4:]}. Retrying with another key.")
    except APIError as e:
        # General API error (e.g., bad request, model not found, internal server error from Gemini)
        error_message = f"Gemini API Error (status {e.status_code if hasattr(e, 'status_code') else 'unknown'}): {e.response}"
        # If it's an error that suggests a problem with the key or service availability,
        # you might choose to mark the key as rate-limited as a fallback.
        # For typical 5xx errors, it might be a temporary service issue, not necessarily key-specific.
        if e.status_code and e.status_code >= 500: # Example: internal server error from the model provider
             await mark_key_rate_limited(api_key, f"Server error: {e.response}")
             raise HTTPException(status_code=502, detail=f"Bad Gateway (upstream server error for key ending ...{api_key[-4:]}). Retrying with another key.")
        send_telegram_notification(f"Gemini API Error for key ending ...{api_key[-4:]}: {error_message}")
        raise HTTPException(status_code=e.status_code if hasattr(e, 'status_code') else 500, detail=error_message)
    except Exception as e:
        send_telegram_notification(f"Unexpected error during Gemini API call for key ending ...{api_key[-4:]}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- FastAPI Endpoint ---

@app.post("/chat")
async def chat_proxy(encrypted_data: str = Body(...)):
    decrypted_data: dict
    try:
        # Decrypt the incoming data
        decrypted_data = decrypt_data(encrypted_data)

        if not isinstance(decrypted_data, list) or not all(isinstance(m, dict) for m in decrypted_data):
            raise HTTPException(status_code=422, detail="Decrypted data must be a list of message dictionaries.")

        max_retries = len(API_KEYS) + 1 # Allow trying each key once plus one more global retry
        for attempt in range(max_retries):
            try:
                # Get the next available API key
                current_api_key = await get_next_available_key()

                # Call the Gemini API with the selected key and decrypted payload
                response_json = await call_gemini_api(current_api_key, decrypted_data)
                
                # If successful, notify and return
                send_telegram_notification(f"Successfully processed request using key ending ...{current_api_key[-4:]} for model {GEMINI_MODEL_NAME}.")
                return json.loads(response_json)

            except HTTPException as e:
                # If it's a 429 (rate limit) or 401 (auth error leading to retry)
                # and we still have attempts left, continue to the next iteration
                if e.status_code in [429, 401] and attempt < max_retries - 1:
                    print(f"Attempt {attempt+1}/{max_retries} failed ({e.status_code}). Trying next key...")
                    # The `mark_key_rate_limited` is called within `call_gemini_api`,
                    # so we just need to loop to `get_next_available_key` again.
                    continue
                else:
                    # Re-raise other HTTP exceptions or if no more retries
                    raise e
            except Exception as e:
                # Catch any unexpected errors during the API call or key selection within the loop
                print(f"Unexpected error in API call attempt {attempt+1}: {e}")
                # Don't retry on generic errors, as they might not be key-specific
                raise HTTPException(status_code=500, detail=f"Internal server error during API call: {str(e)}")
        
        # If loop finishes without success (all keys exhausted)
        raise HTTPException(status_code=429, detail="All configured API keys exhausted after multiple retries.")

    except HTTPException as e:
        # Re-raise HTTP exceptions (e.g., from decryption or initial key exhaustion)
        raise e
    except Exception as e:
        # Catch any other unhandled errors
        print(f"Unhandled error in /chat endpoint: {e}")
        send_telegram_notification(f"Unhandled error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Example of how to run this (if not using uvicorn directly via command line)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)