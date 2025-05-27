# Gemini API Proxy Server

A robust FastAPI-based proxy server for managing multiple Google Gemini API keys with automatic failover, rate limit handling, and encrypted request processing.

## Features

- **Multi-Key Management**: Automatic rotation between multiple Gemini API keys
- **Rate Limit Handling**: Smart cooldown management when keys hit rate limits
- **Encrypted Requests**: AES-256-CBC encryption for secure data transmission
- **Telegram Notifications**: Real-time alerts for key status and system events
- **Automatic Failover**: Seamless switching between available API keys
- **Thread-Safe Operations**: Async locks for concurrent request handling
- **Health Monitoring**: Built-in error tracking and recovery mechanisms

## Prerequisites

- Python 3.8+
- Google Gemini API keys
- Telegram Bot (optional, for notifications)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gemini-proxy-server
```

2. Install dependencies:
```bash
pip install fastapi uvicorn requests pycryptodome openai
```

## Configuration

### Required Environment Variables

```bash
# AES Encryption (CRITICAL: Use secure, random values)
export AES_SECRET_KEY="your-32-byte-secret-key-here-exactly"  # Must be exactly 32 bytes
export AES_IV="your-16-byte-iv-here"                         # Must be exactly 16 bytes

# Gemini API Keys (JSON format)
export GEMINI_API_KEYS='{"api_key_1": "Production Key", "api_key_2": "Backup Key"}'

# Telegram Notifications (Optional)
export TELEGRAM_BOT_TOKEN="your-telegram-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"
```

### Optional Environment Variables

```bash
# Gemini Configuration
export GEMINI_MODEL_NAME="gemini-2.5-flash-preview-05-20"  # Default model
export GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta"

# Rate Limit Settings
export RATE_LIMIT_COOLDOWN_MINUTES="5"  # Cooldown period for rate-limited keys
```

## Quick Start

### 1. Generate Encryption Keys

**⚠️ SECURITY WARNING**: Never use the example keys below in production!

```python
# Generate secure keys
import secrets
import base64

# Generate 32-byte key for AES-256
secret_key = base64.b64encode(secrets.token_bytes(32)).decode()[:32]
print(f"AES_SECRET_KEY={secret_key}")

# Generate 16-byte IV
iv = base64.b64encode(secrets.token_bytes(16)).decode()[:16]
print(f"AES_IV={iv}")
```

### 2. Set Up Environment

Create a `.env` file:
```bash
AES_SECRET_KEY=your-generated-32-byte-key
AES_IV=your-generated-16-byte-iv
GEMINI_API_KEYS={"AIza...": "Primary", "AIza...": "Secondary"}
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
```

### 3. Run the Server

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Usage

### Endpoint: `POST /chat`

Send encrypted chat messages to the Gemini API.

#### Request Format

1. **Prepare your message payload**:
```python
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"},
    {"role": "user", "content": "What's the weather like?"}
]
```

2. **Encrypt the payload**:
```python
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import base64
import json

def encrypt_data(data, secret_key, iv):
    cipher = AES.new(secret_key.encode(), AES.MODE_CBC, iv.encode())
    padded_data = pad(json.dumps(data).encode(), AES.block_size)
    encrypted = cipher.encrypt(padded_data)
    return base64.b64encode(encrypted).decode()

encrypted_payload = encrypt_data(messages, SECRET_KEY, IV)
```

3. **Send the request**:
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json=encrypted_payload,
    headers={"Content-Type": "application/json"}
)

print(response.json())
```

#### Response Format

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1703123456,
  "model": "gemini-2.5-flash-preview-05-20",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The response from Gemini API..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 50,
    "total_tokens": 70
  }
}
```

## Client Example

<function_calls>
<invoke name="artifacts">
<parameter name="command">update