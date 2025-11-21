# test_client.py

import base64
import json
import os

import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from dotenv import load_dotenv

# Load environment variables from .env file to get keys
load_dotenv()

# --- Configuration (should match the server's .env file) ---
API_URL = "http://127.0.0.1:8000/chat"
SECRET_KEY = os.getenv("AES_SECRET_KEY")
IV = os.getenv("AES_IV")

if not SECRET_KEY or not IV:
    raise ValueError("AES_SECRET_KEY and AES_IV must be set in the .env file.")

SECRET_KEY_BYTES = SECRET_KEY.encode("utf-8")
IV_BYTES = IV.encode("utf-8")


def encrypt_payload(data: list) -> str:
    """Encrypts the payload in the same way the server expects to decrypt it."""
    try:
        raw_data = json.dumps(data).encode("utf-8")
        cipher = AES.new(SECRET_KEY_BYTES, AES.MODE_CBC, IV_BYTES)
        padded_data = pad(raw_data, AES.block_size)
        encrypted_bytes = cipher.encrypt(padded_data)
        return base64.b64encode(encrypted_bytes).decode("utf-8")
    except Exception as e:
        print(f"Error during encryption: {e}")
        raise


def run_test():
    """
    Creates a sample payload, encrypts it, and sends it to the local API.
    """
    # Sample payload mimicking the OpenAI message format
    sample_messages = [
        {"role": "user", "content": "Hello, how are you?"},
    ]

    print("--- Client: Starting Test ---")
    print(f"Original payload: {sample_messages}")

    try:
        # 1. Encrypt the data
        encrypted_data = encrypt_payload(sample_messages)
        print(f"\nEncrypted payload (first 30 chars): {encrypted_data[:30]}...")

        # 2. Prepare the request body for FastAPI (as it expects a JSON body with the encrypted data)
        request_body = {"encrypted_data": encrypted_data}

        # 3. Send the request to the API
        print(f"\nSending POST request to {API_URL}...")
        response = requests.post(API_URL, json=request_body)

        # 4. Print the results
        print(f"\n--- Server Response ---")
        print(f"Status Code: {response.status_code}")

        if response.ok:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error Response:")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"\nFailed to connect to the server at {API_URL}.")
        print("Please ensure the FastAPI server is running: `uvicorn main:app --reload`")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    run_test()