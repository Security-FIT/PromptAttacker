#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests

OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.1:70b"   # změň podle toho, co máš stažené

payload = {
    "model": MODEL,
    "prompt": "test",
    "stream": False
}

resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=1200)
resp.raise_for_status()

data = resp.json()
print("Model response:")
print(data.get("response", ""))
