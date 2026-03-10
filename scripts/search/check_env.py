"""
Check that all required API keys are set in .env

Usage:
    uv run scripts/search/check_env.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

checks = [
    ("OPENROUTER_API_KEY", True,  "https://openrouter.ai/keys"),
    ("OPENROUTER_MODEL",   False, "default: qwen/qwen3-30b-a3b"),
    ("HF_TOKEN",           False, "https://huggingface.co/settings/tokens"),
    ("KAGGLE_USERNAME",    False, "https://kaggle.com/settings/account -> API"),
    ("KAGGLE_KEY",         False, "https://kaggle.com/settings/account -> API"),
]

all_ok = True
for key, required, hint in checks:
    val = os.getenv(key, "")
    status = "OK" if val else ("MISSING" if required else "not set")
    marker = "✓" if val else ("✗" if required else "-")
    print(f"  {marker} {key:<25} {status}  ({hint})")
    if required and not val:
        all_ok = False

print()
print("All required keys present." if all_ok else "ERROR: set missing keys in .env")
