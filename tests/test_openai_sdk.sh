#!/bin/sh
set -eu

: "${OPENAI_BASE_URL:?set OPENAI_BASE_URL=http://127.0.0.1:8080/v1}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY to the server API key or unused}"
: "${OPENAI_MODEL:?set OPENAI_MODEL to the served model name}"

exec "${PYTHON:-python3}" tests/test_openai_sdk.py
