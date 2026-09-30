"""GLM-OCR OpenVINO microservice — OpenAI-compatible document OCR on the iGPU.

Modules:
    model.py   -- single-flight OV session (load once, greedy/sample generate, token streaming)
    server.py  -- FastAPI app: /v1/chat/completions, /v1/models, /health
"""

__version__ = "0.1.0"
