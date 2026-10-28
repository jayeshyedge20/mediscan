#!/usr/bin/env python3
"""
MediScan AI - 1-Click Application Launcher
Starts the FastAPI server with both backend API and frontend UI on http://localhost:8000
"""

import os
import sys
import time
import webbrowser
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

# Ensure backend directory is in sys.path
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

def main():
    print("=" * 70)
    print(" MediScan AI: Bone Fracture & Disease Detection Suite")
    print("=" * 70)
    print(f" Working Directory : {BASE_DIR}")
    print(f" Diagnostic Web App : http://localhost:8000/")
    print(f" API Documentation  : http://localhost:8000/docs")
    print("=" * 70)

    # Ensure samples exist
    try:
        from generate_samples import setup_samples
        setup_samples()
    except Exception as e:
        print(f"Notice: Sample check ({e})")

    # Launch browser after a brief delay
    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open("http://localhost:8000/")
        except Exception:
            pass

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start FastAPI server
    from app import app
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
