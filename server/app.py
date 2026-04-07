"""
server/app.py - OpenEnv entry point wrapper.

openenv validate requires:
  - server/app.py to exist
  - a callable main() function
  - if __name__ == '__main__' block

This file imports and re-exports the FastAPI app from main.py
and provides the required main() entry point.
"""

import uvicorn
import sys
import os

# Add parent directory to path so we can import main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: F401  — re-export for openenv


def main() -> None:
    """Start the Digital Forensics Lab FastAPI server."""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
