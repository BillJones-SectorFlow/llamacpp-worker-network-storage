from __future__ import annotations
from typing import Any, Dict

def create_error_response(message: Any) -> Dict[str, Any]:
    """
    Homogeneous error shape for handler responses.
    """
    return {
        "error": {"message": str(message), "type": "internal"},
        "ok": False,
    }
