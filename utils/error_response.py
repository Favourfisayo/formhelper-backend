from fastapi.responses import JSONResponse


def error_response(error_type: str, message: str, details: str = None, status_code: int = 400):
    """Standardized error response"""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "data": None,
            "error": {
                "type": error_type,
                "message": message,
                "details": details,
            },
        })