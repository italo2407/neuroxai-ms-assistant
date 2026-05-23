from fastapi import HTTPException


class ModelNotLoadedError(HTTPException):
    def __init__(self, detail: str = "Model not loaded"):
        super().__init__(status_code=503, detail=detail)


class SessionNotFoundError(HTTPException):
    def __init__(self, session_id: str):
        super().__init__(status_code=404, detail=f"Session '{session_id}' not found or expired")


class XAIComputeError(HTTPException):
    def __init__(self, method: str, detail: str):
        super().__init__(status_code=500, detail=f"XAI method '{method}' failed: {detail}")


class InvalidImageError(HTTPException):
    def __init__(self, detail: str = "Invalid image format"):
        super().__init__(status_code=422, detail=detail)
