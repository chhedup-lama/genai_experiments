import os
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse


API_KEY_ENV_VAR = "SECURE_SERVER_API_KEY"


def get_api_key(x_api_key: str = Header(default="")) -> str:
    expected = os.getenv(API_KEY_ENV_VAR, "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server misconfigured: set {API_KEY_ENV_VAR} environment variable.",
        )
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return x_api_key


app = FastAPI(title="Secure Python Server", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/secure-endpoint", dependencies=[Depends(get_api_key)])
def secure_example() -> JSONResponse:
    # TODO: Import and call your own Python code here.
    # from your_module import your_function
    # result = your_function()
    result = "replace this with your logic"
    return JSONResponse({"result": result})

