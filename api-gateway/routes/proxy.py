import os
from typing import Optional, Dict
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import httpx

UPSTREAM_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "15"))

_hop_by_hop = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

_client: Optional[httpx.AsyncClient] = None

async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_SECONDS, follow_redirects=False)
    return _client

def _filtered_req_headers(headers: Dict[str, str]) -> Dict[str, str]:
    out = {k: v for k, v in headers.items()}
    out.pop("host", None)
    out.pop("connection", None)
    out.pop("content-length", None)
    return out

def _filtered_res_headers(headers: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _hop_by_hop}

async def proxy_request(request: Request, base_url: str, target_path: Optional[str] = None) -> Response:
    base = (base_url or "").rstrip("/")
    path = (target_path or "").lstrip("/")
    url = f"{base}/{path}" if path else base

    try:
        client = await _get_client()
        body = await request.body()

        upstream = await client.request(
            method=request.method,
            url=url,
            params=dict(request.query_params),
            content=body if body else None,
            headers=_filtered_req_headers(dict(request.headers)),
        )

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_filtered_res_headers(dict(upstream.headers)),
            media_type=upstream.headers.get("content-type"),
        )
    except httpx.HTTPStatusError as e:
        return Response(
            content=e.response.content,
            status_code=e.response.status_code,
            headers=_filtered_res_headers(dict(e.response.headers)),
        )
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={
                "error": "Bad Gateway",
                "detail": str(e),
                "target": url,
            },
        )

async def close_client():
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None