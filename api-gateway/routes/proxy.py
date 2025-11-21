import os
import logging
import httpx
from typing import Optional, Dict
from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

UPSTREAM_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "30"))  # Increase from 15 to 30

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

async def proxy_request(
    request: Request,
    target_url: str,
    headers: Optional[Dict[str, str]] = None
) -> Response:
    """Proxy a request to an upstream service"""
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_SECONDS) as client:
            # Forward the request
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers or {},
                content=await request.body() if request.method in ["POST", "PUT", "PATCH"] else None,
                params=request.query_params
            )
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
    except httpx.TimeoutException:
        logger.error(f"Timeout calling {target_url}")
        return JSONResponse(
            status_code=504,
            content={
                "error": "Gateway Timeout",
                "detail": f"Upstream service took too long to respond",
                "target": target_url
            }
        )
    except httpx.RequestError as e:
        logger.error(f"Error calling {target_url}: {str(e)}")
        return JSONResponse(
            status_code=502,
            content={
                "error": "Bad Gateway",
                "detail": str(e),
                "target": target_url
            }
        )

async def close_client():
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None