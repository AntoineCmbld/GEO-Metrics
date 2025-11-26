from typing import Optional, Dict, Any
from fastapi import Request, Response
import httpx
import logging

logger = logging.getLogger(__name__)

async def proxy_request(
    request: Request,
    target_url: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    json: Optional[dict] = None,
    timeout: float = 30.0
) -> Response:
    """
    Proxy a request to a target service
    
    Args:
        request: The incoming FastAPI request
        target_url: Base URL of the target service (e.g., "http://report-service:8080")
        extra_headers: Optional dictionary of additional headers to include
        json: Optional JSON data to send in the request body
        timeout: Request timeout in seconds
    
    Returns:
        Response from the target service
    """
    # Prepare headers
    headers = {}
    excluded_headers = {"host", "content-length", "transfer-encoding", "connection"}
    for key, value in request.headers.items():
        if key.lower() not in excluded_headers:
            headers[key] = value
    if extra_headers:
        headers.update(extra_headers)

    logger.info(f"Proxying {request.method} request to {target_url}")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if request.method in ("POST", "PUT", "PATCH"):
                resp = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    json=json if json is not None else await request.json()
                )
            else:
                resp = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
    except httpx.TimeoutException as e:
        logger.error(f"Timeout proxying to {target_url}: {str(e)}")
        return Response(
            content=f'{{"error": "Gateway timeout", "detail": "Request to {target_url} timed out"}}',
            status_code=504,
            media_type="application/json"
        )
    except httpx.RequestError as e:
        logger.error(f"Error proxying to {target_url}: {str(e)}")
        return Response(
            content=f'{{"error": "Bad gateway", "detail": "Failed to connect to {target_url}"}}',
            status_code=502,
            media_type="application/json"
        )
    except Exception as e:
        logger.error(f"Unexpected error proxying to {target_url}: {str(e)}")
        return Response(
            content=f'{{"error": "Internal server error", "detail": "{str(e)}"}}',
            status_code=500,
            media_type="application/json"
        )