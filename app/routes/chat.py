from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import aiohttp
import json

from starlette.background import BackgroundTask

import logging

log = logging.getLogger(__name__)
router = APIRouter()

async def cleanup_response(
    response: Optional[aiohttp.ClientResponse],
    session: Optional[aiohttp.ClientSession],
):
    if response:
        response.close()
    if session:
        await session.close()


@router.post("/chat/completions")
async def generate_chat_completion(
        form_data: dict,
):
    payload = {**form_data}

    if payload.get("temperature") is None:
        payload["temperature"] = float("0.7")

    if payload.get("top_p") is None:
        payload["top_p"] = int(1.1)

    if payload.get("max_tokens") is None:
        payload["max_tokens"] = int(512)

    if payload.get("frequency_penalty") is None :
        payload["frequency_penalty"] = int(2.0)

    if payload.get("seed") is None:
        payload["seed"] = 10

    if payload.get("stop") is None:
        payload["stop"] = (
            [
                bytes(stop, "utf-8").decode("unicode_escape")
                for stop in ["<|im_end|>", "<|im_start|>"]
            ]
           
        )

    # 提示词模版
    # system = prompt_template(system)
    # if payload.get("messages"):
    #     payload["messages"] = add_or_update_system_message(
    #         system, payload["messages"]
    #     )

    # Convert the modified body back to JSON
    payload = json.dumps(payload)
    key = "xxxx"
    url = "https://"

    headers = {}
    headers["Authorization"] = f"Bearer {key}"
    headers["Content-Type"] = "application/json"

    r = None
    session = None
    streaming = False

    try:
        session = aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=10)
        )
        r = await session.request(
            method="POST",
            url=f"{url}/chat/completions",
            data=payload,
            headers=headers,
        )

        r.raise_for_status()

        # Check if response is SSE
        if "text/event-stream" in r.headers.get("Content-Type", ""):
            streaming = True
            return StreamingResponse(
                r.content,
                status_code=r.status,
                headers=dict(r.headers),
                background=BackgroundTask(
                    cleanup_response, response=r, session=session
                ),
            )
        else:
            response_data = await r.json()
            return response_data
    except Exception as e:
        if r is not None:
            try:
                res = await r.json()
                print(res)
                if "error" in res:
                    error_detail = f"External: {res['error']['message'] if 'message' in res['error'] else res['error']}"
            except:
                error_detail = f"External: {e}"
        raise HTTPException(status_code=r.status if r else 500, detail=error_detail)
    finally:
        if not streaming and session:
            if r:
                r.close()
            await session.close()
