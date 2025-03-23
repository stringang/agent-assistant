from fastapi import FastAPI, APIRouter, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse

import asyncio
import json

router = APIRouter()

@router.post("/chat/completions")
async def generate_chat_completion(
    form_data: dict,
):
    payload = {**form_data}
    if "metadata" in payload:
        del payload["metadata"]

    model_id = form_data.get("model")
    model_info = Models.get_model_by_id(model_id)

    if model_info:
        if model_info.base_model_id:
            payload["model"] = model_info.base_model_id

        model_info.params = model_info.params.model_dump()

        if model_info.params:
            if (
                model_info.params.get("temperature", None) is not None
                and payload.get("temperature") is None
            ):
                payload["temperature"] = float(model_info.params.get("temperature"))

            if model_info.params.get("top_p", None) and payload.get("top_p") is None:
                payload["top_p"] = int(model_info.params.get("top_p", None))

            if (
                model_info.params.get("max_tokens", None)
                and payload.get("max_tokens") is None
            ):
                payload["max_tokens"] = int(model_info.params.get("max_tokens", None))

            if (
                model_info.params.get("frequency_penalty", None)
                and payload.get("frequency_penalty") is None
            ):
                payload["frequency_penalty"] = int(
                    model_info.params.get("frequency_penalty", None)
                )

            if (
                model_info.params.get("seed", None) is not None
                and payload.get("seed") is None
            ):
                payload["seed"] = model_info.params.get("seed", None)

            if model_info.params.get("stop", None) and payload.get("stop") is None:
                payload["stop"] = (
                    [
                        bytes(stop, "utf-8").decode("unicode_escape")
                        for stop in model_info.params["stop"]
                    ]
                    if model_info.params.get("stop", None)
                    else None
                )

        system = model_info.params.get("system", None)
        if system:
            system = prompt_template(
                system,
                **(
                    {
                        "user_name": user.name,
                        "user_location": (
                            user.info.get("location") if user.info else None
                        ),
                    }
                    if user
                    else {}
                ),
            )
            if payload.get("messages"):
                payload["messages"] = add_or_update_system_message(
                    system, payload["messages"]
                )

    else:
        pass

    model = app.state.MODELS[payload.get("model")]
    idx = model["urlIdx"]

    if "pipeline" in model and model.get("pipeline"):
        payload["user"] = {
            "name": user.name,
            "id": user.id,
            "email": user.email,
            "role": user.role,
        }

    # Check if the model is "gpt-4-vision-preview" and set "max_tokens" to 4000
    # This is a workaround until OpenAI fixes the issue with this model
    if payload.get("model") == "gpt-4-vision-preview":
        if "max_tokens" not in payload:
            payload["max_tokens"] = 4000
        log.debug("Modified payload:", payload)

    # Convert the modified body back to JSON
    payload = json.dumps(payload)


    url = app.state.config.OPENAI_API_BASE_URLS[idx]
    key = app.state.config.OPENAI_API_KEYS[idx]

    headers = {}
    headers["Authorization"] = f"Bearer {key}"
    headers["Content-Type"] = "application/json"
    if "openrouter.ai" in app.state.config.OPENAI_API_BASE_URLS[idx]:
        headers["HTTP-Referer"] = "https://openwebui.com/"
        headers["X-Title"] = "Open WebUI"

    r = None
    session = None
    streaming = False

    try:
        session = aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
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
        log.exception(e)
        error_detail = "Open WebUI: Server Connection Error"
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