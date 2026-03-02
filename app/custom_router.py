from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Any
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

from app.agent import get_agent

agent = get_agent()
router = APIRouter()

# @router.get("/agent/query")
# def agent_query(query: str):
#     result = agent.invoke({
#         "messages": [query],
#     })
#
#     # Store the result for later evaluation.
#     try:
#         with open("agent_eval_data.json", "r") as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         data = []
#
#     data.append({
#         "query": query,
#         "response": result["messages"][-1].content
#     })
#
#     json.dump(data, open("agent_eval_data.json", "w"), indent=2)
#
#     return {"response": result["messages"][-1].content}


class OpenAIMessageRole(str, Enum):
    system = "system"
    user = "user"
    ai = "ai"
    assistant = "assistant"
    human = "human"


class OpenAIMessageFormat(BaseModel):
    role: OpenAIMessageRole
    content: str


class OpenAIPayload(BaseModel):
    messages: List[OpenAIMessageFormat]


def serialize_event(obj):
    """Convert non-serializable objects to JSON-compatible format."""
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        return obj.dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


@router.post("/agent/stream2")
def agent_stream2(payload: OpenAIPayload):
    import queue

    # queue used to communicate between worker thread and response generator
    event_queue = queue.Queue()

    def stream_to_queue():
        """Run the agent stream in a background thread and push SSE-ready tuples."""
        try:
            for event in agent.stream({
                "messages": [msg.model_dump() for msg in payload.messages],
            }):
                try:
                    serializable_event = json.loads(
                        json.dumps(event, default=serialize_event)
                    )
                    ev_type = serializable_event.get("type", "message")
                    ev_data = json.dumps(serializable_event)
                    event_queue.put(("sse", ev_type, ev_data))
                except Exception as e:
                    print(f"Error serializing event: {e}")
                    err = json.dumps({"error": str(e)})
                    event_queue.put(("error", err))
            event_queue.put(("done", None, None))
        except Exception as e:
            print(f"Stream error: {e}")
            err = json.dumps({"error": str(e)})
            event_queue.put(("error", err))
            event_queue.put(("done", None, None))

    thread = threading.Thread(target=stream_to_queue, daemon=True)
    thread.start()

    def event_generator():
        """Yield Server-Sent-Event formatted strings until stream end."""
        while True:
            item = event_queue.get()
            if not item:
                break
            typ = item[0]
            if typ == "done":
                break
            elif typ == "sse":
                _, ev_type, ev_data = item
                yield f"event: {ev_type}\n"
                yield f"data: {ev_data}\n\n"
            elif typ == "error":
                _, err = item
                yield "event: error\n"
                yield f"data: {err}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/agent/invoke")
def agent_invoke(payload: OpenAIPayload):
    result = agent.invoke({
        "messages": [msg.model_dump() for msg in payload.messages],
    })

    return { "output": result }

