"""Cancellation state for background Responses API streams.

The browser may stop reading before OpenAI has published the terminal response.
Keeping the provider response id lets us cancel it and still reconcile its exact
usage from the terminal response.
"""
from dataclasses import dataclass, field
from threading import Lock

from loguru import logger


@dataclass
class StreamControl:
    client: object | None = None
    response_id: str | None = None
    cancel_requested: bool = False
    cancel_sent: bool = False
    lock: Lock = field(default_factory=Lock)


_controls: dict[str, StreamControl] = {}
_controls_lock = Lock()


def start(stream_id: str, client: object) -> StreamControl:
    with _controls_lock:
        control = _controls.setdefault(stream_id, StreamControl())
    with control.lock:
        control.client = client
    return control


def attach_response(control: StreamControl, response_id: str) -> None:
    with control.lock:
        control.response_id = response_id
    _send_cancel_if_requested(control)


def request_cancel(stream_id: str) -> bool:
    with _controls_lock:
        control = _controls.setdefault(stream_id, StreamControl())
    with control.lock:
        control.cancel_requested = True
    return _send_cancel_if_requested(control)


def is_cancel_requested(control: StreamControl | None) -> bool:
    if control is None:
        return False
    with control.lock:
        return control.cancel_requested


def finish(stream_id: str, control: StreamControl | None) -> None:
    if control is None:
        return
    with _controls_lock:
        if _controls.get(stream_id) is control:
            _controls.pop(stream_id, None)


def _send_cancel_if_requested(control: StreamControl) -> bool:
    with control.lock:
        if not control.cancel_requested or control.cancel_sent or not control.client or not control.response_id:
            return False
        control.cancel_sent = True
        client, response_id = control.client, control.response_id
    try:
        client.responses.cancel(response_id)
        logger.info("Cancelled background OpenAI response {}", response_id)
    except Exception as error:
        # The response may have already reached a terminal state; retrieval will
        # still reconcile any provider-reported usage.
        logger.warning("Could not cancel OpenAI response {}: {}", response_id, type(error).__name__)
    return True
