"""Is the local model service answering? (#62)

A simulation or preparation started against a dead model server used to run
to the end on failed decisions and report "completed". The check is a cheap
GET of ``<base_url>/models`` and applies to local endpoints only: a hosted
API needs a key and its own error reporting. Any HTTP answer below 500
counts as up: a keyed server answers 401 and a non-OpenAI one 404, and both
are reachable. Loopback probes bypass proxy settings, as the other local
clients do.
"""

from __future__ import annotations

import httpx

from ..profiles import is_local_url


def check_model_service(base_url: str | None, timeout: float = 3.0) -> str | None:
    """None when the service answers (or is not local); otherwise the reason."""

    if not base_url or not is_local_url(base_url):
        return None
    url = base_url.rstrip("/") + "/models"
    try:
        response = httpx.get(url, timeout=timeout, trust_env=False)
    except httpx.HTTPError as error:
        return f"模型服務無回應：{base_url}（{type(error).__name__}）"
    if response.status_code >= 500:
        return f"模型服務回應異常：{base_url}（HTTP {response.status_code}）"
    return None
