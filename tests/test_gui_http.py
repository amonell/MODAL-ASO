"""Exercise real GUI middleware, beyond Streamlit's in-process widget tests."""
import asyncio
import pytest

pytest.importorskip("streamlit")
from starlette.responses import FileResponse
from streamlit.web.server.starlette.starlette_gzip_middleware import MediaAwareGZipMiddleware


@pytest.mark.parametrize("accept_encoding", [b"gzip", b"identity"])
@pytest.mark.parametrize("pre_encoded", [True, False])
def test_gui_file_responses_send_headers_before_body(tmp_path, accept_encoding, pre_encoded):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"browser asset" * 10000)
    messages = []
    response = FileResponse(asset, headers={"content-encoding": "gzip"} if pre_encoded else {})
    middleware = MediaAwareGZipMiddleware(response)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(dict(message))

    asyncio.run(middleware({"type": "http", "method": "GET", "path": "/media/asset.bin",
                            "headers": [(b"accept-encoding", accept_encoding)],
                            "asgi": {"version": "3.0", "spec_version": "2.4"}}, receive, send))
    assert messages[0]["type"] == "http.response.start"
    assert sum(item["type"] == "http.response.start" for item in messages) == 1
    assert messages[-1]["type"] == "http.response.body"
    assert not messages[-1].get("more_body", False)
