"""Mount the UI and its report route on the same Streamlit listener."""
from pathlib import Path

import streamlit as st
from starlette.routing import Route

from aso_agents.gui.artifacts import route_prefix, serve_artifact


app = st.App(Path(__file__).with_name("app.py"), routes=[
    Route(route_prefix(st.get_option("server.baseUrlPath")) + "/{token}/{relative:path}", serve_artifact),
])
