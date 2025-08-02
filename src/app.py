# --- Import Packages --- #
import asyncio
from datetime import datetime
import json
import os
from typing import List, Tuple

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from prophet import Prophet
from streamlit_folium import st_folium

from langchain_community.chat_models.openai import ChatOpenAI as OpenRouterChat
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage

# --- Model config --- #
LLM_MODEL = "mistralai/mixtral-8x7b"

# Create a callback that writes tokens into Streamlit
class StreamlitTokenStreamer(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.current = ""
    def on_llm_new_token(self, token: str, **kwargs):
        self.current += token
        # overwrite previous text
        self.placeholder.markdown(self.current, unsafe_allow_html=True)

# Streaming LLM client
llm_stream = OpenRouterChat(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    streaming=True,
    api_key=st.secrets["OPENROUTER_API_KEY"],
    callback_manager=None  # we’ll attach per-call
)

# --- Constants --- #
LOCATION_IDENTIFIER = "LocationIdentifier"
DATA_ANALYST      = "DataAnalyst"
AVATARS = {
    "user": "🚜",
    LOCATION_IDENTIFIER: "🌍",
    DATA_ANALYST: "📊",
}

NASA_PARAMETER = "GWETROOT"  # Soil moisture 0-100 cm

# --------------------------------------------------------------------------- #
#                                Helper tools                                 #
# --------------------------------------------------------------------------- #

def duckduckgo_local_agri_search(lat: float, lon: float) -> str:
    from duckduckgo_search import DDGS
    geolocator = Nominatim(user_agent="agri_ai_app")
    location = geolocator.reverse((lat, lon), language="en")
    if location is None:
        return "Unable to determine location from coordinates."
    addr = location.raw.get("address", {})
    country = addr.get("country","")
    region  = addr.get("state") or addr.get("province") or ""
    if not country and not region:
        return "Unable to determine country or region from coordinates."
    region_prefix = f"{region}, " if region else ""
    name = location.address
    queries = [
        f"agricultural regulations in {region_prefix}{country}",
        f"recommended crops and market demand in {region_prefix}{country}",
    ]
    results = []
    with DDGS() as ddgs:
        for q in queries:
            hits = ddgs.text(q, max_results=2)
            if hits:
                formatted = "\n\n".join(
                    f"**{h['title']}**\n{h['body']}\n{h['href']}"
                    for h in hits
                )
                results.append(f"### Results for '{q}':\n{formatted}")
            else:
                results.append(f"No results for '{q}'.")
    return f"**Resolved location:** {name}\n\n" + "\n\n".join(results)

def fetch_soil_moisture_forecast(lat: float, lon: float) -> pd.DataFrame:
    start = "19810101"
    end   = datetime.utcnow().strftime("%Y%m%d")
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={NASA_PARAMETER}&community=ag"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start}&end={end}&format=JSON"
    )
    r = requests.get(url); r.raise_for_status()
    series = r.json()["properties"]["parameter"][NASA_PARAMETER]
    df = pd.DataFrame.from_dict(series, orient="index").replace(-999, np.nan)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    prophet_df = df.reset_index().rename(columns={"index":"ds", 0:"y"})
    m = Prophet(weekly_seasonality=False, yearly_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    return forecast[["ds","yhat"]].tail(365)

# --------------------------------------------------------------------------- #
#                      Agent prompt templates                                 #
# --------------------------------------------------------------------------- #

location_identifier_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        content=(
            "You are an agricultural LOCATION IDENTIFIER. Summarise into exactly four "
            "sections with headings:\n"
            "1. Location and Climate Overview\n"
            "2. Agricultural Regulations\n"
            "3. Best Crops & Local Demand Insights\n"
            "4. Recommendation Summary\n"
            "If a query returned no results, state that explicitly."
        )
    ),
    MessagesPlaceholder(variable_name="search_md"),
])

data_analyst_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        content=(
            "You are an agricultural SOIL-MOISTURE DATA ANALYST. You receive a JSON list "
            "of dates (`ds`) and soil-moisture (`yhat`). Interpret trends without numeric "
            "thresholds. Speak in future tense. Detect upcoming dryness and high-moisture "
            "periods with dates, describe overall trend, and give irrigation/drainage advice."
        )
    ),
    MessagesPlaceholder(variable_name="forecast_json"),
])

# --------------------------------------------------------------------------- #
#                       Streamlit UI & streaming logic                        #
# --------------------------------------------------------------------------- #

st.set_page_config(layout="wide")
st.title("🌾 Agricultural Agentic AI")
st.write("### Select a location on the map")

# Map
world_map = folium.Map(location=[20,0], zoom_start=2)
map_data = st_folium(world_map, width=1200, height=600, returned_objects=["last_clicked"])

if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str,str]] = []

def run_streaming_agent(name: str, prompt: ChatPromptTemplate, tool_output: str) -> None:
    """Run one of the agents in streaming mode and append to history."""
    # render a new chat message placeholder
    with st.chat_message(name, avatar=AVATARS[name]):
        placeholder = st.empty()
        streamer = StreamlitTokenStreamer(placeholder)
        # format messages
        msgs = prompt.format_messages(**{prompt.input_variables[0]: [HumanMessage(content=tool_output)]})
        # attach our callback for this call
        llm_stream.callback_manager = llm_stream.callback_manager.clone().add_handler(streamer)
        # trigger streaming call
        llm_stream(messages=msgs)
        # save final text
        st.session_state.history.append((name, streamer.current))

# On click: fetch and stream
if map_data and map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    st.spinner("Fetching location context …")
    md = duckduckgo_local_agri_search(lat, lon)
    run_streaming_agent(LOCATION_IDENTIFIER, location_identifier_prompt, md)

    st.spinner("Fetching soil-moisture forecast …")
    df = fetch_soil_moisture_forecast(lat, lon)
    # convert to JSON text for prompt
    forecast_text = json.dumps(df.assign(ds=df.ds.astype(str)).to_dict(orient="records"))
    run_streaming_agent(DATA_ANALYST, data_analyst_prompt, forecast_text)

# Display history
st.markdown("### 🤖 Chat History")
for sender, msg in st.session_state.history:
    with st.chat_message(sender, avatar=AVATARS.get(sender)):
        st.markdown(msg)

# Follow-up
user_input = st.chat_input("Ask a follow-up question…")
if user_input:
    # echo user
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_input)
    # build context
    context = "\n\n".join(msg for _,msg in st.session_state.history)
    followup_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                "Act as an expert agricultural assistant. Use the existing context to answer. "
                "If unrelated to agriculture, politely refuse."
            )
        ),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{user_input}"),
    ])
    run_streaming_agent("assistant", followup_prompt, None)
