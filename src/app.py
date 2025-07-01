# ===============================================
# Agricultural Agentic AI – LangChain Edition
# ===============================================

# --- Imports --- #
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

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# --- OpenAI model config --- #
LLM_MODEL = "gpt-4o-mini"
llm_stream = ChatOpenAI(
    model_name=LLM_MODEL,
    streaming=True,            # Stream tokens to Streamlit
    temperature=0,
    api_key=st.secrets["openai"]["api_key"],
)

# --- Constants --- #
LOCATION_IDENTIFIER = "LocationIdentifier"
DATA_ANALYST = "DataAnalyst"
TERMINATION_KEYWORD = "yes"

AVATARS = {
    "user": "🚜",
    LOCATION_IDENTIFIER: "🌍",
    DATA_ANALYST: "📊",
}

NASA_PARAMETER = "GWETROOT"  # Soil moisture 0‑100 cm

# --------------------------------------------------------------------------- #
#                                Helper tools                                 #
# --------------------------------------------------------------------------- #

def duckduckgo_local_agri_search(lat: float, lon: float) -> str:
    """
    Reverse‑geocode (lat, lon) and run two DuckDuckGo queries:
      1. Agricultural regulations
      2. Recommended crops / market demand

    Returns a Markdown‑formatted string summarising the top 2 hits per query.
    """
    from duckduckgo_search import DDGS  # local import keeps startup fast

    geolocator = Nominatim(user_agent="agricultural_agentic_ai_app")
    location = geolocator.reverse((lat, lon), language="en")

    if location is None:
        return "Unable to determine location from coordinates."

    address = location.raw.get("address", {})
    country = address.get("country", "")
    region = (
        address.get("state")
        or address.get("province")
        or address.get("region")
        or address.get("county")
        or address.get("district")
        or ""
    )
    if not country and not region:
        return "Unable to determine country or region from coordinates."

    region_prefix = f"{region}, " if region else ""
    location_name = location.address
    queries = [
        f"agricultural regulations in {region_prefix}{country}",
        f"recommended crops and agricultural market demand in {region_prefix}{country}",
    ]
    results = []
    with DDGS() as ddgs:
        for q in queries:
            hits = ddgs.text(q, max_results=2)
            if hits:
                formatted = "\n\n".join(
                    f"**{h['title']}**\n{h['body']}\n{h['href']}" for h in hits
                )
                results.append(f"### Search Results for '{q}':\n{formatted}")
            else:
                results.append(f"No results found for '{q}'.")
    return f"**Resolved location:** {location_name}\n\n" + "\n\n".join(results)


def fetch_soil_moisture_forecast(
    lat: float, lon: float, parameter: str = NASA_PARAMETER
) -> pd.DataFrame:
    """
    Fetch daily soil‑moisture data from NASA POWER, train Prophet,
    and return a one‑year forecast (`ds`, `yhat`).
    """
    start_date = "19810101"
    end_date = datetime.utcnow().strftime("%Y%m%d")
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={parameter}&community=ag&longitude={lon}"
        f"&latitude={lat}&start={start_date}&end={end_date}&format=JSON"
    )
    r = requests.get(url)
    r.raise_for_status()
    series_dict = r.json()["properties"]["parameter"][parameter]
    df = pd.DataFrame.from_dict(series_dict, orient="index").replace(-999, np.nan)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df_prophet = df.reset_index().rename(columns={"index": "ds", 0: "y"})

    model = Prophet(weekly_seasonality=False, yearly_seasonality=True)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]].tail(365)


# --------------------------------------------------------------------------- #
#                      LangChain agent definitions                            #
# --------------------------------------------------------------------------- #

# 1️⃣  LocationIdentifier agent  ------------------------------------------------
location_identifier_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are an agricultural LOCATION IDENTIFIER. "
                "Summarise findings into **exactly four** sections with headings:\n"
                "1. Location and Climate Overview\n"
                "2. Agricultural Regulations\n"
                "3. Best Crops & Local Demand Insights\n"
                "4. Recommendation Summary\n\n"
                "If any query returned no results, state that explicitly."
            )
        ),
        MessagesPlaceholder(variable_name="search_md"),  # tool output
    ]
)


def location_identifier_agent(search_md: str) -> str:
    """Run the LocationIdentifier LLM with `search_md` injected."""
    chain = location_identifier_prompt | llm_stream
    return chain.invoke({"search_md": [HumanMessage(content=search_md)]}).content


# 2️⃣  DataAnalyst agent  -------------------------------------------------------
data_analyst_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are an agricultural SOIL‑MOISTURE DATA ANALYST.\n"
                "You will receive a JSON list of forecasted dates "
                "with the associated soil‑moisture values (0‑1).\n"
                "Interpret trends **without** revealing numeric thresholds.\n"
                "Speak in future tense.  Tasks:\n"
                " • Detect and date upcoming dryness periods.\n"
                " • Detect and date high‑moisture periods.\n"
                " • Describe overall trend (increasing / decreasing / stable).\n"
                " • Give clear, actionable irrigation or drainage advice "
                "for farmers.\n"
            )
        ),
        MessagesPlaceholder(variable_name="forecast_json"),
    ]
)


def data_analyst_agent(forecast_df: pd.DataFrame) -> str:
    forecast_json = forecast_df.to_dict(orient="records")
    chain = data_analyst_prompt | llm_stream
    return chain.invoke({"forecast_json": [HumanMessage(content=json.dumps(forecast_json))]}).content


# --------------------------------------------------------------------------- #
#                       Streamlit user interface                               #
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Agricultural Agentic AI", layout="wide")
st.title("🌾 Agricultural Agentic AI (LangChain Edition)")
st.write("### 1 ⃣ Pick a location on the map")

# Interactive world map
world_map = folium.Map(location=[20, 0], zoom_start=2)
map_data = st_folium(
    world_map,
    width=1200,
    height=600,
    returned_objects=["last_clicked"],
)

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []

# --- When user clicks on the map --- #
if map_data and map_data["last_clicked"]:
    lat = float(map_data["last_clicked"]["lat"])
    lon = float(map_data["last_clicked"]["lng"])

    with st.spinner("Fetching location context …"):
        search_md = duckduckgo_local_agri_search(lat, lon)
    loc_response = location_identifier_agent(search_md)

    with st.spinner("Fetching NASA soil‑moisture forecast …"):
        forecast_df = fetch_soil_moisture_forecast(lat, lon)
    analyst_response = data_analyst_agent(forecast_df)

    # Store in chat history
    st.session_state.history.extend(
        [
            (LOCATION_IDENTIFIER, loc_response),
            (DATA_ANALYST, analyst_response),
        ]
    )

# --- Display conversation --- #
st.markdown("### 🤖 Conversation")
for sender, msg in st.session_state.history:
    with st.chat_message(sender, avatar=AVATARS.get(sender, "🤖")):
        st.markdown(msg)

# --- Follow‑up chat --- #
user_input = st.chat_input(
    "Ask a follow‑up question about farming, regulations, or the forecast..."
)
if user_input:
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_input)

    # Combine all previous AI messages as context
    context = "\n\n".join(msg for _, msg in st.session_state.history)

    followup_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "Act as an expert agricultural assistant.\n"
                    "Use the existing context below to answer the user's question.\n"
                    "If the question is unrelated to agriculture, politely refuse."
                )
            ),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{user_input}"),
        ]
    )
    reply = (followup_prompt | llm_stream).invoke({}).content

    st.session_state.history.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)
