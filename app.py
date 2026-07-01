import os
import json
import re
import streamlit as st
from groq import Groq
from tavily import TavilyClient
from supabase import create_client, Client
from streamlit_supabase_auth import login_form
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") # Ensure this is the anon/publishable key
MODEL_NAME = "llama-3.1-8b-instant" 

if not all([GROQ_API_KEY, TAVILY_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    st.error("Missing environment variables. Check your .env file.")
    st.stop()

# --- CLIENT INITIALIZATION ---
@st.cache_resource
def get_clients():
    return (
        Groq(api_key=GROQ_API_KEY), 
        TavilyClient(api_key=TAVILY_API_KEY),
        create_client(SUPABASE_URL, SUPABASE_KEY)
    )

groq_client, tavily_client, supabase_client = get_clients()

# --- 1. AUTHENTICATION LAYER (Sidebar) ---
st.sidebar.header("Account")
# The login form is now in the sidebar, preventing it from blocking the main UI
session = login_form(
    url=SUPABASE_URL, 
    apiKey=SUPABASE_KEY, 
    redirectUri="http://localhost:8503" # Update to Streamlit Cloud URL when deploying
)

if session:
    user_id = session['user']['id']
    user_email = session['user']['email']
    st.sidebar.success(f"Logged in as: {user_email}")
    
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
else:
    st.sidebar.info("👤 Guest Mode")
    st.sidebar.caption("Log in to save preferences and get personalized results.")

# --- 2. CONDITIONAL DB PROFILE FETCHING ---
USER_CONTEXT = ""

if session:
    # Fetch profile only once per session
    if "user_profile" not in st.session_state:
        try:
            response = supabase_client.table("user_profiles").select("*").eq("id", user_id).execute()
            st.session_state.user_profile = response.data[0] if response.data else {}
        except Exception as e:
            st.sidebar.error(f"Profile fetch error: {str(e)}")
            st.session_state.user_profile = {}

    profile = st.session_state.user_profile
    budget = profile.get('preferred_budget', 'Not specified')
    location = profile.get('default_location', 'Not specified')
    brands = profile.get('favorite_brands', [])
    brands_str = ", ".join(brands) if brands else "None"

    USER_CONTEXT = f"""
    USER PROFILE CONTEXT (Prioritize these constraints in your extraction):
    - Default Budget: {budget}
    - Default Location: {location}
    - Favorite Brands: {brands_str}
    """
else:
    # Guest Context: Explicitly instructs the Model not to hallucinate defaults
    USER_CONTEXT = """
    GUEST CONTEXT: The user is not logged in. Do not assume any specific budget, location, or brands unless explicitly stated in the user's query.
    """

# --- SYSTEM PROMPT (Dynamic Orchestration Contract) ---
SYSTEM_PROMPT = f"""You are an expert data extraction engine. Analyze the user's query and extract the core market sector, the primary intent, and any specific attributes into a strict JSON object.

{USER_CONTEXT}

CRITICAL RULES:
1. Output ONLY valid JSON. No markdown formatting, no explanations.
2. The JSON must strictly follow this schema:
{{
  "sector": "String (e.g., Retail, Real Estate, Healthcare, Automotive)",
  "intent": "String (The core action, e.g., 'buy boots', 'rent house')",
  "attributes": {{
     "key": "value (Dynamically generate keys based on context, e.g., 'budget', 'location', 'specifications')"
  }}
}}
3. If the user's query lacks specific details, use the values provided in the USER PROFILE CONTEXT to fill in the 'attributes' object.
4. If a specific attribute is not mentioned and not in the profile, do not include it.
"""

# --- DOMAIN REGISTRY (The Simulated DB Boundary) ---
DOMAIN_REGISTRY = {
    "Retail": ["takealot.com", "makro.co.za", "loot.co.za", "bobshop.co.za"],
    "Real Estate": ["property24.com", "privateproperty.co.za", "rawson.co.za", "pamgolding.co.za"],
    "Automotive": ["autotrader.co.za", "cars.co.za", "webuycars.co.za"],
    "Default": ["takealot.com", "gumtree.co.za", "junkmail.co.za"]
}

# --- TOOL REGISTRY: QUERY STRATEGIES ---
def build_realestate_query(attributes):
    location = attributes.get('location', '')
    budget = attributes.get('budget', '')
    intent = attributes.get('intent', 'property')
    return f"{intent} in {location} {budget}".strip()

def build_retail_query(attributes):
    product = attributes.get('product', 'item')
    specs = " ".join(attributes.get('specifications', []))
    return f"buy {product} {specs}".strip()

def build_default_query(attributes):
    intent = attributes.get('intent', '')
    extra = " ".join(str(v) for v in attributes.values() if v != intent)
    return f"{intent} {extra}".strip()

QUERY_REGISTRY = {
    "Real Estate": build_realestate_query,
    "Retail": build_retail_query
}

# --- UNIFIED EXECUTION ENGINE ---
def execute_market_search(sector, attributes):
    query_builder = QUERY_REGISTRY.get(sector, build_default_query)
    search_query = query_builder(attributes)
    allowed_domains = DOMAIN_REGISTRY.get(sector, DOMAIN_REGISTRY["Default"])
    
    response = tavily_client.search(
        search_query, max_results=8, include_raw_content=False, include_domains=allowed_domains
    )
    results = response.get('results', [])
    fallback_triggered = False
    
    if not results:
        print(f"[Orchestration] Zero results in restricted domains. Triggering broad web fallback.")
        fallback_triggered = True
        response = tavily_client.search(search_query, max_results=3, include_raw_content=False)
        results = response.get('results', [])
        
    return results, allowed_domains, fallback_triggered

# --- UI LAYOUT ---
st.set_page_config(page_title="Agent Action Engine", layout="wide")

st.markdown(
    """
    <style>
    .main-title { text-align: center; padding-top: 20px; padding-bottom: 20px; }
    </style>
    <h1 class="main-title">Dynamic Extraction & Action Engine (SA PoC)</h1>
    """,
    unsafe_allow_html=True
)

left_spacer, chat_container, right_spacer = st.columns([1, 2, 1])

with chat_container:
    if prompt := st.chat_input("Describe what you are looking for..."):
        st.write(f"**User Query:** {prompt}")
        
        # 1. PERCEPTION: Extract Data via Model
        with st.spinner("Agent analyzing intent..."):
            try:
                completion = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                raw_response = completion.choices[0].message.content
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                json_string = json_match.group(0) if json_match else raw_response
                extracted_data = json.loads(json_string)
                
                st.success("Data Extracted Successfully!")
                
                with st.expander("View Extracted JSON Data", expanded=False):
                    st.json(extracted_data)
                
            except Exception as e:
                st.error(f"Extraction Error: {str(e)}")
                st.stop()

        # 2. ACTION: Route via Unified Execution Engine
        sector = extracted_data.get("sector", "Unknown")
        attributes = extracted_data.get("attributes", {})
        
        with st.spinner(f"Agent searching localized market for {sector} options..."):
            try:
                search_results, searched_domains, fallback_triggered = execute_market_search(sector, attributes)
                
                st.subheader(f"Market Options & Research ({sector})")
                
                if fallback_triggered:
                    st.caption(f"🔍 Tavily searched: Broad Web (no results found in restricted domains)")
                else:
                    domains_str = ", ".join(searched_domains)
                    st.caption(f" Tavily searched: {domains_str}")
                
                if not search_results:
                    st.warning("No market options found for this query.")
                else:
                    for result in search_results:
                        st.markdown(f"### [{result.get('title', 'Untitled')}]({result.get('url', '#')})")
                        st.caption(result.get('url'))
                        st.write(result.get('content', 'No summary available.'))
                        st.divider()
                        
            except Exception as e:
                st.error(f"Action Execution Error (Tavily API): {str(e)}")