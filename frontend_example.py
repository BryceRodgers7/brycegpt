"""
Example Streamlit Frontend for VoyagerGPT Backend
This file demonstrates how to integrate with the VoyagerGPT API
"""

import streamlit as st
import requests
import time

# Configuration - Update this with your deployed Cloud Run URL
API_URL = "http://localhost:8080"  # Change to your Cloud Run URL after deployment
# Example: API_URL = "https://brycegpt-xxxxx-uc.a.run.app"

# Page setup
st.set_page_config(page_title="VoyagerGPT", page_icon="🚀")
st.title('🚀 Voyager GPT')
st.subheader("A bigram GPT built from scratch!")

github = "https://github.com/BryceRodgers7/VoyagerGPT"
st.write("View the code [here](%s)" % github)

st.info("This app now uses a backend API deployed on Google Cloud Run!")

# Sidebar controls
st.sidebar.title('VoyagerGPT Panel')
input_seed = st.sidebar.number_input(
    "Seed number", 
    step=1, 
    value=1337, 
    key="seed_input",
    help="Random seed for reproducible generation"
)
temperature = st.sidebar.slider(
    'Temperature', 
    min_value=0.01, 
    max_value=1.0, 
    value=0.1, 
    step=0.01, 
    key="temp_input",
    help="Higher temperature = more random output"
)
max_tokens = st.sidebar.slider(
    'Max Tokens',
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    key="max_tokens_input",
    help="Maximum number of tokens to generate"
)

st.divider()

# Display vocabulary info
if st.sidebar.checkbox("Show Vocabulary Info"):
    try:
        response = requests.get(f"{API_URL}/vocab", timeout=5)
        if response.status_code == 200:
            vocab_info = response.json()
            st.sidebar.write(f"**Vocab Size:** {vocab_info['vocab_size']}")
            st.sidebar.write(f"**Block Size:** {vocab_info['block_size']}")
            with st.sidebar.expander("View Characters"):
                st.write(" ".join(vocab_info['characters']))
    except Exception as e:
        st.sidebar.error(f"Could not fetch vocab info: {e}")

# Check API health
try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data["status"] == "healthy":
            st.sidebar.success("✅ API Connected")
        else:
            st.sidebar.warning("⚠️ API Unhealthy")
    else:
        st.sidebar.error("❌ API Not Responding")
except Exception as e:
    st.sidebar.error(f"❌ Cannot connect to API: {e}")
    st.warning(f"Make sure the API is running at: {API_URL}")

# Initialize session state for context management
if 'generated_context' not in st.session_state:
    st.session_state.generated_context = None
    st.session_state.prev_seed = input_seed
    st.session_state.prev_temp = temperature

# Reset context if seed or temperature changes
if st.session_state.prev_seed != input_seed or st.session_state.prev_temp != temperature:
    st.session_state.generated_context = None
    st.session_state.prev_seed = input_seed
    st.session_state.prev_temp = temperature

# Generation controls
col1, col2 = st.columns(2)

with col1:
    generate_button = st.button("🎬 Generate Star Trek Text!", use_container_width=True)

with col2:
    if st.button("🔄 Reset Context", use_container_width=True):
        st.session_state.generated_context = None
        st.success("Context reset!")
        st.rerun()

# Generation logic
if generate_button:
    try:
        with st.spinner(f"Generating with seed {input_seed} and temperature {temperature}... please wait"):
            start_time = time.time()
            
            # Prepare request payload
            payload = {
                "seed": input_seed,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "context": st.session_state.generated_context
            }
            
            # Call the API
            response = requests.post(
                f"{API_URL}/generate",
                json=payload,
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Update context for continuation
                st.session_state.generated_context = result["tokens"]
                
                # Display the generated text
                st.success("Generation complete!")
                
                # Format and display text
                formatted_text = result["text"].replace('\n', '<br>')
                st.markdown(
                    f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>"
                    f"<p style='font-family: monospace;'>{formatted_text}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                # Show metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Generation Time", f"{result['generation_time']:.2f}s")
                with col2:
                    st.metric("Tokens Generated", len(result['tokens']))
                with col3:
                    st.metric("Context Size", len(st.session_state.generated_context))
                
            else:
                st.error(f"Generation failed: {response.status_code}")
                st.json(response.json())
                
    except requests.exceptions.Timeout:
        st.error("Request timed out. The generation is taking too long. Try reducing max_tokens.")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_URL}. Make sure it's running!")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Information section
st.divider()
st.markdown("### 📚 About VoyagerGPT")
st.write("""
VoyagerGPT has over **10M parameters** and was trained on Star Trek scripts. 
It uses a Transformer architecture with:
- **6 attention heads**
- **6 transformer layers**
- **384 embedding dimensions**
- **256 token context window**

The model runs on a backend API deployed on Google Cloud Run, which allows for:
- ✅ Scalable inference
- ✅ Separation of concerns
- ✅ Better performance
- ✅ Easy deployment and updates
""")

st.markdown("### 🎮 Tips for Best Results")
st.write("""
- **Lower temperature** (0.1-0.3): More coherent, predictable text
- **Higher temperature** (0.7-1.0): More creative, random text
- **Seed**: Use the same seed for reproducible results
- **Context**: Click "Generate" multiple times to continue the story
- **Reset**: Clear context to start fresh
""")

