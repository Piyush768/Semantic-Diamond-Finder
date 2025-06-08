import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Diamond Suggestion Chatbot",
    page_icon="💎",
    layout="wide"
)

# Enable debugging
DEBUG = True

# Initialize session state variables properly
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_suggestions' not in st.session_state:
    st.session_state.current_suggestions = []
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = {}

# Paths to your data files
MERGED_EMBEDDINGS_PATH = "merged_embeddings.json"
DIAMOND_DATA_PATH = "merged_diamond_data.json"
IMAGES_FOLDER = "images"  # Assuming you store diamond images here

def load_data():
    """Load embeddings and diamond data"""
    embeddings_data = []
    diamond_data = []
    
    try:
        with open(MERGED_EMBEDDINGS_PATH, "r") as f:
            embeddings_data = json.load(f)
        st.sidebar.success(f"✅ Loaded {len(embeddings_data)} embeddings")
    except FileNotFoundError as e:
        st.sidebar.error(f"❌ Error loading embeddings: {e}")
        st.sidebar.info(f"Looking for: {os.path.abspath(MERGED_EMBEDDINGS_PATH)}")
    except json.JSONDecodeError as e:
        st.sidebar.error(f"❌ Invalid JSON in embeddings file: {e}")
    
    try:
        with open(DIAMOND_DATA_PATH, "r") as f:
            diamond_data = json.load(f)
        st.sidebar.success(f"✅ Loaded {len(diamond_data)} diamond records")
    except FileNotFoundError as e:
        st.sidebar.error(f"❌ Error loading diamond data: {e}")
        st.sidebar.info(f"Looking for: {os.path.abspath(DIAMOND_DATA_PATH)}")
    except json.JSONDecodeError as e:
        st.sidebar.error(f"❌ Invalid JSON in diamond data file: {e}")
    
    return embeddings_data, diamond_data

def extract_embeddings(embeddings_data):
    """Extract the text embeddings from the merged embeddings data"""
    text_embeddings = []
    for item in embeddings_data:
        if "text_embedding" in item:
            text_embeddings.append(item["text_embedding"])
    
    if DEBUG and not text_embeddings:
        st.sidebar.warning("⚠️ No text embeddings found in data")
    
    return np.array(text_embeddings)

def query_to_embedding(query, embeddings_data, model=None):
    """
    Convert user query to embedding
    
    Note: This is a placeholder. You would need to use the same model
    that generated your existing embeddings.
    """
    try:
        # This assumes you have a function in another file
        from query_embeddings import get_text_embedding
        embedding = get_text_embedding(query)
        if DEBUG:
            st.sidebar.success("✅ Generated embedding using query_embeddings.py")
        return embedding
    except ImportError:
        st.sidebar.warning("⚠️ Using mock embedding for demo purposes")
        # Return a random embedding of the same dimension as your data
        if embeddings_data and len(embeddings_data) > 0 and "text_embedding" in embeddings_data[0]:
            sample_dimension = len(embeddings_data[0]["text_embedding"])
            if DEBUG:
                st.sidebar.info(f"📏 Using embedding dimension: {sample_dimension}")
            return np.random.rand(sample_dimension)
        else:
            st.sidebar.error("❌ Could not determine embedding dimension")
            return np.random.rand(512)  # Default fallback dimension

def find_similar_diamonds(query_embedding, all_embeddings, embeddings_data, diamond_data, top_n=3):
    """Find the most similar diamonds based on embedding similarity"""
    if len(all_embeddings) == 0:
        st.warning("⚠️ No embeddings available for comparison")
        return []
        
    # Calculate similarity
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]
    
    # Get indices of top matches
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Track matched and unmatched diamonds for debugging
    matched_diamonds = []
    unmatched_ids = []
    
    # Get the corresponding diamond data and similarity scores
    results = []
    for idx in top_indices:
        if idx < len(embeddings_data):
            # Try both potential ID fields
            diamond_id = embeddings_data[idx].get("image_path", "")
            if not diamond_id:
                diamond_id = embeddings_data[idx].get("id", "")
            
            # Save this for debugging
            if DEBUG:
                st.session_state.debug_info["embedding_ids"] = [
                    {"idx": i, "id": item.get("image_path", "") or item.get("id", "Unknown")}
                    for i, item in enumerate(embeddings_data[:5])  # Show first 5 for brevity
                ]
            
            # Find the corresponding diamond in diamond_data
            diamond_info = None
            
            # Try different fields that might contain the ID
            for d in diamond_data:
                # Check all possible ID fields
                potential_ids = [
                    d.get("id", ""),
                    d.get("image_path", ""),
                    d.get("diamond_id", ""),
                    d.get("sku", ""),
                    os.path.basename(d.get("image_path", "")) if d.get("image_path") else ""
                ]
                
                # Check if any of the diamond's IDs match our embedding ID
                if diamond_id in potential_ids or any(pid == diamond_id for pid in potential_ids if pid):
                    diamond_info = d
                    matched_diamonds.append(diamond_id)
                    break
            
            if not diamond_info:
                unmatched_ids.append(diamond_id)
                # Create a minimal record for display if we can't find the diamond
                diamond_info = {
                    "id": diamond_id,
                    "note": "Limited data available",
                    "similarity_score": similarities[idx]
                }
            
            results.append({
                "diamond": diamond_info,
                "similarity": float(similarities[idx]),
                "index": int(idx),
                "image_path": diamond_id
            })
    
    if DEBUG:
        st.session_state.debug_info["matched_diamonds"] = matched_diamonds
        st.session_state.debug_info["unmatched_ids"] = unmatched_ids
        st.sidebar.info(f"🔍 Found {len(results)} similar diamonds")
        if unmatched_ids:
            st.sidebar.warning(f"⚠️ {len(unmatched_ids)} diamonds couldn't be matched to data")
    
    return results

def get_diamond_image_path(image_path):
    """Get the full path to a diamond image"""
    # Extract just the filename if it's a full path
    filename = os.path.basename(image_path)
    
    # Check if image exists in the images folder
    img_path = os.path.join(IMAGES_FOLDER, filename)
    if os.path.exists(img_path):
        return img_path
    
    # If not found, check if the full path exists
    if os.path.exists(image_path):
        return image_path
    
    # Otherwise return the original path for error handling
    return img_path

def display_diamond_suggestion(diamond_info, col, index):
    """Display a diamond suggestion with flexible attribute handling"""
    diamond = diamond_info.get("diamond", {})
    similarity = diamond_info.get("similarity", 0)
    image_path = diamond_info.get("image_path", "")
    
    # Handle different possible diamond data structures
    if not diamond:
        col.warning("⚠️ Diamond information not available")
        if DEBUG:
            col.json(diamond_info)
        return
    
    col.subheader(f"Diamond Match (Similarity: {similarity:.2f})")
    
    # Try to display the diamond image
    if image_path:
        try:
            img_path = get_diamond_image_path(image_path)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                col.image(img, caption=f"Diamond {diamond.get('id', '')}", use_column_width=True)
            else:
                if DEBUG:
                    col.info(f"ℹ️ Image not found: {img_path}")
                # Display a placeholder image
                col.image("https://via.placeholder.com/200x200.png?text=Diamond+Image", use_column_width=True)
        except Exception as e:
            if DEBUG:
                col.error(f"❌ Error loading image: {e}")
            col.image("https://via.placeholder.com/200x200.png?text=Diamond+Image", use_column_width=True)
    
    # Create a card-like container for diamond details
    with col.container():
        # Display all attributes for this diamond
        # We'll use a more flexible approach that doesn't assume specific attributes
        
        # Extract all available attributes
        attributes = {}
        for key, value in diamond.items():
            if key not in ["id", "image_path", "note"] and value is not None:
                # Format key for display
                display_key = key.replace("_", " ").title()
                
                # Format value based on type
                if key == 'price' and isinstance(value, (int, float)):
                    formatted_value = f"${value:,.2f}"
                elif isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                elif isinstance(value, (int, str)):
                    formatted_value = str(value)
                else:
                    formatted_value = str(value)
                
                attributes[display_key] = formatted_value
        
        # If we have a "note" field, display it
        if "note" in diamond:
            col.info(diamond["note"])
        
        # Define key attributes to display prominently if they exist
        key_attributes = ['Carat', 'Cut', 'Color', 'Clarity', 'Price']
        available_key_attrs = [k for k in key_attributes if k in attributes]
        
        # Display key attributes with metrics
        if available_key_attrs:
            key_cols = col.columns(len(available_key_attrs))
            for i, attr in enumerate(available_key_attrs):
                key_cols[i].metric(attr, attributes[attr])
                # Remove from attributes dict to avoid duplicate display
                attributes.pop(attr, None)
        
        # Display remaining attributes
        if attributes:
            with col.expander("All Diamond Details", expanded=True):
                for attr, value in attributes.items():
                    col.write(f"**{attr}:** {value}")
        
        # Display raw data in debug mode
        if DEBUG:
            with col.expander("Raw Diamond Data", expanded=False):
                col.json(diamond)
        
        # Add a select button with a unique key using both diamond_id and index
        diamond_id = diamond.get("id", "") or diamond.get("diamond_id", "") or "unknown"
        unique_key = f"select_{diamond_id}_{index}"  # Adding index to ensure uniqueness
        if col.button("Select This Diamond", key=unique_key):
            st.session_state.messages.append({
                "role": "user", 
                "content": f"I'd like more information about diamond {diamond_id}"
            })
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"Great choice! Here's more information about diamond {diamond_id}. What specific details would you like to know?"
            })
            st.experimental_rerun()

def process_query(query):
    """Process the user query and display results"""
    if not query or not query.strip():
        return
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Convert query to embedding
    with st.spinner("Finding the perfect diamonds for you..."):
        query_embedding = query_to_embedding(query, embeddings_data)
        
        # Find similar diamonds
        all_embeddings = extract_embeddings(embeddings_data)
        similar_diamonds = find_similar_diamonds(
            query_embedding, 
            all_embeddings,
            embeddings_data,
            diamond_data,
            top_n=top_n
        )
    
    # Format response
    if not similar_diamonds:
        response = "I couldn't find any matching diamonds. Could you try describing what you're looking for differently?"
    else:
        response = f"Based on your request, I found these diamonds that might interest you:"
        
        # Debug: Directly display what was found
        if DEBUG:
            st.session_state.debug_info["diamonds_found"] = len(similar_diamonds)
            st.session_state.debug_info["diamond_ids"] = [
                d.get("diamond", {}).get("id", "No ID") for d in similar_diamonds
            ]
    
    # Add chatbot response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Store the current suggestions for display
    st.session_state.current_suggestions = similar_diamonds

# Load data at startup
embeddings_data, diamond_data = load_data()

# Sidebar
with st.sidebar:
    st.title("💎 Diamond Finder")
    st.markdown("---")
    st.markdown("### About")
    st.write("This app helps you find the perfect diamond based on your preferences.")
    st.markdown("---")
    
    # Advanced options
    st.subheader("Settings")
    top_n = st.slider("Number of suggestions", min_value=1, max_value=5, value=3)
    
    # Files information
    st.subheader("Data Files")
    st.write(f"Embeddings: {MERGED_EMBEDDINGS_PATH}")
    st.write(f"Diamond data: {DIAMOND_DATA_PATH}")
    st.write(f"Images folder: {IMAGES_FOLDER}")
    
    if st.button("Reload Data"):
        embeddings_data, diamond_data = load_data()
        st.success("Data reloaded!")

# Main content
st.title("💎 Diamond Suggestion Chatbot")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Display diamond suggestions after the last assistant message
    if len(st.session_state.current_suggestions) > 0:
        diamond_container = st.container()
        with diamond_container:
            # Debug statement to confirm we have suggestions to display
            if DEBUG:
                st.write(f"Displaying {len(st.session_state.current_suggestions)} diamonds")
            
            # Use columns for displaying diamonds
            cols = st.columns(min(3, len(st.session_state.current_suggestions)))
            for i, (col, diamond_info) in enumerate(zip(cols, st.session_state.current_suggestions)):
                display_diamond_suggestion(diamond_info, col, i)  # Pass the index to ensure unique keys
    elif DEBUG and st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        # Display a warning if we should be showing diamonds but aren't
        st.warning("No diamond suggestions to display despite assistant response")

# User input
user_query = st.chat_input("Describe the diamond you're looking for...")
if user_query:
    process_query(user_query)

# Initial welcome message
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("Hello! I'm your diamond assistant. Tell me what kind of diamond you're looking for, and I'll suggest some options that might be perfect for you.")

# Display error if data failed to load
if not embeddings_data or not diamond_data:
    st.error("❌ Failed to load necessary data files. Please check the file paths in the sidebar.")
    st.info("Make sure your data files exist and are in the correct format.")

# Add a sample diamond display for testing
if DEBUG and (not diamond_data or st.button("Show Example Diamond")):
    st.subheader("Demo Diamond (Static Example)")
    
    # Create columns
    cols = st.columns(3)
    
    # Create sample diamond data for testing display
    sample_diamond = {
        "id": "SAMPLE-001",
        "carat": 1.5,
        "cut": "Excellent",
        "color": "D",
        "clarity": "VVS1",
        "price": 12500,
        "depth": 61.2,
        "table": 57.0,
        "length": 7.25,
        "width": 7.21,
        "height": 4.43,
        "certificate": "GIA",
        "cert_number": "12345678"
    }
    
    # Display the sample diamond
    display_diamond_suggestion({
        "diamond": sample_diamond,
        "similarity": 0.95,
        "image_path": "sample.jpg"
    }, cols[0], 0)  # Pass index 0 for the sample diamond

# Add debug information section with embedding details hidden
if DEBUG:
    st.markdown("---")
    with st.expander("🔍 Debug Information", expanded=False):
        st.markdown("### Session State")
        st.write("Messages count:", len(st.session_state.messages))
        st.write("Current suggestions:", len(st.session_state.current_suggestions))
        
        st.markdown("### Debug Info")
        # Create a copy of debug info without embeddings for display
        debug_info_display = st.session_state.debug_info.copy()
        st.json(debug_info_display)
        
        st.markdown("### Data Loading")
        st.write("Embeddings data length:", len(embeddings_data) if embeddings_data else 0)
        st.write("Diamond data length:", len(diamond_data) if diamond_data else 0)
        
        st.markdown("### File System")
        st.write("Working directory:", os.getcwd())
        try:
            st.write("Files in directory:", os.listdir())
        except:
            st.write("Could not list files in directory")
        
        try:
            if os.path.exists(IMAGES_FOLDER):
                st.write(f"Images in {IMAGES_FOLDER}:", os.listdir(IMAGES_FOLDER))
            else:
                st.write(f"Images folder '{IMAGES_FOLDER}' doesn't exist")
        except:
            st.write("Could not check images folder")
        
        # Display sample of diamond data structure if available
        if diamond_data and len(diamond_data) > 0:
            st.markdown("### Sample Diamond Data Structure")
            st.json(diamond_data[0])
        
        # Display sample of embeddings data structure if available
        if embeddings_data and len(embeddings_data) > 0:
            st.markdown("### Sample Embedding Data Structure")
            # Avoid including potentially large embedding arrays
            sample = {k: v for k, v in embeddings_data[0].items() if k != "text_embedding"}
            if "text_embedding" in embeddings_data[0]:
                sample["text_embedding"] = f"[Array with {len(embeddings_data[0]['text_embedding'])} elements]"
            st.json(sample)