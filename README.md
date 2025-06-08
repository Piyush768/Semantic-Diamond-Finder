# Semantic-Diamond-Finder

# 💎 Smart Diamond Recommender with RAG & FAISS

This project is a multimodal AI assistant that helps users find diamonds based on natural language queries. It uses GPT for query understanding, FAISS for fast image similarity search, and Streamlit for a conversational interface — all backed by a local vector database of real diamond images.

---

## 🧠 Project Workflow

1. **Diamond Image Collection**  
   - Images of various diamond types (e.g., cushion, round, princess, oval, pear, etc.)
   - Organized into folders like:  
     `images/cushion`, `images/emerald`, `images/heart`, etc.

2. **Embedding & Vector Indexing**  
   - All images are converted into vector embeddings.
   - Stored locally using **FAISS** for fast semantic retrieval.

3. **GPT-Powered RAG Pipeline**  
   - Natural language queries like  
     _"I need a princess-cut diamond under $2000"_  
     are interpreted and semantically matched against vector embeddings.
   - Relevant image matches and metadata are fetched.

4. **Streamlit Chat Interface**  
   - Users interact with a chatbot.
   - It returns the top 5 matching diamonds with image previews and basic details.

---

## 🖼️ Supported Diamond Shapes

- Cushion  
- Emerald  
- Heart  
- Marquise  
- Oval  
- Pear  
- Princess  
- Round

---

## 💬 Example Queries

- “Show me a round diamond under $1500.”
- “I want a pear-shaped diamond with high clarity.”
- “Best heart diamonds below $2500.”

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **FAISS** – Local vector similarity search
- **OpenAI GPT (or similar)** – Query understanding
- **Streamlit** – Chatbot web UI
- **Pillow, os, json** – For image and path handling

---

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/diamond-chatbot-rag.git
cd diamond-chatbot-rag

pip install -r requirements.txt
streamlit run diamond_chatbot.py
📁 Project Structure
python
Copy
Edit
├── images/                   # Contains folders of diamond images
│   ├── round/
│   ├── cushion/
│   └── ...
├── faiss_index.bin           # Vector index for all embeddings
├── image_paths.json          # Maps FAISS indices to image files
├── query_embeddings.py       # Embeds user questions
├── diamond_chatbot.py        # Streamlit chatbot interface
├── qy.py, pi.py, ja.py       # Support scripts for filtering & logic
📷 Output Example
Ask: “I want a heart diamond under $2000”
Response: Top 5 image previews + descriptions pulled from vector DB.

📄 License
This project is under the MIT License.

