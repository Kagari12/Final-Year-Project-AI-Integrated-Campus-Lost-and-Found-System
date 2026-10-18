# 🛰️ AI-Integrated Web-Based Campus Lost & Found System (6001CEM)

**Author:** Low Justin (Coventry ID: 16290525)  
**Supervisor:** Ms Nor Hanina Zainal Abidin  
**Institution:** School of Computing, INTI International College Penang (in collaboration with Coventry University, UK)

---

Project Overview
An intelligent, centralized university Lost & Found web portal that automates property reporting, spatial tracking, and ownership verification using **Convolutional Neural Networks (CNN)**, **Levenshtein Distance Fuzzy Matching (NLP)**, **6-Level 3D Isometric Floor Mapping**, and **HTML5 Geolocation**.

Key Features
6-Level 3D Isometric Campus Floor Selector (`report_item.html`):** Interactive 3D building blueprint (Level 6 down to Underground Carpark) with single-floor isolation and individual room tags (`LR301`–`LR308`, `LR401`–`LR408`, `LR501`–`LR508`, Labs, Cafeteria, Study Areas).
Automated CNN Image Tagging (`ai_engine.py`):** Extracts object recognition tags automatically from uploaded item photos.
Fault-Tolerant AI Smart Search (`smart_search.html`):** Uses Levenshtein distance fuzzy matching to connect lost and found reports even when queries contain typos or different vocabulary.
360° Multi-Angle Inspection Viewer:** Interactive click-and-drag 360° photo inspection modal across up to 6 uploaded angles.
Live Campus Spatial Analytics (`campus_map.html`):** Real-time Leaflet.js GPS pin map and high-risk campus hotspot tracking.
Category-Specific Ownership Verification & Admin Dashboard (`admin_dashboard.html`):** Dynamic 3-question security claims, multi-claimant review tabs, temporary day-based user blocking, non-destructive Resolved Case archiving, and one-click **CSV / PDF Pattern Report Exports**.

---

Technology Stack
* **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript (ES6+), Leaflet.js, Chart.js, jsPDF (`AutoTable`)
* **Backend:** Python 3.9+, FastAPI, Uvicorn
* **AI & NLP:** HuggingFace `transformers` (Vision CNN), `thefuzz` / Levenshtein Distance
* **Authentication & Database:** Google Firebase Authentication (`@student.newinti.edu.my`) + SQLite (SQLAlchemy ORM)

---

🚀 How to Run Locally

1. **Clone the repository:**
   git clone [https://github.com/Kagari12/Final-Year-Project-AI-Integrated-Campus-Lost-and-Found-System.git](https://github.com/Kagari12/Final-Year-Project-AI-Integrated-Campus-Lost-and-Found-System.git)
   cd Final-Year-Project-AI-Integrated-Campus-Lost-and-Found-System

2. Install Python dependencies:
pip install fastapi uvicorn sqlalchemy pydantic python-multipart transformers torch pillow thefuzz python-Levenshtein

3. Start the FastAPI Backend Server:
uvicorn main:app --reload
