from transformers import pipeline
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from fastapi import FastAPI, Depends, File, UploadFile
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import os
import shutil
import ai_engine  # Add this near your other imports
from thefuzz import fuzz
# Import your custom local files
import models
import schemas
import crud
from database import SessionLocal, engine

# 1. Initialize the Database (Creates tables if they don't exist yet)
models.Base.metadata.create_all(bind=engine)

# 2. Initialize the FastAPI Server
app = FastAPI(title="Campus Lost and Found API", description="Backend for FYP 360-Viewer System")
# Load the pre-trained Image Classification AI (This happens once when server starts)
print("Waking up the Vision AI... This might take a few seconds.")
vision_ai = pipeline("image-classification", model="google/vit-base-patch16-224")
print("Vision AI is ready!")
# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Setup the Image Storage Directory
# This ensures the folder exists so the server doesn't crash when saving files
os.makedirs("static/images", exist_ok=True)

# 4. Serve the static folder so web browsers can view the saved images
app.mount("/static", StaticFiles(directory="static"), name="static")

# 5. Database Connection Dependency
# This opens a fresh connection for every web request, then closes it safely
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- API ENDPOINTS (THE DOORS) ---

# Root Home Page
@app.get("/")
def read_root():
    return {"Message": "Welcome to the Campus Lost and Found API! Go to /docs to test the system."}

# Route 1: CREATE a new text-based item (Lost or Found)
@app.post("/items/", response_model=schemas.ItemResponse)
def create_new_item(item: schemas.ItemCreate, db: Session = Depends(get_db)):
    return crud.create_item(db=db, item=item)

# Route 2: READ all items in the database
@app.get("/items/", response_model=list[schemas.ItemResponse])
def read_all_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_items(db, skip=skip, limit=limit)


# Route 3: UPLOAD images AND run Computer Vision AI
# Route: Upload Images & Save AI Tags
@app.post("/items/{item_id}/images/")
def upload_images(item_id: int, files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    image_folder = "static/images"
    os.makedirs(image_folder, exist_ok=True)
    
    ai_analysis_results = []
    all_tags_list = [] # We will collect all tags here
    
    for file in files:
        file_location = f"{image_folder}/item_{item_id}_{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        # 1. Run the Vision AI
        predictions = vision_ai(file_location)
        tags = [p["label"] for p in predictions if p["score"] > 0.1]
        all_tags_list.extend(tags) # Add tags to our master list
        
        ai_analysis_results.append({"filename": file.filename, "ai_tags": tags})

    # 2. SAVE THE TAGS TO THE DATABASE!
    db_item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if db_item:
        # Get rid of duplicate tags and combine them into one string
        unique_tags = list(set(all_tags_list))
        db_item.ai_tags = ", ".join(unique_tags)
        db.commit() # Lock it into the database

    return {"message": "Images uploaded & Tags saved!", "ai_analysis": ai_analysis_results}

# Route 4: Run the AI NLP Matcher
@app.get("/items/{item_id}/matches/")
def find_matches_for_item(item_id: int, db: Session = Depends(get_db)):
    target_item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if not target_item:
        return {"error": "Item not found"}

    all_items = db.query(models.Item).filter(models.Item.id != item_id).all()
    matches = []

    # What did the loser type?
    search_words = f"{target_item.title or ''} {target_item.description or ''}".lower().strip()

    for other_item in all_items:
        if target_item.item_type != other_item.item_type:
            
            # Combine the Finder's Title, Description, AND AI Tags!
            item_words = f"{other_item.title or ''} {other_item.description or ''} {other_item.ai_tags or ''}".lower().strip()
            
            if not search_words or not item_words:
                continue
            
            # --- THE "SMART" SEARCH ENGINE UPGRADE ---
            # 1. Discovery Math: Finds words hidden in long tags ("bag" inside "mailbag")
            discovery_score = max(
                fuzz.token_set_ratio(search_words, item_words),
                fuzz.partial_ratio(search_words, item_words)
            )
            
            # 2. Strict Math: Penalizes if lengths/words are vastly different
            strict_score = fuzz.ratio(search_words, item_words)
            
            # 3. The Blend: 70% weight to discovering the item, 30% penalty for extra un-searched words
            final_confidence = round((discovery_score * 0.7) + (strict_score * 0.3))
            
            # Only show reasonably good matches (e.g., above 50%)
            if final_confidence >= 50:
                display_title = other_item.title if other_item.title else "Unknown Item (AI Matched)"
                
                matches.append({
                    "matched_item_id": other_item.id,
                    "matched_title": display_title,
                    "matched_description": other_item.description,
                    "latitude": other_item.latitude,
                    "longitude": other_item.longitude,
                    "confidence_score": final_confidence
                })
                
    matches.sort(key=lambda x: x["confidence_score"], reverse=True)
    return {"target_item": target_item.title, "ai_predictions": matches}

# Route 5: Proactive Matching (Upgraded AI & Location Data)
@app.get("/items/{item_id}/matches/")
def find_matches_for_item(item_id: int, db: Session = Depends(get_db)):
    target_item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if not target_item:
        return {"error": "Item not found"}

    all_items = db.query(models.Item).filter(models.Item.id != item_id).all()
    matches = []

    for other_item in all_items:
        if target_item.item_type != other_item.item_type:
            
            # UPGRADE 1: Use token_sort_ratio instead of token_set_ratio for realistic scoring
            title_score = fuzz.token_sort_ratio(target_item.title.lower(), other_item.title.lower())
            
            target_desc = target_item.description.lower() if target_item.description else ""
            other_desc = other_item.description.lower() if other_item.description else ""
            desc_score = fuzz.token_sort_ratio(target_desc, other_desc) if other_desc else 0
            
            final_score = (title_score * 0.7) + (desc_score * 0.3)
            
            # Calculate the final confidence out of 100
            confidence = round(final_score)
            
            if confidence > 40:
                matches.append({
                    "matched_item_id": other_item.id,
                    "matched_title": other_item.title,
                    "matched_description": other_item.description, # UPGRADE 2: Send Description
                    "latitude": other_item.latitude,               # UPGRADE 3: Send GPS
                    "longitude": other_item.longitude,
                    "confidence_score": confidence
                })
                
    matches.sort(key=lambda x: x["confidence_score"], reverse=True)
    return {"target_item": target_item.title, "ai_predictions": matches}
    
# Route 6: The AI Smart Search (Takes raw text from the user)
@app.get("/search/")
def ai_smart_search(q: str, db: Session = Depends(get_db)):
    all_items = db.query(models.Item).all()
    matches = []
    
    for item in all_items:
        # Compare the user's typed search (q) against every item in the database
        title_score = fuzz.token_set_ratio(q, item.title)
        desc_score = fuzz.token_set_ratio(q, item.description) if item.description else 0
        
        # Calculate final confidence score
        final_score = (title_score * 0.7) + (desc_score * 0.3)
        
        # If it's more than a 40% match, send it back!
        if final_score > 40:
            matches.append({
                "id": item.id,
                "title": item.title,
                "type": item.item_type,
                "description": item.description,
                "confidence": round(final_score, 2)
            })
            
    # Sort so the highest match is at the very top
    matches.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {"query": q, "results": matches}

# Route 7: Admin functionality - Delete/Resolve an item
@app.delete("/items/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db)):
    # Find the item in the database
    item = db.query(models.Item).filter(models.Item.id == item_id).first()
    
    if not item:
        return {"error": "Item not found in the database."}
        
    # Delete it and save the changes
    db.delete(item)
    db.commit()
    
    return {"message": f"Item {item_id} has been successfully resolved and removed."}

# Route 8: GET images for the 360 Viewer
@app.get("/items/{item_id}/images/")
def get_item_images(item_id: int):
    image_folder = "static/images"
    
    # If the folder doesn't exist yet, return empty list
    if not os.path.exists(image_folder):
        return {"images": []}
        
    # Look through the folder and find any image that starts with this item's ID
    images = []
    for filename in os.listdir(image_folder):
        if filename.startswith(f"item_{item_id}_"):
            images.append(f"/static/images/{filename}")
            
    return {"images": images}