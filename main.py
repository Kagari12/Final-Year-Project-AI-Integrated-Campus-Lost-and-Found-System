import os
import shutil
from datetime import datetime
from typing import List, Optional

import numpy as np
import tensorflow as tf
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session
from thefuzz import fuzz

import ai_engine
import crud
import models
import schemas
from database import SessionLocal, engine

# 1. Initialize the Database (Creates tables if they don't exist yet)
models.Base.metadata.create_all(bind=engine)

# 2. Initialize the FastAPI Server
app = FastAPI(title="Campus Lost and Found API", description="Backend for FYP 360-Viewer System")

# Load YOUR Custom CNN Image Classification AI
print("Waking up the Custom Campus CNN... This might take a few seconds.")
try:
    # Make sure 'campus_ai_model.h5' is saved in the same folder as this main.py file!
    custom_model = tf.keras.models.load_model('campus_ai_model.h5')
    
    # These must perfectly match the categories you trained your AI on!
    class_names = [
        'Bracelet', 'Earrings', 'Necklace', 'Pendant', 'Plastic Bottle', 
        'Set', 'Tumbler', 'Unlabeled', 'back_pack', 'bike_helmet', 'bottle', 
        'calculator', 'headphones', 'key', 'keyboard', 'laptop_computer', 
        'mobile phone', 'mobile_phone', 'mouse', 'mug', 'paper_notebook', 
        'pen', 'phone', 'ring', 'ruler', 'scissors', 'speaker', 'stapler'
    ]
    print("Custom Vision AI is ready!")
except Exception as e:
    print(f"Warning: Could not load custom AI model. Error: {e}")

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

# --- PYDANTIC REQUEST MODELS ---

class ClaimRequest(BaseModel):
    category: str
    field1: str
    field2: str
    field3: Optional[str] = None

class ResolveItemRequest(BaseModel):
    claim_status: str = "Resolved"
    resolved_date: Optional[str] = None

class UserRequest(BaseModel):
    email: str

class BlockUserRequest(BaseModel):
    days: int
    blocked_until: Optional[str] = None


# --- API ENDPOINTS (THE DOORS) ---

# Root Home Page
@app.get("/items/top-locations/")
def get_top_locations(db: Session = Depends(get_db)):
    # Ask the database for the top 5 most common locations for Lost items
    top_locations = (
        db.query(models.Item.specific_location, func.count(models.Item.id).label('total'))
        .filter(models.Item.item_type == "Lost")
        .filter(models.Item.specific_location != None)
        .filter(models.Item.specific_location != "")
        .group_by(models.Item.specific_location)
        .order_by(func.count(models.Item.id).desc())
        .limit(5)
        .all()
    )

    # Convert the database answer into a clean list for the frontend
    return [{"location": loc[0], "count": loc[1]} for loc in top_locations]

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

# Helper: Maps your exact 28 Custom CNN `class_names` to the 8 Portal Categories
def map_cnn_class_to_category(raw_tag: str) -> Optional[str]:
    # Strip percentage e.g. "key (88%)" -> "key"
    clean_name = raw_tag.split("(")[0].strip().lower()

    cnn_to_category = {
        # Keys
        "key": "Keys",
        # Water Bottle / Drinkware
        "plastic bottle": "Water Bottle",
        "tumbler": "Water Bottle",
        "bottle": "Water Bottle",
        "mug": "Water Bottle",
        # Mobile Phone
        "mobile phone": "Mobile Phone",
        "mobile_phone": "Mobile Phone",
        "phone": "Mobile Phone",
        # Bag / Backpack
        "back_pack": "Bag/Backpack",
        # Jewelry / Accessories
        "bracelet": "Jewelry/Bracelet",
        "earrings": "Jewelry/Bracelet",
        "necklace": "Jewelry/Bracelet",
        "pendant": "Jewelry/Bracelet",
        "ring": "Jewelry/Bracelet",
        "set": "Jewelry/Bracelet",
        # Laptop / Computer Peripherals
        "laptop_computer": "Laptop/Computer",
        "keyboard": "Laptop/Computer",
        "mouse": "Laptop/Computer",
        "headphones": "Laptop/Computer",
        "speaker": "Laptop/Computer",
        # Other Stationery / Misc
        "calculator": "Other",
        "bike_helmet": "Other",
        "paper_notebook": "Other",
        "pen": "Other",
        "ruler": "Other",
        "scissors": "Other",
        "stapler": "Other",
    }
    return cnn_to_category.get(clean_name, None)


# Route 3: UPLOAD images AND run Computer Vision AI
@app.post("/items/{item_id}/images/")
def upload_images(item_id: int, files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    image_folder = "static/images"
    os.makedirs(image_folder, exist_ok=True)
    
    ai_analysis_results = []
    all_tags_list = []  # Collect all tags in order of highest confidence
    best_detected_category = None
    best_score = -1.0
    
    for file in files:
        file_location = f"{image_folder}/item_{item_id}_{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        # 1. Run YOUR Custom Vision AI
        try:
            # Open the image, resize it to 224x224 (standard for MobileNet/ResNet)
            img = Image.open(file_location).convert('RGB').resize((224, 224))
            
            # Convert image to numbers for the AI
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Ask the AI what it sees!
            predictions = custom_model.predict(img_array)[0] 
            
            # This sorts the predictions and grabs the top 3 highest scores
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            
            tags = []
            for idx_rank, i in enumerate(top_3_indices):
                score = float(predictions[i])
                percent = int(score * 100)
                tag_name = class_names[i]
                
                # Create the tag (e.g., "key (85%)")
                tag_str = f"{tag_name} ({percent}%)"
                tags.append(tag_str)

                # Check if this prediction maps to a known category with the highest score
                mapped_cat = map_cnn_class_to_category(tag_name)
                if mapped_cat and score > best_score:
                    best_score = score
                    best_detected_category = mapped_cat
            
            # Save the top 3 tags to the website!
            all_tags_list.extend(tags)
            ai_analysis_results.append({"filename": file.filename, "ai_tags": tags})
                
        except Exception as e:
            print(f"AI Prediction Failed for {file.filename}: {e}")

    # 2. SAVE THE TAGS & AUTO-CORRECT THE CATEGORY IN THE DATABASE!
    auto_corrected_category = None
    db_item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if db_item and all_tags_list:
        # Preserve top-scoring tag order while removing duplicates
        unique_tags = list(dict.fromkeys(all_tags_list))
        db_item.ai_tags = ", ".join(unique_tags)

        # If the AI recognized a specific category (e.g., "Keys") and the item was left on
        # "Auto-Detect" or default "Laptop/Computer" (or mismatches the photo), update it!
        if best_detected_category:
            auto_corrected_category = best_detected_category
            if db_item.title in ["Auto-Detect", "Laptop/Computer", "Other"] or db_item.title != best_detected_category:
                db_item.title = best_detected_category

        db.commit()  # Lock both ai_tags and the corrected category into SQLite!

    return {
        "message": "Images uploaded & Custom AI Tags saved!",
        "ai_analysis": ai_analysis_results,
        "auto_corrected_category": auto_corrected_category
    }

# Route 4: Combined AI NLP & Location Matcher
@app.get("/items/{item_id}/matches/")
def find_matches_for_item(item_id: int, db: Session = Depends(get_db)):
    target_item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if not target_item:
        return {"error": "Item not found"}

    all_items = db.query(models.Item).filter(models.Item.id != item_id).all()
    matches = []

    # Target search string
    search_words = f"{target_item.title or ''} {target_item.description or ''}".lower().strip()

    for other_item in all_items:
        # Match lost items with found items and vice-versa
        if target_item.item_type != other_item.item_type:
            
            # Combine Title, Description, AND saved AI Tags
            item_words = f"{other_item.title or ''} {other_item.description or ''} {other_item.ai_tags or ''}".lower().strip()
            
            if not search_words or not item_words:
                continue
            
            # 1. Fuzzy matching calculation
            discovery_score = max(
                fuzz.token_set_ratio(search_words, item_words),
                fuzz.partial_ratio(search_words, item_words)
            )
            strict_score = fuzz.ratio(search_words, item_words)
            final_confidence = round((discovery_score * 0.7) + (strict_score * 0.3))
            
            # Only include strong matches (> 40%)
            if final_confidence > 40:
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

# Route 9: Submit a security claim for an item
@app.put("/items/{item_id}/claim")
def submit_security_claim(item_id: int, claim: ClaimRequest, db: Session = Depends(get_db)):
    item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Save the custom category-specific verification data
    item.claim_status = "Pending Review"
    item.verify_category = claim.category
    item.verify_field_1 = claim.field1
    item.verify_field_2 = claim.field2
    item.verify_field_3 = claim.field3
    
    db.commit()
    return {"message": "Category-specific security claim submitted!"}

# Route 10: Mark an item case as Resolved (without deleting)
@app.put("/items/{item_id}/resolve")
def resolve_item_case(item_id: int, req: ResolveItemRequest, db: Session = Depends(get_db)):
    item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    # Save as Resolved instead of deleting
    item.claim_status = "Resolved"
    db.commit()
    db.refresh(item)
    return item


# --- USER MANAGEMENT ENDPOINTS ---

# 1. Register a new user in SQLite (Prevents Ghost Users)
@app.post("/users/")
def register_user(user: UserRequest, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user:
        new_user = models.User(email=user.email)
        db.add(new_user)
        db.commit()
    return {"message": "User registered in database"}

# 2. Check if a user is banned (Prevents the Flicker Bug)
@app.get("/users/{email}/status")
def check_user_status(email: str, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == email).first()
    if not db_user:
        return {"status": "Active"}  # Default safe fallback
    return {"status": db_user.status}

# 3. Get all users for the Admin Dashboard
@app.get("/users/")
def get_all_users(db: Session = Depends(get_db)):
    return db.query(models.User).all()

# 4. Toggle Ban Status (For Admin)
@app.put("/users/{email}/toggle-ban")
def toggle_ban(email: str, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == email).first()
    if db_user:
        db_user.status = "Banned" if db_user.status == "Active" else "Active"
        db.commit()
        return {"message": f"User is now {db_user.status}"}
    raise HTTPException(status_code=404, detail="User not found")

# 5. Block User Temporarily (For Admin)
@app.put("/users/{email}/block")
def block_user_temporarily(email: str, req: BlockUserRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if req.days > 0 and req.blocked_until:
        user.status = f"Blocked until {req.blocked_until[:10]}"
    else:
        user.status = "Active"
    db.commit()
    return {"email": user.email, "status": user.status, "blocked_until": req.blocked_until}

# 6. Delete User (For Admin)
@app.delete("/users/{email}")
def delete_user(email: str, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == email).first()
    if db_user:
        db.delete(db_user)
        db.commit()
        return {"message": "User deleted from database"}
    raise HTTPException(status_code=404, detail="User not found")
