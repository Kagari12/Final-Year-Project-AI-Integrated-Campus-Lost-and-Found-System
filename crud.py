from sqlalchemy.orm import Session
import models, schemas

# Function to GET all items from the database
def get_items(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Item).offset(skip).limit(limit).all()

# Function to CREATE a new item in the database
def create_item(db: Session, item: schemas.ItemCreate):
    db_item = models.Item(
        title=item.title, 
        item_type=item.item_type,
        description=item.description,  # ADDED: Save the description!
        latitude=item.latitude,        # ADDED: Save the GPS!
        longitude=item.longitude       # ADDED: Save the GPS!
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item