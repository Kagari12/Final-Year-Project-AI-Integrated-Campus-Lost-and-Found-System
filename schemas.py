from pydantic import BaseModel
from typing import Optional

# This is the data we expect from the user when they submit a form
class ItemCreate(BaseModel):
    title: str
    item_type: str
    description: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

# This is the data we send back to the user (includes the generated ID)
class ItemResponse(ItemCreate):
    id: int

    class Config:
        from_attributes = True