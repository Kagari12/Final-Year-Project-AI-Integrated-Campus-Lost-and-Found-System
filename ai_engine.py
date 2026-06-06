from thefuzz import fuzz
import models

def find_best_matches(target_item: models.Item, all_items: list[models.Item]):
    matches = []
    
    for item in all_items:
        # Rule 1: Only compare "Lost" items against "Found" items.
        # We don't want to match a lost item with another lost item.
        if item.item_type != target_item.item_type:
            
            # Rule 2: Token Set Ratio is perfect for NLP because it ignores word order.
            # "Blue Flask" will perfectly match "Flask Blue".
            title_score = fuzz.token_set_ratio(target_item.title, item.title)
            
            # If descriptions exist, score them too. Otherwise, default to 0.
            desc_score = 0
            if target_item.description and item.description:
                desc_score = fuzz.token_set_ratio(target_item.description, item.description)
            
            # Rule 3: Calculate the final confidence weight.
            # We give the Title more importance (70%) than the Description (30%).
            final_score = (title_score * 0.7) + (desc_score * 0.3)
            
            # Rule 4: The Threshold. Only return matches with a > 60% confidence score.
            if final_score > 60:
                matches.append({
                    "matched_item_id": item.id,
                    "matched_title": item.title,
                    "matched_type": item.item_type,
                    "confidence_score": round(final_score, 2)
                })
                
    # Sort the results so the highest confidence score appears at the top
    matches.sort(key=lambda x: x["confidence_score"], reverse=True)
    return matches
    