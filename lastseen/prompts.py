"""Prompts and JSON schemas for every Muse Spark call. Kept in one place so they can be tuned
after looking at runs/<id>/api/*.json."""
from __future__ import annotations

CATEGORIES = [
    "keys", "wallet", "passport", "id_card", "phone", "charger", "cable", "earbuds", "headphones",
    "laptop", "tablet", "remote", "glasses", "sunglasses", "watch", "mug", "cup", "bottle", "glass",
    "plate", "bowl", "utensil", "book", "notebook", "pen", "paper", "envelope", "bag", "backpack",
    "purse", "toy", "medication", "cosmetics", "hat", "clothing", "shoes", "umbrella", "tool",
    "food", "plant", "candle", "speaker", "camera", "game_controller", "other",
]

# ------------------------------------------------------------ keyframe grounding
GROUNDING_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["room", "surfaces", "objects"],
    "properties": {
        "room": {"type": "string", "description": "kind of room, e.g. kitchen, bedroom, living room, hallway, bathroom, office; 'unknown' if unclear"},
        "surfaces": {"type": "array", "items": {"type": "string"},
                     "description": "surfaces visible where objects can rest: counter, desk, nightstand, sofa, floor, shelf, table, bed, sink..."},
        "objects": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["label", "description", "bbox", "in_hand", "surface", "confidence"],
            "properties": {
                "label": {"type": "string", "description": "short noun phrase, 1-3 words, e.g. 'black charger', 'blue mug'"},
                "description": {"type": "string", "description": "one line: colour, material, distinguishing marks"},
                "bbox": {"type": "object", "additionalProperties": False,
                         "required": ["x_min", "y_min", "x_max", "y_max"],
                         "properties": {k: {"type": "integer"} for k in ("x_min", "y_min", "x_max", "y_max")}},
                "in_hand": {"type": "boolean", "description": "true if a person is holding or carrying it"},
                "surface": {"type": "string", "description": "what it rests on, or 'hand', or 'unknown'"},
                "confidence": {"type": "number"},
            }}},
    },
}

GROUNDING_PROMPT = (
    "You are an object grounding expert. This is a frame from a handheld walk-through video of a home. "
    "List every clearly visible MOVABLE personal object that a person might later ask 'where did I leave it' about: "
    "keys, wallet, passport, phone, charger, cables, earbuds, remote, glasses, watch, mug, bottle, book, notebook, pen, "
    "bag, toy, medication, cosmetics, hat, shoes, tools, food items, clothing left lying around, and similar. "
    "Ignore furniture, appliances, fixtures, walls, floors, people, and decor that never moves. "
    "Also skip: stickers, decals, labels and tape stuck to surfaces; crumbs, scraps and fragments; items that are "
    "mostly out of frame, mostly occluded, or too blurred or too small to name confidently. One box per whole item "
    "(not per part). "
    "For each object give a short label (the kind of thing plus its colour, e.g. 'black massage gun'), a one-line "
    "description with colour/material/marks, and the bounding box as x_min, y_min, x_max, y_max on a normalized 0-1000 "
    "grid with (0,0) at the top-left. Make boxes tight. Set confidence below 0.5 when unsure what the item is. "
    "Also name the room and the visible surfaces."
)

# labels Spark still emits for things we do not want to track (matched as whole words, lower-case)
GROUNDING_SKIP_WORDS = {"sticker", "stickers", "decal", "decals", "fragment", "fragments", "scrap", "scraps", "crumb",
                        "crumbs", "label", "tape", "blurry", "blurred", "partial", "unidentified", "unknown",
                        "smudge", "stain", "reflection", "shadow"}

# ------------------------------------------------------------ attribute record
ATTRIBUTES_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["canonical_label", "category", "dominant_color", "secondary_colors", "material",
                 "brand_text", "marks", "size_class", "description"],
    "properties": {
        "canonical_label": {"type": "string", "description": "2-4 word name a person would use, e.g. 'black Anker charger'"},
        "category": {"type": "string", "enum": CATEGORIES},
        "dominant_color": {"type": "string"},
        "secondary_colors": {"type": "array", "items": {"type": "string"}},
        "material": {"type": "string"},
        "brand_text": {"type": "string", "description": "any readable brand, logo or text; '' if none"},
        "marks": {"type": "array", "items": {"type": "string"}, "description": "stickers, scratches, dents, patterns, anything that distinguishes this exact item"},
        "size_class": {"type": "string", "enum": ["tiny", "small", "medium", "large"]},
        "description": {"type": "string"},
    },
}

ATTRIBUTES_PROMPT = (
    "This is a crop of one household object (the background may be blanked to grey). Describe the exact item so it "
    "can be told apart from a look-alike later: pick the closest category, give the dominant and secondary colours, "
    "material, any readable brand or text, and distinguishing marks (stickers, wear, scratches, patterns). Be literal; "
    "do not guess marks you cannot see. canonical_label is the name the owner would use: the specific kind of thing "
    "plus one colour or brand word ('navy jacket', 'Meyer's soap bottle', 'gray massage gun'), never a generic class "
    "('clothing', 'tool', 'item') when a specific noun fits, and never words about image quality (blurry, partial, "
    "masked, cropped). The detector saw this object in several frames and called it: {labels}. Reuse that noun, and its colour "
    "word when the crop is ambiguous (a black watch with a silver face is a 'black watch'), unless the crop clearly "
    "shows they are wrong."
)

# ------------------------------------------------------------ query intent
INTENT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["intent", "object_candidates", "place", "time_from", "time_to", "clarify"],
    "properties": {
        "intent": {"type": "string", "enum": ["locate", "changes_between", "history", "list_place", "unsupported"]},
        "object_candidates": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["object_id", "match"],
            "properties": {"object_id": {"type": "integer"}, "match": {"type": "number"}}}},
        "place": {"type": "string", "description": "room or surface mentioned, or ''"},
        "time_from": {"type": "string", "description": "one of: '', 'today_start', 'yesterday_start', 'now-<N>h', 'morning', or ISO-8601"},
        "time_to": {"type": "string", "description": "'' or 'now' or ISO-8601"},
        "clarify": {"type": "string", "description": "a question to ask the user if the request is ambiguous, else ''"},
    },
}

INTENT_SYSTEM = (
    "You map a question about where household objects are to a JSON intent. Pick object_candidates ONLY from the catalog "
    "(by id); give match < 0.5 when nothing fits. 'locate' = where is X / where did I leave X. 'changes_between' = what moved, "
    "what changed, what is missing since <time>. 'history' = where has X been. 'list_place' = what is on/in <place>. "
    "Use 'unsupported' for anything else. Put time words into time_from/time_to tokens; leave '' when absent."
)
