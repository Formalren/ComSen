import pandas as pd

df = pd.read_excel(r"")

def normalize_scene(scene):
    if pd.isna(scene):
        return None
    s = str(scene).lower()
    if "kitchen" in s:
        return "Kitchen"
    if "bath" in s:
        return "Bathroom"
    if "living" in s:
        return "LivingRoom"
    if "bed" in s:
        return "Bedroom"
    return None

df["Scene"] = df["Scenes"].apply(normalize_scene)

df = df[df["Scene"].notna()].copy()

def classify(scene, props):
    props = "" if pd.isna(props) else str(props)

    # ----- Kitchen -----
    if scene == "Kitchen":
        if "Sliceable" in props or "Cookable" in props:
            return "Food"
        if "UsedForCooking" in props:
            return "CookingTool"
        if "Receptacle" in props and "Pickupable" in props:
            return "Container"
        if "Appliance" in props:
            return "Appliance"
        return "KitchenObject"

    # ----- Bathroom -----
    if scene == "Bathroom":
        if "Hygiene" in props:
            return "HygieneItem"
        if "Chemical" in props:
            return "CleaningItem"
        if "Receptacle" in props and "Pickupable" in props:
            return "Container"
        return "BathroomObject"

    # ----- Living Room -----
    if scene == "LivingRoom":
        if "Sit" in props or "Support" in props:
            return "Furniture"
        if "Appliance" in props:
            return "Appliance"
        if "Receptacle" in props:
            return "Container"
        return "LivingRoomObject"

    # ----- Bedroom -----
    if scene == "Bedroom":
        if "Sleep" in props or "Rest" in props:
            return "Furniture"
        if "Soft" in props:
            return "SoftItem"
        if "Receptacle" in props:
            return "Container"
        if "Appliance" in props:
            return "Appliance"
        return "BedroomObject"

    return "GeneralObject"


df["Category"] = df.apply(
    lambda r: classify(r["Scene"], r["Actionable Properties"]),
    axis=1
)

output_cols = ["Object Type", "Scene", "Actionable Properties", "Category"]
df[output_cols].to_excel(
    "scene_actionable_object_classification.xlsx",
    index=False
)

print("✅ Scene + Actionable Properties （ count /  risk / ）")
