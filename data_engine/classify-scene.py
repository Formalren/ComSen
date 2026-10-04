import pandas as pd

df = pd.read_excel(r"")

def normalize_scene(scene):
    if pd.isna(scene):
        return "Other"
    s = str(scene).lower()
    if "kitchen" in s:
        return "Kitchen"
    if "bath" in s:
        return "Bathroom"
    if "living" in s:
        return "LivingRoom"
    if "bed" in s:
        return "Bedroom"
    return "Other"

df["Scene"] = df["Scenes"].apply(normalize_scene)

def classify_scene_object(row):
    scene = row["Scene"]
    props = str(row["Actionable Properties"])
    obj = str(row["Object Type"]).lower()

    # ===== Kitchen =====
    if scene == "Kitchen":
        if "Sliceable" in props or "Cookable" in props:
            return "Food"
        if "Receptacle" in props and "Pickupable" in props:
            return "Container"
        if "UsedForCooking" in props or "knife" in obj:
            return "CookingTool"
        if "Appliance" in props:
            return "Appliance"
        if "Sink" in obj or "counter" in obj:
            return "Furniture"
        return "Cleaning"

    # ===== Bathroom =====
    if scene == "Bathroom":
        if "Hygiene" in props or "soap" in obj:
            return "HygieneItem"
        if "Chemical" in props:
            return "Cleaning"
        if "Receptacle" in props:
            return "Container"
        if "sink" in obj or "toilet" in obj or "bathtub" in obj:
            return "Fixture"
        return "Furniture"

    # ===== Living Room =====
    if scene == "LivingRoom":
        if "Sit" in props or "Furniture" in props:
            return "Furniture"
        if "Appliance" in props:
            return "Appliance"
        if "Receptacle" in props:
            return "Container"
        if "Decoration" in props or "plant" in obj:
            return "Decoration"
        return "DailyObject"

    # ===== Bedroom =====
    if scene == "Bedroom":
        if "bed" in obj or "dresser" in obj:
            return "Furniture"
        if "Soft" in props:
            return "SoftItem"
        if "Receptacle" in props:
            return "Container"
        if "Appliance" in props:
            return "Appliance"
        return "DailyObject"

    # ===== Other =====
    return "GeneralObject"

df["Category"] = df.apply(classify_scene_object, axis=1)

HIGH = {"Container", "CookingTool", "Appliance", "Cleaning"}
MEDIUM = {"Food", "SoftItem", "DailyObject", "HygieneItem"}
LOW = {"Furniture", "Fixture", "Decoration", "GeneralObject"}

def risk(cat):
    if cat in HIGH:
        return "High"
    if cat in MEDIUM:
        return "Medium"
    return "Low"

df["CausalConfusionRisk"] = df["Category"].apply(risk)

stats = (
    df.groupby(["Scene", "Category", "Object Type", "CausalConfusionRisk"])
      .size()
      .reset_index(name="Count")
      .sort_values(["Scene", "Category", "Count"], ascending=[True, True, False])
)

stats.to_excel("scene_category_object_stats_scene_aware.xlsx", index=False)

print("✅ Scene-aware （ Other）")