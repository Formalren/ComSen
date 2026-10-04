import pandas as pd


df = pd.read_excel(r"d:\embodied_reasoner\data_engine\cjfl.xlsx")


def classify(obj):
    if pd.isna(obj): 
        return "Unknown"
        
    name = str(obj).lower()


    if any(k in name for k in [
        "apple","bread","egg","meat","tomato","lettuce","steak","cheese",
        "banana","orange","pear","food","slice","milk","cream","juice"
    ]):
        return "Food"


    if any(k in name for k in [
        "bowl","cup","mug","basket","plate","pan","pot",
        "bottle","box","jar","can","container"
    ]):
        return "Container"


    if any(k in name for k in [
        "chair","sofa","bed","table","cabinet","bench",
        "shelf","desk","armchair","dresser","nightstand",
        "counter","stool"
    ]):
        return "Furniture"


    if any(k in name for k in [
        "bleach","soap","toilet","bath","towel","clean","shampoo",
        "detergent","spray","disinfect","brush","sponge"
    ]):
        return "Cleaning/Chemical"

 
    if any(k in name for k in [
        "knife","fork","spoon","scissor","scissors","spatula","tongs",
        "bat","hammer","tool","wrench","pliers","peeler"
    ]):
        return "Tools/Utensils"


    if any(k in name for k in [
        "fridge","microwave","stove","lamp","alarm","coffee","tv",
        "toaster","dishwasher","kettle","dryer","washer","fan"
    ]):
        return "Appliance"

    if any(k in name for k in [
        "pillow","blanket","cloth","teddy","rug","mat","towel",
        "blinds","curtain","sheet","cover"
    ]):
        return "Soft Items"

 
    if any(k in name for k in [
        "book","key","pen","remote","paper","magazine","wallet",
        "charger","phone","coin","card"
    ]):
        return "Small Object"


    if any(k in name for k in [
        "wall","floor","window","door","ceiling"
    ]):
        return "Structure"

    return "Other" 


df["Category"] = df["Object Type"].apply(classify)


output_name = "fljg.xlsx"
df.to_excel(output_name, index=False)

print(f"{output_name}")
