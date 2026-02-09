from openai import OpenAI
from pydantic import BaseModel, Field
import json
from typing import List, Dict
from difflib import get_close_matches
import os
import pydicom
import re
import csv
import requests
from pathlib import Path

url = "https://api.deepseek.com"
api_key = "your_api"

client = OpenAI(
  base_url = url,
  api_key = api_key
)

MODEL="deepseek-reasoner"

## anatomical classification
anatomical_prompt = f"""
You are a radiotherapy assistant specializing in classifying anatomical sites.
Given the following input: classification_query. Determine which one of the following five Anatomical Sites it most likely belongs to:
    •   nasopharynx
    •   breast
    •   lung
    •   rectum
    •   cervix
### Your task:
Output only the most appropriate anatomical site from the list above.
Remember this anatomical site for use in subsequent tasks and reasoning.

### Important: Do not explain your choice. Only respond with the exact anatomical site name from the list. <think>/n/n</think>
"""

## classification
classification_protocol = {
    "GTV": "Target Volume", 
    "CTVnd": "Target Volume", 
    "PGTVnx": "Target Volume", 
    "PGTV-m": "Plan-Specific Auxiliary Results", 
    "PTV-all": "Plan-Specific Auxiliary Results", 
    "PTV-skin": "Plan-Specific Auxiliary Results",
    "Spincal Cord": "Organs at Risk",
    "Brain Stem": "Organs at Risk.",
    "Couch": "Dose Calculation-Specific",
    "Laser": "Dose Calculation-Specific",
    "Body": "Dose Calculation-Specific",
}

classification_prompt = f"""
You are an intelligent assistant specializing in the classification of ROI (Region of Interest) names in radiotherapy. Your task is to classify ROI names provided by the user into one of the following four classes:
    1. **Target Volume**: Includes CTV, PTV, GTV (e.g., PTV_70, GTV_60). These often contain numbers indicating prescription doses.
    2. **Organs at Risk (OAR)**: Normal organs and tissues needing protection (e.g., Lung, Heart, Spinal Cord).
    3. **Plan-Specific Auxiliary Results**: Structures created for optimization or constraints, such as margins, rings, or auxiliary ROIs (e.g., R60, R60+, Paro_M, 56+, 0.3).
    4. **Dose Calculation-Specific ROIs**: Structures for dose calculation, like Couch, laser, or iso.
---
### Your Task
    1. **Matching Analysis**: Match ROI names from classification_protocol of predefined ROI names and categories.
    2. **Classification**: Assign each ROI name to one of the four categories.
---
### Input Format
    **Actual ROI List (To Be Classified)**: A user-provided list of ROI names to classify.
    Example:
    ["PTV_high", "GTV_tumor", "Lung_L", "R60"]
---
### Working Process:
    1. **Dose Calculation-Specific ROIs**: 
        •   Matching device-related terms → filed under 'Dose Calculation-Specific ROIs'. 
        ……
    2. Classify according to the {classification_protocol}. If the {classification_protocol} contains similarly named ROI, directly classify it according to Classification Catogary. Pay attention to "Target Volume" classification. 
    3. When two targets appear in one ROI name (e.g. PTV2-PTVnx, PTV-all), it is the ROI after the target volume processing, also belong to the "Plan-Specific Auxiliary Results".
    4. **Special Rule for Nasopharynx and Breast Only**:
        •   Only if site_result is "nasopharynx" or "breast", apply this rule:
            - If an ROI name contains "PTV" but there is no exactly corresponding "CTV" or "GTV" name (considering digits, format, suffix) in classification_query, classify it as "Plan-Specific Auxiliary Results".            
        •   Otherwise (if site_result is not nasopharynx or breast), skip this special rule.
    5. **Target Volume**: 
        •   ROIs starting with "PTV" , "GTV" ,"CTV" , "ITV" , "IGTV" , "ICTV" or "PGTV" may be Target Volumes. 
     	……
    6. **Plan-Specific Auxiliary Results**: 
        •   ROIs that are not clearly OAR or Target Volume should be considered auxiliary.  
……
    7. **Organs at Risk (OAR)**: 
        •   Organs and tissues that require protection (e.g., Lung, Heart, SpinalCord).  
        ……
    9.  **When encountering a '-' or '_' symbol in an ROI name**:
        ……
---
### Output Requirements
    1.  Classification Results: Check one by one in classification_query to ensure that no ROI is missed. Provide the classification of each ROI in the actual list, Arrange each classification catagory results in the order of the classification_query. The classification results should be selected from these four class: Target Volume, Organs at Risk, Plan-Specific Auxiliary Results, and Dose Calculation-Specific.
    Example output:
……
    2.  The output "Target Volume" must contain "PTV". If not, perhaps some PTVS have been wrongly classified as "Plan Specific". Please reclassify.

### Important
    •   Return ONLY a valid JSON object with no additional text. Do NOT include any explanation, code block markers, or extra characters.
    •   Pay close attention to the naming characteristics (e.g., keywords like “PTV,” “GTV,” or “Dose”).
    •   For ambiguous names, use logical reasoning based on context and the reference list.
    •   Ensure the output is concise and easy to interpret.
"""

##Relabel Standardization Sheets (as provided in Supplementary 3 and Supplementary 4)
ptv_relabel_protocol = {
    'PTV1', 
    'PTVln', 
    'PTVg', 
    'PTV_60Gy', 
    'PTVtb_50Gy', 
    'PTVg^st'
}
oar_relabel_protocol = {
    'LargeBowel', 
    'SmallBowel', 
    'Lens_Left', 
    'Lens_Right', 
    'TemporalLobe_Left', 
    'TemporalLobe_Right', 
    'Lung_Left',
    'Lung_Right',
}

relabel_prompt = f"""
You are an intelligent assistant specializing in refining and correcting ROI (Region of Interest) names in radiotherapy datasets. Your task is to relabel user-provided ROI names into standardized, clean, and accurate names according to a reference dictionary, ensuring consistency and clinical clarity.
---
### Input
**Original ROI Names (To Be Relabeled)**: A list of **target_rename** and another list of **oar_rename**, inconsistent formatting, or non-standard naming.
**Reference Protocol**: A list of standard ROI names used in clinical practice. Use this as the basis for correction.
---
### Standardization Rules
## **Target Volume Nomenclature**
    **Important**: The Target Volumes should be standardized according to the {ptv_relabel_protocol} first (Even if it does not comply with the following rules), and the rest should be standardized according to the AAPM TG-263 naming conventions.
    1. Prefix: Must be one of the following:
        •   GTV, CTV, ITV, IGTV (gross + motion margin), ICTV (clinical + motion margin), PTV, PTV! (low-dose PTV excluding high-dose overlap).
        •   PCTV, PGTV are classified under PTV and renamed as PTVc and PTVg.
    2. Classifier: Directly follows the prefix (no space):
        •   n (nodal), p (primary), sb (surgical bed), par (parenchyma), v (venous thrombosis), vas (vascular) and b, tb , rpn, imn, scn, aln, icv, cw.
    3. Multiple targets: Arabic numerals follow type/classifier (e.g., PTV1, GTVp1).
    4. Imaging Modality & Order: Use _ followed by modality (CT, PT, MR, SP) and sequence number (e.g., PTVp1_CT1PT1).
    5. Structure Indicators: Append _ and indicator (e.g., CTV_A_Aorta, GTV_Preop, PTV_MR2_Prostate).
    6. Dose Level: Appended with _:
        •   Relative dose: _High, _Mid, _Low, or _MidXX (e.g., PTV_High, PTV_Mid01).
        •   Physical dose: Use cGy (e.g., PTV_5040) or Gy (e.g., PTV_50.4Gy, PTV_50p4Gy if periods are restricted).
        •   Dose per fraction: Use x separator (e.g., PTV_Liver_20Gyx3).
    7. Cropped Structures: Append -XX in mm (e.g., PTV-03, CTVp2-05).
    8. Custom Qualifiers: Append ^ (e.g., PTV^Physician1, PTVb^Eval, GTV_Liver^ICG).
    9. Character Limit (16 max): If exceeded, remove underscores from left to right while maintaining order (e.g., PTVLiverR_2000x3).

## **OAR (Organs-at-Risk) Nomenclature**
    **Important**: The OARs should be standardized according to the {oar_relabel_protocol} first (Even if it does not comply with the following rules), and the rest should be standardized according to the AAPM TG-263 naming conventions.
    1. Character Limit: 16 max, ensuring uniqueness (case-insensitive).
    2. Compound Structures: Use plural form (e.g., Lungs, Kidneys, Ribs_L).
    3. Capitalization: First letter of each category is uppercase (e.g., Femur_Head, Ears_External).
    4. No Spaces: Use _ for separation (e.g., Bowel_Bag).
    5. Spatial Indicators: Placed at the end (e.g., Lung_L, Lung_LUL, Lung_RLL, OpticNrv_PRV03_L).
        •   L (Left), R (Right), A (Anterior), P (Posterior), I (Inferior), S (Superior).
        •   RUL, RLL, RML (right lung lobes), LUL, LLL (left lung lobes).
        •   NAdj (non-adjacent), Dist (distal), Prox (proximal).
    6. Consistent Root Names: e.g., SeminalVes & SeminalVes_Dist (not SeminalVesicle & SemVes_Dist).
    7. Standard Roots for Distributed Structures:
        •   A (Artery): A_Aorta, A_Carotid.
        •   V (Vein): V_Portal, V_Pulmonary.
        •   LN (Lymph Node): LN_Ax_L1, LN_IMN.
        •   CN (Cranial Nerve): CN_IX_L, CN_XII_R.
        •   Glnd (Glandular): Glnd_Submand.
        •   Bone: Bone_Hyoid, Bone_Pelvic.
        •   Musc (Muscle): Musc_Masseter, Musc_Sclmast_L.
        •   Spc (Space): Spc_Bowel, Spc_Retrophar_L.
        •   VB (Vertebral Body).
        •   Sinus: Sinus_Frontal, Sinus_Maxillary.
    8. PRV (Planning OAR Volumes): Use _PRV + optional mm expansion (e.g., Brainstem_PRV, SpinalCord_PRV05).
    9. Partial Structures: Use ~ (e.g., Brain~, Lung~_L).
    10. Custom Qualifiers: Use ^ (e.g., Lungs^Ex).
    11. Primary vs. Reverse Order Naming:
        •   Primary: General → Specific → Laterality (e.g., Kidney_R, Kidney_Cortex_L).
        •   Reverse Order: Laterality → Specific → General (e.g., R_Hilum_Kidney, truncated as needed).
    12. Camel Case: Used only when distinct categories don’t exist (e.g., CaudaEquina, not Cauda_Equina).
    13. Non-Dose Evaluation Structures: Prefix with z or _ to separate them (e.g., zPTVopt).
---

### Output Requirements
- Output the result as a valid JSON object in the following format:
```json
    "Target Volume":
        "Input": ["GTV-LN-70", "CTV-LN-54", "PTV-C-60"],
        "Relabeled": ["GTVln_70Gy", "CTVln_54Gy", "PTVc_60Gy"]
    "Organs at Risk":
        "Input": ["Heart"],
        "Relabeled": ["Heart"]
"""

def get_roi_names_from_rtstruct(rtstruct_file):
    ds = pydicom.dcmread(rtstruct_file)
    
    if ds.Modality != "RTSTRUCT":
        raise ValueError(f"{rtstruct_file} is invalid RTSTRUCT file.")
    
    roi_names = [
        roi.ROIName for roi in ds.StructureSetROISequence
    ]
    return roi_names

def get_all_roi_names_from_folder(folder_path):
    all_roi_names = []
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path):
            try:
                roi_names = get_roi_names_from_rtstruct(file_path)
                all_roi_names.extend(roi_names)

            except Exception as e:
                print(f"skip file {file_name}: {e}")

    return list(set(all_roi_names))


folder_path = ""  
output_file = Path(folder_path).parent / (Path(folder_path).name + "_result_customized.csv")

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    print(file_path)
    for dcm_file_name in os.listdir(file_path):
        dcm_file_path = os.path.join(file_path, dcm_file_name)
        if os.path.isfile(dcm_file_path) and ("RS" in dcm_file_name or "RTSTRUCT" in dcm_file_name):
            try:
                classification_query = list(set(get_roi_names_from_rtstruct(dcm_file_path)))
              
                # Step 1: Determine Anatomical Site
                site_messages = [
                    {
                        "role": "system",
                        "content": f"You are a medical assistant specializing in radiotherapy. Given a ROI name or identifier, determine which anatomical site it belongs to: nasopharynx, breast, lung, rectum, or cervix. Work as {anatomical_prompt}. Respond with only the site name."
                    },
                    {
                        "role": "assistant", 
                        "content": "<think>/n/n</think>"
                    },
                    {
                        "role": "user",
                        "content": f"Respond with only the site name of: {classification_query} without thinking process."
                    }
                ]

                site_response = client.chat.completions.create(
                model=MODEL,
                messages=site_messages,
                temperature=0,
                top_p=0.9,
                stream=True,
                response_format={
                    "type": "json_object",
                    "format_instructions": "Output only the most appropriate anatomical site from the list above."
                }
                )

                site_text = ""
                for chunk in site_response:
                    if chunk.choices[0].delta.content is not None:
                        site_text += chunk.choices[0].delta.content
                        #print(chunk.choices[0].delta.content, end="")

                    start_marker = "</think>"
                    start_index = site_text.find(start_marker)

                    if start_index != -1 :
                        site_result = site_text[start_index + len(start_marker):].strip()
                    else:
                        site_result = site_text  

                print("Anatomical sites: ", site_result)

                # Step 2: Classification
                classification_messages=[
                    {
                        "role": "system",
                        "content": f"You are an intelligent assistant specializing in the classification of ROI (Region of Interest) names used in the process of radiotherapy. Your task is to classify a list of ROI names provided by the user into one of the following four categories. response as {classification_prompt}. Return only the standardized names in JSON format."
                    },
                    {
                        "role": "user",
                        "content": f"Classify the ROI names in query {classification_query} and the corresponding site_result {site_result}."
                    }
                ]

                classification_response = client.chat.completions.create(
                model=MODEL,
                messages=classification_messages,
                temperature=0,
                top_p=0.9,
                stream=True
                )

                complete_text = ""
                for chunk in classification_response:
                    if chunk.choices[0].delta.content is not None:
                        complete_text += chunk.choices[0].delta.content

                start_marker = "{"
                end_marker = "}"

                start_index = complete_text.find(start_marker)
                end_index = complete_text.find(end_marker)

                if start_index != -1 and end_index != -1:                    
                    clas_result = complete_text[start_index + len(start_marker):end_index].strip()
                else:
                    clas_result = complete_text  

                print(clas_result)

                target_rename = []
                oar_rename = []
                planspec_list = []
                dosecal_list = []

                clas_pairs = clas_result.split("\n")
                
                classification_data = {}
                for clas_pair in clas_pairs:
                    clas_pair = clas_pair.strip()
                    if clas_pair:
                        key, value = clas_pair.split(":", 1)
                        key = key.strip().strip('"')
                        value = value.strip().strip('"').strip(",")
                        classification_data[key] = value
                    
                for roi_class, roi_name in classification_data.items():
                    if roi_class == "Target Volume":
                        target_rename = roi_name.strip('"')
                    elif roi_class == "Organs at Risk":
                        oar_rename = roi_name.strip('"')
                    elif roi_class == "Plan-Specific Auxiliary Results":
                        planspec_list = roi_name.strip('"')
                    elif roi_class == "Dose Calculation-Specific":
                        dosecal_list = roi_name.strip('"')

                if not target_rename and not oar_rename:
                    print("No Target Volume or Organs at Risk need to be relabeled.")
                    exit(0)

                # Step 3: Relabel
                relabel_messages = [
                    {
                        "role": "system",
                        "content": f"You are an intelligent assistant specializing in the rename of ROI (Region of Interest) names used in the process of radiotherapy. Your task is to systematically rename a list of ROI names provided by the user. Respond according to {relabel_prompt}. Return only the standardized names in JSON format without explaination."
                    },
                    {
                        "role": "user",
                        "content": f"Rename the ROI names in the given Target Volume list {target_rename} and OAR list {oar_rename}."
                    }
                ]

                relabel_response = client.chat.completions.create(
                model=MODEL,
                messages=relabel_messages,
                temperature=0,
                top_p=0.9,
                stream=True
                )
                complete_text = ""
                for chunk in relabel_response:
                    if chunk.choices[0].delta.content is not None:
                        complete_text += chunk.choices[0].delta.content
                        print(chunk.choices[0].delta.content, end="")

                start_marker = "</think>"
                start_index = complete_text.find(start_marker)

                if start_index != -1:
                    rel_result = complete_text[start_index + len(start_marker): ].strip()
                else:
                    rel_result = complete_text  

                print("\nRelabel results: \n", rel_result)
                
                with open(output_file, mode="a+", newline="") as file:
                    writer = csv.writer(file)

                    row = [dcm_file_name, classification_query, site_result, clas_result, rel_result]

                    writer.writerow(row)
                    file.close()
                

            except Exception as e:
                print(f"Skip file {file_name}: {e}")
