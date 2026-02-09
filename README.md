## Foundation Model–Driven ROI Classification and Renaming in Radiotherapy

This repository contains the official implementation of the paper:
## "Foundation Model–Driven Regions of Interests Classification and Renaming in Radiotherapy: A Customizable, Retraining-Free Workflow Across Institutions"

## 📖 Overview

Inconsistent and institution-specific naming of regions of interest (ROIs) limits interoperability and hinders clinical automation in radiotherapy. We propose a modular, multi-stage workflow that utilizes large language models (LLMs) and a CLIP-based image-text module to automate ROI classification and standardization.

### Key Highlights

* 
**Retraining-Free:** Operates on foundation models without the need for task-specific fine-tuning.


* 
**Multi-Stage Pipeline:** Decomposes the complex task into anatomical site determination, semantic classification, and customizable renaming .


* 
**Customizable:** Supports both AAPM TG-263 standards and institution-specific naming conventions via prompt-embedded protocols.


* 
**Safety Verification:** Integrates a CLIP-based module for image-based laterality (Left/Right) check.


* 
**High Performance:** Achieved **99.12%** classification accuracy and **97.92%** renaming accuracy across 600 multi-institutional cases.



---

## 🛠 Workflow Architecture

The workflow consists of four modular stages:

1. 
**Anatomical Site Classification:** Identifies the disease site (e.g., Nasopharynx, Breast) to provide context.


2. 
**ROI Semantic Classification:** Categorizes ROIs into four types: Target Volume (TV), Organ at Risk (OAR), Plan-Specific Auxiliary, or Dose Calculation-Specific.


3. 
**Customizable ROI Renaming:** Standardizes names using a reference protocol (TG-263 or local).


4. 
**CLIP-Based Laterality Verification:** Uses a 3D dual-channel vision transformer to cross-verify the LLM's output against the actual CT anatomy .



---

## 📁 Repository Structure

* `code_prompt.py`: The main execution script containing the LLM logic, prompt engineering for all classification and renaming stages, and DICOM processing.
* `TG263_oar_relabel_protocol.py`: A comprehensive reference library containing standardized OAR names according to TG-263.
* `customized_relabel_protocol.py`: An example of an institution-specific protocol for flexible adaptation.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* `pydicom`
* `openai`
* Access to an LLM API (e.g., DeepSeek-R1 or GPT-4) or a locally deployed model.

### Usage

1. Set your API key in `code_prompt.py`:
```python
api_key = "your_api_key_here"
url = "https://api.deepseek.com" # or your local endpoint

```


2. Configure the `folder_path` to your DICOM RTSTRUCT files.
3. Run the pipeline:
```bash
python code_prompt.py

```



---

## 📊 Results Summary

Evaluation across three institutions (Fudan University Shanghai Cancer Center, Zhejiang University, and Xiangya Hospital):

| Metric | Accuracy |
| --- | --- |
| **Overall Classification** | 99.12% 

 |
| **Overall Renaming** | 97.92% 

 |
| **Target Volume (TV) Renaming** | 97.56% 

 |
| **Laterality Error Detection** | 100% (Stress Test) 

 |

---

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{yang2026foundation,
  title={Foundation Model–Driven Regions of Interests Classification and Renaming in Radiotherapy: A Customizable, Retraining-Free Workflow Across Institutions},
  author={Yang, Dong and Lei, Mingjun and Yang, Qiangxing and Sun, Zihan and Hou, Xuewen and Hu, Weigang and Wang, Jiazhou},
  journal={Medical Physics / Medical Image Analysis (Submitted)},
  year={2026}
}

```
