# 🧬 RX Tracking System v2.0
**AI-Assisted Data Consolidation & Record Linkage Engine**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Machine Learning](https://img.shields.io/badge/AI-Matching-blueviolet?style=for-the-badge)](https://en.wikipedia.org/wiki/Record_linkage)

## 🛠 Tech Stack

| Category | Tools |
| :--- | :--- |
| **Language** | **Python 3.12** |
| **AI / ML** | **Scikit-learn** (TF-IDF, NearestNeighbors), **RapidFuzz** (String Similarity) |
| **Data Processing** | **Pandas** & **NumPy** (High-performance Dataframe Manipulation) |
| **Frontend** | **Streamlit** (Interactive Metrics & Matching GUI) |
| **Database** | **SQL Server** (via SQLAlchemy & pyodbc) |
| **Architecture** | **Multi-step Rules Engine** (Exact ➔ Fuzzy ➔ AI) |

---

## 🎯 Project Overview
The **RX Tracking System v2.0** is an intelligent data-cleaning and reconciliation platform. It automates the matching of messy RX transaction records against a master doctor list using a hybrid approach of **deterministic rules** and **probabilistic AI matching**.

### 🧠 AI-Assisted Matching Pipeline
To achieve high match rates where standard SQL joins fail, the system utilizes a sophisticated pipeline:
1.  **Exact Match:** Deterministic key-based matching for high-confidence links.
2.  **Vectorized Similarity (TF-IDF):** Converts doctor names and addresses into character n-gram vectors.
3.  **K-Nearest Neighbors (k-NN):** Efficiently searches the vector space to find the closest match in the masterlist for unmatched records.
4.  **Quick Suggest:** A word-based fuzzy heuristic for real-time human-in-the-loop verification.

---

## 🚀 Key Professional Capabilities

### 📊 Advanced Analytics Dashboard
* **Real-time Metrics:** Tracks "Masterlist Match Rate" vs "AI Match Rate" to give stakeholders transparency into data quality.
* **Financial Integrity:** Implemented validation checks between matching steps to ensure total amounts and transaction counts remain 100% accurate.

### 🛡️ Enterprise Data Engineering
* **Hybrid Matching Modes:** Offers "Basic" (Strict/Safe) and "Advanced" (AI-Driven) modes to balance speed and accuracy.
* **Memory Management:** Optimized for large pharmaceutical datasets by leveraging efficient vectorized operations in Scikit-Learn.
* **Cross-Reference Logic:** Handles complex many-to-one mappings for Item and PTR cross-references.

### 💼 Professional Workflow Features
* **Audit-Ready Exports:** Standardized column formatting ensuring the output is ready for immediate ingestion into downstream ERP or BI systems.
* **Reference Integration:** Dynamic loading of auxiliary datasets (PPE Doctors, PTR Mappings) to enrich transaction data.

---

## ⚙️ Development & Installation

### Requirements
- **Python 3.10+**
- **ODBC Driver 18 for SQL Server**

### Quick Start
```bash
# 1. Install Dependencies
pip install streamlit pandas sqlalchemy scikit-learn rapidfuzz pyodbc openpyxl

# 2. Configure Secrets
# Add DB credentials to .streamlit/secrets.toml

# 3. Launch the AI Matching Engine
streamlit run RXTracking_WebGUI_Streamlit.py
```

---

## 📜 License & Intellectual Property
**Copyright (c) 2026 Benedic Cater / InnoGen Pharmaceuticals Inc.**

**All Rights Reserved.**
This repository is published for **portfolio review and technical demonstration purposes only.**

**Strict Restrictions:**
- **No Reproduction:** No part of this code may be copied, modified, or distributed.
- **Brand Protection:** Use of the "InnoGen" or "Solvang" name, branding, or logos is strictly prohibited.
- **Data Privacy:** Use of any proprietary data or business logic contained herein for commercial or personal projects is strictly prohibited.

_For professional inquiries or permission requests, please contact Benedic Cater._
