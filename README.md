# RX Tracking System v2.0 (Streamlit) — AI-Assisted Matching

RX Tracking System v2.0 is a Streamlit web application for consolidating RX transaction files and matching doctor records against a masterlist using a multi-step rules engine with AI-assisted matching (TF‑IDF + nearest-neighbor similarity). It produces standardized outputs, matching metrics, and export-ready reports.

## Key Features

- **Two matching modes**
  - **Basic**: exact matching with strict rules (drives `suggest_dn = TRUE`).
  - **Advanced**: optimized batch matching with similarity thresholds (AI-type matches; `suggest_dn` remains `FALSE`).
- **AI-assisted matching for unmatched rows**
  - TF‑IDF character n-grams + `NearestNeighbors` for doctor name and PTR.
  - Optional PPE Doctors matching.
  - Word-based “Quick Suggest” with optional reference CSV integration.
- **Metrics dashboard**
  - Masterlist Match Rate, AI Match Rate, Quick Suggest Rate, total amounts, record counts.
- **Export**
  - Downloadable results in consistent column order.
- **Data integrity safeguards**
  - Amount column preservation + validation checks between matching steps.
- **Reference/auxiliary datasets**
  - `doctors_reference.csv`, `ppe_doctors.csv`, `ptr_with_topmd.csv`, item cross references, masterlist.

## Project Layout

- `RXTracking_WebGUI_Streamlit.py` — main Streamlit application
- `.streamlit/config.toml` — Streamlit configuration
- `.streamlit/secrets.toml` — local/server-only secrets (not committed)
- `rx_md_masterlist.csv` — doctor masterlist
- `ptr_with_topmd.csv` — PTR final mapping (TopMD)
- `ppe_doctors.csv` — PPE doctor list (optional, can auto-download if configured)
- `doctors_reference.csv` — optional reference list for Quick Suggest
- `rx_item_cross_ref.csv`, `Table_Item.csv` — item-related reference data (if used by your workflow)

## Matching Output Columns (Core)

The app uses internal columns and also renames some for display/export:

- `suggested_md` → displayed as `suggest_dn`
- `md_official_name` → displayed as `MD NAME FINAL`
- `md_ptrs` → displayed as `PTR FINAL`

Other common columns:
- `quick_suggest_name`, `suggested_name`
- `DOCTOR_CODE`, `CUSTOMER_CODE`
- Transaction fields: `Doctor Name`, `PTR No`, `Address1`, `Branch Code`, etc.

## Matching Pipeline (High Level)

Typical flow (Basic mode):

1. **Exact Match (Basic)**  
   Exact key match using Doctor Name and/or PTR No (normalized). Sets `suggest_dn = TRUE`.
2. **Exact Address Match**  
   Exact match on `(Doctor Name, Address1)` ↔ `(md_suggest, md_add_1)`. Keeps `suggest_dn = FALSE`.
3. **AI Matching (TF‑IDF Masterlist)**  
   Tries to match unmatched records using similarity (doctor name and/or PTR), with validations.
4. **PPE Doctors Matching (optional)**  
   Matches remaining blanks using PPE Doctors list.
5. **Quick Suggest (final step)**  
   Word-based matching + reference integration; fills quick suggestions without claiming TRUE matches.
6. **Unmatched labeling**  
   True unmatched records are left blank in `suggest_dn`.

## Requirements

This app is a Streamlit + pandas + SQL Server workflow. Install dependencies (example):

```bash
pip install streamlit pandas sqlalchemy scikit-learn rapidfuzz psutil pyodbc openpyxl
```

### SQL Server Driver (Windows)

Install a Microsoft ODBC driver (e.g., “ODBC Driver 18 for SQL Server” or “ODBC Driver 17 for SQL Server”) and ensure it is visible in ODBC Data Sources.

## Secure Configuration (IMPORTANT)

### Do NOT commit secrets
This repo is intended to be public/private on GitHub without exposing credentials.

- Keep `.streamlit/secrets.toml` out of Git (recommended).
- Use environment variables on production servers when possible.

### Option A — Streamlit Secrets (recommended for Streamlit hosting)
Create a local/server file:

`./.streamlit/secrets.toml`

```toml
[db_credentials]
host = "YOUR_SERVER\\INSTANCE"
user = "YOUR_USERNAME"
password = "YOUR_PASSWORD"
rxtracking_database = "RXTracking"
innogen_database = "InnogenBC174"
```

### Option B — Environment Variables

```text
DB_HOST=YOUR_SERVER\INSTANCE
DB_USER=YOUR_USERNAME
DB_PASSWORD=YOUR_PASSWORD
DB_NAME_RXTRACKING=RXTracking
DB_NAME_INNOGEN=InnogenBC174
```

If you previously committed credentials, rotate passwords immediately and purge history before publishing.

## Run Locally

From the project directory:

```bash
streamlit run RXTracking_WebGUI_Streamlit.py
```

## Notes on Data Files

- Keep reference CSVs in the project folder if you want them loaded locally.
- Some features (like PPE auto-update) may depend on your SQL stored procedures and network access.

## Troubleshooting

- **DB connection fails**
  - Confirm ODBC driver installed.
  - Confirm credentials configured via secrets/env.
  - Confirm server/instance name resolves from the machine running Streamlit.
- **Memory issues on large datasets**
  - Use Basic mode first, then AI matching.
  - Reduce dataset size or run on a machine with more RAM.
- **Quick Suggest / Reference not working**
  - Confirm `doctors_reference.csv` exists and has expected columns.


## License

### Copyright (c) 2026 Benedic Cater / InnoGen Pharmaceuticals Inc.

### All Rights Reserved.

This repository and its contents, including all code, assets, and data, are the sole property of the author. This code is made public for portfolio review and demonstration purposes only.

### Restrictions:
- You may not copy, modify, or distribute this code.
- You may not use the "InnoGen" name, branding, or logos for any purpose.
- Use of the data contained within this repository for commercial or personal projects is strictly prohibited.

For inquiries or permission requests, please contact the author.
