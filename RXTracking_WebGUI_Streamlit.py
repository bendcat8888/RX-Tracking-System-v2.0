import streamlit as st
import pandas as pd
import re
from io import StringIO, BytesIO
import zipfile
import os
import urllib.parse
import platform
import tempfile
from sqlalchemy import create_engine, text
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import gc
import signal
import sys
import traceback
import logging
from functools import wraps
import time
from datetime import datetime
import threading
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

# Configure logging for error tracking
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag to track if we're in a critical operation
_in_critical_operation = False

def handle_critical_errors(func):
    """Decorator to handle critical errors and prevent crashes."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _in_critical_operation
        try:
            _in_critical_operation = True
            return func(*args, **kwargs)
        except MemoryError as e:
            logger.error(f"Memory error in {func.__name__}: {str(e)}")
            cleanup_memory()
            # Return safe defaults instead of crashing
            if 'df' in kwargs:
                return kwargs['df'], 0
            return None, 0
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}")
            cleanup_memory()
            # Return safe defaults instead of crashing
            if 'df' in kwargs:
                return kwargs['df'], 0
            return None, 0
        finally:
            _in_critical_operation = False
    return wrapper

def signal_handler(signum, frame):
    """Handle signals gracefully to prevent abrupt termination."""
    global _in_critical_operation
    if _in_critical_operation:
        logger.warning(f"Signal {signum} received during critical operation. Attempting graceful cleanup...")
        cleanup_memory()
    sys.exit(0)

# Register signal handlers for graceful shutdown
# Only register in main thread (signal handlers don't work in Streamlit's worker threads)
try:
    import threading
    if threading.current_thread() is threading.main_thread():
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal_handler)
except (AttributeError, ValueError):
    # Signal registration failed (not in main thread or not supported)
    # This is OK - Streamlit handles signals itself
    pass

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signals."""
    raise TimeoutError("Operation timed out")

def with_timeout(timeout_seconds):
    """Decorator to add timeout to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if hasattr(signal, 'SIGALRM'):  # Unix only
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                return result
            else:
                # Windows doesn't support SIGALRM, use time-based check instead
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    raise TimeoutError(f"Operation took {elapsed:.1f}s, exceeded timeout of {timeout_seconds}s")
                return result
        return wrapper
    return decorator

def get_memory_usage():
    """Get current memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return 0
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except Exception:
        return 0

def check_memory_available(min_mb=500):
    """Check if enough memory is available."""
    if not PSUTIL_AVAILABLE:
        return True  # If psutil not available, assume OK
    try:
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024
        return available_mb >= min_mb
    except Exception:
        return True  # If check fails, assume OK

def cleanup_memory(force=False):
    """
    Force garbage collection to free memory.
    Args:
        force: If True, force collection even if memory usage is low.
    """
    # Check memory usage first to avoid unnecessary overhead
    should_collect = force
    if not should_collect:
        try:
            # OPTIMIZATION: Only run if memory is actually low (> 2GB used or < 500MB free)
            # Increased threshold to reduce frequency
            if get_memory_usage() > 2000:  # > 2GB
                should_collect = True
            elif not check_memory_available(min_mb=500):
                should_collect = True
        except Exception:
            # If checking fails, don't force collect unless necessary to avoid slowdowns
            should_collect = False 
            
    if should_collect:
        # OPTIMIZATION: Run garbage collection ONLY ONCE instead of 3 times
        # This provides the "Major Speed Boost" requested
        gc.collect()


# --------------------------------------------------------------
# CSV File Directory Helper
# --------------------------------------------------------------

def get_csv_dir():
    """
    Get the directory for CSV file storage.
    
    Uses tempfile.gettempdir() to avoid file locking issues in multi-user environments.
    This prevents writing CSV files to the source directory which can cause conflicts.
    
    Returns:
        str: Path to the temporary directory for CSV files
    """
    return tempfile.gettempdir()

def get_reference_csv_dir():
    """
    Get the directory for reference CSV file storage.

    Uses script_dir (persistent location) instead of temp directory because reference files
    need to persist across sessions and should be accessible to all users.
    
    Returns:
        str: Path to the script directory for reference CSV files
    """
    return os.path.dirname(os.path.abspath(__file__))

ITEM_CROSS_REF_FILENAME = 'rx_item_cross_ref.csv'
ITEM_CROSS_REF_KEY_COLUMN = 'Cross-Reference No.'
ITEM_CROSS_REF_REQUIRED_COLUMNS = [ITEM_CROSS_REF_KEY_COLUMN, 'Item No.']
ITEM_CROSS_REF_DEFAULT_COLUMNS = [
    'Cross-Reference No.',
    'Item No.',
    'Description',
    'FINAL GROSS Unit Price',
    'DOH CEILING PRICE',
    'DISC',
    'net of disc',
    'FINAL NET PRICE',
    'VAT Product Posting Group',
    'DIVISION',
    'Standard Cost',
]

def get_item_cross_ref_path():
    """Return the persistent path for rx_item_cross_ref.csv."""
    return os.path.join(get_reference_csv_dir(), ITEM_CROSS_REF_FILENAME)

def _normalize_upload_column_name(column_name):
    return re.sub(r'\s+', ' ', str(column_name).strip()).lower()

def _normalize_item_cross_ref_key(series):
    return series.fillna('').astype(str).str.strip().str.upper()

def get_item_cross_ref_columns():
    """Use the current cross-reference header when available; otherwise use the default schema."""
    cross_ref_path = get_item_cross_ref_path()
    if os.path.exists(cross_ref_path):
        try:
            existing_columns = pd.read_csv(cross_ref_path, nrows=0).columns.tolist()
            existing_columns = [str(col).strip() for col in existing_columns if str(col).strip()]
            if existing_columns:
                return existing_columns
        except Exception as e:
            logger.warning(f"Could not read rx_item_cross_ref.csv columns: {str(e)}")
    return ITEM_CROSS_REF_DEFAULT_COLUMNS.copy()

def build_item_cross_ref_template():
    """Build an Excel template with the current cross-reference columns and one sample row."""
    columns = get_item_cross_ref_columns()
    sample_row = {col: '' for col in columns}
    cross_ref_path = get_item_cross_ref_path()

    if os.path.exists(cross_ref_path):
        try:
            sample_df = pd.read_csv(cross_ref_path, dtype=str, nrows=1).fillna('')
            if not sample_df.empty:
                sample_row.update(sample_df.iloc[0].to_dict())
        except Exception as e:
            logger.warning(f"Could not read rx_item_cross_ref.csv sample row: {str(e)}")

    if not sample_row.get(ITEM_CROSS_REF_KEY_COLUMN):
        sample_row[ITEM_CROSS_REF_KEY_COLUMN] = 'SAMPLE-CROSS-REF'
    if not sample_row.get('Item No.'):
        sample_row['Item No.'] = 'IP0000001'
    if 'Description' in sample_row and not sample_row.get('Description'):
        sample_row['Description'] = 'Sample Item'

    template_df = pd.DataFrame([sample_row], columns=columns)
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            template_df.to_excel(writer, index=False, sheet_name='Item Cross Reference')
        return (
            output.getvalue(),
            'rx_item_cross_ref_template.xlsx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
    except Exception as e:
        logger.warning(f"Could not build Excel template; falling back to CSV template: {str(e)}")
        return (
            template_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
            'rx_item_cross_ref_template.csv',
            'text/csv',
        )

def read_item_cross_ref_upload(uploaded_file):
    """Read an uploaded item cross-reference file into a DataFrame."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'csv':
        return pd.read_csv(uploaded_file, dtype=str, encoding='utf-8-sig', on_bad_lines='skip')
    if file_extension in ['xlsx', 'xls']:
        return pd.read_excel(uploaded_file, dtype=str, engine='openpyxl' if file_extension == 'xlsx' else None)
    raise ValueError("Unsupported file type. Please upload an Excel or CSV file.")

def upsert_item_cross_ref_from_upload(uploaded_file):
    """
    Validate and upsert uploaded item cross-reference rows.

    Existing rows are matched by Cross-Reference No.; matching rows are replaced and
    non-existing rows are appended.
    """
    expected_columns = get_item_cross_ref_columns()
    expected_lookup = {_normalize_upload_column_name(col): col for col in expected_columns}

    uploaded_df = read_item_cross_ref_upload(uploaded_file)
    if uploaded_df is None or uploaded_df.empty:
        return False, "Uploaded file is empty.", None

    uploaded_df.columns = [str(col).strip() for col in uploaded_df.columns]
    uploaded_lookup = {}
    duplicate_columns = []
    for col in uploaded_df.columns:
        normalized_col = _normalize_upload_column_name(col)
        if normalized_col in uploaded_lookup:
            duplicate_columns.append(col)
        uploaded_lookup[normalized_col] = col

    if duplicate_columns:
        return False, f"Duplicate uploaded columns found: {', '.join(duplicate_columns)}", None

    missing_required = [
        col for col in ITEM_CROSS_REF_REQUIRED_COLUMNS
        if _normalize_upload_column_name(col) not in uploaded_lookup
    ]
    if missing_required:
        return False, f"Missing required column(s): {', '.join(missing_required)}", None

    aligned_df = pd.DataFrame(index=uploaded_df.index)
    missing_optional = []
    for expected_col in expected_columns:
        source_col = uploaded_lookup.get(_normalize_upload_column_name(expected_col))
        if source_col:
            aligned_df[expected_col] = uploaded_df[source_col]
        else:
            aligned_df[expected_col] = ''
            if expected_col not in ITEM_CROSS_REF_REQUIRED_COLUMNS:
                missing_optional.append(expected_col)

    extra_columns = [
        col for col in uploaded_df.columns
        if _normalize_upload_column_name(col) not in expected_lookup
    ]

    aligned_df = aligned_df.fillna('')
    for col in aligned_df.columns:
        aligned_df[col] = aligned_df[col].astype(str).str.strip()

    before_blank_filter = len(aligned_df)
    valid_required_mask = pd.Series(True, index=aligned_df.index)
    for required_col in ITEM_CROSS_REF_REQUIRED_COLUMNS:
        valid_required_mask &= aligned_df[required_col].astype(str).str.strip() != ''
    aligned_df = aligned_df[valid_required_mask].copy()
    blank_rows_removed = before_blank_filter - len(aligned_df)

    if aligned_df.empty:
        return False, "No valid rows found. Cross-Reference No. and Item No. must both have values.", None

    uploaded_keys = _normalize_item_cross_ref_key(aligned_df[ITEM_CROSS_REF_KEY_COLUMN])
    duplicate_upload_rows = int(uploaded_keys.duplicated(keep='last').sum())
    if duplicate_upload_rows:
        aligned_df = aligned_df.loc[~uploaded_keys.duplicated(keep='last')].copy()
        uploaded_keys = _normalize_item_cross_ref_key(aligned_df[ITEM_CROSS_REF_KEY_COLUMN])

    cross_ref_path = get_item_cross_ref_path()
    if os.path.exists(cross_ref_path):
        if is_file_in_use(cross_ref_path):
            return False, "rx_item_cross_ref.csv is currently in use. Please close it and try again.", None
        existing_df = pd.read_csv(cross_ref_path, dtype=str).fillna('')
    else:
        existing_df = pd.DataFrame(columns=expected_columns)

    for col in expected_columns:
        if col not in existing_df.columns:
            existing_df[col] = ''
    existing_df = existing_df[expected_columns].fillna('')

    existing_keys = _normalize_item_cross_ref_key(existing_df[ITEM_CROSS_REF_KEY_COLUMN])
    upload_key_set = set(uploaded_keys)
    existing_key_set = set(existing_keys[existing_keys != ''])

    replaced_count = len(upload_key_set & existing_key_set)
    added_count = len(upload_key_set - existing_key_set)
    retained_df = existing_df.loc[~existing_keys.isin(upload_key_set)].copy()
    updated_df = pd.concat([retained_df, aligned_df], ignore_index=True)
    updated_df.to_csv(cross_ref_path, index=False, encoding='utf-8-sig')

    load_cross_reference_csv.clear()

    details = {
        'total_records': len(updated_df),
        'uploaded_rows': len(aligned_df),
        'replaced_count': replaced_count,
        'added_count': added_count,
        'blank_rows_removed': blank_rows_removed,
        'duplicate_upload_rows': duplicate_upload_rows,
        'missing_optional': missing_optional,
        'extra_columns': extra_columns,
        'preview_df': updated_df.tail(min(len(aligned_df), 100)),
    }
    message = (
        f"Successfully updated {ITEM_CROSS_REF_FILENAME}. "
        f"Replaced {replaced_count:,} existing row(s), added {added_count:,} new row(s). "
        f"Total records: {len(updated_df):,}."
    )
    return True, message, details

# --------------------------------------------------------------
# Database Credentials Helper
# --------------------------------------------------------------

def get_db_credentials(db_key="rxtracking"):
    """
    Get database credentials from st.secrets or environment variables.
    
    SECURITY: No credentials are stored in this file. You must configure them via:
    - Streamlit secrets: .streamlit/secrets.toml
    - Environment variables: DB_HOST, DB_USER, DB_PASSWORD, DB_NAME_RXTRACKING, DB_NAME_INNOGEN
    
    Args:
        db_key: "rxtracking" or "innogen" to select the database
        
    Returns:
        dict with keys: host, user, password, database
        
    Raises:
        ValueError: If credentials are not configured
    """
    # Database name mapping
    db_names = {
        "rxtracking": "DB_NAME_RXTRACKING",
        "innogen": "DB_NAME_INNOGEN"
    }
    
    # Try st.secrets first (preferred for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and "db_credentials" in st.secrets:
            secrets = st.secrets["db_credentials"]
            db_name_key = f"{db_key}_database"
            return {
                "host": secrets["host"],
                "user": secrets["user"],
                "password": secrets["password"],
                "database": secrets.get(db_name_key, secrets.get("database", ""))
            }
    except (KeyError, TypeError):
        pass
    
    # Try environment variables as fallback
    import os
    host = os.environ.get("DB_HOST")
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    database = os.environ.get(db_names.get(db_key, "DB_NAME_RXTRACKING"))
    
    if all([host, user, password, database]):
        return {
            "host": host,
            "user": user,
            "password": password,
            "database": database
        }
    
    # No credentials found - show helpful error
    st.error("""
    ⚠️ **Database credentials not configured!**
    
    Please configure credentials using ONE of these methods:
    
    **Option 1: Streamlit secrets** (recommended)
    Create `.streamlit/secrets.toml`:
    ```toml
    [db_credentials]
    host = "your_server\\\\instance"
    user = "your_username"
    password = "your_password"
    rxtracking_database = "RXTracking"
    innogen_database = "InnogenBC174"
    ```
    
    **Option 2: Environment variables**
    ```
    DB_HOST=your_server\\instance
    DB_USER=your_username
    DB_PASSWORD=your_password
    DB_NAME_RXTRACKING=RXTracking
    DB_NAME_INNOGEN=InnogenBC174
    ```
    """)
    raise ValueError(f"Database credentials not configured for '{db_key}'. See error message above.")


# --------------------------------------------------------------
# Amount parsing helpers
# --------------------------------------------------------------

def _extract_numeric_amount(value):
    """Extract the last numeric token from a potentially messy string."""
    if pd.isna(value) or value == '':
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        value_str = str(value)
        # Find all numeric tokens (keep commas/decimals); take the LAST token to avoid preceding codes
        matches = re.findall(r'-?\d[\d,]*\.?\d*', value_str)
        if not matches:
            return 0.0
        token = matches[-1]
        token = token.replace(',', '')
        if token in ['', '.']:
            return 0.0
        return float(token)
    except (ValueError, TypeError):
        return 0.0


def standardize_amount_column(df, col_name='Amount'):
    """
    Unified Amount Pipeline: Standardize column to float64.
    
    1. Cleans string formatting (commas, spaces, symbols)
    2. Converts to numeric (coercing errors)
    3. Fills NaNs with 0.0
    4. Enforces float64 dtype
    
    Args:
        df: DataFrame to process
        col_name: Name of the amount column (default: 'Amount')
        
    Returns:
        DataFrame with standardized amount column
    """
    if df is None or df.empty or col_name not in df.columns:
        return df
        
    # fast path: if already float64, just fillna
    if df[col_name].dtype == 'float64':
        df[col_name] = df[col_name].fillna(0.0)
        return df
        
    # Apply cleaning logic
    # Use _extract_numeric_amount for robust parsing of dirty data
    # (e.g. "PHP 1,234.56" -> 1234.56)
    if df[col_name].dtype == 'object':
        df[col_name] = df[col_name].apply(_extract_numeric_amount)
        
    # Convert to numeric, handle errors, fill NaN, cast to float64
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0.0).astype('float64')
    
    return df


def clean_amount_immediate(value):
    """Clean amount value immediately after reading from file."""
    return _extract_numeric_amount(value)

def get_optimal_n_jobs(min_memory_mb=2000):
    """
    Determine optimal n_jobs for NearestNeighbors based on available memory.
    Returns -1 (all cores) if memory is sufficient, otherwise 1 (single core).
    """
    if not PSUTIL_AVAILABLE:
        return -1  # If psutil not available, use all cores (original behavior)
    try:
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024
        # Use all cores if we have enough memory, otherwise use single core
        if available_mb >= min_memory_mb:
            return -1  # Use all available CPU cores for faster processing
        else:
            return 1   # Use single core to reduce memory usage
    except Exception:
        return -1  # If check fails, use all cores (safer default)

def get_tfidf_batch_size(num_records, available_memory_mb=None):
    """
    Determine optimal batch size for TF-IDF processing based on number of records and available memory.
    For large datasets (500K+ rows), processes in smaller batches to prevent memory issues.
    """
    if available_memory_mb is None:
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                available_memory_mb = memory.available / 1024 / 1024
            except Exception:
                available_memory_mb = 1000  # Conservative default
        else:
            available_memory_mb = 1000  # Conservative default
    
    # For very large datasets, use moderate batches (optimized for speed with float32)
    # With float32 and max_features=10000, larger batches are safe
    if num_records > 150000:  # 150K+ records (very large)
        # Use moderate batches: ~3000 records per batch (increased from 2000)
        # This reduces overhead while staying safe with optimized memory
        return min(3000, max(2000, num_records // 67))  # ~67 batches instead of 100
    elif num_records > 100000:  # 100K-150K records
        # Use medium batches: ~4000 records per batch (increased from 3000)
        return min(4000, max(3000, num_records // 40))  # ~40 batches instead of 50
    elif num_records > 50000:  # 50K-100K records
        # Medium batches: ~6000 records per batch (increased from 5000)
        return min(6000, max(4000, num_records // 15))  # ~15 batches instead of 20
    elif num_records > 20000:  # 20K-50K records
        # Medium-small batches: ~12K records per batch (increased from 10K)
        return min(12000, max(8000, num_records // 5))  # ~5 batches instead of 10
    else:
        # Small datasets: process all at once
        return num_records

def safe_dataframe_copy(df, max_rows=None):
    """Safely copy dataframe with memory limits."""
    if df is None or df.empty:
        return df
    if max_rows and len(df) > max_rows:
        return df.head(max_rows).copy()
    return df.copy()

def clean_dataframe_for_display(df):
    """Clean DataFrame to ensure PyArrow compatibility for Streamlit display."""
    if df is None or df.empty:
        return df
    
    df_clean = df.copy()
    
    # Columns that should be preserved as string (not converted to numeric)
    # Address1, Address2 are address text; PTR No can have leading zeros; Branch Code/Name are identifiers
    preserve_as_string_cols = [
        'suggest_dn', 'suggested_md',
        'Address1', 'Address2', 'Address', 'PTR No', 'PTR',
        'Branch Code', 'Branch Name', 'Doctor Name', "Vendor's Name", 'Supplier Name',
        'Item Name', 'InnoGen Item Name', 'Item Code/Name', 'file_loc'
    ]
    
    # Convert all columns to ensure PyArrow compatibility
    for col in df_clean.columns:
        try:
            # Skip columns that should be preserved as string
            if col in preserve_as_string_cols:
                # Convert to string, preserving special values like 'REF' and ''
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace('nan', '').replace('None', '').replace('NaT', '').replace('<NA>', '')
                continue
            
            # For numeric columns, keep as numeric but ensure no mixed types
            if df_clean[col].dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32', 'float16']:
                # Convert to numeric, coercing errors to NaN, then fill NaN with 0
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].fillna(0)
            elif df_clean[col].dtype == 'bool':
                # Keep boolean as is, but convert to int for better compatibility
                df_clean[col] = df_clean[col].astype(int)
            elif df_clean[col].dtype == 'object':
                # Convert object columns to string, handling NaN properly
                # First try to detect if it's actually numeric
                try:
                    # Try converting to numeric first
                    numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                    if not numeric_series.isna().all():
                        # If most values are numeric, use numeric
                        df_clean[col] = numeric_series.fillna(0)
                    else:
                        # Otherwise convert to string
                        df_clean[col] = df_clean[col].astype(str)
                        df_clean[col] = df_clean[col].replace('nan', '').replace('None', '').replace('NaT', '').replace('<NA>', '')
                except Exception:
                    # If numeric conversion fails, convert to string
                    df_clean[col] = df_clean[col].astype(str)
                    df_clean[col] = df_clean[col].replace('nan', '').replace('None', '').replace('NaT', '').replace('<NA>', '')
            elif df_clean[col].dtype.name.startswith('datetime'):
                # Convert datetime to string
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace('NaT', '').replace('nan', '')
            else:
                # For other types, convert to string
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace('nan', '').replace('None', '').replace('NaT', '').replace('<NA>', '')
        except Exception:
            # If conversion fails, convert to string as fallback
            try:
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace('nan', '').replace('None', '').replace('NaT', '').replace('<NA>', '')
            except Exception:
                # If all else fails, replace the column with empty strings
                df_clean[col] = ''
    
    return df_clean

def update_summary(combined_summary, summary):
    """Update combined summary with file summary data."""
    if 'num_records' in summary:
        combined_summary['num_records'] += summary['num_records']
    if 'total_amount' in summary:
        try:
            amount_str = str(summary['total_amount']).replace(',', '')
            combined_summary['total_amount'] += float(amount_str)
        except (ValueError, TypeError):
            pass

def is_file_in_use(file_path):
    """Check if a file is currently in use (locked) on Windows."""
    if not os.path.exists(file_path):
        return False
    try:
        # Try to open the file in exclusive mode
        # On Windows, this will fail if the file is locked by another process
        with open(file_path, 'r+b'):
            pass
        return False  # File is not locked
    except (IOError, PermissionError, OSError):
        return True  # File is locked/in use

def should_update_file(file_path, hours_threshold=3):
    """
    Check if file should be updated based on modification time.
    
    Args:
        file_path: Path to the file
        hours_threshold: Minimum hours since last update to allow update (default: 3)
    
    Returns:
        tuple: (should_update: bool, status_message: str, last_modified: datetime or None)
    """
    if not os.path.exists(file_path):
        return True, "File does not exist - update required.", None
    
    # Check if file is in use
    if is_file_in_use(file_path):
        return False, "⚠️ File is currently in use by another process - update skipped.", None
    
    # Get file modification time
    try:
        mod_time = os.path.getmtime(file_path)
        last_modified = datetime.fromtimestamp(mod_time)
        current_time = datetime.now()
        time_diff = current_time - last_modified
        hours_diff = time_diff.total_seconds() / 3600
        
        if hours_diff >= hours_threshold:
            return True, f"✅ File last updated {hours_diff:.1f} hours ago - update allowed.", last_modified
        else:
            return False, f"⏭️ File updated {hours_diff:.1f} hours ago (less than {hours_threshold} hours) - update skipped.", last_modified
    except Exception as e:
        return True, f"⚠️ Could not check file modification time: {str(e)} - proceeding with update.", None

def download_masterlist_from_server(force_update=False):
    """
    Download RX MD Masterlist from SQL Server using stored procedure and save to CSV.
    
    Args:
        force_update: If True, bypass file age and in-use checks
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        csv_dir = get_csv_dir()
        csv_path = os.path.join(csv_dir, 'rx_md_masterlist.csv')
        
        # Check if update is needed (unless forced)
        if not force_update and os.path.exists(csv_path):
            should_update, status_msg, last_modified = should_update_file(csv_path, hours_threshold=3)
            if not should_update:
                # Return status message indicating why update was skipped
                if last_modified:
                    return False, f"{status_msg} Last modified: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}"
                else:
                    return False, status_msg
        # Get credentials from helper (supports st.secrets override)
        creds = get_db_credentials("rxtracking")
        password_encoded = urllib.parse.quote_plus(creds["password"])
        
        # Detect OS and use appropriate driver
        # Windows uses "SQL Server", Linux/Ubuntu uses "ODBC Driver 17 for SQL Server" or "ODBC Driver 18 for SQL Server"
        system = platform.system().lower()
        if system == 'windows':
            driver = "SQL Server"
        else:
            # Try ODBC Driver 18 first (newer), fallback to 17
            # You may need to install: sudo apt-get install -y msodbcsql18
            # Or for older versions: sudo apt-get install -y msodbcsql17
            driver = "ODBC Driver 18 for SQL Server"
            # Alternative: driver = "ODBC Driver 17 for SQL Server"
        
        # Create connection string with OS-appropriate driver
        # For Linux, URL-encode the driver name
        driver_encoded = urllib.parse.quote_plus(driver)
        conn_str = f"mssql+pyodbc://{creds['user']}:{password_encoded}@{creds['host']}/{creds['database']}?driver={driver_encoded}"
        
        # Additional connection parameters for Linux/Ubuntu
        if system != 'windows':
            # Add TrustServerCertificate for Linux connections (if needed)
            conn_str += "&TrustServerCertificate=yes"
        
        engine = create_engine(conn_str)
        
        # Stored procedure name
        sproc = "sp_bc365_final_rx_tracking"
        
        # Connect to database and execute stored procedure
        with engine.connect() as connection:
            # Execute stored procedure using text() for parameterized queries
            # If the stored procedure requires parameters, add them here
            # Example with parameters: query = text(f"EXEC [dbo].[{sproc}] @param1, @param2")
            query = text(f"EXEC [dbo].[{sproc}]")
            result_csv = pd.read_sql(query, connection)
        
        # Validate that we got data
        if result_csv is None or result_csv.empty:
            engine.dispose()
            return False, "Stored procedure returned no data. Please check the stored procedure and database."
        
        # Remove date columns if they exist (optional - adjust based on your needs)
        if 'md_enc_date' in result_csv.columns:
            result_csv = result_csv.drop(columns=['md_enc_date'])
        if 'md_upd_date' in result_csv.columns:
            result_csv = result_csv.drop(columns=['md_upd_date'])
        
        # Dispose engine
        engine.dispose()
        
        # Check if file is in use before saving
        csv_dir = get_csv_dir()
        csv_path = os.path.join(csv_dir, 'rx_md_masterlist.csv')
        
        if is_file_in_use(csv_path):
            return False, "⚠️ Cannot update: File is currently in use by another process. Please close any applications using rx_md_masterlist.csv and try again."
        
        # Save to CSV
        try:
            result_csv.to_csv(csv_path, index=False)
            return True, f"Successfully downloaded {len(result_csv)} records from server using stored procedure '{sproc}'."
        except PermissionError:
            return False, "⚠️ Cannot update: File is locked or permission denied. Please close any applications using rx_md_masterlist.csv and try again."
        except Exception as e:
            return False, f"⚠️ Error saving file: {str(e)}"
    except Exception as e:
        error_msg = f"Error connecting to server or executing stored procedure: {str(e)}"
        logger.error(f"download_masterlist_from_server error: {error_msg}\n{traceback.format_exc()}")
        return False, error_msg

def download_ptr_with_topmd_from_server(force_update=False):
    """
    Download PTR with TopMD data from SQL Server using stored procedure sp_PTR_with_TopMD and save to CSV.

    Args:
        force_update: If True, bypass file age and in-use checks

    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        csv_dir = get_reference_csv_dir()
        ptr_topmd_path = os.path.join(csv_dir, 'ptr_with_topmd.csv')

        if not force_update and os.path.exists(ptr_topmd_path):
            should_update, status_msg, last_modified = should_update_file(ptr_topmd_path, hours_threshold=3)
            if not should_update:
                if last_modified:
                    return False, f"{status_msg} Last modified: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}"
                return False, status_msg
        creds = get_db_credentials("rxtracking")
        password_encoded = urllib.parse.quote_plus(creds["password"])
        system = platform.system().lower()
        if system == 'windows':
            driver = "SQL Server"
        else:
            driver = "ODBC Driver 18 for SQL Server"
        driver_encoded = urllib.parse.quote_plus(driver)
        conn_str = f"mssql+pyodbc://{creds['user']}:{password_encoded}@{creds['host']}/{creds['database']}?driver={driver_encoded}"
        if system != 'windows':
            conn_str += "&TrustServerCertificate=yes"
        engine = create_engine(conn_str)
        sproc = "sp_PTR_with_TopMD"
        with engine.connect() as connection:
            query = text(f"EXEC [dbo].[{sproc}]")
            result_csv = pd.read_sql(query, connection)
        if result_csv is None or result_csv.empty:
            engine.dispose()
            return False, "Stored procedure returned no data. Please check the stored procedure and database."
        engine.dispose()
        if is_file_in_use(ptr_topmd_path):
            return False, "⚠️ Cannot update: File is currently in use by another process. Please close any applications using ptr_with_topmd.csv and try again."
        try:
            result_csv.to_csv(ptr_topmd_path, index=False)
            return True, f"Successfully downloaded {len(result_csv)} records from server using stored procedure '{sproc}'."
        except PermissionError:
            return False, "⚠️ Cannot update: File is locked or permission denied. Please close any applications using ptr_with_topmd.csv and try again."
        except Exception as e:
            return False, f"⚠️ Error saving file: {str(e)}"
    except Exception as e:
        error_msg = f"Error connecting to server or executing stored procedure: {str(e)}"
        logger.error(f"download_ptr_with_topmd_from_server error: {error_msg}\n{traceback.format_exc()}")
        return False, error_msg

def update_masterlist_with_ptr_topmd():
    """
    Add [md_ptrs_final] column to masterlist and fill it with PTR_FINAL from TopMD.
    Matches md_official_name (masterlist) = MD_NAME (PTR_with_TopMD).
    Does NOT overwrite [md_ptrs] - [md_ptrs] is preserved; PTR Final goes to [md_ptrs_final] only.

    Returns:
        tuple: (success: bool, message: str, records_updated: int)
    """
    try:
        csv_dir = get_csv_dir()
        masterlist_path = os.path.join(csv_dir, 'rx_md_masterlist.csv')
        ptr_topmd_path = os.path.join(get_reference_csv_dir(), 'ptr_with_topmd.csv')

        if not os.path.exists(masterlist_path):
            return False, "Masterlist file not found.", 0
        if not os.path.exists(ptr_topmd_path):
            return False, "PTR with TopMD file not found.", 0

        if is_file_in_use(masterlist_path):
            return False, "⚠️ Cannot update: Masterlist file is currently in use.", 0

        masterlist_df = pd.read_csv(masterlist_path)
        if masterlist_df.empty:
            return False, "Masterlist is empty.", 0

        ptr_topmd_df = pd.read_csv(ptr_topmd_path)
        if ptr_topmd_df.empty:
            return False, "PTR with TopMD data is empty.", 0

        if 'md_official_name' not in masterlist_df.columns:
            return False, "Masterlist missing required column: md_official_name", 0

        md_name_col = None
        for col in ptr_topmd_df.columns:
            if col.upper() == 'MD_NAME' or col.lower() == 'md_name':
                md_name_col = col
                break
        if md_name_col is None:
            return False, f"PTR with TopMD missing MD_NAME column. Available columns: {', '.join(ptr_topmd_df.columns.tolist())}", 0

        ptr_col_name = None
        for col in ptr_topmd_df.columns:
            if col.upper() == 'PTR_FINAL':
                ptr_col_name = col
                break
        if ptr_col_name is None:
            for col in ptr_topmd_df.columns:
                if col.upper() in ['MD_PTRS', 'PTR', 'PTR_NO', 'PTR_NUMBER']:
                    ptr_col_name = col
                    break
        if ptr_col_name is None:
            return False, f"PTR with TopMD missing PTR_FINAL column. Available columns: {', '.join(ptr_topmd_df.columns.tolist())}", 0

        masterlist_df['md_official_name_normalized'] = masterlist_df['md_official_name'].fillna('').astype(str).str.strip().str.upper()
        ptr_topmd_df['md_name_normalized'] = ptr_topmd_df[md_name_col].fillna('').astype(str).str.strip().str.upper()
        ptr_topmd_df = ptr_topmd_df[ptr_topmd_df['md_name_normalized'] != ''].copy()

        if ptr_topmd_df.empty:
            return False, "PTR with TopMD has no valid MD_NAME values after normalization.", 0

        ptr_topmd_grouped = ptr_topmd_df.groupby('md_name_normalized', as_index=False).agg({ptr_col_name: 'last'})
        ptr_topmd_grouped[ptr_col_name] = ptr_topmd_grouped[ptr_col_name].fillna('').astype(str).str.strip()
        ptr_topmd_grouped = ptr_topmd_grouped[ptr_topmd_grouped[ptr_col_name] != ''].copy()

        if ptr_topmd_grouped.empty:
            return False, "PTR with TopMD has no valid PTR_FINAL values after filtering.", 0

        ptr_lookup = dict(zip(ptr_topmd_grouped['md_name_normalized'], ptr_topmd_grouped[ptr_col_name]))

        # Add md_ptrs_final column - do NOT overwrite md_ptrs
        if 'md_ptrs_final' not in masterlist_df.columns:
            masterlist_df['md_ptrs_final'] = ''
        matched_mask = masterlist_df['md_official_name_normalized'].isin(ptr_lookup.keys())
        masterlist_df.loc[matched_mask, 'md_ptrs_final'] = masterlist_df.loc[matched_mask, 'md_official_name_normalized'].map(ptr_lookup)
        masterlist_df['md_ptrs_final'] = masterlist_df['md_ptrs_final'].fillna('').astype(str).str.strip()

        records_updated = matched_mask.sum()
        matched_doctors = masterlist_df[matched_mask]['md_official_name_normalized'].nunique()
        masterlist_df = masterlist_df.drop(columns=['md_official_name_normalized'])

        try:
            masterlist_df.to_csv(masterlist_path, index=False)
            return True, f"Successfully added md_ptrs_final to {records_updated} masterlist records ({matched_doctors} unique doctors) from TopMD. [md_ptrs] preserved.", records_updated
        except PermissionError:
            return False, "⚠️ Cannot save: File is locked or permission denied.", 0
        except Exception as e:
            return False, f"⚠️ Error saving updated masterlist: {str(e)}", 0

    except Exception as e:
        error_msg = f"Error updating masterlist with PTR TopMD data: {str(e)}"
        logger.error(f"update_masterlist_with_ptr_topmd error: {error_msg}\n{traceback.format_exc()}")
        return False, error_msg, 0

def download_ppe_doctors_from_server(force_update=False):
    """
    Download PPE Doctors data from SQL Server using stored procedure and save to CSV.
    
    Args:
        force_update: If True, bypass file age and in-use checks
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        csv_dir = get_reference_csv_dir()  # Use persistent location, not temp directory
        ppe_path = os.path.join(csv_dir, 'ppe_doctors.csv')
        
        # Check if update is needed (unless forced)
        if not force_update and os.path.exists(ppe_path):
            should_update, status_msg, last_modified = should_update_file(ppe_path, hours_threshold=3)
            if not should_update:
                # Return status message indicating why update was skipped
                if last_modified:
                    return False, f"{status_msg} Last modified: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}"
                else:
                    return False, status_msg
        # Get credentials from helper (supports st.secrets override)
        creds = get_db_credentials("rxtracking")
        password_encoded = urllib.parse.quote_plus(creds["password"])
        
        # Detect OS and use appropriate driver
        system = platform.system().lower()
        if system == 'windows':
            driver = "SQL Server"
        else:
            driver = "ODBC Driver 18 for SQL Server"
        
        # Create connection string with OS-appropriate driver
        driver_encoded = urllib.parse.quote_plus(driver)
        conn_str = f"mssql+pyodbc://{creds['user']}:{password_encoded}@{creds['host']}/{creds['database']}?driver={driver_encoded}"
        
        # Additional connection parameters for Linux/Ubuntu
        if system != 'windows':
            conn_str += "&TrustServerCertificate=yes"
        
        engine = create_engine(conn_str)
        
        # Stored procedure name
        sproc = "sp_ppe_doctors"
        
        # Connect to database and execute stored procedure
        with engine.connect() as connection:
            # Execute stored procedure using text() for parameterized queries
            query = text(f"EXEC [dbo].[{sproc}]")
            result_csv = pd.read_sql(query, connection)
        
        # Validate that we got data
        if result_csv is None or result_csv.empty:
            engine.dispose()
            return False, "Stored procedure returned no data. Please check the stored procedure and database."
        
        # Ensure required columns exist (md_official_name, DOCTOR_CODE, CUSTOMER_CODE)
        required_columns = ['md_official_name', 'DOCTOR_CODE', 'CUSTOMER_CODE']
        missing_columns = [col for col in required_columns if col not in result_csv.columns]
        if missing_columns:
            engine.dispose()
            return False, f"Stored procedure returned data but missing required columns: {', '.join(missing_columns)}"
        
        # Select only the required columns
        result_csv = result_csv[required_columns].copy()
        
        # Clean and normalize md_official_name for matching (lowercase, strip)
        if 'md_official_name' in result_csv.columns:
            result_csv['md_official_name_clean'] = result_csv['md_official_name'].astype(str).str.strip().str.lower()
        
        # Dispose engine
        engine.dispose()
        
        # Check if file is in use before saving
        csv_dir = get_reference_csv_dir()  # Use persistent location, not temp directory
        ppe_path = os.path.join(csv_dir, 'ppe_doctors.csv')
        
        if is_file_in_use(ppe_path):
            return False, "⚠️ Cannot update: File is currently in use by another process. Please close any applications using ppe_doctors.csv and try again."
        
        # Save to CSV
        try:
            result_csv.to_csv(ppe_path, index=False)
            return True, f"Successfully downloaded {len(result_csv)} records from server using stored procedure '{sproc}'."
        except PermissionError:
            return False, "⚠️ Cannot update: File is locked or permission denied. Please close any applications using ppe_doctors.csv and try again."
        except Exception as e:
            return False, f"⚠️ Error saving file: {str(e)}"
    except Exception as e:
        error_msg = f"Error connecting to server or executing stored procedure: {str(e)}"
        logger.error(f"download_ppe_doctors_from_server error: {error_msg}\n{traceback.format_exc()}")
        return False, error_msg

def download_md_code_list_from_server():
    """Download RX MD Code List from SQL Server and store in session state."""
    try:
        # Get credentials from helper (supports st.secrets override)
        creds = get_db_credentials("rxtracking")
        password_encoded = urllib.parse.quote_plus(creds["password"])
        
        # Detect OS and use appropriate driver
        system = platform.system().lower()
        if system == 'windows':
            driver = "SQL Server"
        else:
            driver = "ODBC Driver 18 for SQL Server"
        
        # Create connection string with OS-appropriate driver
        driver_encoded = urllib.parse.quote_plus(driver)
        conn_str = f"mssql+pyodbc://{creds['user']}:{password_encoded}@{creds['host']}/{creds['database']}?driver={driver_encoded}"
        
        # Additional connection parameters for Linux/Ubuntu
        if system != 'windows':
            conn_str += "&TrustServerCertificate=yes"
        
        engine = create_engine(conn_str)
        
        # Query md_code_list table
        sql_query = """SELECT [DOCTOR_CODE], [CUSTOMER_CODE], [CUSTOMER_NAME]
                       FROM [RXTracking].[dbo].[rx_md_code_list]"""
        
        # Connect to database and read data into dataframe
        with engine.connect() as connection:
            result_df = pd.read_sql(sql_query, connection)
        
        # Dispose engine
        engine.dispose()
        
        # Store in session state
        st.session_state.md_code_list = result_df
        
        return True, result_df
    except Exception:
        # Return empty dataframe on error (silently handle - no notifications needed)
        st.session_state.md_code_list = pd.DataFrame(columns=['DOCTOR_CODE', 'CUSTOMER_CODE', 'CUSTOMER_NAME'])
        return False, pd.DataFrame(columns=['DOCTOR_CODE', 'CUSTOMER_CODE', 'CUSTOMER_NAME'])

def _get_innogen_engine():
    """Create SQLAlchemy engine for Innogen database connection."""
    try:
        # Get credentials from helper (supports st.secrets override)
        creds = get_db_credentials("innogen")
        password_encoded = urllib.parse.quote_plus(creds["password"])
        
        # Detect OS and use appropriate driver
        system = platform.system().lower()
        if system == 'windows':
            driver = "SQL Server"
        else:
            driver = "ODBC Driver 18 for SQL Server"
        
        # Create connection string
        driver_encoded = urllib.parse.quote_plus(driver)
        conn_str = f"mssql+pyodbc://{creds['user']}:{password_encoded}@{creds['host']}/{creds['database']}?driver={driver_encoded}"
        
        # Additional connection parameters for Linux/Ubuntu
        if system != 'windows':
            conn_str += "&TrustServerCertificate=yes"
        
        engine = create_engine(conn_str)
        return engine
    except Exception as e:
        logger.error(f"Error creating Innogen engine: {str(e)}")
        raise

def export_table_item_from_innogen():
    """
    Connect to Innogen SQL Server and export Item table data to table_item.csv.
    Filters for items where [No_] contains 'IP' or 'AIP'.
    
    Returns:
        (success: bool, message: str, record_count: int)
    """
    try:
        engine = _get_innogen_engine()
        
        # Table name and fully-qualified table name
        table_name = 'Item'
        table_fqn = f'"Innogen Pharmaceuticals, Inc_${table_name}$437dbf0e-84ff-417a-965d-ed2bb9650972"'
        
        # Query with condition: [No_] contains 'IP' or 'AIP'
        sql = f"""
            SELECT [No_], [Standard Cost]
            FROM {table_fqn}
            WHERE [No_] LIKE '%IP%' OR [No_] LIKE '%AIP%'
        """
        
        # Execute query
        with engine.connect() as connection:
            result_df = pd.read_sql(sql, connection)
        
        # Dispose engine
        engine.dispose()
        
        if result_df.empty:
            return False, "No records found matching the criteria (No_ contains 'IP' or 'AIP')", 0
        
        # Save to CSV
        csv_dir = get_csv_dir()
        csv_path = os.path.join(csv_dir, 'table_item.csv')
        result_df.to_csv(csv_path, index=False)
        
        record_count = len(result_df)
        return True, f"Successfully exported {record_count} records to table_item.csv", record_count
        
    except Exception as e:
        error_msg = f"Error connecting to Innogen database: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return False, error_msg, 0

def merge_standard_cost_to_cross_ref():
    """
    Merge Standard Cost from table_item.csv to rx_item_cross_ref.csv.
    Matches [No_] from table_item.csv with [Item No.] from rx_item_cross_ref.csv.
    
    Returns:
        (success: bool, message: str, updated_count: int)
    """
    try:
        table_item_path = os.path.join(get_csv_dir(), 'table_item.csv')
        cross_ref_path = get_item_cross_ref_path()
        
        # Check if table_item.csv exists
        if not os.path.exists(table_item_path):
            return False, "table_item.csv not found. Please export from database first.", 0
        
        # Check if rx_item_cross_ref.csv exists
        if not os.path.exists(cross_ref_path):
            return False, "rx_item_cross_ref.csv not found.", 0
        
        # Read both CSV files
        table_item_df = pd.read_csv(table_item_path)
        cross_ref_df = pd.read_csv(cross_ref_path)
        
        # Check required columns
        if 'No_' not in table_item_df.columns or 'Standard Cost' not in table_item_df.columns:
            return False, "table_item.csv missing required columns: [No_] or [Standard Cost]", 0
        
        if 'Item No.' not in cross_ref_df.columns:
            return False, "rx_item_cross_ref.csv missing required column: [Item No.]", 0
        
        # Normalize Item No. for matching (uppercase, stripped)
        cross_ref_df['Item No._merge'] = cross_ref_df['Item No.'].astype(str).str.strip().str.upper()
        table_item_df['No._merge'] = table_item_df['No_'].astype(str).str.strip().str.upper()
        
        # Drop existing Standard Cost column if it exists
        if 'Standard Cost' in cross_ref_df.columns:
            cross_ref_df = cross_ref_df.drop(columns=['Standard Cost'])
        
        # Merge Standard Cost
        merge_df = table_item_df[['No._merge', 'Standard Cost']].copy()
        updated_df = cross_ref_df.merge(
            merge_df,
            left_on='Item No._merge',
            right_on='No._merge',
            how='left'
        )
        
        # Drop temporary merge columns
        updated_df = updated_df.drop(columns=['Item No._merge', 'No._merge'], errors='ignore')
        
        # Ensure Standard Cost is numeric
        updated_df['Standard Cost'] = pd.to_numeric(updated_df['Standard Cost'], errors='coerce')
        
        # Save to CSV
        updated_df.to_csv(cross_ref_path, index=False, na_rep='')
        
        # Count how many items got Standard Cost values
        updated_count = updated_df['Standard Cost'].notna().sum()
        
        # Clear cache to reload
        load_cross_reference_csv.clear()
        
        return True, f"Successfully merged Standard Cost. {updated_count} out of {len(updated_df)} items have Standard Cost values.", updated_count
        
    except Exception as e:
        error_msg = f"Error merging Standard Cost: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return False, error_msg, 0

@st.cache_data
def load_cross_reference_csv():
    """Load cross-reference CSV file (rx_item_cross_ref.csv) for Item Code cross-referencing.
    This file is separate from rx_md_masterlist.csv which has a different purpose."""
    try:
        # Cross-reference data is a persistent reference file, not a temp export.
        csv_dir = get_reference_csv_dir()
        # Use rx_item_cross_ref.csv for cross-referencing (separate from masterlist)
        csv_path = os.path.join(csv_dir, 'rx_item_cross_ref.csv')
        
        if os.path.exists(csv_path):
            cross_ref_df = pd.read_csv(csv_path)
            # Rename columns for merging
            cross_ref_df = cross_ref_df.rename(columns={
                'Cross-Reference No.': 'Item Code',
                'Item No.': 'InnoGen Item Code',
                'Description': 'InnoGen Item Name'
            })
            # Filter out rows with empty Item Code
            cross_ref_df = cross_ref_df[cross_ref_df['Item Code'].notna()].copy()
            cross_ref_df = cross_ref_df[cross_ref_df['Item Code'].astype(str).str.strip() != ''].copy()
            
            # Clean Item Code - extract numeric part only for matching
            def clean_item_code(value):
                """Extract numeric Item Code for matching."""
                if pd.isna(value) or value == '':
                    return ''
                value_str = str(value).strip()
                # Extract first numeric sequence (digits only)
                match = re.search(r'(\d+)', value_str)
                if match:
                    return match.group(1)
                return value_str
            
            cross_ref_df['Item Code'] = cross_ref_df['Item Code'].apply(clean_item_code)
            # Filter out empty Item Codes after cleaning
            cross_ref_df = cross_ref_df[cross_ref_df['Item Code'] != ''].copy()
            
            # Select only the columns we need (include VAT Product Posting Group, Standard Cost, and DIVISION if they exist)
            columns_to_select = ['Item Code', 'InnoGen Item Code', 'InnoGen Item Name']
            # Check if VAT Product Posting Group column exists and add it
            vat_posting_col = None
            for col in cross_ref_df.columns:
                if 'VAT' in col.upper() and 'POSTING' in col.upper() and 'GROUP' in col.upper():
                    vat_posting_col = col
                    columns_to_select.append(col)
                    break
            
            # Check if Standard Cost column exists and add it
            if 'Standard Cost' in cross_ref_df.columns:
                columns_to_select.append('Standard Cost')
            
            # Check if DIVISION column exists and add it
            if 'DIVISION' in cross_ref_df.columns:
                columns_to_select.append('DIVISION')
            
            # Only select columns that exist
            available_columns = [col for col in columns_to_select if col in cross_ref_df.columns]
            cross_ref_df = cross_ref_df[available_columns].copy()
            
            # Rename VAT Product Posting Group column if it exists (normalize the name)
            if vat_posting_col and vat_posting_col in cross_ref_df.columns:
                cross_ref_df = cross_ref_df.rename(columns={vat_posting_col: 'VAT Product Posting Group'})
            
            return cross_ref_df
        else:
            return None
    except Exception as e:
        st.warning(f"Could not load cross-reference CSV: {str(e)}")
        return None

@st.cache_data
def load_masterlist_csv():
    """Load RX MD Masterlist CSV file."""
    try:
        csv_dir = get_csv_dir()
        csv_path = os.path.join(csv_dir, 'rx_md_masterlist.csv')
        
        if os.path.exists(csv_path):
            masterlist_df = pd.read_csv(csv_path)
            return masterlist_df
        else:
            return None
    except Exception as e:
        st.warning(f"Could not load masterlist CSV: {str(e)}")
        return None

@st.cache_data
def load_doctors_reference_csv():
    """Load doctors_reference.csv file for Quick Suggest Matching."""
    try:
        csv_dir = get_reference_csv_dir()  # Use persistent location, not temp directory
        csv_path = os.path.join(csv_dir, 'doctors_reference.csv')
        
        if os.path.exists(csv_path):
            reference_df = pd.read_csv(csv_path)
            # Ensure required columns exist
            if all(col in reference_df.columns for col in ['Doctor Name', 'Address', 'PTR']):
                return reference_df
            else:
                return None
        else:
            return None
    except Exception as e:
        logger.warning(f"Could not load doctors_reference.csv: {str(e)}")
        return None

@st.cache_data
def load_ppe_doctors_csv():
    """Load PPE Doctors CSV file for additional doctor name matching."""
    try:
        csv_dir = get_reference_csv_dir()  # Use persistent location, not temp directory
        csv_path = os.path.join(csv_dir, 'ppe_doctors.csv')
        
        if os.path.exists(csv_path):
            ppe_df = pd.read_csv(csv_path)
            # Ensure required columns exist
            required_columns = ['md_official_name', 'DOCTOR_CODE', 'CUSTOMER_CODE']
            if all(col in ppe_df.columns for col in required_columns):
                # Clean and normalize md_official_name for matching
                if 'md_official_name_clean' not in ppe_df.columns:
                    ppe_df['md_official_name_clean'] = ppe_df['md_official_name'].astype(str).str.strip().str.lower()
                return ppe_df
            else:
                st.warning("ppe_doctors.csv missing required columns. Expected: md_official_name, DOCTOR_CODE, CUSTOMER_CODE")
                return None
        else:
            return None
    except Exception as e:
        st.warning(f"Could not load ppe_doctors CSV: {str(e)}")
        return None

def format_time_hhmmss(elapsed_seconds):
    """Format elapsed time as hh:mm:ss."""
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = int(elapsed_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def generate_filename_from_trans_date(df, prefix='Summary_RXTracking_Report', extension='csv'):
    """Generate filename with month and year from Trans Date column."""
    extension = extension.lstrip('.')
    try:
        if 'Trans Date' in df.columns and not df.empty:
            # Get the first non-null trans_date value
            trans_dates = df['Trans Date'].dropna()
            if not trans_dates.empty:
                # Try to parse the date
                first_date = trans_dates.iloc[0]
                if isinstance(first_date, str):
                    # Try common date formats
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                        try:
                            date_obj = pd.to_datetime(first_date, format=fmt)
                            break
                        except (ValueError, TypeError):
                            continue
                    else:
                        # Try pandas automatic parsing
                        date_obj = pd.to_datetime(first_date, errors='coerce')
                else:
                    date_obj = pd.to_datetime(first_date, errors='coerce')
                
                if pd.notna(date_obj):
                    month_name = date_obj.strftime('%B')  # Full month name (e.g., "July")
                    year = date_obj.strftime('%Y')  # Year (e.g., "2025")
                    return f"{prefix}_{month_name}_{year}.{extension}"
    except Exception:
        pass
    
    # Fallback to default filename
    return f"{prefix}.{extension}"


def prepare_matched_report_export_df(matched_df):
    """
    Prepare matched data for final CSV/Excel report export only.
    Upload PTR (PTR No) -> mdc_ptr_no in the first column slot.
    Masterlist/reference PTR (md_ptrs) -> master_ptr_no in the last column slot.
    """
    download_df = matched_df.copy()

    if 'Trans Date' in download_df.columns:
        def split_trans_date(date_value):
            if pd.isna(date_value) or date_value == '':
                return pd.Series({'YEAR': '', 'MONTH': '', 'DAYS': ''})
            try:
                date_obj = pd.to_datetime(date_value, errors='coerce')
                if pd.isna(date_obj):
                    return pd.Series({'YEAR': '', 'MONTH': '', 'DAYS': ''})
                return pd.Series({
                    'YEAR': str(date_obj.year),
                    'MONTH': str(date_obj.month).zfill(2),
                    'DAYS': str(date_obj.day).zfill(2),
                })
            except Exception:
                return pd.Series({'YEAR': '', 'MONTH': '', 'DAYS': ''})

        date_split = download_df['Trans Date'].apply(split_trans_date)
        download_df['YEAR'] = date_split['YEAR']
        download_df['MONTH'] = date_split['MONTH']
        download_df['DAYS'] = date_split['DAYS']

    rename_dict = {
        'PTR No': 'mdc_ptr_no',
        'md_ptrs': 'master_ptr_no',
        'suggested_md': 'suggest_dn',
        'md_official_name': 'MD NAME FINAL',
    }
    download_df = download_df.rename(columns={k: v for k, v in rename_dict.items() if k in download_df.columns})

    export_column_order = [
        'mdc_ptr_no', 'suggest_dn', 'CUSTOMER_CODE', 'MD NAME FINAL', 'suggested_name', 'quick_suggest_name',
        'Doctor Name', 'Branch Code', 'Branch Name', 'YEAR', 'MONTH', 'DAYS', 'Address1', 'Address2',
        'Supplier Code', 'Supplier Name', "Vendor's Name",
        'Item Code', 'InnoGen Item Code', 'Item Name', 'InnoGen Item Name', 'Standard Cost', 'Qty', 'Amount',
        'OSCA DISC', 'DIVISION', 'VAT Product Posting Group', 'PERIOD',
        'DOCTOR_CODE', 'file_loc',
        'master_ptr_no',
    ]
    export_reserved_names = {'PTR No', 'md_ptrs', 'PTR FINAL', 'mdc_ptr_no', 'master_ptr_no'}

    download_columns = [col for col in export_column_order if col in download_df.columns]
    remaining_cols = [
        col for col in download_df.columns
        if col not in download_columns and col not in export_reserved_names
    ]
    if 'master_ptr_no' in download_columns:
        master_ptr_idx = download_columns.index('master_ptr_no')
        download_columns = download_columns[:master_ptr_idx] + remaining_cols + download_columns[master_ptr_idx:]
    else:
        download_columns.extend(remaining_cols)
        if 'master_ptr_no' in download_df.columns:
            download_columns.append('master_ptr_no')

    download_df = download_df[download_columns]

    sort_columns = []
    sort_ascending = []
    if 'suggest_dn' in download_df.columns:
        sort_columns.append('suggest_dn')
        sort_ascending.append(False)
    elif 'suggested_md' in download_df.columns:
        sort_columns.append('suggested_md')
        sort_ascending.append(False)

    if all(col in download_df.columns for col in ['YEAR', 'MONTH', 'DAYS']):
        sort_columns.extend(['YEAR', 'MONTH', 'DAYS'])
        sort_ascending.extend([True, True, True])
    elif 'Trans Date' in download_df.columns:
        try:
            download_df['_temp_sort_date'] = pd.to_datetime(download_df['Trans Date'], errors='coerce')
            sort_columns.append('_temp_sort_date')
            sort_ascending.append(True)
        except Exception:
            sort_columns.append('Trans Date')
            sort_ascending.append(True)

    if sort_columns:
        try:
            download_df = download_df.sort_values(sort_columns, ascending=sort_ascending, na_position='last')
            if '_temp_sort_date' in download_df.columns:
                download_df = download_df.drop(columns=['_temp_sort_date'])
        except Exception:
            pass

    return download_df


def sanitize_matched_report_export_columns(df):
    """Normalize column headers for CSV/Excel export compatibility."""
    df_out = df.copy()
    df_out.columns = [col.replace(' ', '_').replace("'", '') for col in df_out.columns]
    return df_out

def calculate_similarity(str1, str2):
    """Calculate similarity ratio between two strings using SequenceMatcher or rapidfuzz if available."""
    if pd.isna(str1) or pd.isna(str2) or str1 == '' or str2 == '':
        return 0.0
    s1 = str(str1).lower().strip()
    s2 = str(str2).lower().strip()
    
    # Use rapidfuzz if available (much faster)
    if RAPIDFUZZ_AVAILABLE:
        from rapidfuzz import fuzz
        return fuzz.ratio(s1, s2) / 100.0  # Convert 0-100 to 0-1 range
    else:
        return SequenceMatcher(None, s1, s2).ratio()

def fast_similarity_estimate(str1, str2):
    """Fast similarity estimate using simple heuristics (much faster than SequenceMatcher)."""
    if pd.isna(str1) or pd.isna(str2) or str1 == '' or str2 == '':
        return 0.0
    
    s1 = str(str1).lower().strip()
    s2 = str(str2).lower().strip()
    
    # Exact match
    if s1 == s2:
        return 1.0
    
    # Length difference penalty
    len_diff = abs(len(s1) - len(s2))
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    length_penalty = 1.0 - (len_diff / max_len) * 0.3
    
    # Common words/characters
    words1 = set(s1.split())
    words2 = set(s2.split())
    if words1 and words2:
        common_words = len(words1.intersection(words2))
        total_words = len(words1.union(words2))
        word_similarity = common_words / total_words if total_words > 0 else 0.0
    else:
        # Character-based if no words
        chars1 = set(s1.replace(' ', ''))
        chars2 = set(s2.replace(' ', ''))
        if chars1 and chars2:
            common_chars = len(chars1.intersection(chars2))
            total_chars = len(chars1.union(chars2))
            word_similarity = common_chars / total_chars if total_chars > 0 else 0.0
        else:
            word_similarity = 0.0
    
    # Combined estimate (faster but less accurate)
    return min(1.0, length_penalty * word_similarity * 1.2)

def match_doctor_name(doctor_name, ptr_no, branch_code, item_code, masterlist_df, 
                      use_branch_filter=True, use_product_filter=True, use_name_matching=True,
                      use_ptr_matching=True, similarity_threshold=0.7, ptr_threshold=0.4):
    """
    Match doctor name with masterlist using multiple criteria:
    1. Fuzzy matching on doctor name (optional)
    2. PTR No matching (always enabled, exact match with high score)
    3. Branch code filtering (optional)
    4. Product code filtering (optional)
    5. PTR No matching (optional)
    
    Parameters:
    - use_branch_filter: Filter by branch code if True
    - use_product_filter: Filter by product code if True
    - use_name_matching: Use fuzzy name matching if True
    - use_ptr_matching: Use PTR No matching if True
    - similarity_threshold: Minimum similarity score for name matching (default 0.7)
    - ptr_threshold: Minimum similarity score when PTR matches (default 0.4)
    """
    if masterlist_df is None or masterlist_df.empty:
        return False, '', ''
    
    if pd.isna(doctor_name) or str(doctor_name).strip() == '':
        return False, '', ''
    
    doctor_name_clean = str(doctor_name).strip()
    ptr_no_clean = str(ptr_no).strip() if pd.notna(ptr_no) else ''
    
    # Filter by branch code if available and enabled
    filtered_df = masterlist_df.copy()
    if use_branch_filter and pd.notna(branch_code) and str(branch_code).strip() != '':
        branch_code_clean = str(branch_code).strip()
        # Extract numeric part of branch code
        branch_match = re.search(r'(\d+)', branch_code_clean)
        if branch_match:
            branch_code_numeric = branch_match.group(1)
            # Check if md_b_codes contains the branch code (handles comma-separated values)
            if 'md_b_codes' in filtered_df.columns:
                # Create a function to check if branch code exists in md_b_codes
                def branch_code_match(md_b_codes_value):
                    if pd.isna(md_b_codes_value):
                        return False
                    md_b_codes_str = str(md_b_codes_value).strip()
                    # Check if branch code appears as whole word (handles comma-separated)
                    return bool(re.search(r'\b' + re.escape(branch_code_numeric) + r'\b', md_b_codes_str))
                
                filtered_df = filtered_df[filtered_df['md_b_codes'].apply(branch_code_match)]
    
    # Filter by product code if available and enabled
    if use_product_filter and pd.notna(item_code) and str(item_code).strip() != '':
        item_code_clean = str(item_code).strip()
        # Extract numeric part for matching
        match = re.search(r'(\d+)', item_code_clean)
        if match:
            item_code_numeric = match.group(1)
            # Check if md_p_codes contains the item code (handles comma-separated values)
            if 'md_p_codes' in filtered_df.columns:
                # Create a function to check if product code exists in md_p_codes
                def product_code_match(md_p_codes_value):
                    if pd.isna(md_p_codes_value):
                        return False
                    md_p_codes_str = str(md_p_codes_value).strip()
                    # Check if product code appears as whole word (handles comma-separated)
                    return bool(re.search(r'\b' + re.escape(item_code_numeric) + r'\b', md_p_codes_str))
                
                filtered_df = filtered_df[filtered_df['md_p_codes'].apply(product_code_match)]
    
    if filtered_df.empty:
        return False, '', ''
    
    # Calculate similarity scores for all rows
    if 'md_official_name' not in filtered_df.columns:
        return False, '', ''
    
    filtered_df = filtered_df.copy()
    if use_name_matching:
        # Calculate similarity scores against both md_official_name and md_suggest
        # md_suggest contains common encoded names from stores, making matching easier
        def calculate_best_similarity(row):
            similarities = []
            # Compare with official name
            if 'md_official_name' in row and pd.notna(row['md_official_name']):
                official_sim = calculate_similarity(doctor_name_clean, str(row['md_official_name']).strip())
                similarities.append(official_sim)
            # Compare with suggested names (common encoded variations)
            if 'md_suggest' in row and pd.notna(row['md_suggest']):
                suggest_sim = calculate_similarity(doctor_name_clean, str(row['md_suggest']).strip())
                similarities.append(suggest_sim)
            # Return the best (highest) similarity score
            return max(similarities) if similarities else 0.0
        
        filtered_df['similarity'] = filtered_df.apply(calculate_best_similarity, axis=1)
    else:
        # If name matching disabled, set all similarities to 0.0
        # This ensures only PTR matches will succeed (they boost score to >= 0.4)
        filtered_df['similarity'] = 0.0
    
    # Filter by PTR No if available and enabled - PTR No must be EXACT match (equal)
    if use_ptr_matching and ptr_no_clean != '' and 'md_ptrs' in filtered_df.columns:
        # Exact PTR No match - strip and compare exactly (must be equal)
        ptr_matches = filtered_df[
            filtered_df['md_ptrs'].astype(str).str.strip() == ptr_no_clean
        ]
        if not ptr_matches.empty:
            # If PTR matches exactly (equal), calculate similarity and give very high score
            ptr_matches = ptr_matches.copy()
            if use_name_matching:
                # Calculate similarity against both md_official_name and md_suggest
                def calculate_best_similarity_ptr(row):
                    similarities = []
                    # Compare with official name
                    if 'md_official_name' in row and pd.notna(row['md_official_name']):
                        official_sim = calculate_similarity(doctor_name_clean, str(row['md_official_name']).strip())
                        similarities.append(official_sim)
                    # Compare with suggested names (common encoded variations)
                    if 'md_suggest' in row and pd.notna(row['md_suggest']):
                        suggest_sim = calculate_similarity(doctor_name_clean, str(row['md_suggest']).strip())
                        similarities.append(suggest_sim)
                    # Return the best (highest) similarity score
                    return max(similarities) if similarities else 0.0
                
                ptr_matches['similarity'] = ptr_matches.apply(calculate_best_similarity_ptr, axis=1)
            else:
                # If name matching disabled, start with 0.0
                ptr_matches['similarity'] = 0.0
            
            # Boost similarity score significantly for exact PTR matches (add 0.4, cap at 1.0)
            # This makes PTR matches have very high priority
            ptr_matches['similarity'] = ptr_matches['similarity'].apply(
                lambda x: min(1.0, x + 0.4)
            )
            best_match = ptr_matches.loc[ptr_matches['similarity'].idxmax()]
            # Very low threshold for PTR matches since PTR is exact match (equal)
            # Even if name similarity is low, PTR match gives high confidence
            if best_match['similarity'] >= ptr_threshold:
                return True, best_match['md_official_name'], best_match.get('md_ptrs', '')
    
    # Find best match by similarity (when PTR doesn't match or not available)
    # Only proceed if name matching is enabled
    if use_name_matching:
        best_match = filtered_df.loc[filtered_df['similarity'].idxmax()]
        
        # Set threshold for matching (configurable similarity_threshold)
        if best_match['similarity'] >= similarity_threshold:
            md_ptrs = best_match.get('md_ptrs', '') if 'md_ptrs' in best_match.index else ''
            return True, best_match['md_official_name'], md_ptrs
    
    # If name matching is disabled and no PTR match, return False
    return False, '', ''

def match_doctor_name_optimized(doctor_name, ptr_no, branch_code, item_code, masterlist_df, 
                                 use_branch_filter=True, use_product_filter=True, use_name_matching=True,
                                 use_ptr_matching=True, similarity_threshold=0.7, ptr_threshold=0.4):
    """
    Optimized version of match_doctor_name that uses pre-processed clean columns.
    This function expects masterlist_df to have _clean columns already prepared.
    """
    if masterlist_df is None or masterlist_df.empty:
        return False, '', ''
    
    if pd.isna(doctor_name) or str(doctor_name).strip() == '':
        return False, '', ''
    
    doctor_name_clean = str(doctor_name).strip().lower()
    ptr_no_clean = str(ptr_no).strip() if pd.notna(ptr_no) else ''
    
    # Filter by branch code if available and enabled (use pre-processed column)
    filtered_df = masterlist_df.copy()
    if use_branch_filter and pd.notna(branch_code) and str(branch_code).strip() != '':
        branch_code_clean = str(branch_code).strip()
        branch_match = re.search(r'(\d+)', branch_code_clean)
        if branch_match:
            branch_code_numeric = branch_match.group(1)
            if 'md_b_codes_clean' in filtered_df.columns:
                # Use vectorized string operations for better performance
                filtered_df = filtered_df[
                    filtered_df['md_b_codes_clean'].str.contains(
                        r'\b' + re.escape(branch_code_numeric) + r'\b',
                        na=False, regex=True
                    )
                ]
    
    # Filter by product code if available and enabled (use pre-processed column)
    if use_product_filter and pd.notna(item_code) and str(item_code).strip() != '':
        item_code_clean = str(item_code).strip()
        match = re.search(r'(\d+)', item_code_clean)
        if match:
            item_code_numeric = match.group(1)
            if 'md_p_codes_clean' in filtered_df.columns:
                # Use vectorized string operations for better performance
                filtered_df = filtered_df[
                    filtered_df['md_p_codes_clean'].str.contains(
                        r'\b' + re.escape(item_code_numeric) + r'\b',
                        na=False, regex=True
                    )
                ]
    
    if filtered_df.empty:
        return False, '', ''
    
    # Calculate similarity scores using pre-processed columns
    filtered_df = filtered_df.copy()
    if use_name_matching:
        # Vectorized similarity calculation using pre-processed columns
        similarities_official = pd.Series([0.0] * len(filtered_df))
        similarities_suggest = pd.Series([0.0] * len(filtered_df))
        
        if 'md_official_name_clean' in filtered_df.columns:
            similarities_official = filtered_df['md_official_name_clean'].apply(
                lambda x: calculate_similarity(doctor_name_clean, x)
            )
        
        if 'md_suggest_clean' in filtered_df.columns:
            similarities_suggest = filtered_df['md_suggest_clean'].apply(
                lambda x: calculate_similarity(doctor_name_clean, x)
            )
        
        # Take maximum similarity between official and suggest
        if 'md_official_name_clean' in filtered_df.columns and 'md_suggest_clean' in filtered_df.columns:
            filtered_df['similarity'] = pd.concat([similarities_official, similarities_suggest], axis=1).max(axis=1)
        elif 'md_official_name_clean' in filtered_df.columns:
            filtered_df['similarity'] = similarities_official
        elif 'md_suggest_clean' in filtered_df.columns:
            filtered_df['similarity'] = similarities_suggest
        else:
            filtered_df['similarity'] = 0.0
    else:
        filtered_df['similarity'] = 0.0
    
    # Filter by PTR No if available and enabled (use pre-processed column)
    if use_ptr_matching and ptr_no_clean != '' and 'md_ptrs_clean' in filtered_df.columns:
        ptr_matches = filtered_df[filtered_df['md_ptrs_clean'] == ptr_no_clean]
        if not ptr_matches.empty:
            ptr_matches = ptr_matches.copy()
            if use_name_matching:
                # Recalculate similarity for PTR matches if needed
                if 'md_official_name_clean' in ptr_matches.columns or 'md_suggest_clean' in ptr_matches.columns:
                    sim_official = pd.Series([0.0] * len(ptr_matches))
                    sim_suggest = pd.Series([0.0] * len(ptr_matches))
                    if 'md_official_name_clean' in ptr_matches.columns:
                        sim_official = ptr_matches['md_official_name_clean'].apply(
                            lambda x: calculate_similarity(doctor_name_clean, x)
                        )
                    if 'md_suggest_clean' in ptr_matches.columns:
                        sim_suggest = ptr_matches['md_suggest_clean'].apply(
                            lambda x: calculate_similarity(doctor_name_clean, x)
                        )
                    if 'md_official_name_clean' in ptr_matches.columns and 'md_suggest_clean' in ptr_matches.columns:
                        ptr_matches['similarity'] = pd.concat([sim_official, sim_suggest], axis=1).max(axis=1)
                    elif 'md_official_name_clean' in ptr_matches.columns:
                        ptr_matches['similarity'] = sim_official
                    elif 'md_suggest_clean' in ptr_matches.columns:
                        ptr_matches['similarity'] = sim_suggest
            else:
                ptr_matches['similarity'] = 0.0
            
            # Boost similarity score for PTR matches
            ptr_matches['similarity'] = ptr_matches['similarity'].apply(lambda x: min(1.0, x + 0.4))
            best_match = ptr_matches.loc[ptr_matches['similarity'].idxmax()]
            
            if best_match['similarity'] >= ptr_threshold:
                md_official = best_match.get('md_official_name', '') if 'md_official_name' in best_match.index else ''
                md_ptrs = best_match.get('md_ptrs', '') if 'md_ptrs' in best_match.index else ''
                return True, md_official, md_ptrs
    
    # Find best match by similarity (when PTR doesn't match or not available)
    if use_name_matching:
        best_match = filtered_df.loc[filtered_df['similarity'].idxmax()]
        if best_match['similarity'] >= similarity_threshold:
            md_official = best_match.get('md_official_name', '') if 'md_official_name' in best_match.index else ''
            md_ptrs = best_match.get('md_ptrs', '') if 'md_ptrs' in best_match.index else ''
            return True, md_official, md_ptrs
    
    return False, '', ''


def prepare_masterlist_for_matching(masterlist_df):
    """
    Create a copy of the masterlist for use in all matching steps.
    - [ptr_final]: for filling [PTR Final] on matched rows. Uses [md_ptrs_final] if present (TopMD),
      otherwise original [md_ptrs]. Never overwrites [md_ptrs] in source.
    - [md_ptrs]: in the copy, replaced by [md_ptrs_old] when [md_ptrs_old] is non-null and non-blank (for matching only).
    So matching uses md_ptrs_old-first; filling [PTR Final] uses ptr_final (md_ptrs_final or md_ptrs).
    """
    if masterlist_df is None or masterlist_df.empty:
        return masterlist_df
    df = masterlist_df.copy()
    if 'md_ptrs' in df.columns:
        df['ptr_final'] = df['md_ptrs'].fillna('').astype(str).str.strip()
    else:
        df['ptr_final'] = ''
    if 'md_ptrs_final' in df.columns:
        ptrs_final = df['md_ptrs_final'].fillna('').astype(str).str.strip()
        valid_final = ptrs_final.notna() & (ptrs_final != '') & (ptrs_final.str.lower() != 'nan')
        df.loc[valid_final, 'ptr_final'] = ptrs_final.loc[valid_final]
    if 'md_ptrs_old' in df.columns and 'md_ptrs' in df.columns:
        ptrs_old = df['md_ptrs_old'].astype(str).str.strip()
        valid_old = ptrs_old.notna() & (ptrs_old != '') & (ptrs_old.str.lower() != 'nan')
        df.loc[valid_old, 'md_ptrs'] = df.loc[valid_old, 'md_ptrs_old']
    return df


def group_masterlist_for_matching(masterlist_df, matching_mode='Basic'):
    """
    Group masterlist by key fields to reduce rows and speed up matching.
    
    For Basic mode: Group by [md_official_name], [md_suggest], [md_ptrs], [md_add_1]
    For Advanced mode: Group by [md_official_name], [md_suggest], [md_ptrs], [md_add_1] (can be customized)
    
    Args:
        masterlist_df: The masterlist dataframe to group
        matching_mode: 'Basic' or 'Advanced'
    
    Returns:
        Grouped masterlist dataframe with first occurrence of each unique combination
    """
    if masterlist_df is None or masterlist_df.empty:
        return masterlist_df
    
    try:
        # Make a copy to avoid modifying original
        grouped_df = masterlist_df.copy()
        
        # Filter out rows where md_suggest has length <= 2 (only accept more than 3 characters, so len > 2)
        # This reduces the dataset size and improves matching performance
        if 'md_suggest' in grouped_df.columns:
            # Convert to string, handle NaN values, and filter by length
            grouped_df['md_suggest'] = grouped_df['md_suggest'].fillna('').astype(str).str.strip()
            # Keep only rows where md_suggest has more than 2 characters (len > 2 means 3+ characters)
            grouped_df = grouped_df[grouped_df['md_suggest'].str.len() > 2].copy()
        
        # If dataframe is empty after filtering, return it
        if grouped_df.empty:
            return grouped_df
        
        # Define grouping columns based on mode
        if matching_mode == 'Basic':
            # For Basic: Group by md_official_name, md_suggest, md_ptrs, md_add_1
            group_cols = []
            for col in ['md_official_name', 'md_suggest', 'md_ptrs', 'md_add_1']:
                if col in grouped_df.columns:
                    group_cols.append(col)
        else:  # Advanced
            # For Advanced: Same grouping as Basic (can be customized later with different fields)
            group_cols = []
            for col in ['md_official_name', 'md_suggest', 'md_ptrs', 'md_add_1']:
                if col in grouped_df.columns:
                    group_cols.append(col)
        
        # If no grouping columns found, return original
        if not group_cols:
            return grouped_df
        
        # Fill NaN values with empty string for grouping consistency
        for col in group_cols:
            if col in grouped_df.columns:
                grouped_df[col] = grouped_df[col].fillna('').astype(str).str.strip()
        
        # Group by the specified columns and take the first occurrence
        # This reduces the number of rows while preserving all necessary matching fields
        # Use as_index=False to keep grouped columns as regular columns
        grouped_df = grouped_df.groupby(group_cols, dropna=False, as_index=False).first()
        
        return grouped_df
    except Exception as e:
        logger.warning(f"Error grouping masterlist: {str(e)}. Returning original masterlist.")
        import traceback
        logger.error(traceback.format_exc())
        return masterlist_df

def process_doctor_matching_basic(df, masterlist_df, use_doctor_name=True, use_ptr_no=True,
                                   progress_bar=None, status_text=None):
    """
    Basic matching: Simple exact matching using combined keys.
    Logic: 
    - If both enabled: DOCTOR_NAME (uppercase) + PTR_NO (digits only) = MD_SUGGEST (uppercase) + MD_PTRS (digits only)
    - If only doctor name: DOCTOR_NAME (uppercase) = MD_SUGGEST (uppercase)
    - If only PTR No: PTR_NO (digits only) = MD_PTRS (digits only)

    --- TRUE MATCHING TRACE (suggested_md / suggest_dn = TRUE) ---
    suggested_md (displayed as suggest_dn) is set to TRUE *only* in this function, and *only* when:
      1. The row's combine_key (Doctor Name + optional PTR No, normalized) exactly matches a key
         in the masterlist (built from md_suggest/md_official_name + md_ptrs).
      2. [PTR Final] (md_ptrs) and [MD Name Final] (md_official_name) are then filled from the
         same masterlist row via md_ptrs_mapping and md_official_name_mapping (lines ~1676-1678, 1692-1695).
    So: suggested_md = TRUE <=> [PTR Final] and [MD Name Final] were filled from masterlist by
    exact key match in Basic mode. Other matching (Advanced PTR merge, exact address, TF-IDF,
    split_matching) fills [PTR Final]/[MD Name Final] from masterlist but leaves suggested_md = FALSE.
    """
    
    # CRITICAL: Store original Amount column separately to prevent any modification
    # Amount should NEVER be modified during matching - only matching columns are added
    original_amount = None
    if df is not None and not df.empty and 'Amount' in df.columns:
        # Use unified pipeline to ensure we store clean float64 data
        temp_df_for_amount = df[['Amount']].copy()
        temp_df_for_amount = standardize_amount_column(temp_df_for_amount, 'Amount')
        original_amount = temp_df_for_amount['Amount']
    
    if df is None or df.empty:
        return df
    
    if masterlist_df is None or masterlist_df.empty:
        df['suggested_md'] = False
        df['md_official_name'] = ''
        df['md_ptrs'] = ''
        # CRITICAL: Restore original Amount column (NEVER modified during matching)
        if original_amount is not None and len(original_amount) == len(df):
            df['Amount'] = original_amount.values
        return df
    
    # Pre-process masterlist: create combined keys
    masterlist_df = masterlist_df.copy()
    
    # Filter out rows with empty or very short md_suggest (similar to reference code)
    if 'md_suggest' in masterlist_df.columns:
        masterlist_df = masterlist_df[
            ~((masterlist_df['md_suggest'].isnull()) | 
              (masterlist_df['md_suggest'] == '') | 
              (masterlist_df['md_suggest'].astype(str).str.strip().str.len() <= 2))
        ]
    
    if masterlist_df.empty:
        df['suggested_md'] = False
        # Ensure string columns are object type to avoid FutureWarning
        if 'md_official_name' not in df.columns:
            df['md_official_name'] = ''
        else:
            df['md_official_name'] = df['md_official_name'].astype('object')
            df['md_official_name'] = ''
        if 'md_ptrs' not in df.columns:
            df['md_ptrs'] = ''
        else:
            df['md_ptrs'] = df['md_ptrs'].astype('object')
            df['md_ptrs'] = ''
        # CRITICAL: Restore original Amount column (NEVER modified during matching)
        if original_amount is not None and len(original_amount) == len(df):
            df['Amount'] = original_amount.values
        return df
    
    # Extract digits from md_ptrs and normalize by removing leading zeros
    def extract_digits_normalized(value):
        """Extract digits and normalize by removing leading zeros for flexible matching."""
        if pd.isna(value) or str(value).strip() == '':
            return ''
        digits = ''.join(filter(str.isdigit, str(value))) if any(c.isdigit() for c in str(value)) else ''
        # Remove leading zeros to normalize (e.g., "0111025" becomes "111025")
        # But keep at least one digit if all zeros (e.g., "000" stays as "0")
        if digits:
            normalized = digits.lstrip('0') if digits.lstrip('0') else '0'
            return normalized
        return ''
    
    # Prepare masterlist columns - match against both md_suggest and md_official_name
    if 'md_suggest' in masterlist_df.columns:
        masterlist_df['md_suggest_upper'] = masterlist_df['md_suggest'].astype(str).str.upper().str.strip()
    else:
        masterlist_df['md_suggest_upper'] = ''
    
    if 'md_official_name' in masterlist_df.columns:
        masterlist_df['md_official_name_upper'] = masterlist_df['md_official_name'].astype(str).str.upper().str.strip()
    else:
        masterlist_df['md_official_name_upper'] = ''
    
    if 'md_ptrs' in masterlist_df.columns:
        masterlist_df['md_ptrs_digits'] = masterlist_df['md_ptrs'].apply(extract_digits_normalized)
    else:
        masterlist_df['md_ptrs_digits'] = ''
    
    # Create combined keys for masterlist based on enabled options
    # Create keys for both md_suggest and md_official_name to allow matching against either
    if use_doctor_name and use_ptr_no:
        # Both enabled: combine name + PTR for both md_suggest and md_official_name
        masterlist_df['combine_key_suggest'] = masterlist_df['md_suggest_upper'] + ' ' + masterlist_df['md_ptrs_digits']
        masterlist_df['combine_key_official'] = masterlist_df['md_official_name_upper'] + ' ' + masterlist_df['md_ptrs_digits']
    elif use_doctor_name:
        # Only doctor name: use name only for both md_suggest and md_official_name
        masterlist_df['combine_key_suggest'] = masterlist_df['md_suggest_upper']
        masterlist_df['combine_key_official'] = masterlist_df['md_official_name_upper']
    elif use_ptr_no:
        # Only PTR No: use md_ptrs_digits only (same for both)
        masterlist_df['combine_key_suggest'] = masterlist_df['md_ptrs_digits']
        masterlist_df['combine_key_official'] = masterlist_df['md_ptrs_digits']
    else:
        # Both disabled: return all False
        df['suggested_md'] = False
        df['md_official_name'] = ''
        df['md_ptrs'] = ''
        # CRITICAL: Restore original Amount column (NEVER modified during matching)
        if original_amount is not None and len(original_amount) == len(df):
            df['Amount'] = original_amount.values
        return df
    
    # OPTIMIZED: Create mapping dictionaries using vectorized operations instead of iterrows()
    # First create mappings for md_suggest (priority) - use drop_duplicates to get first match
    masterlist_suggest = masterlist_df[masterlist_df['combine_key_suggest'].astype(str).str.strip() != ''].copy()
    if not masterlist_suggest.empty:
        # Drop duplicates keeping first match (consistent with original logic)
        masterlist_suggest_unique = masterlist_suggest.drop_duplicates(subset=['combine_key_suggest'], keep='first')
        
        # Create mappings using to_dict() - much faster than iterrows()
        md_official_name_mapping_suggest = masterlist_suggest_unique.set_index('combine_key_suggest')['md_official_name'].fillna('').astype(str).to_dict()
        # [PTR Final] from masterlist ptr_final/md_ptrs_final consistently
        _ptr_col = 'ptr_final' if 'ptr_final' in masterlist_suggest_unique.columns else ('md_ptrs_final' if 'md_ptrs_final' in masterlist_suggest_unique.columns else 'md_ptrs')
        md_ptrs_mapping_suggest = masterlist_suggest_unique.set_index('combine_key_suggest')[_ptr_col].fillna('').astype(str).str.strip().to_dict()
        
        # Handle DOCTOR_CODE and CUSTOMER_CODE (fillna with empty string)
        md_doctor_code_mapping_suggest = masterlist_suggest_unique.set_index('combine_key_suggest')['DOCTOR_CODE'].fillna('').astype(str).to_dict()
        md_customer_code_mapping_suggest = masterlist_suggest_unique.set_index('combine_key_suggest')['CUSTOMER_CODE'].fillna('').astype(str).to_dict()
    else:
        md_official_name_mapping_suggest = {}
        md_ptrs_mapping_suggest = {}
        md_doctor_code_mapping_suggest = {}
        md_customer_code_mapping_suggest = {}
    
    # Then create mappings for md_official_name (fallback, only if not already in suggest mapping)
    masterlist_official = masterlist_df[
        (masterlist_df['combine_key_official'].astype(str).str.strip() != '') &
        (~masterlist_df['combine_key_official'].isin(md_official_name_mapping_suggest.keys()))
    ].copy()
    
    if not masterlist_official.empty:
        # Drop duplicates keeping first match
        masterlist_official_unique = masterlist_official.drop_duplicates(subset=['combine_key_official'], keep='first')
        
        # Create mappings using to_dict()
        md_official_name_mapping_official = masterlist_official_unique.set_index('combine_key_official')['md_official_name'].fillna('').astype(str).to_dict()
        _ptr_col = 'ptr_final' if 'ptr_final' in masterlist_official_unique.columns else ('md_ptrs_final' if 'md_ptrs_final' in masterlist_official_unique.columns else 'md_ptrs')
        md_ptrs_mapping_official = masterlist_official_unique.set_index('combine_key_official')[_ptr_col].fillna('').astype(str).str.strip().to_dict()
        md_doctor_code_mapping_official = masterlist_official_unique.set_index('combine_key_official')['DOCTOR_CODE'].fillna('').astype(str).to_dict()
        md_customer_code_mapping_official = masterlist_official_unique.set_index('combine_key_official')['CUSTOMER_CODE'].fillna('').astype(str).to_dict()
    else:
        md_official_name_mapping_official = {}
        md_ptrs_mapping_official = {}
        md_doctor_code_mapping_official = {}
        md_customer_code_mapping_official = {}
    
    # Combine both mappings (suggest takes priority)
    md_official_name_mapping = {**md_official_name_mapping_official, **md_official_name_mapping_suggest}
    md_ptrs_mapping = {**md_ptrs_mapping_official, **md_ptrs_mapping_suggest}
    md_doctor_code_mapping = {**md_doctor_code_mapping_official, **md_doctor_code_mapping_suggest}
    md_customer_code_mapping = {**md_customer_code_mapping_official, **md_customer_code_mapping_suggest}
    
    # Initialize new columns
    df['suggested_md'] = False
    df['md_official_name'] = ''
    df['md_ptrs'] = ''
    # Initialize DOCTOR_CODE and CUSTOMER_CODE columns if they don't exist
    if 'DOCTOR_CODE' not in df.columns:
        df['DOCTOR_CODE'] = ''
    if 'CUSTOMER_CODE' not in df.columns:
        df['CUSTOMER_CODE'] = ''
    
    # OPTIMIZED: Vectorized operations instead of iterrows()
    # Prepare data columns
    doctor_names = df['Doctor Name'].fillna('').astype(str).str.strip()
    ptr_nos = df['PTR No'].fillna('').astype(str).str.strip()
    
    # Extract digits from PTR No and normalize (remove leading zeros) - vectorized
    ptr_no_digits_series = ptr_nos.apply(extract_digits_normalized)
    
    # Create combine key based on enabled options - vectorized
    if use_doctor_name and use_ptr_no:
        # Both enabled: combine both
        doctor_names_upper = doctor_names.str.upper().str.strip()
        combine_keys = doctor_names_upper + ' ' + ptr_no_digits_series
    elif use_doctor_name:
        # Only doctor name: use name only
        combine_keys = doctor_names.str.upper().str.strip()
    elif use_ptr_no:
        # Only PTR No: use PTR digits only
        combine_keys = ptr_no_digits_series
    else:
        # Both disabled (should not reach here, but handle anyway)
        combine_keys = pd.Series([''] * len(df), index=df.index)
    
    # OPTIMIZED: Use vectorized .map() for dictionary lookups (much faster than iterrows)
    # Map combine_keys to md_official_name
    df['md_official_name'] = combine_keys.map(md_official_name_mapping).fillna('')
    df['md_ptrs'] = combine_keys.map(md_ptrs_mapping).fillna('')
    
    # Map DOCTOR_CODE and CUSTOMER_CODE
    doctor_codes_mapped = combine_keys.map(md_doctor_code_mapping).fillna('')
    customer_codes_mapped = combine_keys.map(md_customer_code_mapping).fillna('')
    
    # Set DOCTOR_CODE (only non-empty values)
    df.loc[doctor_codes_mapped != '', 'DOCTOR_CODE'] = doctor_codes_mapped[doctor_codes_mapped != ''].astype(str).str.strip()
    
    # Set CUSTOMER_CODE (only non-empty values)
    df.loc[customer_codes_mapped != '', 'CUSTOMER_CODE'] = customer_codes_mapped[customer_codes_mapped != ''].astype(str).str.strip()
    
    # Set suggested_md = True only for rows where combine_key exists in mapping and is not empty
    # This maintains the exact same logic: if combine_key and combine_key in md_official_name_mapping
    valid_keys_mask = (combine_keys != '') & combine_keys.isin(md_official_name_mapping.keys())
    df.loc[valid_keys_mask, 'suggested_md'] = True
    df.loc[~valid_keys_mask, 'suggested_md'] = False
    
    # Count matches from md_suggest vs md_official_name (for logging/reporting)
    matches_from_suggest = 0
    matches_from_official = 0
    if valid_keys_mask.any():
        # Check which mapping was used for each matched key
        matched_keys = combine_keys[valid_keys_mask]
        for key in matched_keys:
            if key in md_official_name_mapping_suggest:
                matches_from_suggest += 1
            elif key in md_official_name_mapping_official:
                matches_from_official += 1
    
    # Update progress (single update instead of per-row)
    if progress_bar is not None and status_text is not None:
        progress_bar.progress(1.0)
        total_matches = matches_from_suggest + matches_from_official
        match_info = f'Exact matching completed! Total matches: {total_matches}'
        if matches_from_suggest > 0 or matches_from_official > 0:
            match_info += f' (md_suggest: {matches_from_suggest}, md_official_name: {matches_from_official})'
        status_text.text(match_info)
    
    # CRITICAL: Restore original Amount column (NEVER modified during matching)
    # Use the original Amount we stored at the start, not from df
    # This ensures Amount is never corrupted by row iterations or column assignments
    if original_amount is not None and len(original_amount) == len(df):
        df['Amount'] = original_amount.values  # Restore original Amount values
    elif 'Amount' in df.columns:
        # Fallback: if original_amount wasn't stored, ensure Amount is numeric
        df = standardize_amount_column(df, 'Amount')
    
    return df

class DoctorNameMatchRevalidator:
    def __init__(self, min_word_length=3, fuzzy_threshold=0.85, max_len_diff=2):
        self.min_word_length = int(min_word_length)
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.max_len_diff = int(max_len_diff)
        self.leading_num_pattern = re.compile(r"^\d+[:\s]*", re.IGNORECASE)
        self.excluded_words = frozenset(
            {
                "dr", "dra", "sr", "sra", "mr", "mrs", "ms", "miss",
                "md", "do", "phd", "jr", "ii", "iii", "iv",
                "ma", "rn", "lpn", "np", "pa", "dds", "dvm",
            }
        )

    def _normalize_and_extract_words(self, text):
        if pd.isna(text):
            return set()
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            return set()
        cleaned = self.leading_num_pattern.sub("", text).upper()
        words = set()
        for word in re.split(r"[\s,;.]+", cleaned):
            w = word.strip()
            if len(w) >= self.min_word_length and w not in self.excluded_words:
                words.add(w)
        return words

    def _row_match_exact(self, name_final, doctor_name):
        wf = self._normalize_and_extract_words(name_final)
        wd = self._normalize_and_extract_words(doctor_name)
        return bool(wf & wd)

    def _row_match_fuzzy(self, name_final, doctor_name, cache):
        wf = self._normalize_and_extract_words(name_final)
        wd = self._normalize_and_extract_words(doctor_name)
        if not wf or not wd:
            return False
        for a in wf:
            for b in wd:
                if abs(len(a) - len(b)) > self.max_len_diff:
                    continue
                key = (a, b) if a <= b else (b, a)
                cached = cache.get(key)
                if cached is None:
                    if RAPIDFUZZ_AVAILABLE:
                        cached = (fuzz.ratio(a, b) / 100.0) >= self.fuzzy_threshold
                    else:
                        cached = SequenceMatcher(None, a, b).ratio() >= self.fuzzy_threshold
                    cache[key] = cached
                if cached:
                    return True
        return False

    def revalidate_true_matches(self, df, suggested_col="suggested_md", md_name_col="md_official_name", doctor_name_col=None):
        if df is None or df.empty:
            return df, 0
        if suggested_col not in df.columns or md_name_col not in df.columns:
            return df, 0

        if doctor_name_col is None:
            for col in ["Doctor Name", "doctor_name", "DOCTOR_NAME", "Doctor_Name", "doc_name"]:
                if col in df.columns:
                    doctor_name_col = col
                    break
        if not doctor_name_col:
            return df, 0

        is_true = df[suggested_col].astype(str).str.strip().str.lower() == "true"
        if not is_true.any():
            return df, 0

        fuzzy_cache = {}
        reset_count = 0
        for idx in df.index[is_true]:
            md_val = df.at[idx, md_name_col]
            doc_val = df.at[idx, doctor_name_col]
            if self._row_match_exact(md_val, doc_val):
                continue
            if self._row_match_fuzzy(md_val, doc_val, fuzzy_cache):
                continue
            df.at[idx, suggested_col] = False
            reset_count += 1

        return df, reset_count

def process_doctor_matching(df, masterlist_df, use_branch_filter=True, use_product_filter=True, 
                            use_name_matching=True, use_ptr_matching=True,
                            similarity_threshold=0.7, ptr_threshold=0.4,
                            progress_bar=None, status_text=None):
    """Process doctor name matching using BATCH PROCESSING (like SQL JOIN) for maximum speed."""
    
    # Initialize df_work to None - will be set in try block
    df_work: pd.DataFrame = None  # type: ignore
    
    # CRITICAL: Initialize original_amount at function start to prevent NameError in exception handlers
    original_amount = None
    
    def copy_partial_results_to_df(df_target, df_work_var):
        """Helper function to copy partial results from df_work to df if available."""
        if df_work_var is not None and isinstance(df_work_var, pd.DataFrame) and not df_work_var.empty:
            try:
                # Copy columns from df_work to df if they exist
                if 'suggested_md' in df_work_var.columns:
                    df_target['suggested_md'] = df_work_var['suggested_md']
                else:
                    df_target['suggested_md'] = False
                
                if 'md_official_name' in df_work_var.columns:
                    df_target['md_official_name'] = df_work_var['md_official_name']
                else:
                    df_target['md_official_name'] = ''
                
                if 'md_ptrs' in df_work_var.columns:
                    df_target['md_ptrs'] = df_work_var['md_ptrs']
                else:
                    df_target['md_ptrs'] = ''
            except (KeyError, AttributeError):
                df_target['suggested_md'] = False
                df_target['md_official_name'] = ''
                df_target['md_ptrs'] = ''
        else:
            df_target['suggested_md'] = False
            df_target['md_official_name'] = ''
            df_target['md_ptrs'] = ''
    
    try:
        if df is None or df.empty:
            return df
        
        if masterlist_df is None or masterlist_df.empty:
            df['suggested_md'] = False
            df['md_official_name'] = ''
            df['md_ptrs'] = ''
            return df
        
        # Check memory before starting
        if not check_memory_available(500):
            if status_text is not None:
                status_text.text('Warning: Low memory available. Processing may be slower...')
        
        if progress_bar is not None:
            progress_bar.progress(0.1)
        if status_text is not None:
            status_text.text('Preparing data for batch matching...')
        
        # Pre-process masterlist for optimization: normalize strings once
        masterlist_df = masterlist_df.copy()
        if 'md_official_name' in masterlist_df.columns:
            masterlist_df['md_official_name_clean'] = masterlist_df['md_official_name'].astype(str).str.strip().str.lower()
        if 'md_suggest' in masterlist_df.columns:
            masterlist_df['md_suggest_clean'] = masterlist_df['md_suggest'].astype(str).str.strip().str.lower()
        if 'md_ptrs' in masterlist_df.columns:
            masterlist_df['md_ptrs_clean'] = masterlist_df['md_ptrs'].astype(str).str.strip()
        if 'md_b_codes' in masterlist_df.columns:
            masterlist_df['md_b_codes_clean'] = masterlist_df['md_b_codes'].astype(str).str.strip()
        if 'md_p_codes' in masterlist_df.columns:
            masterlist_df['md_p_codes_clean'] = masterlist_df['md_p_codes'].astype(str).str.strip()
        
        # Pre-extract numeric parts from branch and product codes
        def extract_numeric(s):
            if pd.isna(s) or str(s).strip() == '':
                return ''
            match = re.search(r'(\d+)', str(s).strip())
            return match.group(1) if match else ''
        
        if 'md_b_codes_clean' in masterlist_df.columns:
            masterlist_df['md_b_codes_numeric'] = masterlist_df['md_b_codes_clean'].apply(extract_numeric)
        if 'md_p_codes_clean' in masterlist_df.columns:
            masterlist_df['md_p_codes_numeric'] = masterlist_df['md_p_codes_clean'].apply(extract_numeric)
        
        # CRITICAL: Store original Amount column separately to prevent any modification
        # Amount should NEVER be modified during matching - only matching columns are added
        # Note: original_amount was initialized at function start, now we populate it
        if 'Amount' in df.columns:
            # Store original Amount as a separate Series (preserve original values)
            # Use unified pipeline
            temp_df_for_amount = df[['Amount']].copy()
            temp_df_for_amount = standardize_amount_column(temp_df_for_amount, 'Amount')
            original_amount = temp_df_for_amount['Amount']
        
        # Pre-process input dataframe
        df_work = df.copy()
        
        # CRITICAL: Ensure Amount column is numeric and preserved (DO NOT MODIFY)
        # This prevents data corruption from string concatenation or formatting issues
        if 'Amount' in df_work.columns:
            df_work = standardize_amount_column(df_work, 'Amount')
        
        df_work['doctor_name_clean'] = df_work['Doctor Name'].astype(str).str.strip().str.lower()
        df_work['ptr_no_clean'] = df_work['PTR No'].astype(str).str.strip()
        df_work['branch_code_clean'] = df_work['Branch Code'].astype(str).str.strip()
        df_work['item_code_combined'] = (df_work.get('InnoGen Item Code', pd.Series([''] * len(df_work))).fillna('').astype(str) + 
                                         df_work.get('Item Code', pd.Series([''] * len(df_work))).fillna('').astype(str))
        df_work['branch_code_numeric'] = df_work['branch_code_clean'].apply(extract_numeric)
        df_work['item_code_numeric'] = df_work['item_code_combined'].apply(extract_numeric)
        
        # Initialize result columns
        df_work['suggested_md'] = False
        df_work['md_official_name'] = ''
        df_work['md_ptrs'] = ''
        df_work['match_similarity'] = 0.0
        
        if progress_bar is not None:
            progress_bar.progress(0.2)
        
        # BATCH PTR MATCHING (like SQL JOIN) - Fastest path
        if use_ptr_matching and 'md_ptrs_clean' in masterlist_df.columns:
            if status_text is not None:
                status_text.text('Batch matching by PTR No...')
            
            # Create PTR matching dataframe (like SQL JOIN on PTR)
            ptr_cols = ['md_ptrs_clean', 'md_official_name', 'md_ptrs',
                        'md_official_name_clean', 'md_suggest_clean']
            if 'ptr_final' in masterlist_df.columns:
                ptr_cols.append('ptr_final')
            if 'md_ptrs_final' in masterlist_df.columns:
                ptr_cols.append('md_ptrs_final')
            # Include DOCTOR_CODE and CUSTOMER_CODE if available in masterlist
            if 'DOCTOR_CODE' in masterlist_df.columns:
                ptr_cols.append('DOCTOR_CODE')
            if 'CUSTOMER_CODE' in masterlist_df.columns:
                ptr_cols.append('CUSTOMER_CODE')
            if use_branch_filter and 'md_b_codes_numeric' in masterlist_df.columns:
                ptr_cols.append('md_b_codes_numeric')
            if use_product_filter and 'md_p_codes_numeric' in masterlist_df.columns:
                ptr_cols.append('md_p_codes_numeric')
            
            ptr_master = masterlist_df[ptr_cols].copy()
            ptr_master = ptr_master[ptr_master['md_ptrs_clean'] != '']
            
            # Validation: frequency of (PTR, MD NAME FINAL) in masterlist - when same PTR has multiple md_official_name, prefer the pair that appears most often in masterlist
            ptr_md_freq = masterlist_df.groupby(['md_ptrs_clean', 'md_official_name'], dropna=False).size().reset_index(name='ptr_md_frequency')
            ptr_master = ptr_master.merge(ptr_md_freq, on=['md_ptrs_clean', 'md_official_name'], how='left')
            ptr_master['ptr_md_frequency'] = ptr_master['ptr_md_frequency'].fillna(0).astype(int)
            
            # Reset index to preserve original indices after merge
            df_work_reset = df_work.reset_index()
            original_index_name = df_work.index.name if df_work.index.name else 'original_index'
            if 'index' in df_work_reset.columns:
                df_work_reset = df_work_reset.rename(columns={'index': original_index_name})
            else:
                df_work_reset[original_index_name] = df_work.index.values
            
            # Merge on PTR (like SQL JOIN) - optimize memory by selecting only needed columns
            try:
                ptr_merge = df_work_reset.merge(
                    ptr_master,
                    left_on='ptr_no_clean',
                    right_on='md_ptrs_clean',
                    how='left',
                    suffixes=('', '_master')
                )
                
                # Cleanup intermediate dataframes
                del df_work_reset, ptr_master
                cleanup_memory()
                
                # Filter by branch/product if enabled (whole word match)
                if use_branch_filter:
                    def branch_match(row):
                        if pd.isna(row['branch_code_numeric']) or row['branch_code_numeric'] == '':
                            return True
                        if pd.isna(row['md_b_codes_numeric']) or row['md_b_codes_numeric'] == '':
                            return False
                        return bool(re.search(r'\b' + re.escape(str(row['branch_code_numeric'])) + r'\b', str(row['md_b_codes_numeric'])))
                    ptr_merge = ptr_merge[ptr_merge.apply(branch_match, axis=1)]
                    cleanup_memory()
                
                if use_product_filter:
                    def product_match(row):
                        if pd.isna(row['item_code_numeric']) or row['item_code_numeric'] == '':
                            return True
                        if pd.isna(row['md_p_codes_numeric']) or row['md_p_codes_numeric'] == '':
                            return False
                        return bool(re.search(r'\b' + re.escape(str(row['item_code_numeric'])) + r'\b', str(row['md_p_codes_numeric'])))
                    ptr_merge = ptr_merge[ptr_merge.apply(product_match, axis=1)]
                    cleanup_memory()
            except MemoryError:
                if status_text is not None:
                    status_text.text('Memory error during PTR matching. Skipping...')
                ptr_merge = pd.DataFrame()
            
            # Calculate similarity for PTR matches (batch)
            if not ptr_merge.empty:
                if use_name_matching:
                    # OPTIMIZED: Vectorized similarity calculation using rapidfuzz when available
                    if RAPIDFUZZ_AVAILABLE:
                        from rapidfuzz import fuzz
                        # Vectorized calculation: apply rapidfuzz to entire columns at once
                        doctor_names = ptr_merge['doctor_name_clean'].fillna('').astype(str)
                        official_names = ptr_merge['md_official_name_clean'].fillna('').astype(str)
                        suggest_names = ptr_merge['md_suggest_clean'].fillna('').astype(str)
                        
                        # Calculate similarities using list comprehension (faster than apply)
                        sim_official = pd.Series([
                            fuzz.ratio(doc, off) / 100.0 if doc and off else 0.0
                            for doc, off in zip(doctor_names, official_names)
                        ], index=ptr_merge.index)
                        
                        sim_suggest = pd.Series([
                            fuzz.ratio(doc, sug) / 100.0 if doc and sug else 0.0
                            for doc, sug in zip(doctor_names, suggest_names)
                        ], index=ptr_merge.index)
                        
                        ptr_merge['match_similarity'] = pd.concat([sim_official, sim_suggest], axis=1).max(axis=1) + 0.4
                        ptr_merge['match_similarity'] = ptr_merge['match_similarity'].clip(upper=1.0)
                        del sim_official, sim_suggest, doctor_names, official_names, suggest_names
                    else:
                        # Fallback to apply() if rapidfuzz not available
                        sim_official = ptr_merge.apply(
                            lambda row: calculate_similarity(row['doctor_name_clean'], row['md_official_name_clean']) 
                            if pd.notna(row['md_official_name_clean']) else 0.0, axis=1
                        )
                        sim_suggest = ptr_merge.apply(
                            lambda row: calculate_similarity(row['doctor_name_clean'], row['md_suggest_clean']) 
                            if pd.notna(row['md_suggest_clean']) else 0.0, axis=1
                        )
                        ptr_merge['match_similarity'] = pd.concat([sim_official, sim_suggest], axis=1).max(axis=1) + 0.4
                        ptr_merge['match_similarity'] = ptr_merge['match_similarity'].clip(upper=1.0)
                        del sim_official, sim_suggest
                else:
                    ptr_merge['match_similarity'] = 0.4  # Base score for PTR match
                
                # Best match per row: (1) prefer highest ptr_md_frequency, (2) when using name similarity, compare from TOP to lowest and stop at first that meets threshold; (3) if no similarity (name matching off), use TOP frequency only
                if not ptr_merge.empty:
                    ptr_merge_sorted = ptr_merge.sort_values(['ptr_md_frequency', 'match_similarity'], ascending=[False, False])
                    if use_name_matching:
                        # From TOP to lowest (by frequency then similarity), take first row that meets threshold; if none, take TOP frequency (vectorized: idxmax on boolean gives first True)
                        ptr_best_idx = ptr_merge_sorted.groupby(original_index_name).apply(
                            lambda g: (g['match_similarity'] >= ptr_threshold).idxmax()
                        ).values
                    else:
                        # No similarity: just use TOP frequency
                        ptr_best_idx = ptr_merge_sorted.groupby(original_index_name).head(1).index
                    ptr_best = ptr_merge.loc[ptr_best_idx]
                    
                    # Get original indices
                    original_indices = ptr_best[original_index_name].values
                    
                    # CRITICAL: TF-IDF Masterlist (PTR) should set suggested_md = FALSE (not True)
                    # According to matching rules: TF-IDF matches are NOT TRUE matches
                    # Only Exact Match (Basic/Advanced) sets suggested_md = True
                    df_work.loc[original_indices, 'suggested_md'] = False
                    df_work.loc[original_indices, 'md_official_name'] = ptr_best['md_official_name'].values
                    # [PTR Final] from masterlist ptr_final/md_ptrs_final consistently
                    if 'ptr_final' in ptr_best.columns:
                        _pf = ptr_best['ptr_final'].fillna('').astype(str).str.strip()
                    elif 'md_ptrs_final' in ptr_best.columns:
                        _pf = ptr_best['md_ptrs_final'].fillna('').astype(str).str.strip()
                    else:
                        _pf = ptr_best['md_ptrs'].fillna('').astype(str).str.strip()
                    df_work.loc[original_indices, 'md_ptrs'] = _pf.values
                    df_work.loc[original_indices, 'match_similarity'] = ptr_best['match_similarity'].values
                    # Fill DOCTOR_CODE and CUSTOMER_CODE from masterlist
                    # Ensure columns exist
                    if 'DOCTOR_CODE' not in df_work.columns:
                        df_work['DOCTOR_CODE'] = ''
                    if 'CUSTOMER_CODE' not in df_work.columns:
                        df_work['CUSTOMER_CODE'] = ''
                    # Apply DOCTOR_CODE even if CUSTOMER_CODE is null/empty
                    if 'DOCTOR_CODE' in ptr_best.columns:
                        doctor_codes = ptr_best['DOCTOR_CODE'].fillna('').astype(str).str.strip()
                        df_work.loc[original_indices, 'DOCTOR_CODE'] = doctor_codes.values
                    # Apply CUSTOMER_CODE if available (can be empty/null)
                    if 'CUSTOMER_CODE' in ptr_best.columns:
                        customer_codes = ptr_best['CUSTOMER_CODE'].fillna('').astype(str).str.strip()
                        df_work.loc[original_indices, 'CUSTOMER_CODE'] = customer_codes.values
                    
                    # Cleanup
                    del ptr_merge, ptr_best
                    cleanup_memory()
        
        if progress_bar is not None:
            progress_bar.progress(0.5)
        
        # BATCH NAME FUZZY MATCHING (for rows without PTR match or below threshold)
        if use_name_matching:
            if status_text is not None:
                status_text.text('Batch fuzzy matching by Doctor Name...')
            
            # Get rows that need name matching (no match yet or similarity below threshold)
            # Exclude 'REF' from unmatched (treat as matched, just from reference data)
            if df_work['suggested_md'].dtype == bool:
                unmatched_mask = (~df_work['suggested_md']) | (df_work['match_similarity'] < similarity_threshold)
            else:
                # Handle mixed types (bool, string 'REF', etc.)
                not_matched = (df_work['suggested_md'].astype(str).str.lower() != 'true') & \
                             (df_work['suggested_md'].astype(str).str.lower() != 'ref')
                unmatched_mask = not_matched | (df_work['match_similarity'] < similarity_threshold)
            unmatched_df = df_work[unmatched_mask].copy()
            
            if not unmatched_df.empty:
                # Group by unique doctor names to avoid duplicate calculations
                unique_doctors = unmatched_df[['doctor_name_clean', 'branch_code_numeric', 'item_code_numeric']].drop_duplicates()
                
                # Filter masterlist by branch/product if enabled - use view instead of copy when possible
                filtered_master = masterlist_df  # Use reference instead of copy to save memory
                
                # Batch calculate similarities for unique doctor names (OPTIMIZED)
                doctor_matches = {}
                total_unique = len(unique_doctors)
                processed_unique = 0
                
                # Process in chunks to avoid memory issues
                chunk_size = 100  # Process 100 unique doctors at a time
                for chunk_start in range(0, total_unique, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_unique)
                    unique_doctors_chunk = unique_doctors.iloc[chunk_start:chunk_end]
                    
                    for _, doctor_row in unique_doctors_chunk.iterrows():
                        doctor_name_clean = doctor_row['doctor_name_clean']
                        branch_numeric = doctor_row['branch_code_numeric']
                        item_numeric = doctor_row['item_code_numeric']
                        
                        # Filter masterlist - use boolean indexing instead of copy when possible
                        candidate_mask = pd.Series([True] * len(filtered_master), index=filtered_master.index)
                        
                        if use_branch_filter and branch_numeric:
                            branch_mask = filtered_master['md_b_codes_numeric'].astype(str).str.contains(
                                r'\b' + re.escape(branch_numeric) + r'\b', na=False, regex=True
                            )
                            candidate_mask = candidate_mask & branch_mask
                        
                        if use_product_filter and item_numeric:
                            product_mask = filtered_master['md_p_codes_numeric'].astype(str).str.contains(
                                r'\b' + re.escape(item_numeric) + r'\b', na=False, regex=True
                            )
                            candidate_mask = candidate_mask & product_mask
                        
                        candidate_master = filtered_master[candidate_mask]
                        
                        if candidate_master.empty:
                            processed_unique += 1
                            continue
                        
                        # OPTIMIZED: Use rapidfuzz.process.extractOne for fast batch matching
                        best_match = None
                        best_similarity = 0.0
                        
                        if RAPIDFUZZ_AVAILABLE:
                            from rapidfuzz import process
                            from rapidfuzz import fuzz
                            
                            # OPTIMIZED: Prepare candidate strings using vectorized operations (much faster than iterrows)
                            # Use both names, prefer official (vectorized)
                            candidate_strings = (
                                candidate_master['md_official_name_clean']
                                .fillna(candidate_master['md_suggest_clean'])
                                .fillna('')
                                .astype(str)
                                .tolist()
                            )
                            candidate_indices = candidate_master.index.tolist()
                            
                            if candidate_strings:
                                # Use extractOne to find best match (much faster than iterating)
                                result = process.extractOne(
                                    doctor_name_clean,
                                    candidate_strings,
                                    scorer=fuzz.ratio,
                                    score_cutoff=int(similarity_threshold * 100)  # Convert to 0-100 range
                                )
                                
                                if result:
                                    best_match_str, best_score, best_idx = result
                                    best_similarity = best_score / 100.0  # Convert back to 0-1 range
                                    best_match_idx = candidate_indices[best_idx]
                                    best_match = candidate_master.loc[best_match_idx]
                                    
                                    # Also check suggest name if official was used
                                    if best_match_str == str(best_match.get('md_official_name_clean', '')):
                                        # Check suggest name too and take the max
                                        suggest_str = str(best_match.get('md_suggest_clean', ''))
                                        if suggest_str:
                                            suggest_score = fuzz.ratio(doctor_name_clean, suggest_str) / 100.0
                                            best_similarity = max(best_similarity, suggest_score)
                                else:
                                    processed_unique += 1
                                    continue
                            else:
                                processed_unique += 1
                                continue
                        else:
                            # Fallback to original method if rapidfuzz not available
                            # STEP 1: Fast pre-filtering using fast similarity estimate
                            # This eliminates most candidates quickly
                            fast_sim_official = candidate_master['md_official_name_clean'].apply(
                                lambda x: fast_similarity_estimate(doctor_name_clean, x) if pd.notna(x) else 0.0
                            )
                            fast_sim_suggest = candidate_master['md_suggest_clean'].apply(
                                lambda x: fast_similarity_estimate(doctor_name_clean, x) if pd.notna(x) else 0.0
                            )
                            candidate_master = candidate_master.copy()  # Copy only after filtering
                            candidate_master['fast_similarity'] = pd.concat([fast_sim_official, fast_sim_suggest], axis=1).max(axis=1)
                            
                            # Filter to only promising candidates (fast similarity > threshold * 0.7)
                            # This reduces candidates by 80-90% before expensive calculation
                            min_fast_threshold = similarity_threshold * 0.7
                            promising_candidates = candidate_master[candidate_master['fast_similarity'] >= min_fast_threshold]
                            
                            # Cleanup intermediate dataframes
                            del candidate_master, fast_sim_official, fast_sim_suggest
                            
                            # If no promising candidates, skip
                            if promising_candidates.empty:
                                processed_unique += 1
                                continue
                            
                            # Limit to top 50 candidates by fast similarity (prevents checking too many)
                            # This ensures we don't waste time on poor matches
                            if len(promising_candidates) > 50:
                                promising_candidates = promising_candidates.nlargest(50, 'fast_similarity')
                            
                            # STEP 2: Accurate similarity only on promising candidates (much smaller set)
                            sim_official = promising_candidates['md_official_name_clean'].apply(
                                lambda x: calculate_similarity(doctor_name_clean, x) if pd.notna(x) else 0.0
                            )
                            sim_suggest = promising_candidates['md_suggest_clean'].apply(
                                lambda x: calculate_similarity(doctor_name_clean, x) if pd.notna(x) else 0.0
                            )
                            promising_candidates['similarity'] = pd.concat([sim_official, sim_suggest], axis=1).max(axis=1)
                            
                            # Get best match
                            best_match = promising_candidates.loc[promising_candidates['similarity'].idxmax()]
                            best_similarity = best_match['similarity']
                        
                        # Check if similarity meets threshold and best_match was found
                        if best_match is not None and best_similarity >= similarity_threshold:
                            # [PTR Final] from masterlist ptr_final/md_ptrs_final consistently
                            _ptr_val = best_match.get('ptr_final', '') or best_match.get('md_ptrs_final', '') or best_match.get('md_ptrs', '')
                            _ptr_val = str(_ptr_val).strip() if pd.notna(_ptr_val) else ''
                            doctor_matches[doctor_name_clean] = {
                                'md_official_name': best_match['md_official_name'],
                                'md_ptrs': _ptr_val,
                                'similarity': best_similarity,  # Use calculated similarity, not from dataframe
                                'DOCTOR_CODE': best_match.get('DOCTOR_CODE', '') if pd.notna(best_match.get('DOCTOR_CODE', '')) else '',
                                'CUSTOMER_CODE': best_match.get('CUSTOMER_CODE', '') if pd.notna(best_match.get('CUSTOMER_CODE', '')) else ''
                            }
                        
                        # Cleanup (only for fallback path - rapidfuzz path doesn't create these variables)
                        if not RAPIDFUZZ_AVAILABLE:
                            try:
                                del promising_candidates, sim_official, sim_suggest
                            except NameError:
                                pass
                        
                        # Update progress for unique doctors
                        processed_unique += 1
                        if progress_bar is not None and status_text is not None and processed_unique % 10 == 0:
                            progress = 0.5 + (processed_unique / total_unique) * 0.4  # 50% to 90%
                            progress_bar.progress(progress)
                            status_text.text(f'Batch fuzzy matching by Doctor Name... ({processed_unique}/{total_unique} unique names)')
                        cleanup_memory()  # Periodic cleanup
                
                # Cleanup after each chunk
                cleanup_memory()
            
            # Apply matches to all rows with same doctor name
            # Ensure DOCTOR_CODE and CUSTOMER_CODE columns exist
            if 'DOCTOR_CODE' not in df_work.columns:
                df_work['DOCTOR_CODE'] = ''
            if 'CUSTOMER_CODE' not in df_work.columns:
                df_work['CUSTOMER_CODE'] = ''
            
            # OPTIMIZED: Batch update using .loc instead of .at in loop
            # Filter unmatched_df to only rows that have matches
            matched_mask = unmatched_df['doctor_name_clean'].isin(doctor_matches.keys())
            matched_indices = unmatched_df.index[matched_mask]
            
            if len(matched_indices) > 0:
                # Prepare batch update data
                matched_doctor_names = unmatched_df.loc[matched_indices, 'doctor_name_clean']
                match_data_list = [doctor_matches[name] for name in matched_doctor_names]
                
                # Extract values for batch update
                md_official_names_batch = [data['md_official_name'] for data in match_data_list]
                md_ptrs_batch = [data['md_ptrs'] for data in match_data_list]
                similarities_batch = [data['similarity'] for data in match_data_list]
                
                # Batch update using .loc (much faster than .at in loop)
                df_work.loc[matched_indices, 'suggested_md'] = False
                df_work.loc[matched_indices, 'md_official_name'] = md_official_names_batch
                df_work.loc[matched_indices, 'md_ptrs'] = md_ptrs_batch
                df_work.loc[matched_indices, 'match_similarity'] = similarities_batch
                
                # Batch update DOCTOR_CODE and CUSTOMER_CODE
                doctor_codes_batch = []
                customer_codes_batch = []
                for data in match_data_list:
                    doctor_codes_batch.append(str(data.get('DOCTOR_CODE', '')).strip() if data.get('DOCTOR_CODE') else '')
                    customer_codes_batch.append(str(data.get('CUSTOMER_CODE', '')).strip() if data.get('CUSTOMER_CODE') else '')
                
                # Only update non-empty codes
                if 'DOCTOR_CODE' not in df_work.columns:
                    df_work['DOCTOR_CODE'] = ''
                if 'CUSTOMER_CODE' not in df_work.columns:
                    df_work['CUSTOMER_CODE'] = ''
                
                df_work.loc[matched_indices, 'DOCTOR_CODE'] = doctor_codes_batch
                df_work.loc[matched_indices, 'CUSTOMER_CODE'] = customer_codes_batch
    
                # Cleanup after name matching
                del unmatched_df, unique_doctors
                cleanup_memory()
        
        if progress_bar is not None:
            progress_bar.progress(1.0)
        if status_text is not None:
            status_text.text('Matching completed!')
        
        # Return only original columns plus matching results
        df['suggested_md'] = df_work['suggested_md']
        df['md_official_name'] = df_work['md_official_name']
        df['md_ptrs'] = df_work['md_ptrs']
        
        # CRITICAL: Restore original Amount column (NEVER modified during matching)
        # Use the original Amount we stored at the start, not from df_work
        # This ensures Amount is never corrupted by merge operations or row duplications
        if original_amount is not None and len(original_amount) == len(df):
            df['Amount'] = original_amount.values  # Restore original Amount values
        elif 'Amount' in df.columns:
            # Fallback: if original_amount wasn't stored, ensure df_work Amount is numeric
            if 'Amount' in df_work.columns:
                df['Amount'] = df_work['Amount']
            df = standardize_amount_column(df, 'Amount')
        else:
            # If Amount doesn't exist, ensure it's created as numeric
            if 'Amount' in df_work.columns:
                df['Amount'] = df_work['Amount']
                df = standardize_amount_column(df, 'Amount')
        
        # Cleanup working dataframe
        del df_work
        cleanup_memory()
        
        # Ensure Qty column is numeric type (fixes PyArrow conversion issues)
        if 'Qty' in df.columns:
            df = standardize_amount_column(df, 'Qty')
        
        return df
    
    except MemoryError as e:
        if status_text is not None:
            status_text.text(f'Memory error: {str(e)}. Try processing smaller batches.')
        cleanup_memory()
        # Return partial results if available
        copy_partial_results_to_df(df, df_work)  # noqa: F821
        return df
    except Exception as e:
        if status_text is not None:
            status_text.text(f'Error during matching: {str(e)}')
        cleanup_memory()
        # Return partial results if available
        copy_partial_results_to_df(df, df_work)  # noqa: F821
        return df

def validate_amount_total(df, original_total, step_name, raise_error=True):
    """
    Validate that the Amount total hasn't changed after a matching step.
    
    Args:
        df: DataFrame to check
        original_total: Original total amount to compare against
        step_name: Name of the step being validated (for error messages)
        raise_error: If True, raise an error if totals don't match. If False, return validation result.
    
    Returns:
        (is_valid, current_total, difference) tuple if raise_error=False
        Raises ValueError if raise_error=True and totals don't match
    """
    if df is None or df.empty:
        if raise_error:
            raise ValueError(f"ERROR in {step_name}: DataFrame is None or empty!")
        return False, 0.0, original_total
    
    if 'Amount' not in df.columns:
        if raise_error:
            raise ValueError(f"ERROR in {step_name}: Amount column is missing!")
        return False, 0.0, original_total
    
    # Calculate current total
    # Use standardized amount pipeline guarantees (float64)
    current_total = float(df['Amount'].sum())
    
    # Compare with original (allow small floating point differences)
    difference = abs(current_total - original_total)
    tolerance = 0.01  # Allow 0.01 difference for floating point precision
    
    if difference > tolerance:
        error_msg = (
            f"❌ AMOUNT VALIDATION FAILED in {step_name}!\n"
            f"   Original Total: {original_total:,.2f}\n"
            f"   Current Total:  {current_total:,.2f}\n"
            f"   Difference:     {difference:,.2f}\n"
            f"   Rows:           {len(df):,}\n"
            f"   This indicates Amount was modified during {step_name}!"
        )
        if raise_error:
            raise ValueError(error_msg)
        return False, current_total, difference
    
    return True, current_total, 0.0

def add_md_code_columns(df):
    """Add DOCTOR_CODE and CUSTOMER_CODE columns by matching md_official_name with CUSTOMER_NAME from md_code_list.
    Only fills for rows where md_official_name is non-empty (Official Matches rule). Does not overwrite Quick Suggest codes.
    """
    df = df.copy()
    
    # Ensure columns exist (do not overwrite - preserve Quick Suggest codes for rows with blank md_official_name)
    if 'DOCTOR_CODE' not in df.columns:
        df['DOCTOR_CODE'] = ''
    if 'CUSTOMER_CODE' not in df.columns:
        df['CUSTOMER_CODE'] = ''
    
    # Check if md_code_list exists and is not empty
    if 'md_code_list' in st.session_state and st.session_state.md_code_list is not None and not st.session_state.md_code_list.empty:
        md_code_list = st.session_state.md_code_list.copy() 
        
        # Create a mapping dictionary: CUSTOMER_NAME -> (DOCTOR_CODE, CUSTOMER_CODE)
        # Handle case-insensitive matching
        md_code_list['CUSTOMER_NAME_clean'] = md_code_list['CUSTOMER_NAME'].astype(str).str.strip().str.upper()
        md_code_list = md_code_list[md_code_list['CUSTOMER_NAME_clean'] != 'NAN']  # Remove invalid entries
        
        # OPTIMIZED: Create mapping dictionary using vectorized operations (much faster than iterrows)
        # Take first match if duplicates exist (drop_duplicates keeps first)
        md_code_list_unique = md_code_list.drop_duplicates(subset=['CUSTOMER_NAME_clean'], keep='first')
        
        # Create mapping dictionary with tuple values
        name_to_code = {}
        for name, doctor_code, customer_code in zip(
            md_code_list_unique['CUSTOMER_NAME_clean'],
            md_code_list_unique['DOCTOR_CODE'].fillna('').astype(str),
            md_code_list_unique['CUSTOMER_CODE'].fillna('').astype(str)
        ):
            if name and name != 'NAN':  # Only add non-empty, valid names
                name_to_code[name] = (doctor_code, customer_code)
        
        # OPTIMIZED: Match md_official_name with CUSTOMER_NAME using vectorized operations
        # Only fill DOCTOR_CODE, CUSTOMER_CODE for rows where md_official_name is non-empty (Official Matches rule)
        if 'md_official_name' in df.columns:
            # Vectorized cleaning
            md_official_name_clean = (
                df['md_official_name']
                .astype(str)
                .str.strip()
                .str.upper()
            )
            # Only update rows with non-empty md_official_name (Official Matches: processes 1-5)
            mask_official = (md_official_name_clean != '') & (md_official_name_clean != 'NAN')
            if mask_official.any():
                code_matches = md_official_name_clean[mask_official].map(name_to_code)
                df.loc[mask_official, 'DOCTOR_CODE'] = [x[0] if pd.notna(x) and x else '' for x in code_matches]
                df.loc[mask_official, 'CUSTOMER_CODE'] = [x[1] if pd.notna(x) and x else '' for x in code_matches]
    
    return df

def exact_match_suggest_and_address(df, masterlist_df, progress_bar=None, status_text=None):
    """
    Exact matching of md_suggest + md_add_1 to Doctor Name + Address1.
    Only applies to records where suggested_md == False.
    Does not overwrite suggested_md (keeps it as False).
    Returns: (updated_df, matched_count) tuple
    """
    if df is None or df.empty:
        return df, 0
    
    if masterlist_df is None or masterlist_df.empty:
        return df, 0
    
    # CRITICAL: Store original Amount column separately to prevent any modification
    # Amount should NEVER be modified during matching - only matching columns are added
    original_amount = None
    if 'Amount' in df.columns:
        # Use unified pipeline to ensure we store clean float64 data
        temp_df_for_amount = df[['Amount']].copy()
        temp_df_for_amount = standardize_amount_column(temp_df_for_amount, 'Amount')
        original_amount = temp_df_for_amount['Amount']
    
    # Create a copy to work with
    df_work = df.copy()
    
    # Find records where suggested_md = False (exclude 'REF' - treat as matched)
    if df_work['suggested_md'].dtype == bool:
        unmatched_mask = ~df_work['suggested_md']
    else:
        # Handle mixed types: exclude 'REF' from unmatched (treat as matched)
        unmatched_mask = (df_work['suggested_md'].astype(str).str.lower() == 'false') | \
                        (df_work['suggested_md'].isna()) | \
                        (df_work['suggested_md'].astype(str).str.strip() == '')
    
    unmatched_df = df_work[unmatched_mask].copy()
    
    if unmatched_df.empty:
        return df_work, 0
    
    # Check required columns
    doctor_name_col = None
    for col in ['Doctor Name', 'doctor_name', 'DOCTOR_NAME']:
        if col in unmatched_df.columns:
            doctor_name_col = col
            break
    
    address_col = None
    for col in ['Address1', 'Address', 'address1', 'ADDRESS1']:
        if col in unmatched_df.columns:
            address_col = col
            break
    
    if not doctor_name_col or not address_col:
        return df_work, 0
    
    # Check masterlist columns
    if 'md_suggest' not in masterlist_df.columns or 'md_add_1' not in masterlist_df.columns:
        return df_work, 0
    
    if progress_bar is not None and status_text is not None:
        status_text.text('Performing exact matching (MD suggest + address)...')
        progress_bar.progress(0.88)
    
    # OPTIMIZED: Prepare masterlist lookup dictionary using vectorized operations
    # Key: (md_suggest_upper, md_add_1_upper) -> (md_official_name, md_ptrs)
    masterlist_temp = masterlist_df.copy()
    masterlist_temp['md_suggest_upper'] = masterlist_temp['md_suggest'].fillna('').astype(str).str.strip().str.upper()
    masterlist_temp['md_add_1_upper'] = masterlist_temp['md_add_1'].fillna('').astype(str).str.strip().str.upper()
    
    # Filter to rows with both md_suggest and md_add_1
    masterlist_temp = masterlist_temp[
        (masterlist_temp['md_suggest_upper'] != '') & 
        (masterlist_temp['md_add_1_upper'] != '')
    ].copy()
    
    if not masterlist_temp.empty:
        # Create composite key column
        masterlist_temp['composite_key'] = list(zip(masterlist_temp['md_suggest_upper'], masterlist_temp['md_add_1_upper']))
        
        # Drop duplicates keeping first match (consistent with original logic)
        masterlist_temp_unique = masterlist_temp.drop_duplicates(subset=['composite_key'], keep='first')
        
        # Create lookup dictionary using to_dict() - much faster than iterrows()
        md_official_names = masterlist_temp_unique['md_official_name'].fillna('').astype(str)
        # [PTR Final] from masterlist ptr_final/md_ptrs_final consistently
        _ptr_col = 'ptr_final' if 'ptr_final' in masterlist_temp_unique.columns else ('md_ptrs_final' if 'md_ptrs_final' in masterlist_temp_unique.columns else 'md_ptrs')
        md_ptrs_values = masterlist_temp_unique[_ptr_col].fillna('').astype(str).str.strip()
        # Include DOCTOR_CODE and CUSTOMER_CODE (fill only when md_official_name filled - Official Matches rule)
        dc_col = 'DOCTOR_CODE' if 'DOCTOR_CODE' in masterlist_temp_unique.columns else None
        cc_col = 'CUSTOMER_CODE' if 'CUSTOMER_CODE' in masterlist_temp_unique.columns else None
        masterlist_lookup = {}
        for i, key in enumerate(masterlist_temp_unique['composite_key']):
            dc = str(masterlist_temp_unique.iloc[i][dc_col]).strip() if dc_col and pd.notna(masterlist_temp_unique.iloc[i][dc_col]) else ''
            cc = str(masterlist_temp_unique.iloc[i][cc_col]).strip() if cc_col and pd.notna(masterlist_temp_unique.iloc[i][cc_col]) else ''
            masterlist_lookup[key] = (md_official_names.iloc[i], md_ptrs_values.iloc[i], dc, cc)
    else:
        masterlist_lookup = {}
        dc_col = cc_col = None
    
    if not masterlist_lookup:
        return df_work, 0
    
    # OPTIMIZED: Vectorized operations instead of row-by-row iteration
    # Prepare lookup keys as tuples - vectorized
    doctor_names_clean = unmatched_df[doctor_name_col].fillna('').astype(str).str.strip().str.upper()
    address1_clean = unmatched_df[address_col].fillna('').astype(str).str.strip().str.upper()
    
    # Create lookup_key tuples as Series - vectorized
    lookup_keys_series = pd.Series(list(zip(doctor_names_clean, address1_clean)), index=unmatched_df.index)
    
    # OPTIMIZED: Use vectorized dictionary lookup (much faster than iterrows)
    # Filter to only rows where both doctor_name and address1 are non-empty
    valid_mask = (doctor_names_clean != '') & (address1_clean != '')
    
    if valid_mask.any():
        # Map lookup keys to masterlist values using list comprehension (faster than apply)
        matched_results = pd.Series([
            masterlist_lookup.get(key_tuple, None) if key_tuple[0] and key_tuple[1] else None
            for key_tuple in lookup_keys_series[valid_mask]
        ], index=unmatched_df.index[valid_mask])
        
        # Filter to only matched rows (where lookup found a result)
        matched_mask = matched_results.notna()
        matched_indices = matched_results[matched_mask].index
        
        # Update matched records - vectorized
        if len(matched_indices) > 0:
            # Extract md_official_name, md_ptrs, DOCTOR_CODE, CUSTOMER_CODE from matched results (Official Matches rule)
            md_official_names = pd.Series([x[0] for x in matched_results[matched_mask]], index=matched_indices)
            md_ptrs_values = pd.Series([x[1] if pd.notna(x[1]) else '' for x in matched_results[matched_mask]], index=matched_indices)
            doctor_codes = pd.Series([x[2] if len(x) > 2 else '' for x in matched_results[matched_mask]], index=matched_indices)
            customer_codes = pd.Series([x[3] if len(x) > 3 else '' for x in matched_results[matched_mask]], index=matched_indices)
            
            df_work.loc[matched_indices, 'md_official_name'] = md_official_names.values
            df_work.loc[matched_indices, 'md_ptrs'] = md_ptrs_values.values
            if 'DOCTOR_CODE' not in df_work.columns:
                df_work['DOCTOR_CODE'] = ''
            if 'CUSTOMER_CODE' not in df_work.columns:
                df_work['CUSTOMER_CODE'] = ''
            df_work.loc[matched_indices, 'DOCTOR_CODE'] = doctor_codes.values
            df_work.loc[matched_indices, 'CUSTOMER_CODE'] = customer_codes.values
            
            matched_count = len(matched_indices)
        else:
            matched_count = 0
    else:
        matched_count = 0
    
    if progress_bar is not None:
        progress_bar.progress(0.89)
    if status_text is not None:
        status_text.text(f'Exact matching completed! {matched_count} records matched.')
    
    # CRITICAL: Restore original Amount column (NEVER modified during matching)
    if original_amount is not None and len(original_amount) == len(df_work):
        df_work['Amount'] = original_amount.values  # Restore original Amount values
    
    return df_work, matched_count

def split_matching_for_unmatched(df, masterlist_df, progress_bar=None, status_text=None, threshold=0.4, use_doctor_name=True, use_ptr_no=True, use_tfidf_matching=True, use_quick_suggest_matching=True):
    """
    Perform fast TF-IDF with N-Grams matching using NearestNeighbors for records where suggested_md = False.
    Supports matching by doctor name, PTR number, or both.
    Uses tree-based search for much faster matching than nested loops or full similarity matrices.
    
    Args:
        use_tfidf_matching: If False, skips TF-IDF matching (useful for low-memory systems)
    
    Returns: (updated_df, ai_matched_count, completion_info) tuple
        completion_info: dict with keys 'doctor_name_matched', 'ptr_matched', 'total_matched'
    """
    # CRITICAL: Store original Amount column separately to prevent any modification
    # Amount should NEVER be modified during matching - only matching columns are added
    original_amount = None
    if df is not None and not df.empty and 'Amount' in df.columns:
        # Use unified pipeline to ensure we store clean float64 data
        temp_df_for_amount = df[['Amount']].copy()
        temp_df_for_amount = standardize_amount_column(temp_df_for_amount, 'Amount')
        original_amount = temp_df_for_amount['Amount']
    
    if df is None or df.empty:
        return df, 0, {'doctor_name_matched': 0, 'ptr_matched': 0, 'total_matched': 0, 'use_doctor_name': use_doctor_name, 'use_ptr_no': use_ptr_no}
    
    if masterlist_df is None or masterlist_df.empty:
        return df, 0, {'doctor_name_matched': 0, 'ptr_matched': 0, 'total_matched': 0, 'use_doctor_name': use_doctor_name, 'use_ptr_no': use_ptr_no}
    
    # At least one matching method must be enabled
    if not use_doctor_name and not use_ptr_no:
        return df, 0, {'doctor_name_matched': 0, 'ptr_matched': 0, 'total_matched': 0, 'use_doctor_name': use_doctor_name, 'use_ptr_no': use_ptr_no}
    
    # If TF-IDF matching is disabled, return early (only exact matches will be performed)
    if not use_tfidf_matching:
        if status_text is not None:
            status_text.text('TF-IDF AI Matching is disabled. Skipping AI matching step...')
        if progress_bar is not None:
            progress_bar.progress(1.0)
        # CRITICAL: Restore original Amount column (NEVER modified during matching)
        if original_amount is not None and len(original_amount) == len(df):
            df['Amount'] = original_amount.values  # Restore original Amount values
        return df, 0, {'doctor_name_matched': 0, 'ptr_matched': 0, 'total_matched': 0, 'use_doctor_name': use_doctor_name, 'use_ptr_no': use_ptr_no}
    
    # Adjust threshold based on checkbox settings
    # If only Doctor Name is checked (no PTR validation), use higher threshold (85%) for accuracy
    # If both are checked, use default threshold (70%) since PTR validation adds extra accuracy
    name_threshold = 0.7  # Default threshold for name matching (70%)
    if use_doctor_name and not use_ptr_no:
        name_threshold = 0.85  # 85% when only name matching (no PTR validation)
    
    ptr_threshold = 0.85  # High threshold (85%) for PTR matching (PTR numbers should match closely)
    doctor_similarity_threshold = 0.7  # When matching by PTR, also validate doctor name similarity (70%)
    doctor_similarity_threshold_exact_ptr = 0.7  # Threshold (70%) when PTR matches exactly
    
    
    # Create a copy to work with
    df_work = df.copy()
    
    # Find records where suggested_md = False AND both md_official_name and md_ptrs are blank
    # Exclude 'REF' from unmatched (treat as matched, just from reference data)
    if df_work['suggested_md'].dtype == bool:
        unmatched_mask = ~df_work['suggested_md']
    else:
        # Handle string or other types - exclude 'REF' from unmatched (treat as matched)
        unmatched_mask = (df_work['suggested_md'].astype(str).str.lower() == 'false') | \
                        (df_work['suggested_md'].isna()) | \
                        (df_work['suggested_md'].astype(str).str.strip() == '')
    
    # Also check that md_official_name and md_ptrs are blank
    if 'md_official_name' in df_work.columns:
        unmatched_mask = unmatched_mask & ((df_work['md_official_name'] == '') | (df_work['md_official_name'].isna()))
    if 'md_ptrs' in df_work.columns:
        unmatched_mask = unmatched_mask & ((df_work['md_ptrs'] == '') | (df_work['md_ptrs'].isna()))
    
    unmatched_df = df_work[unmatched_mask].copy()
    
    if unmatched_df.empty:
        return df_work, 0, {'doctor_name_matched': 0, 'ptr_matched': 0, 'total_matched': 0, 'use_doctor_name': use_doctor_name, 'use_ptr_no': use_ptr_no}
    
    # Prepare masterlist for matching
    masterlist_work = masterlist_df.copy()
    if 'md_official_name' not in masterlist_work.columns:
        return df_work, 0, {'doctor_name_matched': 0, 'ptr_matched': 0, 'total_matched': 0, 'use_doctor_name': use_doctor_name, 'use_ptr_no': use_ptr_no}
    
    # Build ptr_final lookup from masterlist (md_official_name -> ptr_final) for consistent [PTR Final] filling
    _ptr_src = 'ptr_final' if 'ptr_final' in masterlist_work.columns else ('md_ptrs_final' if 'md_ptrs_final' in masterlist_work.columns else 'md_ptrs')
    if _ptr_src not in masterlist_work.columns:
        _ptr_src = 'md_ptrs' if 'md_ptrs' in masterlist_work.columns else None
    masterlist_work['_md_official_name_norm'] = masterlist_work['md_official_name'].fillna('').astype(str).str.strip().str.upper()
    ptr_final_lookup = masterlist_work.set_index('_md_official_name_norm')[_ptr_src].fillna('').astype(str).str.strip().to_dict() if _ptr_src else {}
    ptr_final_lookup = {k: v for k, v in ptr_final_lookup.items() if k and v and str(v).lower() != 'nan'}
    
    # Get 'Doctor Name' and 'PTR No' column names
    doctor_name_col = None
    for col in ['Doctor Name', 'doctor_name', 'DOCTOR_NAME']:
        if col in unmatched_df.columns:
            doctor_name_col = col
            break
    
    ptr_no_col = None
    for col in ['PTR No', 'ptr_no', 'PTR_NO']:
        if col in unmatched_df.columns:
            ptr_no_col = col
            break
    
    if progress_bar is not None and status_text is not None:
        status_text.text('Performing TF-IDF AI matching for unmatched records...')
        progress_bar.progress(0.85)
    
    # Clean function for names (vectorized - very fast)
    def clean_name(name):
        """Clean doctor name for matching."""
        if pd.isna(name) or str(name).strip() == '' or str(name).lower() == 'nan':
            return ''
        name_str = str(name).lower().strip()
        # Remove common prefixes like "dr.", "dr", "doctor", "md"
        name_str = re.sub(r'\b(dr|md|doctor)\b', '', name_str, flags=re.IGNORECASE)
        # Remove non-alphanumeric characters except spaces
        name_str = re.sub(r'[^a-z0-9\s]', '', name_str)
        return name_str.strip()
    
    # Extract digits function for PTR numbers
    def extract_digits(value):
        """Extract digits from PTR number."""
        if pd.isna(value) or str(value).strip() == '':
            return ''
        return ''.join(filter(str.isdigit, str(value)))
    
    # Extract and normalize digits (remove leading zeros) for PTR numbers
    def extract_digits_normalized(value):
        """Extract digits and normalize by removing leading zeros for flexible matching."""
        if pd.isna(value) or str(value).strip() == '':
            return ''
        digits = ''.join(filter(str.isdigit, str(value)))
        # Remove leading zeros to normalize (e.g., "0111025" becomes "111025")
        # But keep at least one digit if all zeros (e.g., "000" stays as "0")
        if digits:
            normalized = digits.lstrip('0') if digits.lstrip('0') else '0'
            return normalized
        return ''
    
    updated_count = 0
    doctor_name_matched_count = 0  # Track matches from doctor name matching
    ptr_matched_count = 0  # Track matches from PTR matching
    ppe_matched_count = 0  # Track matches from PPE Doctors matching
    
    try:
        # ===== MATCHING BY DOCTOR NAME (if enabled) =====
        if use_doctor_name and doctor_name_col:
            if progress_bar is not None:
                progress_bar.progress(0.86)
            if status_text is not None:
                status_text.text('TF-IDF AI matching by doctor name...')
                time.sleep(0.5)  # Allow UI to refresh and show the message
            
            # Clean data
            try:
                unmatched_names_list = unmatched_df[doctor_name_col].apply(clean_name).tolist()
                master_names_list = masterlist_work['md_official_name'].apply(clean_name).tolist()
            except Exception as e:
                if status_text is not None:
                    status_text.text(f'Doctor\'s Name: Error cleaning data: {str(e)}')
                unmatched_names_list = []
                master_names_list = []
            
            # Also prepare md_suggest column if available (collected data from store branches)
            master_suggest_list = []
            if 'md_suggest' in masterlist_work.columns:
                master_suggest_list = masterlist_work['md_suggest'].apply(clean_name).tolist()
            else:
                master_suggest_list = ['' for _ in range(len(master_names_list))]
            
            # Filter out empty names but keep track of original indices
            unmatched_indices = unmatched_df.index.tolist()
            unmatched_names_clean = []
            unmatched_indices_clean = []
            
            for i, name in enumerate(unmatched_names_list):
                if name:  # Only include non-empty names
                    unmatched_names_clean.append(name)
                    unmatched_indices_clean.append(unmatched_indices[i])
            
            # Filter masterlist names and keep track of indices (keep all masterlist entries)
            master_indices = masterlist_work.index.tolist()
            master_names_clean = []
            master_suggest_clean = []
            
            for i in range(len(master_indices)):
                master_names_clean.append(master_names_list[i] if i < len(master_names_list) else '')
                master_suggest_clean.append(master_suggest_list[i] if i < len(master_suggest_list) else '')
            
            # Check if we have data to process
            if not unmatched_names_clean:
                if status_text is not None:
                    status_text.text('Doctor\'s Name: No unmatched names to process for TF-IDF matching.')
            elif not (master_names_clean or master_suggest_clean):
                if status_text is not None:
                    status_text.text('Doctor\'s Name: No master names available for TF-IDF matching.')
            elif unmatched_names_clean and (master_names_clean or master_suggest_clean):
                # Update status to show we're processing (include doctor name context)
                if status_text is not None:
                    status_text.text(f'TF-IDF AI matching by doctor name: Preparing {len(unmatched_names_clean)} unmatched records against {len(master_names_clean)} master records...')
                    # time.sleep(0.3)  # Brief pause to ensure message is visible
                
                # Vectorize md_official_name using N-Grams (only non-empty names)
                official_name_indices_map = []  # Maps filtered index to original masterlist index
                master_names_nonempty = []
                for i, name in enumerate(master_names_clean):
                    if name:
                        master_names_nonempty.append(name)
                        official_name_indices_map.append(i)
                
                official_similarities = [0.0] * len(unmatched_names_clean)
                official_match_indices = [0] * len(unmatched_names_clean)
                
                if master_names_nonempty:
                    if status_text is not None:
                        status_text.text(f'Doctor\'s Name: Starting TF-IDF vectorization: {len(master_names_nonempty)} non-empty master names...')
                    try:
                        # Check memory before TF-IDF
                        available_memory = None
                        if PSUTIL_AVAILABLE:
                            try:
                                memory = psutil.virtual_memory()
                                available_memory = memory.available / 1024 / 1024
                            except Exception:
                                pass
                        
                        # Determine batch size for large datasets (500K+ rows)
                        batch_size = get_tfidf_batch_size(len(unmatched_names_clean), available_memory)
                        
                        # Use conservative max_features for low-memory VM
                        max_features = 10000  # Reduced for low-memory Ubuntu VM
                        
                        # Always show batch processing status
                        num_batches_calc = (len(unmatched_names_clean) + batch_size - 1) // batch_size
                        if status_text is not None:
                            if num_batches_calc > 1:
                                status_text.text(f'Doctor\'s Name: Processing TF-IDF in {num_batches_calc} batches ({len(unmatched_names_clean)} records, batch size: {batch_size}, max_features: {max_features})...')
                            else:
                                status_text.text(f'Doctor\'s Name: Processing TF-IDF ({len(unmatched_names_clean)} records, max_features: {max_features})...')
                        
                        # Check memory before starting
                        if not check_memory_available(500):
                            if status_text is not None:
                                status_text.text('Warning: Low memory. Using conservative settings...')
                            batch_size = min(batch_size, 1000)  # Smaller batches
                        
                        # Fit vectorizer on masterlist once
                        vectorizer = TfidfVectorizer(
                            analyzer='char', 
                            ngram_range=(2, 3),
                            min_df=1,
                            lowercase=True,
                            max_features=max_features  # Conservative limit for low-memory VM
                        )
                        master_matrix = vectorizer.fit_transform(master_names_nonempty)
                        # Convert to float32 to reduce memory usage (halves memory needed)
                        master_matrix = master_matrix.astype('float32')
                        
                        # Use NearestNeighbors for md_official_name (fit once on master)
                        # For very large datasets, use single core to reduce memory
                        num_unmatched = len(unmatched_names_clean)
                        if num_unmatched > 100000:
                            optimal_n_jobs = 1  # Force single core for very large datasets
                        else:
                            optimal_n_jobs = get_optimal_n_jobs(min_memory_mb=2000)
                        nbrs = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=optimal_n_jobs).fit(master_matrix)
                        
                        # Process unmatched records in batches
                        num_batches = (len(unmatched_names_clean) + batch_size - 1) // batch_size
                        
                        # Ensure batch processing always runs (even if num_batches is 1)
                        if num_batches == 0:
                            num_batches = 1  # Process at least one batch
                        
                        # Update status before starting batch processing
                        if status_text is not None:
                            status_text.text(f'Doctor\'s Name: Starting batch processing: {num_batches} batches, {len(unmatched_names_clean)} records...')
                        
                        for batch_idx in range(num_batches):
                            # Status update at start of each batch to confirm execution
                            if status_text is not None:
                                status_text.text(f'Doctor\'s Name: Processing batch {batch_idx + 1}/{num_batches} ({batch_size} records per batch)...')
                            
                            # Check memory before each batch
                            if PSUTIL_AVAILABLE and batch_idx > 0:
                                try:
                                    memory = psutil.virtual_memory()
                                    if memory.available / 1024 / 1024 < 500:  # Less than 500MB available
                                        cleanup_memory()  # Force cleanup before continuing
                                except Exception:
                                    pass
                            
                            # Periodic keepalive update to prevent connection timeout
                            # Update less frequently to reduce memory/UI overhead
                            update_frequency = max(10, num_batches // 20)  # Update ~20 times total, or every 10 batches minimum
                            if status_text is not None and batch_idx % update_frequency == 0:
                                try:
                                    progress_pct = min(0.90 + (batch_idx / num_batches) * 0.05, 0.95)
                                    if progress_bar is not None:
                                        progress_bar.progress(progress_pct)
                                    status_text.text(f'Doctor\'s Name: TF-IDF AI matching: Processing batch {batch_idx + 1}/{num_batches} ({int((batch_idx+1)/num_batches*100)}%)... Please wait, this may take several minutes.')
                                    # Minimal delay - just enough to keep connection alive
                                    time.sleep(0.05)  # Reduced delay to prevent memory issues
                                except Exception:
                                    pass
                            
                            batch_start = batch_idx * batch_size
                            batch_end = min(batch_start + batch_size, len(unmatched_names_clean))
                            batch_names = unmatched_names_clean[batch_start:batch_end]
                            
                            # Transform batch
                            unmatched_batch_matrix = vectorizer.transform(batch_names)
                            # Convert to float32 to reduce memory usage
                            unmatched_batch_matrix = unmatched_batch_matrix.astype('float32')
                            
                            # Find nearest neighbors for batch
                            distances, indices = nbrs.kneighbors(unmatched_batch_matrix)
                            
                            # Store results
                            for j, (dist, idx) in enumerate(zip(distances, indices)):
                                global_idx = batch_start + j
                                official_similarities[global_idx] = 1 - dist[0]
                                official_match_indices[global_idx] = official_name_indices_map[idx[0]] if idx[0] < len(official_name_indices_map) else 0
                            
                            # Cleanup batch matrix immediately
                            del unmatched_batch_matrix, distances, indices
                            cleanup_memory()  # Cleanup after EVERY batch for large datasets
                            
                            # Update progress more frequently for large datasets
                            if progress_bar is not None and status_text is not None:
                                if num_batches > 20:  # Update every batch if many batches
                                    progress = 0.86 + (batch_idx / num_batches) * 0.02
                                    progress_bar.progress(progress)
                                    status_text.text(f'Doctor\'s Name: TF-IDF (AI Matching) batch {batch_idx + 1}/{num_batches} ({batch_end}/{len(unmatched_names_clean)} records)...')
                                elif batch_idx % 5 == 0:  # Update every 5 batches for smaller datasets
                                    progress = 0.86 + (batch_idx / num_batches) * 0.02
                                    progress_bar.progress(progress)
                                    status_text.text(f'Doctor\'s Name: TF-IDF (AI Matching) batch {batch_idx + 1}/{num_batches}...')
                        
                        # Cleanup master matrices
                        del vectorizer, master_matrix, nbrs
                        cleanup_memory()
                    except MemoryError as e:
                        if status_text is not None:
                            status_text.text(f'Doctor\'s Name: Memory error during TF-IDF. Skipping name matching... Error: {str(e)}')
                        # Set all similarities to 0
                        official_similarities = [0.0] * len(unmatched_names_clean)
                        official_match_indices = [0] * len(unmatched_names_clean)
                    except Exception as e:
                        if status_text is not None:
                            status_text.text(f'Doctor\'s Name: Error during TF-IDF batch processing: {str(e)}')
                        # Set all similarities to 0 on error
                        official_similarities = [0.0] * len(unmatched_names_clean)
                        official_match_indices = [0] * len(unmatched_names_clean)
                else:
                    # No non-empty master names found
                    if status_text is not None:
                        status_text.text('Doctor\'s Name: Warning: No non-empty master names found for TF-IDF matching.')
                    official_similarities = [0.0] * len(unmatched_names_clean)
                    official_match_indices = [0] * len(unmatched_names_clean)
                
                # Also vectorize md_suggest if available (only non-empty suggests)
                suggest_similarities = [0.0] * len(unmatched_names_clean)
                suggest_match_indices = [0] * len(unmatched_names_clean)
                
                if 'md_suggest' in masterlist_work.columns:
                    suggest_indices_map = []  # Maps filtered index to original masterlist index
                    master_suggest_nonempty = []
                    for i, suggest in enumerate(master_suggest_clean):
                        if suggest:
                            master_suggest_nonempty.append(suggest)
                            suggest_indices_map.append(i)
                    
                    if master_suggest_nonempty:
                        try:
                            # Determine batch size for large datasets
                            available_memory = None
                            if PSUTIL_AVAILABLE:
                                try:
                                    memory = psutil.virtual_memory()
                                    available_memory = memory.available / 1024 / 1024
                                except Exception:
                                    pass
                            
                            batch_size = get_tfidf_batch_size(len(unmatched_names_clean), available_memory)
                            
                            # Use conservative max_features for low-memory VM
                            max_features = 10000  # Reduced for low-memory Ubuntu VM
                            
                            # Check memory before starting
                            if not check_memory_available(500):
                                if status_text is not None:
                                    status_text.text('Doctor\'s Name: Warning: Low memory. Using conservative settings...')
                                batch_size = min(batch_size, 1000)  # Smaller batches
                            
                            # Fit vectorizer on masterlist once
                            suggest_vectorizer = TfidfVectorizer(
                                analyzer='char',
                                ngram_range=(2, 3),
                                min_df=1,
                                lowercase=True,
                                max_features=max_features  # Conservative limit for low-memory VM
                            )
                            master_suggest_matrix = suggest_vectorizer.fit_transform(master_suggest_nonempty)
                            # Convert to float32 to reduce memory usage (halves memory needed)
                            master_suggest_matrix = master_suggest_matrix.astype('float32')
                            
                            # Use NearestNeighbors (fit once on master)
                            # For very large datasets, use single core to reduce memory
                            num_unmatched = len(unmatched_names_clean)
                            if num_unmatched > 100000:
                                optimal_n_jobs = 1  # Force single core for very large datasets
                            else:
                                optimal_n_jobs = get_optimal_n_jobs(min_memory_mb=2000)
                            suggest_nbrs = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=optimal_n_jobs).fit(master_suggest_matrix)
                            
                            # Process unmatched records in batches
                            num_batches = (len(unmatched_names_clean) + batch_size - 1) // batch_size
                            
                            # Ensure batch processing always runs (even if num_batches is 1)
                            if num_batches == 0:
                                num_batches = 1  # Process at least one batch
                            
                            for batch_idx in range(num_batches):
                                # Check memory before each batch
                                if PSUTIL_AVAILABLE and batch_idx > 0:
                                    try:
                                        memory = psutil.virtual_memory()
                                        if memory.available / 1024 / 1024 < 500:  # Less than 500MB available
                                            cleanup_memory()  # Force cleanup before continuing
                                    except Exception:
                                        pass
                                
                                batch_start = batch_idx * batch_size
                                batch_end = min(batch_start + batch_size, len(unmatched_names_clean))
                                batch_names = unmatched_names_clean[batch_start:batch_end]
                                
                                # Transform batch
                                unmatched_suggest_batch_matrix = suggest_vectorizer.transform(batch_names)
                                # Convert to float32 to reduce memory usage
                                unmatched_suggest_batch_matrix = unmatched_suggest_batch_matrix.astype('float32')
                                
                                # Find nearest neighbors for batch
                                suggest_distances, suggest_idx_mapped = suggest_nbrs.kneighbors(unmatched_suggest_batch_matrix)
                                
                                # Store results
                                for j, (suggest_dist, suggest_idx) in enumerate(zip(suggest_distances, suggest_idx_mapped)):
                                    global_idx = batch_start + j
                                    suggest_similarities[global_idx] = 1 - suggest_dist[0]
                                    suggest_match_indices[global_idx] = suggest_indices_map[suggest_idx[0]] if suggest_idx[0] < len(suggest_indices_map) else 0
                                
                                # Cleanup batch matrix immediately
                                del unmatched_suggest_batch_matrix, suggest_distances, suggest_idx_mapped
                                cleanup_memory()  # Cleanup after EVERY batch for large datasets
                            
                            # Cleanup master matrices
                            del suggest_vectorizer, master_suggest_matrix, suggest_nbrs
                            cleanup_memory()
                        except MemoryError:
                            if status_text is not None:
                                status_text.text('Doctor\'s Name: Memory error during md_suggest TF-IDF. Continuing...')
                            # Set all similarities to 0
                            suggest_similarities = [0.0] * len(unmatched_names_clean)
                            suggest_match_indices = [0] * len(unmatched_names_clean)
                
                # Process name matches - use the best similarity from md_official_name or md_suggest
                # Then apply cascading matching: PTR first, then Address if PTR doesn't match
                for i in range(len(unmatched_names_clean)):
                    official_similarity = official_similarities[i]
                    suggest_similarity = suggest_similarities[i]
                    
                    unmatched_name_clean = unmatched_names_clean[i]
                    official_master_idx = official_match_indices[i]
                    suggest_master_idx = suggest_match_indices[i]
                    
                    master_name_clean = master_names_clean[official_master_idx] if official_master_idx < len(master_names_clean) else ''
                    master_suggest_clean_val = master_suggest_clean[suggest_master_idx] if suggest_master_idx < len(master_suggest_clean) else ''
                    
                    # Use the higher similarity score (best match from either md_official_name or md_suggest)
                    similarity = max(official_similarity, suggest_similarity)
                    # Use the index that gave the better match
                    best_master_idx = official_master_idx if official_similarity >= suggest_similarity else suggest_master_idx
                    
                    # Additional check: substring match with both md_official_name and md_suggest
                    is_substring_match_official = (unmatched_name_clean in master_name_clean) and len(unmatched_name_clean) >= 4 if master_name_clean else False
                    is_substring_match_suggest = (unmatched_name_clean in master_suggest_clean_val) and len(unmatched_name_clean) >= 4 if master_suggest_clean_val else False
                    is_substring_match = is_substring_match_official or is_substring_match_suggest
                    
                    name_match = (similarity >= name_threshold or is_substring_match)
                    
                    if name_match:
                        try:
                            original_idx = unmatched_indices_clean[i]
                            master_idx = master_indices[best_master_idx]  # Use best_master_idx to get original masterlist index
                            master_row = masterlist_work.loc[master_idx]
                            
                            # Cascading matching logic:
                            # 1. Doctor name matched with md_suggest ✓
                            # 2. Check PTR if enabled
                            # 3. If PTR doesn't match, check Address (md_add_1 = Address1)
                            
                            should_match = True
                            
                            # Step 2: Validate PTR if enabled
                            if use_ptr_no and ptr_no_col:
                                unmatched_ptr = extract_digits_normalized(unmatched_df.loc[original_idx, ptr_no_col]) if original_idx in unmatched_df.index else ''
                                master_ptr = extract_digits_normalized(master_row['md_ptrs']) if 'md_ptrs' in master_row.index and pd.notna(master_row['md_ptrs']) else ''
                                
                                if unmatched_ptr and master_ptr:
                                    # Both have PTR - check if they match
                                    if unmatched_ptr != master_ptr:
                                        # PTR doesn't match - proceed to Step 3: Address matching
                                        should_match = False
                                        
                                        # Step 3: Check Address matching (md_add_1 = Address1)
                                        address_col = None
                                        for col in ['Address1', 'Address', 'address1', 'ADDRESS1']:
                                            if col in unmatched_df.columns:
                                                address_col = col
                                                break
                                        
                                        if address_col and 'md_add_1' in master_row.index:
                                            unmatched_address = str(unmatched_df.loc[original_idx, address_col]).strip().upper() if original_idx in unmatched_df.index and pd.notna(unmatched_df.loc[original_idx, address_col]) else ''
                                            master_address = str(master_row['md_add_1']).strip().upper() if pd.notna(master_row['md_add_1']) else ''
                                            
                                            # Check if addresses match (exact or substring)
                                            if unmatched_address and master_address:
                                                # Exact match or substring match
                                                if unmatched_address == master_address or unmatched_address in master_address or master_address in unmatched_address:
                                                    should_match = True  # Address matches, proceed with match
                                                else:
                                                    # Try partial matching (check if key words match)
                                                    unmatched_words = set(unmatched_address.split())
                                                    master_words = set(master_address.split())
                                                    # If at least 2 words match, consider it a match
                                                    if len(unmatched_words & master_words) >= 2:
                                                        should_match = True
                                                    else:
                                                        should_match = False
                                            else:
                                                # No address data - skip this match
                                                should_match = False
                                        else:
                                            # No address column or md_add_1 - skip this match
                                            should_match = False
                                elif unmatched_ptr or master_ptr:
                                    # One has PTR, one doesn't - check address as fallback
                                    should_match = False
                                    
                                    # Step 3: Check Address matching
                                    address_col = None
                                    for col in ['Address1', 'Address', 'address1', 'ADDRESS1']:
                                        if col in unmatched_df.columns:
                                            address_col = col
                                            break
                                    
                                    if address_col and 'md_add_1' in master_row.index:
                                        unmatched_address = str(unmatched_df.loc[original_idx, address_col]).strip().upper() if original_idx in unmatched_df.index and pd.notna(unmatched_df.loc[original_idx, address_col]) else ''
                                        master_address = str(master_row['md_add_1']).strip().upper() if pd.notna(master_row['md_add_1']) else ''
                                        
                                        if unmatched_address and master_address:
                                            if unmatched_address == master_address or unmatched_address in master_address or master_address in unmatched_address:
                                                should_match = True
                                            else:
                                                unmatched_words = set(unmatched_address.split())
                                                master_words = set(master_address.split())
                                                if len(unmatched_words & master_words) >= 2:
                                                    should_match = True
                                                else:
                                                    should_match = False
                                        else:
                                            should_match = False
                                    else:
                                        should_match = False
                            
                            # Update the record only if all matching criteria are met (Official Matches: fill DOCTOR_CODE, CUSTOMER_CODE)
                            if should_match:
                                df_work.at[original_idx, 'md_official_name'] = master_row['md_official_name']
                                # [PTR Final] from masterlist ptr_final/md_ptrs_final consistently
                                _ptr_val = master_row.get('ptr_final', '') or master_row.get('md_ptrs_final', '') or master_row.get('md_ptrs', '')
                                df_work.at[original_idx, 'md_ptrs'] = str(_ptr_val).strip() if pd.notna(_ptr_val) and str(_ptr_val).strip() else ''
                                if 'DOCTOR_CODE' in master_row.index and pd.notna(master_row.get('DOCTOR_CODE')):
                                    df_work.at[original_idx, 'DOCTOR_CODE'] = str(master_row['DOCTOR_CODE']).strip()
                                if 'CUSTOMER_CODE' in master_row.index and pd.notna(master_row.get('CUSTOMER_CODE')):
                                    df_work.at[original_idx, 'CUSTOMER_CODE'] = str(master_row['CUSTOMER_CODE']).strip()
                                updated_count += 1
                                doctor_name_matched_count += 1  # Track doctor name matches
                        except (IndexError, KeyError):
                            continue
            
            # Completion status for doctor name matching
            if status_text is not None:
                completion_msg = f'✅ AI Matching by Doctor Name completed! {doctor_name_matched_count} records matched via doctor name.'
                status_text.text(completion_msg)
                # Also log to console for debugging
                print(f"[STATUS] {completion_msg}")
                time.sleep(1.0)  # Longer pause to ensure message is visible
        else:
            # Doctor name matching is disabled or column not found
            if not use_doctor_name:
                if status_text is not None:
                    status_text.text('⚠️ Doctor name matching is disabled.')
            elif not doctor_name_col:
                if status_text is not None:
                    status_text.text('⚠️ Doctor Name column not found in data.')
        
        # ===== MATCHING BY PTR NUMBER (if enabled) =====
        if use_ptr_no and ptr_no_col and 'md_ptrs' in masterlist_work.columns:
            if progress_bar is not None:
                progress_bar.progress(0.92)
            if status_text is not None:
                status_text.text('TF-IDF AI matching by PTR number...')
                time.sleep(0.5)  # Allow UI to refresh and show the message
            
            # Get unmatched records that still have blank md_official_name and md_ptrs
            still_unmatched_mask = ((df_work['md_official_name'] == '') | (df_work['md_official_name'].isna())) & \
                                  ((df_work['md_ptrs'] == '') | (df_work['md_ptrs'].isna()))
            still_unmatched_df = df_work[still_unmatched_mask].copy()
            
            if not still_unmatched_df.empty:
                # Prepare PTR numbers - normalize by removing leading zeros
                unmatched_ptrs_list = still_unmatched_df[ptr_no_col].apply(extract_digits_normalized).tolist()
                master_ptrs_list = masterlist_work['md_ptrs'].apply(extract_digits_normalized).tolist()
                
                # Filter out empty PTRs but keep track of original indices
                still_unmatched_indices = still_unmatched_df.index.tolist()
                unmatched_ptrs_clean = []
                unmatched_ptr_indices_clean = []
                
                for i, ptr in enumerate(unmatched_ptrs_list):
                    if ptr and len(ptr) >= 4:  # Only include PTRs with at least 4 digits
                        unmatched_ptrs_clean.append(ptr)
                        unmatched_ptr_indices_clean.append(still_unmatched_indices[i])
                
                # Filter masterlist PTRs and keep track of indices
                master_ptr_indices = masterlist_work.index.tolist()
                master_ptrs_clean = []
                master_ptr_indices_clean = []
                
                for i, ptr in enumerate(master_ptrs_list):
                    if ptr and len(ptr) >= 4:  # Only include PTRs with at least 4 digits
                        master_ptrs_clean.append(ptr)
                        master_ptr_indices_clean.append(master_ptr_indices[i])
                
                if unmatched_ptrs_clean and master_ptrs_clean:
                    # First, try exact PTR matching (after normalization) - this is faster and more accurate
                    # Create a dictionary for fast exact lookup
                    ptr_to_master_idx = {}
                    for i, ptr in enumerate(master_ptrs_clean):
                        master_idx = master_ptr_indices_clean[i]
                        if ptr not in ptr_to_master_idx:
                            ptr_to_master_idx[ptr] = []
                        ptr_to_master_idx[ptr].append(master_idx)
                    
                    # Process exact PTR matches first
                    exact_matched_indices = set()
                    for i, unmatched_ptr in enumerate(unmatched_ptrs_clean):
                        if unmatched_ptr in ptr_to_master_idx:
                            # Exact match found - check doctor name similarity if enabled
                            original_idx = unmatched_ptr_indices_clean[i]
                            master_idx_candidates = ptr_to_master_idx[unmatched_ptr]
                            
                            # If doctor name matching is enabled, find best match by name similarity
                            best_master_idx = None
                            best_name_similarity = 0.0
                            
                            if use_doctor_name and doctor_name_col:
                                unmatched_doc_name = clean_name(still_unmatched_df.loc[original_idx, doctor_name_col]) if original_idx in still_unmatched_df.index else ''
                                
                                for master_idx_candidate in master_idx_candidates:
                                    master_row_candidate = masterlist_work.loc[master_idx_candidate]
                                    # Use md_suggest for comparison (data from store branches) - more accurate for matching
                                    master_doc_name = ''
                                    if 'md_suggest' in master_row_candidate.index and pd.notna(master_row_candidate['md_suggest']):
                                        master_doc_name = clean_name(master_row_candidate['md_suggest'])
                                    elif 'md_official_name' in master_row_candidate.index and pd.notna(master_row_candidate['md_official_name']):
                                        # Fallback to md_official_name if md_suggest is not available
                                        master_doc_name = clean_name(master_row_candidate['md_official_name'])
                                    
                                    if unmatched_doc_name and master_doc_name:
                                        name_similarity = SequenceMatcher(None, unmatched_doc_name, master_doc_name).ratio()
                                        if name_similarity > best_name_similarity:
                                            best_name_similarity = name_similarity
                                            best_master_idx = master_idx_candidate
                                
                                # If name similarity check is enabled, use lower threshold for exact PTR matches
                                # Since PTR matches exactly, we can be more lenient with name matching
                                if best_name_similarity < doctor_similarity_threshold_exact_ptr:
                                    continue  # Skip if doctor names don't match well enough
                            else:
                                # No doctor name check - just take first match
                                best_master_idx = master_idx_candidates[0]
                            
                            if best_master_idx is not None:
                                try:
                                    master_row = masterlist_work.loc[best_master_idx]
                                    # [PTR Final] from masterlist ptr_final/md_ptrs_final consistently
                                    _ptr_val = master_row.get('ptr_final', '') or master_row.get('md_ptrs_final', '') or master_row.get('md_ptrs', '')
                                    _ptr_val = str(_ptr_val).strip() if pd.notna(_ptr_val) and str(_ptr_val).strip() else ''
                                    # Update the record (Official Matches: fill DOCTOR_CODE, CUSTOMER_CODE)
                                    df_work.at[original_idx, 'md_official_name'] = master_row['md_official_name']
                                    df_work.at[original_idx, 'md_ptrs'] = _ptr_val
                                    if 'DOCTOR_CODE' in master_row.index and pd.notna(master_row.get('DOCTOR_CODE')):
                                        df_work.at[original_idx, 'DOCTOR_CODE'] = str(master_row['DOCTOR_CODE']).strip()
                                    if 'CUSTOMER_CODE' in master_row.index and pd.notna(master_row.get('CUSTOMER_CODE')):
                                        df_work.at[original_idx, 'CUSTOMER_CODE'] = str(master_row['CUSTOMER_CODE']).strip()
                                    updated_count += 1
                                    ptr_matched_count += 1  # Track PTR matches
                                    exact_matched_indices.add(i)  # Mark as matched
                                except (IndexError, KeyError):
                                    continue
                    
                    # Then, use TF-IDF for non-exact matches (similar but not identical PTRs)
                    # Filter out already matched records
                    unmatched_ptrs_for_tfidf = []
                    unmatched_ptr_indices_for_tfidf = []
                    for i, ptr in enumerate(unmatched_ptrs_clean):
                        if i not in exact_matched_indices:
                            unmatched_ptrs_for_tfidf.append(ptr)
                            unmatched_ptr_indices_for_tfidf.append(unmatched_ptr_indices_clean[i])
                    
                    if unmatched_ptrs_for_tfidf and master_ptrs_clean:
                        try:
                            # Vectorize PTR numbers using character n-grams
                            ptr_vectorizer = TfidfVectorizer(
                                analyzer='char',
                                ngram_range=(2, 4),  # Slightly longer ngrams for numbers
                                min_df=1,
                                lowercase=False,
                                max_features=10000  # Reduced for low-memory Ubuntu VM
                            )
                            master_ptr_matrix = ptr_vectorizer.fit_transform(master_ptrs_clean)
                            # Convert to float32 to reduce memory usage (halves memory needed)
                            master_ptr_matrix = master_ptr_matrix.astype('float32')
                            unmatched_ptr_matrix = ptr_vectorizer.transform(unmatched_ptrs_for_tfidf)
                            # Convert to float32 to reduce memory usage
                            unmatched_ptr_matrix = unmatched_ptr_matrix.astype('float32')
                            
                            # Cleanup vectorizer
                            del ptr_vectorizer
                            cleanup_memory()
                            
                            # Use NearestNeighbors for PTR matching
                            optimal_n_jobs = get_optimal_n_jobs(min_memory_mb=1500)
                            ptr_nbrs = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=optimal_n_jobs).fit(master_ptr_matrix)
                            ptr_distances, ptr_indices = ptr_nbrs.kneighbors(unmatched_ptr_matrix)
                            
                            # Process TF-IDF PTR matches
                            for i, (dist, idx) in enumerate(zip(ptr_distances, ptr_indices)):
                                ptr_similarity = 1 - dist[0]
                                
                                if ptr_similarity >= ptr_threshold:
                                    try:
                                        original_idx = unmatched_ptr_indices_for_tfidf[i]
                                        master_idx = master_ptr_indices_clean[idx[0]]
                                        master_row = masterlist_work.loc[master_idx]
                                        
                                        # When matching by PTR, also validate doctor name similarity
                                        if use_doctor_name and doctor_name_col:
                                            unmatched_doc_name = clean_name(still_unmatched_df.loc[original_idx, doctor_name_col]) if original_idx in still_unmatched_df.index else ''
                                            # Use md_suggest for comparison (data from store branches) - more accurate for matching
                                            master_doc_name = ''
                                            if 'md_suggest' in master_row.index and pd.notna(master_row['md_suggest']):
                                                master_doc_name = clean_name(master_row['md_suggest'])
                                            elif 'md_official_name' in master_row.index and pd.notna(master_row['md_official_name']):
                                                # Fallback to md_official_name if md_suggest is not available
                                                master_doc_name = clean_name(master_row['md_official_name'])
                                            
                                            if unmatched_doc_name and master_doc_name:
                                                # Calculate name similarity using SequenceMatcher
                                                name_similarity = SequenceMatcher(None, unmatched_doc_name, master_doc_name).ratio()
                                                
                                                # Require minimum doctor name similarity
                                                if name_similarity < doctor_similarity_threshold:
                                                    continue  # Skip if doctor names don't match well enough
                                        
                                        # [PTR Final] from masterlist ptr_final/md_ptrs_final consistently
                                        _ptr_val = master_row.get('ptr_final', '') or master_row.get('md_ptrs_final', '') or master_row.get('md_ptrs', '')
                                        _ptr_val = str(_ptr_val).strip() if pd.notna(_ptr_val) and str(_ptr_val).strip() else ''
                                        # Update the record (Official Matches: fill DOCTOR_CODE, CUSTOMER_CODE)
                                        df_work.at[original_idx, 'md_official_name'] = master_row['md_official_name']
                                        df_work.at[original_idx, 'md_ptrs'] = _ptr_val
                                        if 'DOCTOR_CODE' in master_row.index and pd.notna(master_row.get('DOCTOR_CODE')):
                                            df_work.at[original_idx, 'DOCTOR_CODE'] = str(master_row['DOCTOR_CODE']).strip()
                                        if 'CUSTOMER_CODE' in master_row.index and pd.notna(master_row.get('CUSTOMER_CODE')):
                                            df_work.at[original_idx, 'CUSTOMER_CODE'] = str(master_row['CUSTOMER_CODE']).strip()
                                        updated_count += 1
                                        ptr_matched_count += 1  # Track PTR matches
                                    except (IndexError, KeyError):
                                        continue
                            
                            # Cleanup matrices
                            del master_ptr_matrix, unmatched_ptr_matrix, ptr_nbrs, ptr_distances, ptr_indices
                            cleanup_memory()
                        except MemoryError:
                            if status_text is not None:
                                status_text.text('Memory error during PTR TF-IDF. Skipping...')
                            # Continue without TF-IDF matching
                            pass
            
            # Completion status for PTR matching
            if status_text is not None:
                completion_msg = f'✅ AI Matching by PTR Number completed! {ptr_matched_count} records matched via PTR number.'
                status_text.text(completion_msg)
                # Also log to console for debugging
                print(f"[STATUS] {completion_msg}")
                time.sleep(1.0)  # Longer pause to ensure message is visible
        else:
            # PTR matching is disabled or column not found
            if not use_ptr_no:
                if status_text is not None:
                    status_text.text('⚠️ PTR number matching is disabled.')
            elif not ptr_no_col:
                if status_text is not None:
                    status_text.text('⚠️ PTR No column not found in data.')
            elif 'md_ptrs' not in masterlist_work.columns:
                if status_text is not None:
                    status_text.text('⚠️ md_ptrs column not found in masterlist.')
        
        if progress_bar is not None:
            progress_bar.progress(1.0)
        if status_text is not None:
            # Show comprehensive summary of all matching results
            summary_parts = []
            if use_doctor_name and doctor_name_col:
                summary_parts.append(f'Doctor Name: {doctor_name_matched_count}')
            if use_ptr_no and ptr_no_col:
                summary_parts.append(f'PTR Number: {ptr_matched_count}')
            
            # Create comprehensive final summary message
            if summary_parts:
                summary_text = ' | '.join(summary_parts)
                final_msg = f'✅ TF-IDF AI Matching COMPLETED!\n\nTotal Records Matched: {updated_count}\n\nBreakdown by Method:\n  • {summary_text.replace(" | ", "\n  • ")}'
            else:
                final_msg = f'✅ TF-IDF AI matching completed! Total: {updated_count} records matched.'
            
            # Show final summary with all details
            status_text.text(final_msg)
            # Also log to console for debugging
            print(f"[STATUS] {final_msg}")
            # Force UI refresh by updating progress bar
            if progress_bar is not None:
                progress_bar.progress(1.0)
            time.sleep(1.0)  # Longer pause to ensure final message is visible
        
        # Final cleanup
        cleanup_memory()
    
    except MemoryError as e:
        # If TF-IDF fails due to memory, log error but don't fail completely
        if status_text is not None:
            status_text.text(f'Memory error during TF-IDF AI matching: {str(e)}. Returning partial results...')
        cleanup_memory()
        # CRITICAL: Restore original Amount column (NEVER modified during matching)
        if original_amount is not None and len(original_amount) == len(df_work):
            df_work['Amount'] = original_amount.values  # Restore original Amount values
        completion_info = {
            'doctor_name_matched': doctor_name_matched_count,
            'ptr_matched': ptr_matched_count,
            'ppe_doctors_matched': ppe_matched_count,
            'total_matched': updated_count,
            'use_doctor_name': use_doctor_name and doctor_name_col is not None,
            'use_ptr_no': use_ptr_no and ptr_no_col is not None
        }
        return df_work, updated_count, completion_info
    except Exception as e:
        # If TF-IDF fails, log error but don't fail completely
        if status_text is not None:
            status_text.text(f'TF-IDF AI matching encountered an error: {str(e)}. Continuing...')
        cleanup_memory()
        # CRITICAL: Restore original Amount column (NEVER modified during matching)
        if original_amount is not None and len(original_amount) == len(df_work):
            df_work['Amount'] = original_amount.values  # Restore original Amount values
        completion_info = {
            'doctor_name_matched': doctor_name_matched_count,
            'ptr_matched': ptr_matched_count,
            'ppe_doctors_matched': ppe_matched_count,
            'total_matched': updated_count,
            'use_doctor_name': use_doctor_name and doctor_name_col is not None,
            'use_ptr_no': use_ptr_no and ptr_no_col is not None
        }
        return df_work, updated_count, completion_info
    
    # ===== PPE DOCTORS MATCHING (Additional AI Matching) =====
    # Match against ppe_doctors data for records still unmatched
    ppe_matched_count = 0
    if use_doctor_name and doctor_name_col:
        try:
            # Load ppe_doctors data
            ppe_doctors_df = load_ppe_doctors_csv()
            
            if ppe_doctors_df is not None and not ppe_doctors_df.empty:
                if status_text is not None:
                    status_text.text('Performing PPE Doctors AI matching for unmatched records...')
                if progress_bar is not None:
                    progress_bar.progress(0.97)
                
                # Ensure DOCTOR_CODE and CUSTOMER_CODE columns exist
                if 'DOCTOR_CODE' not in df_work.columns:
                    df_work['DOCTOR_CODE'] = ''
                if 'CUSTOMER_CODE' not in df_work.columns:
                    df_work['CUSTOMER_CODE'] = ''
                
                # Find records that are still unmatched (no md_official_name filled)
                still_unmatched_mask = ((df_work['md_official_name'] == '') | (df_work['md_official_name'].isna()))
                still_unmatched_df = df_work[still_unmatched_mask].copy()
                
                if not still_unmatched_df.empty and 'md_official_name_clean' in ppe_doctors_df.columns:
                    # Clean doctor names from unmatched data
                    still_unmatched_df['doctor_name_clean'] = still_unmatched_df[doctor_name_col].astype(str).str.strip().str.lower()
                    
                    # OPTIMIZED: Create mapping dictionary using vectorized operations (much faster than iterrows)
                    # Take first match if duplicates exist (drop_duplicates keeps first)
                    ppe_doctors_unique = ppe_doctors_df.drop_duplicates(subset=['md_official_name_clean'], keep='first')
                    ppe_mapping = (
                        ppe_doctors_unique
                        .set_index('md_official_name_clean')[['md_official_name', 'DOCTOR_CODE', 'CUSTOMER_CODE']]
                        .to_dict('index')
                    )
                    
                    # Match using TF-IDF similarity (similar to main matching)
                    if len(still_unmatched_df) > 0 and len(ppe_doctors_df) > 0:
                        # Track matched indices to avoid double-counting
                        matched_indices = set()
                        
                        # Get unique doctor names from unmatched records (to avoid processing duplicates)
                        unique_unmatched_names = still_unmatched_df['doctor_name_clean'].drop_duplicates().tolist()
                        ppe_names_list = ppe_doctors_df['md_official_name_clean'].tolist()
                        
                        # Use TF-IDF for matching (similar to main matching logic)
                        try:
                            # Filter out empty names
                            unique_unmatched_names_clean = [n for n in unique_unmatched_names if n and len(n) >= 3]
                            ppe_names_clean = [n for n in ppe_names_list if n and len(n) >= 3]
                            
                            if unique_unmatched_names_clean and ppe_names_clean:
                                # Vectorize using character n-grams
                                vectorizer = TfidfVectorizer(
                                    analyzer='char',
                                    ngram_range=(2, 3),
                                    min_df=1,
                                    lowercase=True,
                                    max_features=10000
                                )
                                ppe_matrix = vectorizer.fit_transform(ppe_names_clean)
                                ppe_matrix = ppe_matrix.astype('float32')
                                
                                # Use NearestNeighbors
                                optimal_n_jobs = get_optimal_n_jobs(min_memory_mb=1500)
                                ppe_nbrs = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=optimal_n_jobs).fit(ppe_matrix)
                                
                                # Process in batches
                                batch_size = min(1000, len(unique_unmatched_names_clean))
                                num_batches = (len(unique_unmatched_names_clean) + batch_size - 1) // batch_size
                                
                                for batch_idx in range(num_batches):
                                    batch_start = batch_idx * batch_size
                                    batch_end = min(batch_start + batch_size, len(unique_unmatched_names_clean))
                                    batch_names = unique_unmatched_names_clean[batch_start:batch_end]
                                    
                                    unmatched_batch_matrix = vectorizer.transform(batch_names)
                                    unmatched_batch_matrix = unmatched_batch_matrix.astype('float32')
                                    
                                    distances, indices = ppe_nbrs.kneighbors(unmatched_batch_matrix)
                                    
                                    # Process matches - process each unique name only once
                                    for j, (dist, idx) in enumerate(zip(distances, indices)):
                                        similarity = 1 - dist[0]
                                        ppe_idx = indices[j][0]
                                        
                                        # Use name_threshold for matching (same as main matching)
                                        if similarity >= name_threshold:
                                            global_idx = batch_start + j
                                            unmatched_name_clean = unique_unmatched_names_clean[global_idx]
                                            
                                            # Find ALL records with this doctor name that haven't been matched yet
                                            matching_rows = still_unmatched_df[
                                                (still_unmatched_df['doctor_name_clean'] == unmatched_name_clean) &
                                                (~still_unmatched_df.index.isin(matched_indices))
                                            ]
                                            
                                            if not matching_rows.empty:
                                                # Get matched ppe_doctors data
                                                matched_ppe_name_clean = ppe_names_clean[ppe_idx]
                                                if matched_ppe_name_clean in ppe_mapping:
                                                    ppe_data = ppe_mapping[matched_ppe_name_clean]
                                                    
                                                    # Update all records with this doctor name (only once per record)
                                                    for orig_idx in matching_rows.index:
                                                        if orig_idx not in matched_indices:
                                                            # [PTR Final] from masterlist ptr_final consistently (lookup by md_official_name)
                                                            ppe_name_norm = str(ppe_data['md_official_name']).strip().upper() if pd.notna(ppe_data['md_official_name']) else ''
                                                            ptr_val = ptr_final_lookup.get(ppe_name_norm, '') if ppe_name_norm else ''
                                                            if not ptr_val and ptr_no_col and orig_idx in df_work.index:
                                                                ptr_val = str(df_work.loc[orig_idx, ptr_no_col]).strip() if pd.notna(df_work.loc[orig_idx, ptr_no_col]) else ''
                                                            
                                                            # Update the record
                                                            df_work.at[orig_idx, 'md_official_name'] = ppe_data['md_official_name']
                                                            df_work.at[orig_idx, 'md_ptrs'] = ptr_val
                                                            df_work.at[orig_idx, 'DOCTOR_CODE'] = ppe_data['DOCTOR_CODE']
                                                            df_work.at[orig_idx, 'CUSTOMER_CODE'] = ppe_data['CUSTOMER_CODE']
                                                            
                                                            # Mark as matched to avoid double-counting
                                                            matched_indices.add(orig_idx)
                                                            ppe_matched_count += 1
                                                            updated_count += 1
                                
                                # Cleanup
                                del vectorizer, ppe_matrix, ppe_nbrs, unmatched_batch_matrix, distances, indices
                                cleanup_memory()
                        except Exception as e:
                            # If TF-IDF fails, fall back to simple similarity matching
                            if status_text is not None:
                                status_text.text(f'PPE Doctors: TF-IDF failed, using simple matching. Error: {str(e)}')
                            
                            # Simple similarity matching as fallback (only process unmatched records)
                            for idx in still_unmatched_df.index:
                                # Skip if already matched
                                if idx in matched_indices:
                                    continue
                                
                                doctor_name_clean = still_unmatched_df.loc[idx, 'doctor_name_clean']
                                
                                if doctor_name_clean in ppe_mapping:
                                    ppe_data = ppe_mapping[doctor_name_clean]
                                    
                                    # [PTR Final] from masterlist ptr_final consistently (lookup by md_official_name)
                                    ppe_name_norm = str(ppe_data['md_official_name']).strip().upper() if pd.notna(ppe_data['md_official_name']) else ''
                                    ptr_val = ptr_final_lookup.get(ppe_name_norm, '') if ppe_name_norm else ''
                                    if not ptr_val and ptr_no_col and idx in df_work.index:
                                        ptr_val = str(df_work.loc[idx, ptr_no_col]).strip() if pd.notna(df_work.loc[idx, ptr_no_col]) else ''
                                    
                                    # Update the record
                                    df_work.at[idx, 'md_official_name'] = ppe_data['md_official_name']
                                    df_work.at[idx, 'md_ptrs'] = ptr_val
                                    df_work.at[idx, 'DOCTOR_CODE'] = ppe_data['DOCTOR_CODE']
                                    df_work.at[idx, 'CUSTOMER_CODE'] = ppe_data['CUSTOMER_CODE']
                                    
                                    # Mark as matched to avoid double-counting
                                    matched_indices.add(idx)
                                    ppe_matched_count += 1
                                    updated_count += 1
                    
                    if status_text is not None and ppe_matched_count > 0:
                        status_text.text(f'✅ PPE Doctors AI Matching completed! {ppe_matched_count} records matched via PPE Doctors.')
        except Exception as e:
            if status_text is not None:
                status_text.text(f'⚠️ PPE Doctors matching encountered an error: {str(e)}. Continuing...')
            logger.error(f"PPE Doctors matching error: {str(e)}\n{traceback.format_exc()}")
    
    # ===== WORD-BASED QUICK SUGGEST MATCHING (Final AI Matching Step) =====
    # This is the LAST matching process - only processes records with blank md_official_name
    # Fills quick_suggest_name instead of md_official_name
    quick_suggest_matched_count = 0
    tfidf_reference_matched_count = 0  # Track TF-IDF matches using reference data only
    
    # Ensure doctor_name_col is available at function level for both Quick Suggest and TF-IDF Reference matching
    # (doctor_name_col was already defined at line 2520, but ensure it's available here)
    if doctor_name_col is None:
        for col in ['Doctor Name', 'doctor_name', 'DOCTOR_NAME']:
            if col in df_work.columns:
                doctor_name_col = col
                break
    
    if use_quick_suggest_matching:
        try:
            # Ensure quick_suggest_name column exists
            if 'quick_suggest_name' not in df_work.columns:
                df_work['quick_suggest_name'] = ''
            
            # Ensure suggested_name column exists (for highly considered matches)
            if 'suggested_name' not in df_work.columns:
                df_work['suggested_name'] = ''
            
            # Ensure DOCTOR_CODE and CUSTOMER_CODE columns exist
            if 'DOCTOR_CODE' not in df_work.columns:
                df_work['DOCTOR_CODE'] = ''
            if 'CUSTOMER_CODE' not in df_work.columns:
                df_work['CUSTOMER_CODE'] = ''
            
            # Find records that are still unmatched (md_official_name is blank)
            still_unmatched_mask = ((df_work['md_official_name'] == '') | (df_work['md_official_name'].isna()))
            still_unmatched_df = df_work[still_unmatched_mask].copy()
            
            if not still_unmatched_df.empty and masterlist_df is not None and not masterlist_df.empty:
                if status_text is not None:
                    status_text.text('Performing word-based quick suggest matching (final step)...')
                if progress_bar is not None:
                    progress_bar.progress(0.99)
                
                # Get required column names (doctor_name_col already defined at function level, but verify it exists in still_unmatched_df)
                if doctor_name_col is None or doctor_name_col not in still_unmatched_df.columns:
                    # Re-check if column exists with different name
                    for col in ['Doctor Name', 'doctor_name', 'DOCTOR_NAME']:
                        if col in still_unmatched_df.columns:
                            doctor_name_col = col
                            break
                
                ptr_no_col = None
                for col in ['PTR No', 'ptr_no', 'PTR_NO', 'PTR']:
                    if col in still_unmatched_df.columns:
                        ptr_no_col = col
                        break
                
                address_col = None
                for col in ['Address1', 'Address', 'address1', 'ADDRESS1']:
                    if col in still_unmatched_df.columns:
                        address_col = col
                        break
                
                branch_code_col = None
                for col in ['Branch Code', 'branch_code', 'BRANCH_CODE']:
                    if col in still_unmatched_df.columns:
                        branch_code_col = col
                        break
                
                # Check if masterlist has required columns
                if doctor_name_col and 'md_official_name' in masterlist_df.columns:
                    # Pre-process masterlist: create clean versions
                    masterlist_work = masterlist_df.copy()
                    masterlist_work['md_official_name_clean'] = masterlist_work['md_official_name'].astype(str).str.strip().str.lower()
                    
                    # ===== LOAD AND MERGE DOCTORS REFERENCE DATA =====
                    # Track original masterlist length to identify reference data matches
                    original_masterlist_length = len(masterlist_work)
                    reference_df = load_doctors_reference_csv()
                    if reference_df is not None and not reference_df.empty:
                        if status_text is not None:
                            status_text.text(f'Loading reference data ({len(reference_df)} records)...')
                        
                        # Map reference columns to masterlist format
                        reference_mapped = pd.DataFrame()
                        reference_mapped['md_official_name'] = reference_df['Doctor Name'].astype(str).fillna('').str.strip()
                        reference_mapped['md_add_1'] = reference_df['Address'].astype(str).fillna('').str.strip()
                        reference_mapped['md_ptrs'] = reference_df['PTR'].astype(str).fillna('').str.strip()
                        reference_mapped['md_official_name_clean'] = reference_mapped['md_official_name'].str.lower()
                        
                        # CRITICAL: Remove rows where md_official_name (Doctor Name) is empty or null
                        # Doctor Name is required - filter out any blank/null values
                        initial_ref_count = len(reference_mapped)
                        reference_mapped = reference_mapped[
                            (reference_mapped['md_official_name'] != '') & 
                            (reference_mapped['md_official_name'].notna()) &
                            (reference_mapped['md_official_name'].str.strip() != '')
                        ]
                        removed_ref_count = initial_ref_count - len(reference_mapped)
                        
                        if removed_ref_count > 0 and status_text is not None:
                            status_text.text(f'Removed {removed_ref_count} reference record(s) with blank/null Doctor Name')
                        
                        # Add empty columns if they don't exist in reference (for compatibility)
                        for col in ['md_suggest', 'DOCTOR_CODE', 'CUSTOMER_CODE']:
                            if col not in reference_mapped.columns:
                                reference_mapped[col] = ''
                        
                        # Combine with masterlist (append reference data)
                        # Reset index to ensure proper indexing after concatenation
                        masterlist_work = masterlist_work.reset_index(drop=True)
                        reference_mapped = reference_mapped.reset_index(drop=True)
                        masterlist_work = pd.concat([masterlist_work, reference_mapped], ignore_index=True)
                        
                        if status_text is not None:
                            status_text.text(f'Reference data merged. Total masterlist records: {len(masterlist_work):,}')
                    else:
                        # No reference data, so all matches will be from masterlist
                        original_masterlist_length = len(masterlist_work)
                    
                    # ===== OPTIMIZATION: BUILD INVERTED INDEX (ONE TIME) =====
                    if status_text is not None:
                        status_text.text('Building word index for fast matching...')
                    
                    # Build inverted index: word -> set of masterlist indices
                    inverted_index = {}  # {word: set of masterlist indices}
                    
                    for master_idx in masterlist_work.index:
                        master_name_clean = masterlist_work.loc[master_idx, 'md_official_name_clean']
                        if pd.notna(master_name_clean) and master_name_clean.strip():
                            # Extract words from masterlist name (same logic as unmatched records)
                            words = re.findall(r'\b\w+\b', master_name_clean)
                            significant_words = {w for w in words if len(w) > 3}
                            
                            # Add to inverted index
                            for word in significant_words:
                                if word not in inverted_index:
                                    inverted_index[word] = set()
                                inverted_index[word].add(master_idx)
                    
                    # Pre-normalize PTR and Address columns for faster validation (OPTIMIZED: vectorized operations)
                    # Vectorized PTR normalization (much faster than apply)
                    masterlist_work['md_ptrs_normalized'] = (
                        masterlist_work['md_ptrs']
                        .astype(str)
                        .str.replace(' ', '', regex=False)
                        .str.replace('-', '', regex=False)
                        .str.replace(r'^0+', '', regex=True)
                        .replace('nan', '')
                        .str.strip()
                    )
                    # Vectorized Address upper casing (much faster than apply)
                    masterlist_work['md_add_1_upper'] = (
                        masterlist_work['md_add_1']
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .replace('NAN', '')
                    )
                    
                    # Build PTR index: normalized_ptr -> set of masterlist indices
                    ptr_index = {}  # {normalized_ptr: set of masterlist indices}
                    for master_idx in masterlist_work.index:
                        ptr_normalized = masterlist_work.loc[master_idx, 'md_ptrs_normalized']
                        if ptr_normalized:
                            if ptr_normalized not in ptr_index:
                                ptr_index[ptr_normalized] = set()
                            ptr_index[ptr_normalized].add(master_idx)
                    
                    # Build Branch Code index: normalized_branch_code -> set of masterlist indices (for fallback matching)
                    branch_code_index = {}  # {normalized_branch_code: set of masterlist indices}
                    if branch_code_col and 'md_b_codes' in masterlist_work.columns:
                        for master_idx in masterlist_work.index:
                            md_b_codes_value = masterlist_work.loc[master_idx, 'md_b_codes']
                            if pd.notna(md_b_codes_value) and str(md_b_codes_value).strip():
                                md_b_codes_str = str(md_b_codes_value).strip()
                                # Extract all numeric branch codes from md_b_codes (handles comma-separated values)
                                branch_codes = re.findall(r'\b(\d+)\b', md_b_codes_str)
                                for branch_code_numeric in branch_codes:
                                    if branch_code_numeric not in branch_code_index:
                                        branch_code_index[branch_code_numeric] = set()
                                    branch_code_index[branch_code_numeric].add(master_idx)
                    
                    if status_text is not None:
                        index_msg = f'Index built: {len(inverted_index)} unique words, {len(ptr_index)} unique PTRs'
                        if branch_code_index:
                            index_msg += f', {len(branch_code_index)} unique Branch Codes'
                        index_msg += '. Starting fast matching...'
                        status_text.text(index_msg)
                    
                    # Process each unmatched record
                    total_unmatched = len(still_unmatched_df)
                    processed = 0
                    
                    for idx in still_unmatched_df.index:
                        if idx not in df_work.index:
                            continue
                        
                        # Skip if already has quick_suggest_name filled
                        if pd.notna(df_work.loc[idx, 'quick_suggest_name']) and str(df_work.loc[idx, 'quick_suggest_name']).strip() != '':
                            continue
                        
                        # Get Doctor Name from app data
                        doctor_name = str(still_unmatched_df.loc[idx, doctor_name_col]).strip() if pd.notna(still_unmatched_df.loc[idx, doctor_name_col]) else ''
                        if not doctor_name:
                            continue
                        
                        # Extract words from Doctor Name (words with > 3 characters)
                        doctor_name_lower = doctor_name.lower()
                        words = re.findall(r'\b\w+\b', doctor_name_lower)
                        significant_words = {w for w in words if len(w) > 3}  # Use set for faster operations
                        
                        if not significant_words:
                            continue
                        
                        # Get PTR No, Address1, and Branch Code from app data
                        app_ptr_no = ''
                        if ptr_no_col and pd.notna(still_unmatched_df.loc[idx, ptr_no_col]):
                            app_ptr_no = str(still_unmatched_df.loc[idx, ptr_no_col]).strip()
                        
                        app_address1 = ''
                        if address_col and pd.notna(still_unmatched_df.loc[idx, address_col]):
                            app_address1 = str(still_unmatched_df.loc[idx, address_col]).strip().upper()
                        
                        app_branch_code = ''
                        app_branch_code_numeric = ''
                        if branch_code_col and pd.notna(still_unmatched_df.loc[idx, branch_code_col]):
                            app_branch_code = str(still_unmatched_df.loc[idx, branch_code_col]).strip()
                            # Extract numeric part of branch code
                            branch_match = re.search(r'(\d+)', app_branch_code)
                            if branch_match:
                                app_branch_code_numeric = branch_match.group(1)
                        
                        # ===== OPTIMIZED: FAST WORD LOOKUP USING INVERTED INDEX (with fuzzy matching) =====
                        candidate_indices = set()
                        # Fuzzy matching threshold for word-level matching (85% similarity)
                        fuzzy_word_threshold = 85
                        
                        for word in significant_words:
                            # First try exact match (fastest - O(1))
                            if word in inverted_index:
                                candidate_indices.update(inverted_index[word])
                            elif RAPIDFUZZ_AVAILABLE:
                                # If exact match fails, try fuzzy matching with token_set_ratio
                                # This handles typos like "ceasar" vs "cesar" or "ermelita" vs "emelita"
                                try:
                                    # Use process.extract for efficient fuzzy matching
                                    # Limit to top 5 matches per word to balance speed and accuracy
                                    similar_words = process.extract(
                                        word,
                                        list(inverted_index.keys()),  # All indexed words
                                        limit=5,  # Top 5 matches per word
                                        scorer=fuzz.token_set_ratio,  # Use token_set_ratio for better word matching
                                        score_cutoff=fuzzy_word_threshold  # Minimum 85% similarity
                                    )
                                    
                                    # Add indices for all similar words found
                                    for similar_word, score, _ in similar_words:
                                        candidate_indices.update(inverted_index[similar_word])
                                except Exception as e:
                                    # If fuzzy matching fails, continue without it (fallback to exact only)
                                    logger.warning(f"Fuzzy matching error for word '{word}': {str(e)}")
                                    pass
                        
                        if not candidate_indices:
                            continue
                        
                        # ===== OPTIMIZED: FAST PTR AND ADDRESS VALIDATION =====
                        # Pre-normalize app PTR and Address once
                        app_ptr_clean = re.sub(r'^0+', '', app_ptr_no.replace(' ', '').replace('-', '')) if app_ptr_no else ''
                        app_address1_upper = app_address1.upper() if app_address1 else ''
                        
                        # Fast PTR lookup using pre-built index
                        ptr_matching_indices = set()
                        ptr_exists = False
                        if app_ptr_clean and app_ptr_clean in ptr_index:
                            ptr_matching_indices = ptr_index[app_ptr_clean] & candidate_indices
                            ptr_exists = len(ptr_matching_indices) > 0
                        
                        # Fast Address lookup (check only candidates)
                        address_matching_indices = set()
                        address_exists = False
                        if app_address1_upper:
                            for master_idx in candidate_indices:
                                master_address = masterlist_work.loc[master_idx, 'md_add_1_upper']
                                if master_address and (app_address1_upper in master_address or master_address in app_address1_upper):
                                    address_matching_indices.add(master_idx)
                            address_exists = len(address_matching_indices) > 0
                        
                        # Fast Branch Code lookup (check only candidates) - for fallback matching
                        branch_code_matching_indices = set()
                        branch_code_exists = False
                        if app_branch_code_numeric and app_branch_code_numeric in branch_code_index:
                            branch_code_matching_indices = branch_code_index[app_branch_code_numeric] & candidate_indices
                            branch_code_exists = len(branch_code_matching_indices) > 0
                        
                        # ===== OPTIMIZED: DETERMINE MATCH CONDITION USING PRE-COMPUTED SETS =====
                        match_condition = None
                        match_found = False
                        matched_idx = None
                        
                        if app_ptr_no and app_address1:
                            # Both PTR No and Address1 are provided
                            if ptr_exists and address_exists:
                                # Check if both exist in SAME row
                                same_row_indices = ptr_matching_indices & address_matching_indices
                                if same_row_indices:
                                    # Both in same row - use first match
                                    matched_idx = next(iter(same_row_indices))
                                    match_found = True
                                    match_condition = 'both'
                                else:
                                    # Both exist but in different rows
                                    match_found = True
                                    match_condition = 'different_rows'
                                    # Prefer PTR match
                                    matched_idx = next(iter(ptr_matching_indices)) if ptr_matching_indices else next(iter(address_matching_indices))
                            else:
                                # Both provided but not both exist - FALLBACK: check Branch Code
                                if branch_code_exists:
                                    match_found = True
                                    match_condition = 'branch_code_fallback'
                                    matched_idx = next(iter(branch_code_matching_indices))
                                else:
                                    # No fallback match - skip this record (strict validation)
                                    continue
                        elif app_ptr_no:
                            # Only PTR No available
                            if ptr_exists:
                                match_found = True
                                match_condition = 'ptr_only'
                                matched_idx = next(iter(ptr_matching_indices))
                            else:
                                # PTR No provided but doesn't exist - FALLBACK: check Branch Code
                                if branch_code_exists:
                                    match_found = True
                                    match_condition = 'branch_code_fallback'
                                    matched_idx = next(iter(branch_code_matching_indices))
                                else:
                                    # No fallback match - skip this record (strict validation)
                                    continue
                        elif app_address1:
                            # Only Address1 available
                            if address_exists:
                                match_found = True
                                match_condition = 'address_only'
                                matched_idx = next(iter(address_matching_indices))
                            else:
                                # Address1 provided but doesn't exist - FALLBACK: check Branch Code
                                if branch_code_exists:
                                    match_found = True
                                    match_condition = 'branch_code_fallback'
                                    matched_idx = next(iter(branch_code_matching_indices))
                                else:
                                    # No fallback match - skip this record (strict validation)
                                    continue
                        else:
                            # Neither PTR No nor Address1 available - FALLBACK: check Branch Code
                            if branch_code_exists:
                                match_found = True
                                match_condition = 'branch_code_fallback'
                                matched_idx = next(iter(branch_code_matching_indices))
                            else:
                                # No fallback match - skip this record (strict validation)
                                continue
                        
                        # If match found, fill fields based on match condition
                        if match_found and matched_idx is not None and match_condition:
                            # Get matched row data (direct access, no copy needed)
                            matched_row = masterlist_work.loc[matched_idx]
                            
                            # Get md_official_name value from masterlist
                            md_official_name_val = str(matched_row['md_official_name']).strip() if pd.notna(matched_row['md_official_name']) else ''
                            
                            if md_official_name_val:
                                # CRITICAL: Set suggested_md based on match source
                                # If match comes from reference data (doctors_reference.csv), set to 'REF'
                                # If match comes from masterlist, set to False
                                # Quick Suggest Matching should never set suggested_md to True
                                if 'suggested_md' in df_work.columns:
                                    # Check if matched_idx is in the reference data range
                                    if matched_idx >= original_masterlist_length:
                                        # Match came from reference data
                                        df_work.at[idx, 'suggested_md'] = 'REF'
                                    else:
                                        # Match came from masterlist
                                        df_work.at[idx, 'suggested_md'] = False
                                
                                # Condition 1: Both PTR No and Address1 exist in SAME row - fill BOTH suggested_name AND quick_suggest_name
                                # Leave md_official_name blank
                                if match_condition == 'both':
                                    df_work.at[idx, 'suggested_name'] = md_official_name_val
                                    df_work.at[idx, 'quick_suggest_name'] = md_official_name_val
                                    # Ensure md_official_name remains blank
                                    if 'md_official_name' in df_work.columns:
                                        df_work.at[idx, 'md_official_name'] = ''
                                
                                # Condition 1b: Both PTR No and Address1 exist but in DIFFERENT rows - fill ONLY quick_suggest_name
                                elif match_condition == 'different_rows':
                                    df_work.at[idx, 'quick_suggest_name'] = md_official_name_val
                                    # Ensure md_official_name and suggested_name remain blank
                                    if 'md_official_name' in df_work.columns:
                                        df_work.at[idx, 'md_official_name'] = ''
                                    if 'suggested_name' in df_work.columns:
                                        df_work.at[idx, 'suggested_name'] = ''
                                
                                # Condition 2 & 3: Only PTR No or only Address1 exists - fill ONLY quick_suggest_name
                                elif match_condition in ['ptr_only', 'address_only']:
                                    df_work.at[idx, 'quick_suggest_name'] = md_official_name_val
                                
                                # Condition 4: Branch Code fallback (PTR/Address don't match, but Branch Code matches) - fill ONLY quick_suggest_name
                                elif match_condition == 'branch_code_fallback':
                                    df_work.at[idx, 'quick_suggest_name'] = md_official_name_val
                                    # Ensure md_official_name and suggested_name remain blank (Branch Code match is less reliable)
                                    if 'md_official_name' in df_work.columns:
                                        df_work.at[idx, 'md_official_name'] = ''
                                    if 'suggested_name' in df_work.columns:
                                        df_work.at[idx, 'suggested_name'] = ''
                                
                                # [PTR Final] from masterlist ptr_final/md_ptrs_final consistently (never app_ptr_no first)
                                ptr_val = ''
                                if 'ptr_final' in matched_row.index and pd.notna(matched_row.get('ptr_final')) and str(matched_row['ptr_final']).strip():
                                    ptr_val = str(matched_row['ptr_final']).strip()
                                elif 'md_ptrs_final' in matched_row.index and pd.notna(matched_row.get('md_ptrs_final')) and str(matched_row['md_ptrs_final']).strip():
                                    ptr_val = str(matched_row['md_ptrs_final']).strip()
                                elif 'md_ptrs' in matched_row.index and pd.notna(matched_row.get('md_ptrs')) and str(matched_row['md_ptrs']).strip():
                                    ptr_val = str(matched_row['md_ptrs']).strip()
                                if not ptr_val and app_ptr_no:
                                    ptr_val = app_ptr_no
                                df_work.at[idx, 'md_ptrs'] = ptr_val
                                
                                # Fill DOCTOR_CODE and CUSTOMER_CODE only when suggested_name is filled (both PTR+Address in same row)
                                if match_condition == 'both':
                                    if 'DOCTOR_CODE' in masterlist_work.columns and pd.notna(matched_row['DOCTOR_CODE']):
                                        df_work.at[idx, 'DOCTOR_CODE'] = str(matched_row['DOCTOR_CODE']).strip()
                                    if 'CUSTOMER_CODE' in masterlist_work.columns and pd.notna(matched_row['CUSTOMER_CODE']):
                                        df_work.at[idx, 'CUSTOMER_CODE'] = str(matched_row['CUSTOMER_CODE']).strip()
                                
                                quick_suggest_matched_count += 1
                                updated_count += 1
                        
                        # Progress update
                        processed += 1
                        if progress_bar is not None and status_text is not None and processed % 100 == 0:
                            progress = 0.99 + (processed / total_unmatched) * 0.01
                            progress_bar.progress(min(progress, 1.0))
                            status_text.text(f'Quick Suggest: Processed {processed}/{total_unmatched} records ({processed*100//total_unmatched if total_unmatched > 0 else 0}%), matched {quick_suggest_matched_count} so far...')
            
            if status_text is not None and quick_suggest_matched_count > 0:
                status_text.text(f'✅ Quick Suggest Matching completed! {quick_suggest_matched_count} records matched via word-based matching.')
        except Exception as e:
            if status_text is not None:
                status_text.text(f'⚠️ Quick Suggest matching encountered an error: {str(e)}. Continuing...')
            logger.error(f"Quick Suggest matching error: {str(e)}\n{traceback.format_exc()}")
    else:
        if status_text is not None:
            status_text.text('⏭️ Skipping Quick Suggest matching (disabled by user)')
    
    # ===== ADDITIONAL STEP: TF-IDF MATCHING FOR REMAINING BLANKS USING REFERENCE DATA ONLY =====
    # This step matches remaining unmatched records using TF-IDF on Doctor Name only
    # Uses only doctors_reference.csv data (not full masterlist)
    # No PTR or Address validation - only Doctor Name similarity
    if use_quick_suggest_matching:
        try:
            # Find records that are still unmatched (quick_suggest_name is blank)
            still_unmatched_after_quick = ((df_work['quick_suggest_name'] == '') | (df_work['quick_suggest_name'].isna())) & \
                                        ((df_work['md_official_name'] == '') | (df_work['md_official_name'].isna()))
            still_unmatched_df_ref = df_work[still_unmatched_after_quick].copy()
            
            if not still_unmatched_df_ref.empty:
                # Load reference data only
                reference_df = load_doctors_reference_csv()
                
                # Ensure doctor_name_col is defined (should already be set at function level, but verify)
                if doctor_name_col is None or doctor_name_col not in still_unmatched_df_ref.columns:
                    # Re-check if column exists with different name
                    for col in ['Doctor Name', 'doctor_name', 'DOCTOR_NAME']:
                        if col in still_unmatched_df_ref.columns:
                            doctor_name_col = col
                            break
                
                if reference_df is not None and not reference_df.empty and doctor_name_col:
                    if status_text is not None:
                        status_text.text(f'Performing TF-IDF matching on {len(still_unmatched_df_ref)} remaining records using reference data only (Doctor Name only)...')
                    if progress_bar is not None:
                        progress_bar.progress(0.995)
                    
                    # OPTIMIZED: Vectorized name cleaning (much faster than apply)
                    def clean_names_vectorized(series):
                        """Vectorized version of clean_name function - maintains exact same logic."""
                        return (
                            series
                            .astype(str)
                            .str.lower()
                            .str.strip()
                            .str.replace(r'\b(dr|md|doctor)\b', '', regex=True, case=False)
                            .str.replace(r'[^a-z0-9\s]', '', regex=True)
                            .str.strip()
                            .replace('nan', '')
                        )
                    
                    # Prepare reference data (vectorized)
                    reference_names_series = clean_names_vectorized(reference_df['Doctor Name'])
                    reference_indices = reference_df.index.tolist()
                    
                    # Filter out empty names (vectorized boolean indexing)
                    valid_mask = reference_names_series != ''
                    reference_names_clean = reference_names_series[valid_mask].tolist()
                    reference_indices_clean = [reference_indices[i] for i in range(len(reference_indices)) if valid_mask.iloc[i]]
                    
                    # Prepare unmatched records (vectorized)
                    unmatched_names_series = clean_names_vectorized(still_unmatched_df_ref[doctor_name_col])
                    unmatched_indices = still_unmatched_df_ref.index.tolist()
                    
                    # Filter out empty names (vectorized boolean indexing)
                    valid_unmatched_mask = unmatched_names_series != ''
                    unmatched_names_clean = unmatched_names_series[valid_unmatched_mask].tolist()
                    unmatched_indices_clean = [unmatched_indices[i] for i in range(len(unmatched_indices)) if valid_unmatched_mask.iloc[i]]
                    
                    if unmatched_names_clean and reference_names_clean:
                        try:
                            # Use TF-IDF for matching
                            if status_text is not None:
                                status_text.text(f'TF-IDF matching: {len(unmatched_names_clean)} unmatched vs {len(reference_names_clean)} reference records...')
                            
                            # Determine batch size
                            available_memory = None
                            if PSUTIL_AVAILABLE:
                                try:
                                    memory = psutil.virtual_memory()
                                    available_memory = memory.available / 1024 / 1024
                                except Exception:
                                    pass
                            
                            batch_size = get_tfidf_batch_size(len(unmatched_names_clean), available_memory)
                            max_features = 10000
                            
                            # Fit vectorizer on reference data
                            vectorizer = TfidfVectorizer(
                                analyzer='char',
                                ngram_range=(2, 3),
                                min_df=1,
                                lowercase=True,
                                max_features=max_features
                            )
                            reference_matrix = vectorizer.fit_transform(reference_names_clean)
                            reference_matrix = reference_matrix.astype('float32')
                            
                            # Use NearestNeighbors
                            num_unmatched = len(unmatched_names_clean)
                            if num_unmatched > 100000:
                                optimal_n_jobs = 1
                            else:
                                optimal_n_jobs = get_optimal_n_jobs(min_memory_mb=2000)
                            nbrs = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=optimal_n_jobs).fit(reference_matrix)
                            
                            # Process in batches
                            num_batches = (len(unmatched_names_clean) + batch_size - 1) // batch_size
                            if num_batches == 0:
                                num_batches = 1
                            
                            # Similarity threshold for Doctor Name only matching (higher threshold for accuracy)
                            name_similarity_threshold = 0.75  # 75% similarity required
                            
                            for batch_idx in range(num_batches):
                                batch_start = batch_idx * batch_size
                                batch_end = min(batch_start + batch_size, len(unmatched_names_clean))
                                batch_names = unmatched_names_clean[batch_start:batch_end]
                                
                                # Transform batch
                                unmatched_batch_matrix = vectorizer.transform(batch_names)
                                unmatched_batch_matrix = unmatched_batch_matrix.astype('float32')
                                
                                # Find nearest neighbors
                                distances, indices = nbrs.kneighbors(unmatched_batch_matrix)
                                
                                # Process matches
                                for j, (dist, idx) in enumerate(zip(distances, indices)):
                                    global_idx = batch_start + j
                                    similarity = 1 - dist[0]
                                    
                                    # Check if similarity meets threshold
                                    if similarity >= name_similarity_threshold:
                                        original_idx = unmatched_indices_clean[global_idx]
                                        reference_idx = reference_indices_clean[idx[0]]
                                        
                                        # Get matched reference data
                                        matched_reference = reference_df.loc[reference_idx]
                                        matched_doctor_name = str(matched_reference['Doctor Name']).strip() if pd.notna(matched_reference['Doctor Name']) else ''
                                        
                                        if matched_doctor_name:
                                            # Fill quick_suggest_name only (no PTR/Address validation)
                                            df_work.at[original_idx, 'quick_suggest_name'] = matched_doctor_name
                                            
                                            # Set suggested_md (suggest_dn) to 'REF' for TF-IDF reference matches
                                            # This indicates it's a match from reference data (Doctor Name only, no PTR/Address validation)
                                            # TF-IDF Reference Matching sets suggested_md to 'REF' (same as Quick Suggest reference matches)
                                            if 'suggested_md' in df_work.columns:
                                                df_work.at[original_idx, 'suggested_md'] = 'REF'  # Set to 'REF' to flag reference data match
                                            
                                            # REF matches: do NOT fill or replace Address1, Address2, or PTR from reference
                                            # (Doctor Name only match; app address/PTR data must remain unchanged)
                                            
                                            tfidf_reference_matched_count += 1
                                            updated_count += 1
                                
                                # Cleanup batch
                                del unmatched_batch_matrix, distances, indices
                                cleanup_memory()
                                
                                # Progress update
                                if progress_bar is not None and status_text is not None and batch_idx % 10 == 0:
                                    progress = 0.995 + (batch_idx / num_batches) * 0.005
                                    progress_bar.progress(min(progress, 1.0))
                                    status_text.text(f'TF-IDF Reference Matching: Batch {batch_idx + 1}/{num_batches}, matched {tfidf_reference_matched_count} so far...')
                            
                            # Cleanup
                            del vectorizer, reference_matrix, nbrs
                            cleanup_memory()
                            
                            if status_text is not None and tfidf_reference_matched_count > 0:
                                status_text.text(f'✅ TF-IDF Reference Matching completed! {tfidf_reference_matched_count} additional records matched via Doctor Name similarity.')
                        except MemoryError as e:
                            if status_text is not None:
                                status_text.text(f'⚠️ Memory error during TF-IDF reference matching: {str(e)}. Skipping...')
                            logger.error(f"TF-IDF reference matching memory error: {str(e)}")
                        except Exception as e:
                            if status_text is not None:
                                status_text.text(f'⚠️ Error during TF-IDF reference matching: {str(e)}. Continuing...')
                            logger.error(f"TF-IDF reference matching error: {str(e)}\n{traceback.format_exc()}")
                    else:
                        if status_text is not None:
                            if not unmatched_names_clean:
                                status_text.text('ℹ️ No unmatched records with valid Doctor Names for TF-IDF reference matching.')
                            elif not reference_names_clean:
                                status_text.text('ℹ️ No reference data available for TF-IDF matching.')
                else:
                    if status_text is not None:
                        if reference_df is None or reference_df.empty:
                            status_text.text('ℹ️ No reference data available. Skipping TF-IDF reference matching.')
                        elif not doctor_name_col:
                            status_text.text('ℹ️ Doctor Name column not found. Skipping TF-IDF reference matching.')
        except Exception as e:
            if status_text is not None:
                status_text.text(f'⚠️ TF-IDF reference matching encountered an error: {str(e)}. Continuing...')
            logger.error(f"TF-IDF reference matching error: {str(e)}\n{traceback.format_exc()}")
    
    # ===== FINAL MATCHING STEP: Doctor Name = md_suggest AND Branch Code = md_b_codes =====
    # This is the last matching step for all unmatched records
    try:
        # Find still unmatched records (empty quick_suggest_name and md_official_name)
        final_unmatched_mask = ((df_work['quick_suggest_name'] == '') | (df_work['quick_suggest_name'].isna())) & \
                               ((df_work['md_official_name'] == '') | (df_work['md_official_name'].isna()))
        final_unmatched_df = df_work[final_unmatched_mask].copy()
        
        if not final_unmatched_df.empty and masterlist_df is not None and not masterlist_df.empty:
            # Get required column names
            doctor_name_col_final = None
            for col in ['Doctor Name', 'doctor_name', 'DOCTOR_NAME']:
                if col in final_unmatched_df.columns:
                    doctor_name_col_final = col
                    break
            
            branch_code_col_final = None
            for col in ['Branch Code', 'branch_code', 'BRANCH_CODE']:
                if col in final_unmatched_df.columns:
                    branch_code_col_final = col
                    break
            
            # Check if required columns exist
            if doctor_name_col_final and branch_code_col_final and 'md_suggest' in masterlist_df.columns and 'md_b_codes' in masterlist_df.columns:
                if status_text is not None:
                    status_text.text('Performing final matching: Doctor Name = md_suggest AND Branch Code = md_b_codes...')
                if progress_bar is not None:
                    progress_bar.progress(0.999)
                
                # OPTIMIZED: Prepare masterlist with vectorized operations
                masterlist_final = masterlist_df.copy()
                masterlist_final['md_suggest_upper'] = masterlist_final['md_suggest'].fillna('').astype(str).str.upper().str.strip()
                
                # OPTIMIZED: Build composite key index: (doctor_name, branch_code) -> masterlist index (first match)
                # This allows O(1) lookup instead of O(n) nested loops
                composite_key_to_master = {}  # {(doctor_name_upper, branch_code_numeric): masterlist_index}
                
                # Vectorized branch code extraction and key building
                for master_idx in masterlist_final.index:
                    md_suggest_val = masterlist_final.loc[master_idx, 'md_suggest_upper']
                    if not md_suggest_val:
                        continue
                    
                    md_b_codes_value = masterlist_final.loc[master_idx, 'md_b_codes']
                    if pd.notna(md_b_codes_value) and str(md_b_codes_value).strip():
                        md_b_codes_str = str(md_b_codes_value).strip()
                        # Extract all numeric branch codes from md_b_codes (handles comma-separated values)
                        branch_codes = re.findall(r'\b(\d+)\b', md_b_codes_str)
                        for branch_code_numeric in branch_codes:
                            composite_key = (md_suggest_val, branch_code_numeric)
                            # Only store first match (consistent with original logic)
                            if composite_key not in composite_key_to_master:
                                composite_key_to_master[composite_key] = master_idx
                
                # OPTIMIZED: Vectorized processing of unmatched records
                # Prepare app data columns (vectorized)
                doctor_names_upper = final_unmatched_df[doctor_name_col_final].fillna('').astype(str).str.strip().str.upper()
                
                # Extract branch codes (vectorized)
                branch_codes_str = final_unmatched_df[branch_code_col_final].fillna('').astype(str).str.strip()
                branch_codes_numeric = branch_codes_str.str.extract(r'(\d+)', expand=False).fillna('')
                
                # Create composite keys for lookup
                composite_keys_app = list(zip(doctor_names_upper, branch_codes_numeric))
                
                # OPTIMIZED: Batch lookup using dictionary (O(1) per lookup)
                matched_indices_list = []
                matched_app_indices = []
                
                for i, (app_idx, composite_key) in enumerate(zip(final_unmatched_df.index, composite_keys_app)):
                    if composite_key in composite_key_to_master:
                        matched_app_indices.append(app_idx)
                        matched_indices_list.append(composite_key_to_master[composite_key])
                
                # OPTIMIZED: Batch update using .loc with list of indices (much faster than .at in loop)
                if matched_app_indices:
                    final_match_count = len(matched_app_indices)
                    
                    # Get matched masterlist rows
                    matched_master_indices = matched_indices_list
                    matched_rows = masterlist_final.loc[matched_master_indices]
                    
                    # Batch fill quick_suggest_name
                    md_official_names = matched_rows['md_official_name'].fillna('').astype(str).str.strip()
                    valid_matches = md_official_names != ''
                    
                    if valid_matches.any():
                        valid_app_indices = [matched_app_indices[i] for i in range(len(matched_app_indices)) if valid_matches.iloc[i]]
                        
                        # Batch update quick_suggest_name only (per rules: no PTR Final, DOCTOR_CODE, CUSTOMER_CODE)
                        df_work.loc[valid_app_indices, 'quick_suggest_name'] = md_official_names[valid_matches].values
                        df_work.loc[valid_app_indices, 'suggested_md'] = False
                        final_match_count = len(valid_app_indices)
                    else:
                        final_match_count = 0
                else:
                    final_match_count = 0
                
                if status_text is not None and final_match_count > 0:
                    status_text.text(f'✅ Branch Code matching completed! {final_match_count} additional records matched via Doctor Name = md_suggest AND Branch Code = md_b_codes.')
                
                # ===== ADDITIONAL FINAL MATCHING: Doctor Name = md_suggest AND Address1 substring in md_add_1 =====
                # Check for still unmatched records after Branch Code matching
                final_unmatched_after_branch = ((df_work['quick_suggest_name'] == '') | (df_work['quick_suggest_name'].isna())) & \
                                              ((df_work['md_official_name'] == '') | (df_work['md_official_name'].isna()))
                final_unmatched_df_address = df_work[final_unmatched_after_branch].copy()
                
                if not final_unmatched_df_address.empty:
                    # Get Address1 column name
                    address_col_final = None
                    for col in ['Address1', 'Address', 'address1', 'ADDRESS1']:
                        if col in final_unmatched_df_address.columns:
                            address_col_final = col
                            break
                    
                    # Check if required columns exist
                    if doctor_name_col_final and address_col_final and 'md_suggest' in masterlist_df.columns and 'md_add_1' in masterlist_df.columns:
                        if status_text is not None:
                            status_text.text('Performing final matching: Doctor Name = md_suggest AND Address1 substring in md_add_1...')
                        
                        # OPTIMIZED: Prepare masterlist with vectorized operations
                        masterlist_address = masterlist_df.copy()
                        masterlist_address['md_suggest_upper'] = masterlist_address['md_suggest'].fillna('').astype(str).str.upper().str.strip()
                        masterlist_address['md_add_1_upper'] = masterlist_address['md_add_1'].fillna('').astype(str).str.upper().str.strip()
                        
                        # OPTIMIZED: Build index: doctor_name -> list of (md_add_1, masterlist_index) tuples
                        # This allows filtering by doctor name first, then checking address substring
                        doctor_name_to_addresses = {}  # {doctor_name_upper: [(md_add_1_upper, masterlist_index), ...]}
                        
                        for master_idx in masterlist_address.index:
                            md_suggest_val = masterlist_address.loc[master_idx, 'md_suggest_upper']
                            md_add_1_val = masterlist_address.loc[master_idx, 'md_add_1_upper']
                            
                            if md_suggest_val and md_add_1_val:
                                if md_suggest_val not in doctor_name_to_addresses:
                                    doctor_name_to_addresses[md_suggest_val] = []
                                doctor_name_to_addresses[md_suggest_val].append((md_add_1_val, master_idx))
                        
                        # OPTIMIZED: Vectorized processing of unmatched records
                        if not final_unmatched_df_address.empty:
                            # Prepare app data columns (vectorized)
                            doctor_names_addr = final_unmatched_df_address[doctor_name_col_final].fillna('').astype(str).str.strip().str.upper()
                            app_addresses = final_unmatched_df_address[address_col_final].fillna('').astype(str).str.strip().str.upper()
                            
                            # Filter out records with missing data
                            valid_mask = (doctor_names_addr != '') & (app_addresses != '')
                            valid_indices = final_unmatched_df_address.index[valid_mask]
                            valid_doctor_names = doctor_names_addr[valid_mask]
                            valid_addresses = app_addresses[valid_mask]
                            
                            # OPTIMIZED: Batch matching using dictionary lookup and vectorized substring check
                            matched_app_indices_addr = []
                            matched_master_indices_addr = []
                            
                            for app_idx, doctor_name_addr, app_address1 in zip(valid_indices, valid_doctor_names, valid_addresses):
                                if doctor_name_addr in doctor_name_to_addresses:
                                    # Check address substring match for this doctor name
                                    for md_add_1_val, master_idx in doctor_name_to_addresses[doctor_name_addr]:
                                        # Bidirectional substring match
                                        if app_address1 in md_add_1_val or md_add_1_val in app_address1:
                                            matched_app_indices_addr.append(app_idx)
                                            matched_master_indices_addr.append(master_idx)
                                            break  # Use first match (consistent with original logic)
                            
                            # OPTIMIZED: Batch update using .loc with list of indices
                            if matched_app_indices_addr:
                                address_match_count = len(matched_app_indices_addr)
                                
                                # Get matched masterlist rows
                                matched_rows_addr = masterlist_address.loc[matched_master_indices_addr]
                                
                                # Batch fill quick_suggest_name
                                md_official_names_addr = matched_rows_addr['md_official_name'].fillna('').astype(str).str.strip()
                                valid_matches_addr = md_official_names_addr != ''
                                
                                if valid_matches_addr.any():
                                    valid_app_indices_addr = [matched_app_indices_addr[i] for i in range(len(matched_app_indices_addr)) if valid_matches_addr.iloc[i]]
                                    
                                    # Batch update quick_suggest_name only (per rules: no PTR Final, DOCTOR_CODE, CUSTOMER_CODE)
                                    df_work.loc[valid_app_indices_addr, 'quick_suggest_name'] = md_official_names_addr[valid_matches_addr].values
                                    df_work.loc[valid_app_indices_addr, 'suggested_md'] = False
                                    address_match_count = len(valid_app_indices_addr)
                                else:
                                    address_match_count = 0
                            else:
                                address_match_count = 0
                        else:
                            address_match_count = 0
                        
                        if status_text is not None:
                            total_final_matches = final_match_count + address_match_count
                            if address_match_count > 0:
                                status_text.text(f'✅ Final matching completed! Total: {total_final_matches} matches (Branch Code: {final_match_count}, Address: {address_match_count}).')
                            elif final_match_count > 0:
                                status_text.text(f'✅ Final matching completed! {final_match_count} records matched via Branch Code matching.')
                            else:
                                status_text.text('ℹ️ Final matching: No additional matches found.')
                    elif status_text is not None:
                        missing_cols_addr = []
                        if not doctor_name_col_final:
                            missing_cols_addr.append('Doctor Name')
                        if not address_col_final:
                            missing_cols_addr.append('Address1')
                        if 'md_suggest' not in masterlist_df.columns:
                            missing_cols_addr.append('md_suggest')
                        if 'md_add_1' not in masterlist_df.columns:
                            missing_cols_addr.append('md_add_1')
                        if missing_cols_addr:
                            status_text.text(f'ℹ️ Address matching skipped: Missing required columns ({", ".join(missing_cols_addr)}).')
                elif status_text is not None:
                    if final_match_count > 0:
                        status_text.text(f'✅ Final matching completed! {final_match_count} records matched via Branch Code matching.')
                    else:
                        status_text.text('ℹ️ Final matching: No additional matches found.')
            else:
                if status_text is not None:
                    missing_cols = []
                    if not doctor_name_col_final:
                        missing_cols.append('Doctor Name')
                    if not branch_code_col_final:
                        missing_cols.append('Branch Code')
                    if 'md_suggest' not in masterlist_df.columns:
                        missing_cols.append('md_suggest')
                    if 'md_b_codes' not in masterlist_df.columns:
                        missing_cols.append('md_b_codes')
                    status_text.text(f'ℹ️ Final matching skipped: Missing required columns ({", ".join(missing_cols)}).')
    except Exception as e:
        if status_text is not None:
            status_text.text(f'⚠️ Final matching encountered an error: {str(e)}. Continuing...')
        logger.error(f"Final matching error: {str(e)}\n{traceback.format_exc()}")
    
    # Final cleanup before return
    cleanup_memory()
    
    # CRITICAL: Restore original Amount column (NEVER modified during matching)
    if original_amount is not None and len(original_amount) == len(df_work):
        df_work['Amount'] = original_amount.values  # Restore original Amount values
    
    # Set suggested_md to blank ('') for rows that remain unmatched after all matching processes
    # This flags rows with no matches from any process (exact, TF-IDF, PPE, Quick Suggest, or TF-IDF Reference)
    # IMPORTANT: Only convert False/'false' to '' if ALL match columns are blank (truly unmatched)
    # Keep False/'false' if any match column is populated (AI match or Quick Suggest match).
    # suggested_md is effectively a string column: can contain True, False, 'REF', '' (blank).
    # Always use string comparison; do not use dtype == bool or ~.
    if 'suggested_md' in df_work.columns:
        # Check which match columns exist
        md_name_col = 'md_official_name' if 'md_official_name' in df_work.columns else None
        quick_suggest_col = 'quick_suggest_name' if 'quick_suggest_name' in df_work.columns else None
        suggested_name_col = 'suggested_name' if 'suggested_name' in df_work.columns else None
        
        # Mask for rows where suggested_md is False (string 'false' only; column has 'REF', '', etc.)
        is_false_mask = df_work['suggested_md'].astype(str).str.strip().str.lower() == 'false'
        
        # Convert False/'false' to '' (blank) ONLY for rows where ALL match columns are blank (truly unmatched)
        unmatched_conditions = []
        if md_name_col:
            unmatched_conditions.append((df_work[md_name_col] == '') | (df_work[md_name_col].isna()))
        if quick_suggest_col:
            unmatched_conditions.append((df_work[quick_suggest_col] == '') | (df_work[quick_suggest_col].isna()))
        if suggested_name_col:
            unmatched_conditions.append((df_work[suggested_name_col] == '') | (df_work[suggested_name_col].isna()))
        
        if unmatched_conditions:
            all_blank = unmatched_conditions[0]
            for cond in unmatched_conditions[1:]:
                all_blank = all_blank & cond
            unmatched_mask = is_false_mask & all_blank
            df_work.loc[unmatched_mask, 'suggested_md'] = ''
        else:
            df_work.loc[is_false_mask, 'suggested_md'] = ''
    
    # Return completion info for display (include ppe_matched_count, quick_suggest_matched_count, and tfidf_reference_matched_count)
    completion_info = {
        'doctor_name_matched': doctor_name_matched_count,
        'ptr_matched': ptr_matched_count,
        'ppe_doctors_matched': ppe_matched_count,
        'quick_suggest_matched': quick_suggest_matched_count,
        'tfidf_reference_matched': tfidf_reference_matched_count,
        'total_matched': updated_count,
        'use_doctor_name': use_doctor_name and doctor_name_col is not None,
        'use_ptr_no': use_ptr_no and ptr_no_col is not None
    }
    
    return df_work, updated_count, completion_info

def _parse_claimed_total_amount(total_amount_value):
    """Convert header Total Amount string to float."""
    if total_amount_value is None:
        return None
    try:
        return float(str(total_amount_value).replace(',', '').replace('P', '').replace(' ', '').strip())
    except (ValueError, TypeError):
        return None

def validate_txt_file_header_claims(content, file_location='Unknown'):
    """Validate header Number of Records and Total Amount against parsed data."""
    df, summary = load_data_from_content(content, file_location)

    claimed_records = summary.get('num_records')
    claimed_amount = _parse_claimed_total_amount(summary.get('total_amount'))

    result = {
        'file_location': file_location,
        'claimed_records': claimed_records,
        'claimed_amount': claimed_amount,
        'actual_records': None,
        'actual_amount': None,
        'records_match': None,
        'amount_match': None,
        'records_diff': None,
        'amount_diff': None,
        'parse_error': summary.get('error'),
    }

    if df is None or df.empty:
        return result

    result['actual_records'] = len(df)

    if 'Amount' in df.columns:
        amounts = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        result['actual_amount'] = float(amounts.sum())
    else:
        result['actual_amount'] = 0.0

    if claimed_records is not None and result['actual_records'] is not None:
        result['records_diff'] = result['actual_records'] - claimed_records
        result['records_match'] = result['records_diff'] == 0

    if claimed_amount is not None and result['actual_amount'] is not None:
        result['amount_diff'] = result['actual_amount'] - claimed_amount
        result['amount_match'] = abs(result['amount_diff']) <= 0.01

    return result

def collect_txt_header_validations(uploaded_files, zip_files_cache):
    """Run header validation for every TXT file in uploads (including inside ZIPs)."""
    validations = []

    for zip_file_obj in [f for f in uploaded_files if f.name.lower().endswith('.zip')]:
        try:
            zip_content = zip_files_cache.get(zip_file_obj.name)
            if zip_content is None:
                zip_content = zip_file_obj.read()
                zip_files_cache[zip_file_obj.name] = zip_content

            with zipfile.ZipFile(BytesIO(zip_content)) as zip_file:
                for file_path_in_zip in [f for f in zip_file.namelist() if f.lower().endswith('.txt')]:
                    content = zip_file.read(file_path_in_zip)
                    file_location = f"{zip_file_obj.name}/{file_path_in_zip}".replace('\\', '/')
                    validations.append(validate_txt_file_header_claims(content, file_location))
        except Exception as e:
            validations.append({
                'file_location': zip_file_obj.name,
                'claimed_records': None,
                'claimed_amount': None,
                'actual_records': None,
                'actual_amount': None,
                'records_match': None,
                'amount_match': None,
                'records_diff': None,
                'amount_diff': None,
                'parse_error': f'Error reading ZIP: {e}',
            })

    for txt_file_obj in [f for f in uploaded_files if f.name.lower().endswith('.txt')]:
        try:
            content = txt_file_obj.read()
            file_location = txt_file_obj.name.replace('\\', '/')
            validations.append(validate_txt_file_header_claims(content, file_location))
        except Exception as e:
            validations.append({
                'file_location': txt_file_obj.name.replace('\\', '/'),
                'claimed_records': None,
                'claimed_amount': None,
                'actual_records': None,
                'actual_amount': None,
                'records_match': None,
                'amount_match': None,
                'records_diff': None,
                'amount_diff': None,
                'parse_error': f'Error reading file: {e}',
            })

    return validations

def _upload_files_fingerprint(uploaded_files):
    return tuple(sorted((f.name, f.size) for f in uploaded_files))

def _format_records_validation_cell(validation):
    if validation.get('parse_error'):
        return f"❌ Parse error: {validation['parse_error']}"
    if validation.get('records_match') is True:
        return f"✅ Match ({validation['claimed_records']:,})"
    if validation.get('records_match') is False:
        claimed = validation.get('claimed_records')
        actual = validation.get('actual_records')
        diff = validation.get('records_diff')
        return (
            f"❌ Header: {claimed:,} | Actual: {actual:,} | Diff: {diff:+,}"
        )
    return 'N/A'

def _format_amount_validation_cell(validation):
    if validation.get('parse_error'):
        return 'N/A'
    if validation.get('amount_match') is True:
        return f"✅ Match (P {validation['claimed_amount']:,.2f})"
    if validation.get('amount_match') is False:
        claimed = validation.get('claimed_amount')
        actual = validation.get('actual_amount')
        diff = validation.get('amount_diff')
        return (
            f"❌ Header: P {claimed:,.2f} | Actual: P {actual:,.2f} | Diff: P {diff:+,.2f}"
        )
    return 'N/A'

def _validation_has_mismatch(validation):
    if validation.get('parse_error'):
        return True
    if validation.get('records_match') is False:
        return True
    if validation.get('amount_match') is False:
        return True
    return False

def summarize_amounts_by_zip(header_validations):
    """Sum validated TXT amounts grouped by parent ZIP file name."""
    zip_totals = {}
    for validation in header_validations or []:
        file_location = validation.get('file_location', '')
        if '/' not in file_location:
            continue
        zip_name, _, _ = file_location.partition('/')
        amount = validation.get('actual_amount')
        if amount is None:
            amount = validation.get('claimed_amount')
        if amount is None:
            continue
        zip_totals[zip_name] = zip_totals.get(zip_name, 0.0) + float(amount)
    return zip_totals

def get_validation_amount_for_file(header_validations, file_location):
    """Get validated amount for a single TXT file path."""
    normalized = file_location.replace('\\', '/')
    for validation in header_validations or []:
        if validation.get('file_location', '').replace('\\', '/') == normalized:
            amount = validation.get('actual_amount')
            if amount is None:
                amount = validation.get('claimed_amount')
            return amount
    return None

def format_total_amount_display(amount):
    """Format amount for display in file summary tables."""
    if amount is None:
        return 'N/A'
    return f"P {float(amount):,.2f}"

def validate_combined_summary_totals(combined_summary, combined_df):
    """Compare summary header totals against combined transaction data."""
    result = {
        'records_match': None,
        'amount_match': None,
        'actual_records': None,
        'actual_amount': None,
    }

    if combined_df is None or combined_df.empty:
        return result

    result['actual_records'] = len(combined_df)

    claimed_records = combined_summary.get('num_records', 0)
    if claimed_records > 0:
        result['records_match'] = claimed_records == result['actual_records']

    if 'Amount' in combined_df.columns:
        result['actual_amount'] = float(
            pd.to_numeric(combined_df['Amount'], errors='coerce').fillna(0).sum()
        )
        claimed_amount = combined_summary.get('total_amount', 0.0)
        if claimed_amount > 0:
            result['amount_match'] = abs(claimed_amount - result['actual_amount']) <= 0.01

    return result

def validate_matched_process_totals(combined_df, matched_df, combined_summary=None):
    """Validate matched results preserve row count and total amount from combined data."""
    result = {
        'records_match': None,
        'amount_match': None,
        'combined_records': None,
        'matched_records': None,
        'combined_amount': None,
        'matched_amount': None,
        'records_diff': None,
        'amount_diff': None,
    }

    if combined_df is None or combined_df.empty or matched_df is None or matched_df.empty:
        return result

    result['combined_records'] = len(combined_df)
    result['matched_records'] = len(matched_df)
    result['records_diff'] = result['matched_records'] - result['combined_records']
    result['records_match'] = result['records_diff'] == 0

    if 'Amount' in combined_df.columns and 'Amount' in matched_df.columns:
        result['combined_amount'] = float(
            pd.to_numeric(combined_df['Amount'], errors='coerce').fillna(0).sum()
        )
        result['matched_amount'] = float(
            pd.to_numeric(matched_df['Amount'], errors='coerce').fillna(0).sum()
        )
        result['amount_diff'] = result['matched_amount'] - result['combined_amount']
        result['amount_match'] = abs(result['amount_diff']) <= 0.01

    if combined_summary and result['matched_records'] is not None:
        claimed_records = combined_summary.get('num_records', 0)
        if claimed_records > 0 and result['records_match']:
            result['records_match'] = claimed_records == result['matched_records']

    if combined_summary and result['matched_amount'] is not None:
        claimed_amount = combined_summary.get('total_amount', 0.0)
        if claimed_amount > 0 and result['amount_match']:
            result['amount_match'] = abs(claimed_amount - result['matched_amount']) <= 0.01

    return result

def build_header_validation_display_df(validations):
    rows = []
    for idx, validation in enumerate(validations, start=1):
        file_location = validation.get('file_location', 'Unknown')
        rows.append({
            'File #': idx,
            'File Location': file_location,
            'File Name': file_location.split('/')[-1],
            'Records Validation': _format_records_validation_cell(validation),
            'Amount Validation': _format_amount_validation_cell(validation),
        })
    return pd.DataFrame(rows)

def style_header_validation_df(display_df, validations):
    mismatch_rows = {
        idx for idx, validation in enumerate(validations) if _validation_has_mismatch(validation)
    }

    def highlight_mismatch_rows(row):
        styles = [''] * len(row)
        if row.name in mismatch_rows:
            styles = ['background-color: #ffebee'] * len(row)
            for col_idx, col_name in enumerate(row.index):
                if col_name in ('Records Validation', 'Amount Validation') and '❌' in str(row[col_name]):
                    styles[col_idx] = 'background-color: #ffcdd2; color: #b71c1c; font-weight: bold'
        return styles

    return display_df.style.apply(highlight_mismatch_rows, axis=1)

def load_data_from_content(content, file_location='Unknown'):
    """Load and parse the text file using pandas from file content."""
    try:
        # Try different encodings
        encodings = ['cp1252', 'latin-1', 'utf-8', 'cp437', 'iso-8859-1']
        df = None
        summary = {}
        
        for encoding in encodings:
            try:
                # Decode content
                if isinstance(content, bytes):
                    text = content.decode(encoding, errors='ignore')
                else:
                    text = content
                
                lines = text.splitlines(keepends=True)
                
                # Check if we have enough lines
                if len(lines) < 6:
                    continue
                
                # Extract summary from first 2 lines
                if len(lines) > 0:
                    match = re.search(r'Number of Records:\s*(\d+)', lines[0])
                    if match:
                        summary['num_records'] = int(match.group(1))
                
                if len(lines) > 1:
                    match = re.search(r'Total Amount:\s*P\s*([\d,]+\.\d+)', lines[1])
                    if match:
                        summary['total_amount'] = match.group(1)
                
                # Skip header lines (0-4) and empty line (5), start from line 6
                data_lines = [line.rstrip('\n\r') for line in lines[5:] if line.strip()]
                
                # Check if we have data lines
                if not data_lines:
                    continue
                
                # Define column widths based on the header structure
                colspecs = [
                    (0, 40),    # Branch Code/Name
                    (40, 52),   # Trans Date
                    (52, 127),  # Doctor Name, PTR No, and Address1 combined
                    (127, 157), # Address2
                    (157, 193), # Vendor's Name
                    (193, 230), # Supplier Code/Name
                    (230, 275), # Item Code/Name and Qty combined
                    (275, None) # Amount (to end)
                ]
                
                # Column names
                column_names = [
                    'Branch Code/Name',
                    'Trans Date',
                    'Doctor Name/PTR No/Address1',
                    'Address2',
                    "Vendor's Name",
                    'Supplier Code/Name',
                    'Item Code/Name/Qty',
                    'Amount'
                ]
                
                # Read fixed-width file
                try:
                    df = pd.read_fwf(
                        StringIO('\n'.join(data_lines)),
                        colspecs=colspecs,
                        names=column_names
                    )
                except Exception:
                    # If fixed-width parsing fails, try without colspecs
                    try:
                        df = pd.read_fwf(
                            StringIO('\n'.join(data_lines)),
                            names=column_names,
                            infer_nrows=min(10, len(data_lines))
                        )
                    except Exception:
                        continue
                
                # Check if dataframe is empty
                if df is None or df.empty:
                    continue
                
                # Clean up the dataframe
                df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
                df = df.replace('', pd.NA)
                
                # Parse raw Amount column: source layout can have "Qty  Amount" in one column (e.g. "40  1400.00").
                # Capture Qty (first number) and Amount (last number) so we don't wrongly use pack size (e.g. 20) as Qty.
                qty_from_amount_col = None
                if 'Amount' in df.columns:
                    def _parse_qty_and_amount_from_cell(value):
                        """Extract first and last numeric tokens; if 2+, return (qty, amount) else (None, amount)."""
                        if pd.isna(value) or value == '':
                            return None, 0.0
                        if isinstance(value, (int, float)):
                            return None, float(value)
                        matches = re.findall(r'-?\d[\d,]*\.?\d*', str(value))
                        if not matches:
                            return None, 0.0
                        if len(matches) >= 2:
                            try:
                                first = float(matches[0].replace(',', ''))
                                last = float(matches[-1].replace(',', ''))
                                return first, last
                            except (ValueError, TypeError):
                                return None, 0.0
                        try:
                            return None, float(matches[-1].replace(',', ''))
                        except (ValueError, TypeError):
                            return None, 0.0
                    parsed = df['Amount'].apply(_parse_qty_and_amount_from_cell)
                    qty_from_amount_col = parsed.map(lambda x: x[0] if x[0] is not None else pd.NA)
                    amount_values = parsed.map(lambda x: x[1])
                    df['Amount'] = amount_values
                
                # CRITICAL: Clean Amount column IMMEDIATELY after reading fixed-width file
                # This prevents string concatenation issues during pd.concat()
                if 'Amount' in df.columns:
                    # Unified Amount Pipeline: Standardize to float64
                    df = standardize_amount_column(df, 'Amount')
                
                # Check if we have any valid rows
                if df.dropna(how='all').empty:
                    continue
                
                # Split Branch Code/Name into separate columns
                def split_branch_code_name(branch_str):
                    """Split Branch Code/Name into Branch Code and Branch Name.
                    The first numeric value is the code."""
                    if pd.isna(branch_str) or branch_str == '':
                        return pd.Series(['', ''])
                    
                    branch_str = str(branch_str).strip()
                    
                    if not branch_str:
                        return pd.Series(['', ''])
                    
                    # Find the first numeric sequence (digits)
                    match = re.search(r'(\d+)', branch_str)
                    
                    if match:
                        # Found numeric value
                        numeric_end = match.end()
                        
                        # Code includes everything up to and including the first numeric sequence
                        branch_code = branch_str[:numeric_end].strip()
                        # Name is everything after the first numeric sequence
                        branch_name = branch_str[numeric_end:].strip()
                        
                        # Remove leading/trailing spaces and common separators
                        if branch_name.startswith('-') or branch_name.startswith(' '):
                            branch_name = branch_name.lstrip('- ').strip()
                        
                        return pd.Series([branch_code, branch_name])
                    else:
                        # No numeric value found, put everything in name (code empty)
                        return pd.Series(['', branch_str])
                
                # Apply splitting to Branch Code/Name column
                if 'Branch Code/Name' in df.columns:
                    df[['Branch Code', 'Branch Name']] = df['Branch Code/Name'].apply(split_branch_code_name)
                    df = df.drop(columns=['Branch Code/Name'])
                
                # Split Doctor Name/PTR No/Address1 into separate columns
                def split_doctor_ptr_address(doctor_ptr_address_str):
                    """Split Doctor Name/PTR No/Address1 by finding numeric value in the middle."""
                    if pd.isna(doctor_ptr_address_str) or doctor_ptr_address_str == '':
                        return pd.Series(['', '', ''])
                    
                    doctor_ptr_address_str = str(doctor_ptr_address_str).strip()
                    
                    if not doctor_ptr_address_str:
                        return pd.Series(['', '', ''])
                    
                    # Find numeric value in the middle
                    match = re.search(r'(.+?)\s+(\d+)\s+(.+)$', doctor_ptr_address_str)
                    
                    if match:
                        doctor_name = match.group(1).strip()
                        ptr_no = match.group(2).strip()
                        address = match.group(3).strip()
                        return pd.Series([doctor_name, ptr_no, address])
                    
                    # Alternative: Try splitting by spaces and find the numeric part
                    parts = doctor_ptr_address_str.split()
                    if len(parts) >= 3:
                        for i in range(1, len(parts) - 1):
                            if parts[i].isdigit():
                                doctor_name = ' '.join(parts[:i]).strip()
                                ptr_no = parts[i]
                                address = ' '.join(parts[i+1:]).strip()
                                return pd.Series([doctor_name, ptr_no, address])
                    
                    # If no numeric found in middle, try at the end
                    if len(parts) >= 2:
                        if parts[-1].isdigit():
                            doctor_name = ' '.join(parts[:-1]).strip()
                            ptr_no = parts[-1]
                            address = ''
                            return pd.Series([doctor_name, ptr_no, address])
                        elif len(parts) >= 2 and parts[-2].isdigit():
                            doctor_name = ' '.join(parts[:-2]).strip()
                            ptr_no = parts[-2]
                            address = parts[-1] if len(parts) > 1 else ''
                            return pd.Series([doctor_name, ptr_no, address])
                    
                    # Fallback: Use fixed-width positions
                    padded_str = doctor_ptr_address_str.ljust(75) if len(doctor_ptr_address_str) < 75 else doctor_ptr_address_str[:75]
                    doctor_name = padded_str[0:30].strip()
                    ptr_no = ''
                    address = padded_str[30:75].strip()
                    
                    return pd.Series([doctor_name, ptr_no, address])
                
                # Apply splitting to Doctor Name/PTR No/Address1 column
                if 'Doctor Name/PTR No/Address1' in df.columns:
                    df[['Doctor Name', 'PTR No', 'Address1']] = df['Doctor Name/PTR No/Address1'].apply(split_doctor_ptr_address)
                    df = df.drop(columns=['Doctor Name/PTR No/Address1'])
                
                # Split Item Code/Name/Qty into separate columns
                def split_item_qty(item_qty_str):
                    """Split Item Code/Name/Qty by extracting numeric Qty from the rightmost side."""
                    if pd.isna(item_qty_str) or item_qty_str == '':
                        return pd.Series(['', ''])
                    
                    item_qty_str = str(item_qty_str).strip()
                    
                    if not item_qty_str:
                        return pd.Series(['', ''])
                    
                    # Extract numeric value from the rightmost side
                    qty_pattern = r'(\d+\.?\d*)\s*$'
                    match = re.search(qty_pattern, item_qty_str)
                    
                    if match:
                        qty_value = match.group(1)
                        item_name = item_qty_str[:match.start()].strip()
                        return pd.Series([item_name, qty_value])
                    
                    # Try splitting by space and check if last part is numeric
                    parts = item_qty_str.split()
                    if len(parts) > 1:
                        last_part = parts[-1].strip()
                        try:
                            float(last_part)
                            item_name = ' '.join(parts[:-1]).strip()
                            return pd.Series([item_name, last_part])
                        except ValueError:
                            return pd.Series([item_qty_str, ''])
                    
                    return pd.Series([item_qty_str, ''])
                
                # Apply splitting to Item Code/Name/Qty column
                if 'Item Code/Name/Qty' in df.columns:
                    df[['Item Code/Name', 'Qty']] = df['Item Code/Name/Qty'].apply(split_item_qty)
                    df = df.drop(columns=['Item Code/Name/Qty'])
                    # Convert Qty to numeric immediately (fixes PyArrow conversion issues)
                    # Remove commas, handle empty strings, and convert to float
                    if 'Qty' in df.columns:
                        # Replace empty strings with '0' before removing commas
                        df['Qty'] = df['Qty'].astype(str).str.strip()
                        df['Qty'] = df['Qty'].replace('', '0')
                        # Remove commas from numbers like "20,700.00"
                        df['Qty'] = df['Qty'].str.replace(',', '')
                        # Convert to numeric, coercing errors to NaN, then fill NaN with 0
                        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
                    # Overwrite Qty with value parsed from Amount column when source had "Qty  Amount" in one column
                    if qty_from_amount_col is not None and 'Qty' in df.columns:
                        df['Qty'] = df['Qty'].where(qty_from_amount_col.isna(), qty_from_amount_col)
                        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
                
                # Split Supplier Code/Name into separate columns
                def split_supplier_code_name(supplier_str):
                    """Split Supplier Code/Name into Supplier Code and Supplier Name.
                    The first numeric value is the code."""
                    if pd.isna(supplier_str) or supplier_str == '':
                        return pd.Series(['', ''])
                    
                    supplier_str = str(supplier_str).strip()
                    
                    if not supplier_str:
                        return pd.Series(['', ''])
                    
                    # Find the first numeric sequence (digits)
                    match = re.search(r'(\d+)', supplier_str)
                    
                    if match:
                        # Found numeric value
                        numeric_end = match.end()
                        
                        # Code includes everything up to and including the first numeric sequence
                        supplier_code = supplier_str[:numeric_end].strip()
                        # Name is everything after the first numeric sequence
                        supplier_name = supplier_str[numeric_end:].strip()
                        
                        # Remove leading/trailing spaces and common separators
                        if supplier_name.startswith('-') or supplier_name.startswith(' '):
                            supplier_name = supplier_name.lstrip('- ').strip()
                        
                        return pd.Series([supplier_code, supplier_name])
                    else:
                        # No numeric value found, put everything in name (code empty)
                        return pd.Series(['', supplier_str])
                
                # Apply splitting to Supplier Code/Name column
                if 'Supplier Code/Name' in df.columns:
                    df[['Supplier Code', 'Supplier Name']] = df['Supplier Code/Name'].apply(split_supplier_code_name)
                    # Drop the original combined column
                    df = df.drop(columns=['Supplier Code/Name'])
                
                # Split Item Code/Name into separate columns
                def split_item_code_name(item_str):
                    """Split Item Code/Name into Item Code and Item Name.
                    The first numeric value is the code."""
                    if pd.isna(item_str) or item_str == '':
                        return pd.Series(['', ''])
                    
                    item_str = str(item_str).strip()
                    
                    if not item_str:
                        return pd.Series(['', ''])
                    
                    # Find the first numeric sequence (digits)
                    match = re.search(r'(\d+)', item_str)
                    
                    if match:
                        # Found numeric value
                        numeric_end = match.end()
                        
                        # Code includes everything up to and including the first numeric sequence
                        item_code = item_str[:numeric_end].strip()
                        # Name is everything after the first numeric sequence
                        item_name = item_str[numeric_end:].strip()
                        
                        # Remove leading/trailing spaces and common separators
                        if item_name.startswith('-') or item_name.startswith(' '):
                            item_name = item_name.lstrip('- ').strip()
                        
                        return pd.Series([item_code, item_name])
                    else:
                        # No numeric value found, put everything in name (code empty)
                        return pd.Series(['', item_str])
                
                # Apply splitting to Item Code/Name column (after Qty has been separated)
                if 'Item Code/Name' in df.columns:
                    df[['Item Code', 'Item Name']] = df['Item Code/Name'].apply(split_item_code_name)
                    # Drop the original combined column
                    df = df.drop(columns=['Item Code/Name'])
                
                # Add file location column
                df['file_loc'] = file_location
                
                # Clean special characters from dataframe
                def clean_special_chars(value):
                    """Remove special characters like ÿ, ӱ, and other non-printable characters."""
                    if pd.isna(value):
                        return value
                    value_str = str(value)
                    # Remove common problematic characters: ÿ, ӱ, and other non-ASCII control characters
                    # Keep only printable ASCII characters and common punctuation
                    cleaned = ''.join(char for char in value_str if ord(char) >= 32 and ord(char) < 127 or char in ['\n', '\t'])
                    return cleaned.strip()
                
                # Clean all string columns (except Qty which we'll handle separately)
                for col in df.columns:
                    if df[col].dtype == 'object' and col != 'Qty':
                        df[col] = df[col].apply(clean_special_chars)
                
                # Specifically clean Qty column - extract numeric values and convert to integer
                if 'Qty' in df.columns:
                    def clean_qty(value):
                        """Clean Qty column to extract numeric values and convert to float (allows decimals)."""
                        if pd.isna(value):
                            return value
                        
                        value_str = str(value).strip()
                        
                        # If already empty, return as is
                        if not value_str:
                            return value
                        
                        # Try to extract numeric value (digits and decimal point)
                        # Look for patterns like: "100", "100.0", "100.5", "0.5", etc.
                        numeric_match = re.search(r'(\d+\.?\d*)', value_str)
                        
                        if numeric_match:
                            numeric_value = numeric_match.group(1)
                            # Try to convert to float (preserves decimals)
                            try:
                                float_val = float(numeric_value)
                                return float_val
                            except (ValueError, TypeError):
                                return value_str
                        else:
                            # If no numeric found, return original value (don't remove it)
                            return value_str
                    
                    df['Qty'] = df['Qty'].apply(clean_qty)
                    # Convert to numeric type to ensure consistent dtype (handles mixed types)
                    df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
                
                if 'Amount' in df.columns:
                    # CRITICAL: Clean and convert Amount to numeric IMMEDIATELY after parsing
                    # This prevents string concatenation issues during pd.concat()
                    # Unified Amount Pipeline: Standardize to float64
                    df = standardize_amount_column(df, 'Amount')
                
                # Remove rows where critical columns are corrupted (all special chars)
                # Keep rows that have at least some valid data
                df = df[df.apply(lambda row: any(pd.notna(val) and str(val).strip() != '' for val in row), axis=1)]
                
                break
                
            except (UnicodeDecodeError, LookupError):
                continue
            except Exception:
                if df is None:
                    continue
                else:
                    break
        
        if df is None or df.empty:
            error_msg = 'Could not read or parse the file. Please ensure the file format matches the expected fixed-width format.'
            try:
                if isinstance(content, bytes):
                    text = content.decode('cp1252', errors='ignore')
                    lines = text.splitlines()
                    if len(lines) > 0:
                        error_msg += f' File has {len(lines)} lines. First line preview: {lines[0][:100]}'
            except Exception:
                pass
            return None, {'error': error_msg}
        
        return df, summary
        
    except Exception as e:
        return None, {'error': f'Error reading file: {e}'}

# Page configuration
st.set_page_config(
    page_title='RX Tracking System v2.0',
    page_icon='📊',
    layout='wide'
)

# Initialize session state
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = pd.DataFrame()
if 'failed_files' not in st.session_state:
    st.session_state.failed_files = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'show_reference_upload' not in st.session_state:
    st.session_state.show_reference_upload = False
if 'reference_upload_success' not in st.session_state:
    st.session_state.reference_upload_success = None
if 'show_item_cross_ref_upload' not in st.session_state:
    st.session_state.show_item_cross_ref_upload = False
if 'item_cross_ref_upload_success' not in st.session_state:
    st.session_state.item_cross_ref_upload_success = None
if 'combined_summary' not in st.session_state:
    st.session_state.combined_summary = {
        'num_records': 0,
        'total_amount': 0.0,
        'file_count': 0,
    }
if 'zip_files_cache' not in st.session_state:
    st.session_state.zip_files_cache = {}
if 'file_header_validations' not in st.session_state:
    st.session_state.file_header_validations = []
if 'file_header_validation_fingerprint' not in st.session_state:
    st.session_state.file_header_validation_fingerprint = None
if 'matched_df' not in st.session_state:
    st.session_state.matched_df = None
if 'md_code_list' not in st.session_state:
    st.session_state.md_code_list = pd.DataFrame(columns=['DOCTOR_CODE', 'CUSTOMER_CODE', 'CUSTOMER_NAME'])

# Header
st.markdown('<h1 style="margin-bottom: 0.5rem;">📊 RX Tracking System v2.0 <span style="font-size: 0.7em;">(w/ AI Matching)</span></h1>', unsafe_allow_html=True)

# Check and load masterlist CSV
csv_dir = get_csv_dir()
masterlist_path = os.path.join(csv_dir, 'rx_md_masterlist.csv')
masterlist_exists = os.path.exists(masterlist_path)

# Initialize masterlist update status in session state
if 'masterlist_updated' not in st.session_state:
    st.session_state.masterlist_updated = False
if 'masterlist_auto_updated' not in st.session_state:
    st.session_state.masterlist_auto_updated = False
# Track if this is the initial page load (not a rerun)
if 'page_initialized' not in st.session_state:
    st.session_state.page_initialized = False

# Automatically update masterlist on initial load only (not on st.rerun())
# Check both flags: page not initialized yet AND auto_update not done
if not st.session_state.page_initialized and not st.session_state.masterlist_auto_updated:
    spinner_auto = st.empty()
    
    # Check masterlist file status before updating
    masterlist_status_msg = ""
    should_update_ml = True
    if masterlist_exists:
        should_update_ml, status_msg_ml, last_modified_ml = should_update_file(masterlist_path, hours_threshold=3)
        if last_modified_ml:
            masterlist_status_msg = f"📅 Last modified: {last_modified_ml.strftime('%Y-%m-%d %H:%M:%S')}\n"
        masterlist_status_msg += status_msg_ml
        
        if not should_update_ml:
            spinner_auto.info(f"**Masterlist Status:**\n{masterlist_status_msg}")
        else:
            spinner_auto.markdown("""
            <div style="text-align: center; padding: 15px;">
                <div class="spinner-large">⏳</div>
                <div style="font-size: 18px; margin-top: 10px; font-weight: bold;">Auto-updating masterlist from server...</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        spinner_auto.markdown("""
        <div style="text-align: center; padding: 15px;">
            <div class="spinner-large">⏳</div>
            <div style="font-size: 18px; margin-top: 10px; font-weight: bold;">Auto-updating masterlist from server...</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Download masterlist only if update is needed
    if should_update_ml:
        success_ml_auto, message_ml_auto = download_masterlist_from_server(force_update=False)
    else:
        success_ml_auto = False
        message_ml_auto = masterlist_status_msg
    
    # Check PPE Doctors file status before updating
    csv_dir = get_reference_csv_dir()  # Use persistent location, not temp directory
    ppe_path_check = os.path.join(csv_dir, 'ppe_doctors.csv')
    ppe_exists_check = os.path.exists(ppe_path_check)
    ppe_status_msg = ""
    should_update_ppe = True
    if ppe_exists_check:
        should_update_ppe, status_msg_ppe, last_modified_ppe = should_update_file(ppe_path_check, hours_threshold=3)
        if last_modified_ppe:
            ppe_status_msg = f"📅 Last modified: {last_modified_ppe.strftime('%Y-%m-%d %H:%M:%S')}\n"
        ppe_status_msg += status_msg_ppe
        
        if not should_update_ppe:
            spinner_auto.info(f"**PPE Doctors Status:**\n{ppe_status_msg}")
        else:
            spinner_auto.markdown("""
            <div style="text-align: center; padding: 15px;">
                <div class="spinner-large">⏳</div>
                <div style="font-size: 18px; margin-top: 10px; font-weight: bold;">Auto-updating PPE Doctors from server...</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        spinner_auto.markdown("""
        <div style="text-align: center; padding: 15px;">
            <div class="spinner-large">⏳</div>
            <div style="font-size: 18px; margin-top: 10px; font-weight: bold;">Auto-updating PPE Doctors from server...</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Download ppe_doctors only if update is needed
    if should_update_ppe:
        success_ppe_auto, message_ppe_auto = download_ppe_doctors_from_server()
    else:
        success_ppe_auto = False
        message_ppe_auto = ppe_status_msg

    # Download PTR with TopMD
    ptr_topmd_path_auto = os.path.join(csv_dir, 'ptr_with_topmd.csv')
    should_update_ptr_topmd = True
    if os.path.exists(ptr_topmd_path_auto):
        should_update_ptr_topmd, _, _ = should_update_file(ptr_topmd_path_auto, hours_threshold=3)
    if should_update_ptr_topmd:
        success_ptr_topmd_auto, message_ptr_topmd_auto = download_ptr_with_topmd_from_server(force_update=False)
    else:
        success_ptr_topmd_auto = False
        message_ptr_topmd_auto = "PTR with TopMD: skipped (updated recently)"

    # Update masterlist with PTR TopMD data (add md_ptrs_final) if both downloads succeeded
    masterlist_update_success = False
    masterlist_update_msg = ""
    masterlist_records_updated = 0
    if success_ml_auto and success_ptr_topmd_auto:
        masterlist_update_success, masterlist_update_msg, masterlist_records_updated = update_masterlist_with_ptr_topmd()
    
    spinner_auto.empty()
    
    # Clear cache to reload CSVs (if updates were successful)
    if success_ml_auto or masterlist_update_success:
        load_masterlist_csv.clear()
    if success_ppe_auto:
        load_ppe_doctors_csv.clear()
    
    # Mark as updated and page as initialized (before showing messages to prevent rerun loops)
    st.session_state.masterlist_auto_updated = True
    st.session_state.masterlist_updated = True
    st.session_state.page_initialized = True
    
    # Show success/info message with status details
    line_ml = f"📥 **Masterlist**: {message_ml_auto}"
    if masterlist_update_success and masterlist_records_updated > 0:
        line_ml += f"\n   └─ 🔄 Added md_ptrs_final to {masterlist_records_updated} records from TopMD"
    elif masterlist_update_success:
        line_ml += f"\n   └─ ℹ️ {masterlist_update_msg}"
    line_ppe = f"📥 **PPE Doctors**: {message_ppe_auto}"
    line_ptr_topmd = f"📥 **PTR with TopMD**: {message_ptr_topmd_auto}"
    all_ok = success_ml_auto and success_ppe_auto and success_ptr_topmd_auto
    any_ok = success_ml_auto or success_ppe_auto or success_ptr_topmd_auto
    if all_ok:
        st.success(f"✅ **Masterlist Auto-Updated on Load!**\n\n{line_ml}\n{line_ppe}\n{line_ptr_topmd}")
    elif any_ok:
        st.info(f"ℹ️ **Reference data updated**\n\n{line_ml}\n{line_ppe}\n{line_ptr_topmd}")
    else:
        status_display = [f"**Masterlist**: {message_ml_auto}", f"**PPE Doctors**: {message_ppe_auto}", f"**PTR with TopMD**: {message_ptr_topmd_auto}"]
        if not should_update_ml and not should_update_ppe and not should_update_ptr_topmd:
            st.info("ℹ️ **Auto-update skipped**")
        else:
            st.warning("⚠️ **Auto-update status**:\n\n" + "\n\n".join(status_display))

# Mark page as initialized if not already (for fallback checks)
if not st.session_state.page_initialized:
    st.session_state.page_initialized = True

# Try to download masterlist on first load if it doesn't exist (fallback - only if auto-update didn't run)
if not masterlist_exists and not st.session_state.masterlist_updated:
    spinner_ml = st.empty()
    spinner_ml.markdown("""
    <div style="text-align: center; padding: 15px;">
        <div class="spinner-large">⏳</div>
        <div style="font-size: 18px; margin-top: 10px; font-weight: bold;">Attempting to download RX MD Masterlist from server...</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Force update if file doesn't exist
    success, message = download_masterlist_from_server(force_update=True)
    spinner_ml.empty()
    
    if success:
        st.success(message)
        st.session_state.masterlist_updated = True
        # Clear cache to reload the CSV
        load_cross_reference_csv.clear()
        masterlist_exists = True
    else:
        st.warning(f"{message}. App will continue without masterlist. You can try updating later.")

# Try to download ppe_doctors on first load if it doesn't exist (fallback - only if auto-update didn't run)
csv_dir = get_reference_csv_dir()  # Use persistent location, not temp directory
ppe_path = os.path.join(csv_dir, 'ppe_doctors.csv')
ppe_exists = os.path.exists(ppe_path)

if not ppe_exists and not st.session_state.masterlist_updated:
    # Download ppe_doctors silently (no spinner, just attempt)
    try:
        success_ppe, message_ppe = download_ppe_doctors_from_server(force_update=True)
        if success_ppe:
            load_ppe_doctors_csv.clear()
    except Exception:
        pass

# Try to download ptr_with_topmd on first load if it doesn't exist (fallback)
ptr_topmd_path_fallback = os.path.join(get_reference_csv_dir(), 'ptr_with_topmd.csv')
if not os.path.exists(ptr_topmd_path_fallback) and not st.session_state.masterlist_updated:
    try:
        download_ptr_with_topmd_from_server(force_update=True)
    except Exception:
        pass

# Load md_code_list from server (silently, no notifications)
if st.session_state.md_code_list.empty:
    try:
        success, md_code_list_df = download_md_code_list_from_server()
        if success:
            st.session_state.md_code_list = md_code_list_df
    except Exception:
        # Silently fail - initialize empty dataframe
        st.session_state.md_code_list = pd.DataFrame(columns=['DOCTOR_CODE', 'CUSTOMER_CODE', 'CUSTOMER_NAME'])

# Show masterlist update option if CSV exists
csv_dir = get_reference_csv_dir()
cross_ref_path = get_item_cross_ref_path()
cross_ref_exists = os.path.exists(cross_ref_path)

if masterlist_exists:
    # Add custom CSS for both buttons - light green background and green icon for Update Masterlist
    st.markdown("""
    <style>
    </style>
    <script>
    // Apply light green background to both buttons with different icon colors
    setTimeout(function() {
        var buttons = document.querySelectorAll('button');
        buttons.forEach(function(btn) {
            var btnText = btn.textContent || btn.innerText;
            if (btnText.includes('Update Masterlist') || btnText.includes('Update Standard Cost')) {
                // Apply light green background to both buttons
                btn.style.backgroundColor = '#90EE90';
                btn.style.color = '#000000';
                btn.style.borderColor = '#90EE90';
                
                var html = btn.innerHTML;
                // Change icon color to green for Update Masterlist button
                if (btnText.includes('Update Masterlist')) {
                    if (html.includes('🔄')) {
                        btn.innerHTML = html.replace('🔄', '<span style="color: #228B22; font-size: 1.1em;">🔄</span>');
                    }
                }
                // Change icon color to blue for Update Standard Cost button
                else if (btnText.includes('Update Standard Cost')) {
                    if (html.includes('🔄')) {
                        btn.innerHTML = html.replace('🔄', '<span style="color: #1E90FF; font-size: 1.1em;">🔄</span>');
                    }
                }
                
                // Hover effect
                btn.addEventListener('mouseenter', function() {
                    this.style.backgroundColor = '#7FCD7F';
                    this.style.borderColor = '#7FCD7F';
                });
                btn.addEventListener('mouseleave', function() {
                    this.style.backgroundColor = '#90EE90';
                    this.style.borderColor = '#90EE90';
                });
            }
        });
    }, 100);
    </script>
    """, unsafe_allow_html=True)
    
    col_master1, col_master2, col_master3, col_master4, col_master5 = st.columns([3, 1, 1, 1.25, 1.6])
    with col_master1:
        st.info("📋 **Manual Update**: Standard Cost, Masterlist & Reference Files")
    with col_master3:
        if st.button('🔄 Update Masterlists', help='Force download latest RX MD Masterlist and PPE Doctors from SQL Server (bypasses file age checks)', key='update_masterlist_btn'):
            spinner_update = st.empty()
            spinner_update.markdown("""
            <div style="text-align: center; padding: 15px;">
                <div class="spinner-large">⏳</div>
                <div style="font-size: 18px; margin-top: 10px; font-weight: bold;">Updating masterlist from server...</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Force update: always download (bypass file age checks)
            success_ml, message_ml = download_masterlist_from_server(force_update=True)
            
            spinner_update.markdown("""
            <div style="text-align: center; padding: 15px;">
                <div class="spinner-large">⏳</div>
                <div style="font-size: 18px; margin-top: 10px; font-weight: bold;">Updating PPE Doctors from server...</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Force update: always download (bypass file age checks)
            success_ppe, message_ppe = download_ppe_doctors_from_server(force_update=True)

            # Download PTR with TopMD (always force update for manual)
            success_ptr_topmd, message_ptr_topmd = download_ptr_with_topmd_from_server(force_update=True)

            # Update masterlist with PTR TopMD data (add md_ptrs_final) if both downloads succeeded
            masterlist_update_success_manual = False
            masterlist_update_msg_manual = ""
            masterlist_records_updated_manual = 0
            if success_ml and success_ptr_topmd:
                masterlist_update_success_manual, masterlist_update_msg_manual, masterlist_records_updated_manual = update_masterlist_with_ptr_topmd()
            
            spinner_update.empty()
            
            # Clear cache to reload CSVs (only if updates were successful)
            if success_ml or masterlist_update_success_manual:
                load_masterlist_csv.clear()
            if success_ppe:
                load_ppe_doctors_csv.clear()
            
            # Build status lines
            line_ml = f"📥 **Masterlist**: {message_ml}"
            if masterlist_update_success_manual and masterlist_records_updated_manual > 0:
                line_ml += f"\n   └─ 🔄 Added md_ptrs_final to {masterlist_records_updated_manual} records from TopMD"
            elif masterlist_update_success_manual:
                line_ml += f"\n   └─ ℹ️ {masterlist_update_msg_manual}"
            line_ppe = f"📥 **PPE Doctors**: {message_ppe}"
            line_ptr_topmd = f"📥 **PTR with TopMD**: {message_ptr_topmd}"
            all_ok_manual = success_ml and success_ppe and success_ptr_topmd
            any_ok_manual = success_ml or success_ppe or success_ptr_topmd
            
            # Show results with status details
            if all_ok_manual:
                st.success(f"✅ **Update Completed Successfully!**\n\n{line_ml}\n{line_ppe}\n{line_ptr_topmd}")
                st.rerun()
            elif any_ok_manual:
                st.warning(f"⚠️ **Update status**\n\n{line_ml}\n{line_ppe}\n{line_ptr_topmd}")
                st.rerun()
            else:
                status_display = [f"**Masterlist**: {message_ml}", f"**PPE Doctors**: {message_ppe}", f"**PTR with TopMD**: {message_ptr_topmd}"]
                st.error("❌ **Update Failed**\n\n" + "\n\n".join(status_display))
    
    with col_master4:
        if st.button('📤 Upload Reference Files', help='Upload CSV or Excel files with Doctor Name, Address, and PTR columns for Quick Suggest Matching', key='upload_reference_files_btn'):
            st.session_state.show_reference_upload = True

    with col_master5:
        if st.button('🔗 Upload New Item Cross Reference', help='Upload an item cross-reference Excel file and refresh Standard Cost from SQL', key='upload_item_cross_ref_btn'):
            st.session_state.show_item_cross_ref_upload = True
            st.session_state.item_cross_ref_upload_success = None

    # Item Cross-Reference Upload Modal/Dialog
    if st.session_state.get('show_item_cross_ref_upload', False):
        with st.expander("Upload New Item Cross Reference", expanded=True):
            item_cross_ref_path = get_item_cross_ref_path()
            item_cross_ref_exists = os.path.exists(item_cross_ref_path)

            if item_cross_ref_exists:
                try:
                    current_cross_ref_df = pd.read_csv(item_cross_ref_path, dtype=str)
                    st.info(f"Current item cross-reference table: {len(current_cross_ref_df):,} record(s)")
                except Exception as e:
                    st.warning(f"Could not read current item cross-reference table: {str(e)}")
            else:
                st.info("Current item cross-reference table: No data")

            try:
                template_data, template_file_name, template_mime = build_item_cross_ref_template()
                st.download_button(
                    label='Download Item Cross-Reference Template',
                    data=template_data,
                    file_name=template_file_name,
                    mime=template_mime,
                    key='download_item_cross_ref_template',
                    help='Download a template with the current rx_item_cross_ref.csv headers and one sample row'
                )
            except Exception as e:
                st.error(f"Could not build item cross-reference template: {str(e)}")

            uploaded_item_cross_ref = st.file_uploader(
                "Choose item cross-reference Excel file",
                type=['xlsx', 'xls', 'csv'],
                key='item_cross_ref_file_uploader',
                help='Upload an Excel or CSV file with Cross-Reference No. and Item No. columns'
            )

            col_item_ref_upload1, col_item_ref_upload2 = st.columns(2)
            with col_item_ref_upload1:
                if st.button('Process and Save', key='process_item_cross_ref_upload', type='primary'):
                    if uploaded_item_cross_ref:
                        spinner_item_ref = st.empty()
                        spinner_item_ref.markdown("""
                        <div style="text-align: center; padding: 15px;">
                            <div class="spinner-large">⏳</div>
                            <div style="font-size: 18px; margin-top: 10px; font-weight: bold;">Saving item cross-reference...</div>
                        </div>
                        """, unsafe_allow_html=True)

                        try:
                            success_upload, message_upload, upload_details = upsert_item_cross_ref_from_upload(uploaded_item_cross_ref)
                        except Exception as e:
                            spinner_item_ref.empty()
                            st.error(f"Error processing item cross-reference upload: {str(e)}")
                            logger.error(f"Item cross-reference upload error: {str(e)}\n{traceback.format_exc()}")
                            success_upload = False
                            message_upload = ""
                            upload_details = None

                        if success_upload:
                            spinner_item_ref.markdown("""
                            <div style="text-align: center; padding: 15px;">
                                <div class="spinner-large">⏳</div>
                                <div style="font-size: 18px; margin-top: 10px; font-weight: bold;">Updating Standard Cost from SQL...</div>
                            </div>
                            """, unsafe_allow_html=True)

                            success_export, message_export, count_export = export_table_item_from_innogen()
                            if success_export:
                                success_merge, message_merge, updated_count = merge_standard_cost_to_cross_ref()
                            else:
                                success_merge = False
                                message_merge = "Skipped because SQL export failed."
                                updated_count = 0

                            spinner_item_ref.empty()

                            notes = []
                            if upload_details:
                                if upload_details.get('blank_rows_removed'):
                                    notes.append(f"Removed {upload_details['blank_rows_removed']:,} row(s) with blank required fields.")
                                if upload_details.get('duplicate_upload_rows'):
                                    notes.append(f"Handled {upload_details['duplicate_upload_rows']:,} duplicate uploaded Cross-Reference No. row(s); kept the last one.")
                                if upload_details.get('missing_optional'):
                                    notes.append(f"Missing optional columns were saved blank: {', '.join(upload_details['missing_optional'])}.")
                                if upload_details.get('extra_columns'):
                                    notes.append(f"Ignored extra uploaded columns: {', '.join(upload_details['extra_columns'])}.")

                            preview_df = upload_details['preview_df'] if upload_details else pd.DataFrame()
                            try:
                                refreshed_cross_ref_df = pd.read_csv(get_item_cross_ref_path(), dtype=str).fillna('')
                                preview_df = refreshed_cross_ref_df.tail(min(len(preview_df), 100))
                            except Exception:
                                pass

                            status_lines = [f"Upload: {message_upload}"]
                            if notes:
                                status_lines.append("\n".join(notes))
                            status_lines.append(f"SQL Export: {'OK' if success_export else 'Failed'} - {message_export}")
                            status_lines.append(f"Standard Cost Merge: {'OK' if success_merge else 'Failed'} - {message_merge}")

                            st.session_state.item_cross_ref_upload_success = {
                                'type': 'success' if success_export and success_merge else 'warning',
                                'message': "\n\n".join(status_lines),
                                'preview_df': preview_df,
                                'total_records': upload_details['total_records'] if upload_details else 0,
                            }
                            st.session_state.show_item_cross_ref_upload = False
                            st.rerun()
                        else:
                            spinner_item_ref.empty()
                            st.error(message_upload)
                    else:
                        st.warning("Please upload an item cross-reference Excel or CSV file.")

            with col_item_ref_upload2:
                if st.button('Close', key='cancel_item_cross_ref_upload'):
                    st.session_state.show_item_cross_ref_upload = False
                    st.session_state.item_cross_ref_upload_success = None
                    st.rerun()

    # Display item cross-reference upload result outside expander
    if st.session_state.get('item_cross_ref_upload_success') is not None:
        item_cross_ref_success = st.session_state.item_cross_ref_upload_success
        if item_cross_ref_success.get('type') == 'success':
            st.success(item_cross_ref_success['message'])
        else:
            st.warning(item_cross_ref_success['message'])

        if item_cross_ref_success.get('preview_df') is not None and not item_cross_ref_success['preview_df'].empty:
            st.markdown("**Preview of latest item cross-reference rows:**")
            st.dataframe(item_cross_ref_success['preview_df'], use_container_width=True, height=300)

        if st.button('🏠 Return Home', key='dismiss_item_cross_ref_success'):
            st.session_state.item_cross_ref_upload_success = None
            st.rerun()

    # Reference Files Upload Modal/Dialog
    if st.session_state.get('show_reference_upload', False):
        with st.expander("📤 Upload Reference Files", expanded=True):
            st.markdown("**Upload CSV or Excel files with the following headers:** **Doctor Name** (required), **Address** (required), **PTR** (required). *Uploaded data will be merged/appended with the existing reference list.*")
            
            # Check if doctors_reference.csv exists and show table info
            csv_dir = get_reference_csv_dir()  # Use persistent location, not temp directory
            reference_csv_path = os.path.join(csv_dir, 'doctors_reference.csv')
            reference_exists = os.path.exists(reference_csv_path)
            
            # Display current table information
            if reference_exists:
                try:
                    reference_df = load_doctors_reference_csv()
                    if reference_df is not None and not reference_df.empty:
                        row_count = len(reference_df)
                        st.info(f"📊 **Current Reference Table**: {row_count:,} record(s)")
                    else:
                        st.info("📊 **Current Reference Table**: 0 records (file exists but is empty)")
                except Exception as e:
                    st.warning(f"⚠️ Could not read reference table: {str(e)}")
            else:
                st.info("📊 **Current Reference Table**: No data (file does not exist)")
            
            # Action buttons for reference table
            col_ref_info1, col_ref_info2, col_ref_info3 = st.columns(3)
            
            with col_ref_info1:
                # Download button for reference CSV
                if reference_exists:
                    try:
                        with open(reference_csv_path, 'rb') as f:
                            reference_csv_data = f.read()
                        st.download_button(
                            label='📥 Download Reference CSV',
                            data=reference_csv_data,
                            file_name='doctors_reference.csv',
                            mime='text/csv',
                            key='download_reference_csv',
                            help='Download the current doctors_reference.csv file'
                        )
                    except Exception as e:
                        st.error(f'Error reading reference file: {str(e)}')
                else:
                    st.download_button(
                        label='📥 Download Reference CSV',
                        data='',
                        file_name='doctors_reference.csv',
                        mime='text/csv',
                        key='download_reference_csv_disabled',
                        disabled=True,
                        help='No reference file available to download'
                    )
            
            with col_ref_info2:
                # Clear all data button
                if reference_exists:
                    if st.button('🗑️ Clear All Data', key='clear_reference_data', help='Delete all data from doctors_reference.csv'):
                        try:
                            # Delete the CSV file
                            os.remove(reference_csv_path)
                            # Clear cache
                            if 'load_doctors_reference_csv' in globals():
                                load_doctors_reference_csv.clear()
                            st.success("✅ Reference table data cleared successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error clearing reference data: {str(e)}")
                else:
                    st.button('🗑️ Clear All Data', key='clear_reference_data_disabled', disabled=True, help='No reference file to clear')
            
            with col_ref_info3:
                # Refresh button to reload cache
                if st.button('🔄 Refresh Info', key='refresh_reference_info', help='Refresh reference table information'):
                    if 'load_doctors_reference_csv' in globals():
                        load_doctors_reference_csv.clear()
                    st.rerun()
            
            st.divider()
            
            uploaded_files = st.file_uploader(
                "Choose CSV or Excel files",
                type=['csv', 'xlsx', 'xls'],
                accept_multiple_files=True,
                key='reference_files_uploader',
                help="Upload one or more CSV/Excel files. Files will be combined into one dataset."
            )
            
            col_upload1, col_upload2 = st.columns(2)
            with col_upload1:
                if st.button('✅ Process and Save', key='process_reference_files', type='primary'):
                    if uploaded_files:
                        try:
                            combined_df = None
                            processed_files = []
                            errors = []
                            
                            for uploaded_file in uploaded_files:
                                try:
                                    file_extension = uploaded_file.name.split('.')[-1].lower()
                                    
                                    if file_extension == 'csv':
                                        # Read CSV
                                        df = pd.read_csv(uploaded_file, encoding='utf-8-sig', on_bad_lines='skip')
                                    elif file_extension in ['xlsx', 'xls']:
                                        # Read Excel
                                        try:
                                            df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else None)
                                        except ImportError:
                                            st.error("⚠️ openpyxl library is required for Excel files. Please install it: pip install openpyxl")
                                            errors.append(f"❌ {uploaded_file.name}: openpyxl not installed")
                                            continue
                                    else:
                                        errors.append(f"❌ {uploaded_file.name}: Unsupported file type")
                                        continue
                                    
                                    # Normalize column names (case-insensitive, strip whitespace)
                                    df.columns = df.columns.str.strip()
                                    
                                    # Find matching columns (case-insensitive)
                                    doctor_name_col = None
                                    address_col = None
                                    ptr_col = None
                                    
                                    for col in df.columns:
                                        col_lower = col.lower()
                                        if 'doctor' in col_lower and 'name' in col_lower:
                                            doctor_name_col = col
                                        elif 'address' in col_lower:
                                            address_col = col
                                        elif 'ptr' in col_lower:
                                            ptr_col = col
                                    
                                    # Validate required columns
                                    missing_cols = []
                                    if not doctor_name_col:
                                        missing_cols.append('Doctor Name')
                                    if not address_col:
                                        missing_cols.append('Address')
                                    if not ptr_col:
                                        missing_cols.append('PTR')
                                    
                                    if missing_cols:
                                        errors.append(f"⚠️ {uploaded_file.name}: Missing columns: {', '.join(missing_cols)}")
                                        continue
                                    
                                    # Extract and clean required columns
                                    extracted_df = pd.DataFrame()
                                    extracted_df['Doctor Name'] = df[doctor_name_col].astype(str).fillna('').str.strip()
                                    extracted_df['Address'] = df[address_col].astype(str).fillna('').str.strip()
                                    extracted_df['PTR'] = df[ptr_col].astype(str).fillna('').str.strip()
                                    
                                    # CRITICAL: Remove rows where Doctor Name is blank or null (Doctor Name is required)
                                    initial_count = len(extracted_df)
                                    extracted_df = extracted_df[
                                        (extracted_df['Doctor Name'] != '') & 
                                        (extracted_df['Doctor Name'].notna()) &
                                        (extracted_df['Doctor Name'].str.strip() != '')
                                    ]
                                    removed_count = initial_count - len(extracted_df)
                                    
                                    if removed_count > 0:
                                        st.info(f"ℹ️ {uploaded_file.name}: Removed {removed_count} row(s) with blank/null Doctor Name")
                                    
                                    # Combine with previous data
                                    if combined_df is None:
                                        combined_df = extracted_df.copy()
                                    else:
                                        combined_df = pd.concat([combined_df, extracted_df], ignore_index=True)
                                    
                                    processed_files.append(uploaded_file.name)
                                    
                                except Exception as e:
                                    errors.append(f"❌ {uploaded_file.name}: {str(e)}")
                                    continue
                            
                            if combined_df is not None and not combined_df.empty:
                                # Final validation: Remove any rows with blank/null Doctor Name (safety net)
                                before_validation = len(combined_df)
                                combined_df = combined_df[
                                    (combined_df['Doctor Name'] != '') & 
                                    (combined_df['Doctor Name'].notna()) &
                                    (combined_df['Doctor Name'].str.strip() != '')
                                ]
                                after_validation = len(combined_df)
                                
                                if before_validation > after_validation:
                                    st.warning(f"⚠️ Removed {before_validation - after_validation} additional row(s) with blank/null Doctor Name during final validation")
                                
                                # Merge with existing reference data (append, don't overwrite)
                                existing_df = load_doctors_reference_csv()
                                new_records_count = len(combined_df)
                                if existing_df is not None and not existing_df.empty:
                                    # Ensure existing has same columns; concat and dedupe
                                    existing_df = existing_df[['Doctor Name', 'Address', 'PTR']].copy()
                                    existing_df['Doctor Name'] = existing_df['Doctor Name'].astype(str).fillna('').str.strip()
                                    existing_df['Address'] = existing_df['Address'].astype(str).fillna('').str.strip()
                                    existing_df['PTR'] = existing_df['PTR'].astype(str).fillna('').str.strip()
                                    combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
                                
                                # Remove duplicates (based on all three columns; keep first = prefer existing)
                                initial_count = len(combined_df)
                                combined_df = combined_df.drop_duplicates(subset=['Doctor Name', 'Address', 'PTR'], keep='first')
                                duplicates_removed = initial_count - len(combined_df)
                                
                                # Save to CSV (use persistent location, not temp directory)
                                csv_dir = get_reference_csv_dir()
                                csv_path = os.path.join(csv_dir, 'doctors_reference.csv')
                                combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                                
                                # Clear cache if exists
                                if 'load_doctors_reference_csv' in globals():
                                    load_doctors_reference_csv.clear()
                                
                                # Store success message in session state to show after expander closes
                                merge_info = ""
                                save_note = "doctors_reference.csv"
                                if existing_df is not None and not existing_df.empty:
                                    merge_info = f"📥 **New Records**: {new_records_count:,}\n"
                                    save_note = "doctors_reference.csv (merged with existing)"
                                success_msg = (f"✅ **Successfully processed and saved!**\n\n"
                                             f"📊 **Total Records**: {len(combined_df):,}\n"
                                             f"{merge_info}"
                                             f"📁 **Files Processed**: {len(processed_files)}\n"
                                             f"🔄 **Duplicates Removed**: {duplicates_removed:,}\n"
                                             f"💾 **Saved to**: {save_note}")
                                
                                # Store preview data in session state
                                st.session_state.reference_upload_success = {
                                    'message': success_msg,
                                    'preview_df': combined_df.head(100),
                                    'total_records': len(combined_df),
                                    'errors': errors if errors else None
                                }
                                
                                # Close modal and rerun to show success message outside expander
                                st.session_state.show_reference_upload = False
                                st.rerun()
                            else:
                                st.error("❌ No valid data found in uploaded files. Please check file formats and column names.")
                                if errors:
                                    st.error("**Errors:**\n" + "\n".join(errors))
                        except Exception as e:
                            st.error(f"❌ Error processing files: {str(e)}")
                            logger.error(f"Reference files upload error: {str(e)}\n{traceback.format_exc()}")
                    else:
                        st.warning("⚠️ Please upload at least one CSV or Excel file.")
            
            with col_upload2:
                if st.button('❌ Close', key='cancel_reference_upload'):
                    st.session_state.show_reference_upload = False
                    st.session_state.reference_upload_success = None  # Clear any previous success message
                    st.rerun()
    
    # Display success message outside expander (after it closes)
    if st.session_state.get('reference_upload_success') is not None:
        success_data = st.session_state.reference_upload_success
        st.success(success_data['message'])
        
        # Display preview
        st.markdown("**Preview of combined data:**")
        st.dataframe(success_data['preview_df'], use_container_width=True, height=300)
        
        if success_data['total_records'] > 100:
            st.caption(f"Showing first 100 of {success_data['total_records']:,} records")
        
        # Show errors if any
        if success_data['errors']:
            st.warning("**Warnings/Errors:**\n" + "\n".join(success_data['errors']))
        
        # Add button to clear the success message
        if st.button('✖️ Dismiss', key='dismiss_reference_success'):
            st.session_state.reference_upload_success = None
            st.rerun()
    
    with col_master2:
        if st.button('🔄 Update Standard Cost', help='Connect to Innogen SQL Server, export Item table, and merge Standard Cost to rx_item_cross_ref.csv', key='update_standard_cost_btn'):
            if not cross_ref_exists:
                st.error('❌ rx_item_cross_ref.csv not found.')
            else:
                spinner_update = st.empty()
                spinner_update.markdown("""
                <div style="text-align: center; padding: 15px;">
                    <div class="spinner-large">⏳</div>
                    <div style="font-size: 18px; margin-top: 10px; font-weight: bold;">Connecting to Innogen database...</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Step 1: Export from database
                success_export, message_export, count_export = export_table_item_from_innogen()
                
                if success_export:
                    spinner_update.markdown("""
                    <div style="text-align: center; padding: 15px;">
                        <div class="spinner-large">⏳</div>
                        <div style="font-size: 18px; margin-top: 10px; font-weight: bold;">Merging Standard Cost...</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Step 2: Merge Standard Cost
                    success_merge, message_merge, updated_count = merge_standard_cost_to_cross_ref()
                    spinner_update.empty()
                    
                    if success_merge:
                        st.success(f"✅ **Update Completed Successfully!**\n\n📥 **Export**: {message_export}\n🔄 **Merge**: {message_merge}")
                        # Clear cache to reload cross-reference data
                        load_cross_reference_csv.clear()
                        st.rerun()
                    else:
                        st.warning(f"⚠️ **Export succeeded but merge failed**\n\n📥 **Export**: ✅ {message_export}\n🔄 **Merge**: ❌ {message_merge}")
                else:
                    spinner_update.empty()
                    st.error(f"❌ **Export Failed**\n\n{message_export}")

st.divider()

# Shared spinner styles for validation and processing
st.markdown("""
<style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .spinner {
        display: inline-block;
        animation: spin 1s linear infinite;
        font-size: 40px;
        margin-right: 10px;
    }
    .spinner-large {
        display: inline-block;
        animation: spin 1s linear infinite;
        font-size: 60px;
        margin: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# File upload section
col1, col2 = st.columns(2)

with col1:
    # Hide the "Browse files" button using CSS
    st.markdown("""
    <style>
        [data-testid="stFileUploader"] button[kind="secondary"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("📁 Drag and Drop ZIP file or Folder containing TXT files",
        type=['zip', 'txt'],
        accept_multiple_files=True,
        help='Upload a ZIP file (preserves folder structure) or select multiple TXT files from a folder.',
        key='file_uploader'
    )

with col2:
    st.subheader('📄 File Information')
    if uploaded_files:
        # Separate ZIP files from TXT files
        zip_files_list = [f for f in uploaded_files if f.name.lower().endswith('.zip')]
        txt_files_list = [f for f in uploaded_files if f.name.lower().endswith('.txt')]
        
        total_txt_in_zips = 0
        zip_info_list = []
        
        # Process ZIP files for information display
        if zip_files_list:
            for zip_file_obj in zip_files_list:
                try:
                    # Read ZIP to count .txt files
                    # Store the file bytes in session state cache to reuse later
                    if zip_file_obj.name not in st.session_state.zip_files_cache:
                        zip_content = zip_file_obj.read()
                        st.session_state.zip_files_cache[zip_file_obj.name] = zip_content
                    else:
                        zip_content = st.session_state.zip_files_cache[zip_file_obj.name]
                    
                    zip_file = zipfile.ZipFile(BytesIO(zip_content))
                    txt_files = [f for f in zip_file.namelist() if f.lower().endswith('.txt')]
                    zip_file.close()
                    total_txt_in_zips += len(txt_files)
                    zip_info_list.append({
                        'name': zip_file_obj.name,
                        'size': zip_file_obj.size,
                        'txt_count': len(txt_files)
                    })
                except Exception as e:
                    zip_info_list.append({
                        'name': zip_file_obj.name,
                        'size': zip_file_obj.size,
                        'txt_count': 0,
                        'error': str(e)
                    })
        
        # Display summary
        if zip_files_list:
            st.info(f'**{len(zip_files_list)} ZIP file(s) selected**')
            if total_txt_in_zips > 0:
                st.caption(f'Contains {total_txt_in_zips} .txt file(s) total')
            zip_total_size = sum(f.size for f in zip_files_list)
            st.caption(f'ZIP total size: {zip_total_size / (1024*1024):.2f} MB')
        
        if txt_files_list:
            if zip_files_list:
                st.info(f'**{len(txt_files_list)} .txt file(s) selected**')
            else:
                st.info(f'**{len(txt_files_list)} .txt file(s) selected**')
            txt_total_size = sum(f.size for f in txt_files_list)
            st.caption(f'TXT total size: {txt_total_size / (1024*1024):.2f} MB')
        
        if not zip_files_list and not txt_files_list:
            st.warning('No ZIP or TXT files found in upload.')

        header_validations = []
        if zip_files_list or txt_files_list:
            upload_fingerprint = _upload_files_fingerprint(uploaded_files)
            if st.session_state.file_header_validation_fingerprint != upload_fingerprint:
                with st.spinner("Validating Data"):
                    st.session_state.file_header_validations = collect_txt_header_validations(
                        uploaded_files,
                        st.session_state.zip_files_cache,
                    )
                    st.session_state.file_header_validation_fingerprint = upload_fingerprint
                st.rerun()

            header_validations = st.session_state.file_header_validations
            if header_validations:
                mismatch_count = sum(
                    1 for validation in header_validations if _validation_has_mismatch(validation)
                )
                if mismatch_count:
                    st.warning(
                        f"⚠️ {mismatch_count} of {len(header_validations)} TXT file(s) "
                        f"have header claims that do not match parsed data."
                    )
                else:
                    st.success(
                        f"✅ All {len(header_validations)} TXT file(s) match their header claims."
                    )

        # File details expander
        if zip_files_list or txt_files_list:
            with st.expander("📋 View File Details", expanded=False):
                file_info_data = []
                zip_amount_totals = summarize_amounts_by_zip(header_validations)
                
                # Add ZIP files
                for idx, zip_info in enumerate(zip_info_list):
                    zip_total_amount = zip_amount_totals.get(zip_info['name'])
                    file_info_data.append({
                        'File #': idx + 1,
                        'File Name': zip_info['name'],
                        'Size (MB)': f"{zip_info['size'] / (1024*1024):.2f}",
                        'Total Amount': format_total_amount_display(zip_total_amount),
                        'Type': 'application/zip',
                        'Note': f"Contains {zip_info['txt_count']} .txt files" if zip_info['txt_count'] > 0 else f"Error: {zip_info.get('error', 'Unknown')}"
                    })
                
                # Add TXT files
                for idx, file in enumerate(txt_files_list):
                    file_path = file.name
                    file_path = file_path.replace('\\', '/')
                    file_name_only = file_path.split('/')[-1]
                    txt_total_amount = get_validation_amount_for_file(header_validations, file_path)
                    
                    # Check if this file failed to process
                    failed_note = 'Direct TXT file'
                    for failed_file in st.session_state.failed_files:
                        if failed_file.get('file_name', '') == file_path or failed_file.get('file_name', '').endswith(file_name_only):
                            failed_note = f"❌ Failed: {failed_file.get('error_reason', 'Unknown error')}"
                            break
                    
                    file_info_data.append({
                        'File #': len(zip_files_list) + idx + 1,
                        'File Name': file_name_only,
                        'Size (MB)': f"{file.size / (1024*1024):.2f}",
                        'Total Amount': format_total_amount_display(txt_total_amount),
                        'Type': file.type if hasattr(file, 'type') else 'text/plain',
                        'Note': failed_note
                    })
                
                # Add failed TXT files from ZIP files
                if st.session_state.failed_files:
                    failed_count = 0
                    for failed_file in st.session_state.failed_files:
                        file_name = failed_file.get('file_name', '')
                        # Check if it's a file from a ZIP (contains /)
                        if '/' in file_name and file_name.lower().endswith('.txt'):
                            failed_count += 1
                            file_name_only = file_name.split('/')[-1]
                            zip_name = file_name.split('/')[0]
                            
                            file_info_data.append({
                                'File #': len(zip_files_list) + len(txt_files_list) + failed_count,
                                'File Name': file_name_only,
                                'Size (MB)': 'N/A',
                                'Total Amount': 'N/A',
                                'Type': 'text/plain',
                                'Note': f"❌ Failed: {failed_file.get('error_reason', 'Unknown error')} (from {zip_name})"
                            })
                
                file_info_df = pd.DataFrame(file_info_data)
                st.dataframe(file_info_df, use_container_width=True, hide_index=True)

                if header_validations:
                    st.markdown("**Header Validation (Number of Records / Total Amount)**")
                    validation_display_df = build_header_validation_display_df(header_validations)
                    styled_validation_df = style_header_validation_df(
                        validation_display_df,
                        header_validations,
                    )
                    st.dataframe(styled_validation_df, use_container_width=True, hide_index=True)
                
                # Show summary of failed files if any
                if st.session_state.failed_files:
                    failed_count = len(st.session_state.failed_files)
                    st.warning(f"⚠️ {failed_count} file(s) failed to process. See details above.")
    else:
        st.info('No file selected')

# Clear button
if st.button('🗑️ Clear All', type='secondary'):
    st.session_state.combined_df = pd.DataFrame()
    st.session_state.failed_files = []
    st.session_state.processed_files = []
    st.session_state.combined_summary = {
        'num_records': 0,
        'total_amount': 0.0,
        'file_count': 0,
    }
    # Clear matched dataframe
    if 'matched_df' in st.session_state:
        st.session_state.matched_df = None
    # Clear ZIP file cache
    if 'zip_files_cache' in st.session_state:
        st.session_state.zip_files_cache = {}
    st.session_state.file_header_validations = []
    st.session_state.file_header_validation_fingerprint = None
    st.rerun()

# Process files
if uploaded_files:
    all_dataframes = []
    combined_summary = {
        'num_records': 0,
        'total_amount': 0.0,
        'file_count': 0,
    }
    
    # CRITICAL: Track individual file totals to validate against summary
    file_amount_totals = []
    
    # Progress bar with spinner animation
    spinner_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Add CSS for spinning animation
    st.markdown("""
    <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .spinner {
                display: inline-block;
                animation: spin 1s linear infinite;
                font-size: 40px;
                margin-right: 10px;
            }
        .spinner-large {
            display: inline-block;
            animation: spin 1s linear infinite;
            font-size: 60px;
            margin: 20px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Show spinner animation (will be visible during processing)
    spinner_placeholder.markdown('<div class="spinner">⏳</div> <span style="font-size: 18px;">Data Cleaning and Processing...</span>', unsafe_allow_html=True)
    
    # Separate ZIP files from TXT files
    zip_files_to_process = [f for f in uploaded_files if f.name.lower().endswith('.zip')]
    txt_files_to_process = [f for f in uploaded_files if f.name.lower().endswith('.txt')]
    
    # Count total files to process
    total_txt_count = 0
    zip_txt_counts = {}
    
    # Count TXT files in each ZIP
    for zip_file_obj in zip_files_to_process:
        try:
            if zip_file_obj.name not in st.session_state.zip_files_cache:
                zip_content = zip_file_obj.read()
                st.session_state.zip_files_cache[zip_file_obj.name] = zip_content
            else:
                zip_content = st.session_state.zip_files_cache[zip_file_obj.name]
            
            zip_file = zipfile.ZipFile(BytesIO(zip_content))
            txt_files_in_zip = [f for f in zip_file.namelist() if f.lower().endswith('.txt')]
            zip_file.close()
            zip_txt_counts[zip_file_obj.name] = len(txt_files_in_zip)
            total_txt_count += len(txt_files_in_zip)
        except Exception:
            zip_txt_counts[zip_file_obj.name] = 0
    
    total_txt_count += len(txt_files_to_process)
    
    # Check if we have any files to process
    if total_txt_count == 0:
        st.error('No .txt files found. Please upload .txt files or ZIP files containing .txt files.')
        progress_bar.progress(1.0)
        status_text.text('No .txt files found.')
        spinner_placeholder.empty()
        st.stop()
    
    status_text.text(f'Found {total_txt_count} .txt file(s) to process. Processing...')
    
    processed_count = 0
    
    # Process ZIP files first
    for zip_idx, zip_file_obj in enumerate(zip_files_to_process):
        try:
            # Read ZIP file content (reuse from cache if already read)
            if zip_file_obj.name not in st.session_state.zip_files_cache:
                zip_content = zip_file_obj.read()
                st.session_state.zip_files_cache[zip_file_obj.name] = zip_content
            else:
                zip_content = st.session_state.zip_files_cache[zip_file_obj.name]
            
            zip_file = zipfile.ZipFile(BytesIO(zip_content))
            
            # Get list of .txt files from ZIP (preserving folder structure)
            txt_files_in_zip = [f for f in zip_file.namelist() if f.lower().endswith('.txt')]
            
            if not txt_files_in_zip:
                zip_file.close()
                continue
            
            # Process each .txt file from the ZIP
            for file_idx, file_path_in_zip in enumerate(txt_files_in_zip):
                processed_count += 1
                progress = max(0.0, min(1.0, (processed_count - 1) / total_txt_count)) if total_txt_count > 0 else 0.0
                
                progress_bar.progress(progress)
                status_text.text(f'Processing ZIP {zip_idx + 1}/{len(zip_files_to_process)}: {zip_file_obj.name} - File {processed_count}/{total_txt_count} - {file_path_in_zip}')
                
                try:
                    # Read file content from ZIP
                    content = zip_file.read(file_path_in_zip)
                    # Normalize path separators and include ZIP name in path
                    file_location = f"{zip_file_obj.name}/{file_path_in_zip}".replace('\\', '/')
                    
                    # Load and parse data
                    df, summary = load_data_from_content(content, file_location)
                    
                    if df is not None and not df.empty:
                        # CRITICAL: Ensure Amount is numeric BEFORE adding to all_dataframes
                        # This prevents string concatenation during pd.concat()
                        if 'Amount' in df.columns:
                            # Unified Amount Pipeline: Standardize to float64
                            df = standardize_amount_column(df, 'Amount')
                            
                            # CRITICAL: Check for string concatenation - if sum looks wrong, investigate
                            # Sample a few Amount values to check if they're strings
                            sample_amounts = df['Amount'].head(5).tolist()
                            has_string_values = any(isinstance(val, str) for val in sample_amounts)
                            
                            if has_string_values:
                                # CRITICAL ERROR: Amount column still contains strings
                                error_msg = (
                                    f"❌ CRITICAL ERROR: Amount Column Contains Strings!\n\n"
                                    f"   File:              {file_location}\n"
                                    f"   Records:           {len(df):,}\n"
                                    f"   Sample Amounts:    {sample_amounts}\n\n"
                                    f"   The Amount column in this file contains string values.\n"
                                    f"   This will cause string concatenation instead of numeric addition.\n\n"
                                    f"   Processing has been HALTED.\n\n"
                                    f"   Please check this file for Amount formatting issues."
                                )
                                progress_bar.progress(1.0)
                                status_text.text(f'ERROR: Amount contains strings in {file_location}')
                                spinner_placeholder.empty()
                                st.error(error_msg)
                                st.stop()
                            
                            # Track this file's total for validation
                            file_total = float(df['Amount'].sum())
                            file_amount_totals.append(file_total)
                            
                            # CRITICAL: Validate against file summary - HALT if discrepancy found
                            # This catches corruption during file loading phase
                            if 'total_amount' in summary:
                                summary_total = summary.get('total_amount', 0.0)
                                if isinstance(summary_total, str):
                                    summary_total = float(str(summary_total).replace(',', '').replace('P', '').replace(' ', '').strip())
                                if summary_total > 0 and abs(file_total - summary_total) > 0.01:
                                    # CRITICAL ERROR: Amount corruption detected in this file
                                    error_msg = (
                                        f"❌ CRITICAL ERROR: Amount Corruption Detected in File!\n\n"
                                        f"   File:              {file_location}\n"
                                        f"   DataFrame Total:   P {file_total:,.2f}\n"
                                        f"   Summary Total:     P {summary_total:,.2f}\n"
                                        f"   Difference:        P {abs(file_total - summary_total):,.2f}\n"
                                        f"   Records:           {len(df):,}\n\n"
                                        f"   The Amount column in this file was corrupted during loading.\n"
                                        f"   Amount values were concatenated as strings instead of added as numbers.\n\n"
                                        f"   Processing has been HALTED.\n\n"
                                        f"   Please check this file for Amount formatting issues:\n"
                                        f"   - Verify Amount values are properly formatted\n"
                                        f"   - Check for extra spaces or characters in Amount column\n"
                                        f"   - Ensure Amount values are numeric (not formatted strings)"
                                    )
                                    progress_bar.progress(1.0)
                                    status_text.text(f'ERROR: Amount corruption in {file_location}')
                                    spinner_placeholder.empty()
                                    st.error(error_msg)
                                    st.stop()  # Halt processing immediately
                        
                        all_dataframes.append(df)
                        combined_summary['file_count'] += 1
                        update_summary(combined_summary, summary)
                        st.session_state.processed_files.append(file_location)
                    elif 'error' in summary:
                        st.session_state.failed_files.append({
                            'file_name': file_location,
                            'error_reason': summary['error']
                        })
                except Exception as e:
                    file_location_error = f"{zip_file_obj.name}/{file_path_in_zip}".replace('\\', '/')
                    st.session_state.failed_files.append({
                        'file_name': file_location_error,
                        'error_reason': f'Error: {str(e)}'
                    })
            
            zip_file.close()
            
        except zipfile.BadZipFile:
            st.session_state.failed_files.append({
                'file_name': zip_file_obj.name,
                'error_reason': 'Invalid ZIP file format'
            })
        except Exception as e:
            st.session_state.failed_files.append({
                'file_name': zip_file_obj.name,
                'error_reason': f'Error reading ZIP: {str(e)}'
            })
    
    # Process direct TXT files
    for file_idx, uploaded_file in enumerate(txt_files_to_process):
        processed_count += 1
        progress = max(0.0, min(1.0, (processed_count - 1) / total_txt_count)) if total_txt_count > 0 else 0.0
        
        progress_bar.progress(progress)
        status_text.text(f'Processing: {processed_count}/{total_txt_count} files - {uploaded_file.name}')
        
        try:
            # Read file content
            content = uploaded_file.read()
            file_location = uploaded_file.name.replace('\\', '/')
            
            # Load and parse data
            df, summary = load_data_from_content(content, file_location)
            
            if df is not None and not df.empty:
                # CRITICAL: Ensure Amount is numeric BEFORE adding to all_dataframes
                # This prevents string concatenation during pd.concat()
                if 'Amount' in df.columns:
                    # Unified Amount Pipeline: Standardize to float64
                    df = standardize_amount_column(df, 'Amount')
                    
                    # CRITICAL: Check for string concatenation - if sum looks wrong, investigate
                    # Sample a few Amount values to check if they're strings
                    sample_amounts = df['Amount'].head(5).tolist()
                    has_string_values = any(isinstance(val, str) for val in sample_amounts)
                    
                    if has_string_values:
                        # CRITICAL ERROR: Amount column still contains strings
                        error_msg = (
                            f"❌ CRITICAL ERROR: Amount Column Contains Strings!\n\n"
                            f"   File:              {file_location}\n"
                            f"   Records:           {len(df):,}\n"
                            f"   Sample Amounts:    {sample_amounts}\n\n"
                            f"   The Amount column in this file contains string values.\n"
                            f"   This will cause string concatenation instead of numeric addition.\n\n"
                            f"   Processing has been HALTED.\n\n"
                            f"   Please check this file for Amount formatting issues."
                        )
                        progress_bar.progress(1.0)
                        status_text.text(f'ERROR: Amount contains strings in {file_location}')
                        spinner_placeholder.empty()
                        st.error(error_msg)
                        st.stop()
                    
                    # Track this file's total for validation
                    file_total = float(df['Amount'].sum())
                    file_amount_totals.append(file_total)
                    
                    # CRITICAL: Validate against file summary - HALT if discrepancy found
                    # This catches corruption during file loading phase
                    if 'total_amount' in summary:
                        summary_total = summary.get('total_amount', 0.0)
                        if isinstance(summary_total, str):
                            summary_total = float(str(summary_total).replace(',', '').replace('P', '').replace(' ', '').strip())
                        if summary_total > 0 and abs(file_total - summary_total) > 0.01:
                            # CRITICAL ERROR: Amount corruption detected in this file
                            error_msg = (
                                f"❌ CRITICAL ERROR: Amount Corruption Detected in File!\n\n"
                                f"   File:              {file_location}\n"
                                f"   DataFrame Total:   P {file_total:,.2f}\n"
                                f"   Summary Total:     P {summary_total:,.2f}\n"
                                f"   Difference:        P {abs(file_total - summary_total):,.2f}\n"
                                f"   Records:           {len(df):,}\n\n"
                                f"   The Amount column in this file was corrupted during loading.\n"
                                f"   Amount values were concatenated as strings instead of added as numbers.\n\n"
                                f"   Processing has been HALTED.\n\n"
                                f"   Please check this file for Amount formatting issues:\n"
                                f"   - Verify Amount values are properly formatted\n"
                                f"   - Check for extra spaces or characters in Amount column\n"
                                f"   - Ensure Amount values are numeric (not formatted strings)"
                            )
                            progress_bar.progress(1.0)
                            status_text.text(f'ERROR: Amount corruption in {file_location}')
                            spinner_placeholder.empty()
                            st.error(error_msg)
                            st.stop()  # Halt processing immediately
                
                all_dataframes.append(df)
                combined_summary['file_count'] += 1
                update_summary(combined_summary, summary)
                st.session_state.processed_files.append(file_location)
            elif 'error' in summary:
                st.session_state.failed_files.append({
                    'file_name': file_location,
                    'error_reason': summary['error']
                })
        except Exception as e:
            file_location_error = uploaded_file.name.replace('\\', '/')
            st.session_state.failed_files.append({
                'file_name': file_location_error,
                'error_reason': f'Error: {str(e)}'
            })
    
    # Combine all dataframes
    if all_dataframes:
        status_text.text('Combining dataframes...')
        progress_bar.progress(0.9)
        
        try:
            # CRITICAL: Ensure all Amount columns are numeric before concatenation
            # This prevents pandas from converting to object/string type during concat
            
            # Calculate total BEFORE concatenation for validation
            total_before_concat = 0.0
            for df in all_dataframes:
                if 'Amount' in df.columns:
                    # Unified Amount Pipeline: Standardize to float64
                    df = standardize_amount_column(df, 'Amount')
                    total_before_concat += float(df['Amount'].sum())
            
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # CRITICAL: Immediately ensure Amount is numeric after concatenation
            # This prevents any string concatenation issues
            if 'Amount' in combined_df.columns:
                # Unified Amount Pipeline: Standardize to float64
                combined_df = standardize_amount_column(combined_df, 'Amount')
                
                # Validate: Total after concat should equal sum of individual totals
                total_after_concat = float(combined_df['Amount'].sum())
                if abs(total_before_concat - total_after_concat) > 0.01:
                    # Corruption detected during concatenation
                    error_msg = (
                        f"❌ CRITICAL ERROR: Amount Corruption During Concatenation!\n\n"
                        f"   Total Before Concat: P {total_before_concat:,.2f}\n"
                        f"   Total After Concat:  P {total_after_concat:,.2f}\n"
                        f"   Difference:          P {abs(total_before_concat - total_after_concat):,.2f}\n"
                        f"   Records:             {len(combined_df):,}\n\n"
                        f"   The Amount column was corrupted during pd.concat() operation.\n"
                        f"   This indicates some Amount values were strings instead of numbers.\n\n"
                        f"   Processing has been HALTED.\n\n"
                        f"   Please check source files for Amount formatting issues."
                    )
                    progress_bar.progress(1.0)
                    status_text.text('ERROR: Amount corruption during concatenation!')
                    spinner_placeholder.empty()
                    st.error(error_msg)
                    st.stop()
            
            # Cleanup individual dataframes after combining
            del all_dataframes
            cleanup_memory()
        except MemoryError:
            # If memory error, try processing in chunks
            status_text.text('Memory error. Combining in chunks...')
            
            # CRITICAL: Ensure all Amount columns are numeric before concatenation
            
            # Calculate total BEFORE concatenation for validation
            total_before_concat = 0.0
            for df in all_dataframes:
                if 'Amount' in df.columns:
                    # Unified Amount Pipeline: Standardize to float64
                    df = standardize_amount_column(df, 'Amount')
                    total_before_concat += float(df['Amount'].sum())
            
            chunk_size = len(all_dataframes) // 2
            if chunk_size > 0:
                combined_df = pd.concat(all_dataframes[:chunk_size], ignore_index=True)
                # Ensure Amount is numeric after first concat
                if 'Amount' in combined_df.columns:
                    combined_df = standardize_amount_column(combined_df, 'Amount')
                combined_df = pd.concat([combined_df] + all_dataframes[chunk_size:], ignore_index=True)
            else:
                combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # CRITICAL: Ensure Amount is numeric after final concatenation
            if 'Amount' in combined_df.columns:
                combined_df = standardize_amount_column(combined_df, 'Amount')
                
                # Validate: Total after concat should equal sum of individual totals
                total_after_concat = float(combined_df['Amount'].sum())
                if abs(total_before_concat - total_after_concat) > 0.01:
                    # Corruption detected during concatenation
                    error_msg = (
                        f"❌ CRITICAL ERROR: Amount Corruption During Concatenation!\n\n"
                        f"   Total Before Concat: P {total_before_concat:,.2f}\n"
                        f"   Total After Concat:  P {total_after_concat:,.2f}\n"
                        f"   Difference:          P {abs(total_before_concat - total_after_concat):,.2f}\n"
                        f"   Records:             {len(combined_df):,}\n\n"
                        f"   The Amount column was corrupted during pd.concat() operation.\n"
                        f"   This indicates some Amount values were strings instead of numbers.\n\n"
                        f"   Processing has been HALTED.\n\n"
                        f"   Please check source files for Amount formatting issues."
                    )
                    progress_bar.progress(1.0)
                    status_text.text('ERROR: Amount corruption during concatenation!')
                    spinner_placeholder.empty()
                    st.error(error_msg)
                    st.stop()
            
            del all_dataframes
            cleanup_memory()
        
        # Add cross-reference data (InnoGen Item Code and InnoGen Item Name)
        status_text.text('Adding cross-reference data...')
        progress_bar.progress(0.92)
        
        cross_ref_df = load_cross_reference_csv()
        if cross_ref_df is not None and 'Item Code' in combined_df.columns:
            # Clean Item Codes for matching - extract numeric part only
            def clean_item_code_for_matching(value):
                """Extract numeric Item Code for matching."""
                if pd.isna(value) or value == '':
                    return ''
                value_str = str(value).strip()
                # Extract first numeric sequence (digits only)
                match = re.search(r'(\d+)', value_str)
                if match:
                    return match.group(1)
                return value_str
            
            # Create cleaned versions for matching
            combined_df['Item Code_Clean'] = combined_df['Item Code'].apply(clean_item_code_for_matching)
            
            # The cross_ref_df already has cleaned Item Code from load function
            # Merge cross-reference data on cleaned Item Code (include VAT Product Posting Group, Standard Cost, and DIVISION if available)
            merge_columns = ['Item Code', 'InnoGen Item Code', 'InnoGen Item Name']
            if 'VAT Product Posting Group' in cross_ref_df.columns:
                merge_columns.append('VAT Product Posting Group')
            if 'Standard Cost' in cross_ref_df.columns:
                merge_columns.append('Standard Cost')
            if 'DIVISION' in cross_ref_df.columns:
                merge_columns.append('DIVISION')
            
            # CRITICAL: Store Amount before merge to prevent corruption
            amount_before_merge = None
            if 'Amount' in combined_df.columns:
                amount_before_merge = combined_df['Amount'].copy()
            
            combined_df = combined_df.merge(
                cross_ref_df[merge_columns],
                left_on='Item Code_Clean',
                right_on='Item Code',
                how='left',
                suffixes=('', '_ref')
            )
            
            # CRITICAL: Restore Amount after merge (merge should not modify Amount)
            if amount_before_merge is not None and len(amount_before_merge) == len(combined_df):
                combined_df['Amount'] = amount_before_merge.values
            
            # Drop the temporary cleaning column and duplicate Item Code column from merge
            combined_df = combined_df.drop(columns=['Item Code_Clean', 'Item Code_ref'], errors='ignore')
        
        # Ensure Amount column remains numeric (DO NOT format as string - causes data corruption)
        status_text.text('Formatting records...')
        progress_bar.progress(0.95)
        
        if 'Amount' in combined_df.columns:
            # Keep Amount as numeric - do NOT convert to formatted string
            # Formatting will be done only for display purposes
            # Unified Amount Pipeline: Standardize to float64
            combined_df = standardize_amount_column(combined_df, 'Amount')
        
        # Ensure Qty column is numeric type (fixes PyArrow conversion issues)
        if 'Qty' in combined_df.columns:
            combined_df['Qty'] = pd.to_numeric(combined_df['Qty'], errors='coerce')
        
        # CRITICAL: Validate Amount total BEFORE storing in session state
        # This catches corruption during file loading/combining phase
        if 'Amount' in combined_df.columns:
            # Calculate total from dataframe Amount column
            # Unified pipeline guarantees numeric values
            dataframe_total = float(combined_df['Amount'].sum())
            
            # Get summary total for comparison
            summary_total = 0.0
            if 'total_amount' in combined_summary:
                summary_total = combined_summary.get('total_amount', 0.0)
                if isinstance(summary_total, str):
                    summary_total = float(str(summary_total).replace(',', '').replace('P', '').replace(' ', '').strip())
            
            # If summary total is available and there's a discrepancy, show error and halt
            if summary_total > 0 and abs(dataframe_total - summary_total) > 0.01:
                error_msg = (
                    f"❌ CRITICAL ERROR: Amount Corruption Detected During File Loading!\n\n"
                    f"   DataFrame Total: P {dataframe_total:,.2f}\n"
                    f"   Summary Total:   P {summary_total:,.2f}\n"
                    f"   Difference:      P {abs(dataframe_total - summary_total):,.2f}\n"
                    f"   Records:         {len(combined_df):,}\n\n"
                    f"   The Amount column was corrupted during file loading/combining phase.\n"
                    f"   This indicates Amount values were concatenated as strings instead of added as numbers.\n\n"
                    f"   Processing has been HALTED.\n\n"
                    f"   Please:\n"
                    f"   1. Clear all files and re-upload\n"
                    f"   2. Check if source files have properly formatted Amount values\n"
                    f"   3. Verify Amount column parsing in source files"
                )
                progress_bar.progress(1.0)
                status_text.text('ERROR: Amount corruption detected!')
                spinner_placeholder.empty()
                st.error(error_msg)
                st.stop()  # Halt processing immediately
        
        # Store in session state
        st.session_state.combined_df = combined_df
        st.session_state.combined_summary = combined_summary
        
        progress_bar.progress(1.0)
        status_text.text('Complete!')
        # Hide spinner when complete
        spinner_placeholder.empty()
    
    # Display summary
    st.divider()
    st.subheader('📈 Summary')
    
    combined_df_for_validation = st.session_state.combined_df
    summary_validation = validate_combined_summary_totals(
        combined_summary,
        combined_df_for_validation,
    )
    records_validated = summary_validation.get('records_match') is True
    amount_validated = summary_validation.get('amount_match') is True

    if combined_summary['file_count'] > 0:
        col_sum1, col_sum2, col_sum3 = st.columns(3)
    
    with col_sum1:
        st.metric('Files Processed', combined_summary['file_count'])
    
    with col_sum2:
        if combined_summary['num_records'] > 0:
            records_label = '✅ Total Records' if records_validated else 'Total Records'
            st.metric(records_label, f"{combined_summary['num_records']:,}")
    
    with col_sum3:
        if combined_summary['total_amount'] > 0:
            formatted_amount = f"{combined_summary['total_amount']:,.2f}"
            amount_label = '✅ Total Amount' if amount_validated else 'Total Amount'
            st.metric(amount_label, f"P {formatted_amount}")

    if records_validated and amount_validated:
        st.success('✅ Summary totals match combined transaction data.')
    elif combined_df_for_validation is not None and not combined_df_for_validation.empty:
        if summary_validation.get('records_match') is False:
            st.warning(
                f"⚠️ Total Records mismatch: header sum {combined_summary['num_records']:,} "
                f"vs combined data {summary_validation['actual_records']:,}."
            )
        if summary_validation.get('amount_match') is False:
            st.warning(
                f"⚠️ Total Amount mismatch: header sum P {combined_summary['total_amount']:,.2f} "
                f"vs combined data P {summary_validation['actual_amount']:,.2f}."
            )
    
    # Display failed files
    if st.session_state.failed_files:
        st.divider()
        st.subheader('⚠️ Failed Files')
        st.error(f"{len(st.session_state.failed_files)} file(s) failed to process")
        
        failed_df = pd.DataFrame(st.session_state.failed_files)
        st.dataframe(failed_df, use_container_width=True)
    
    # Display table (original data without matching)
    if st.session_state.combined_df is not None and not st.session_state.combined_df.empty:
        st.divider()
        st.subheader('📋 Combined Transaction Data')
        
        # Fill NaN values for display
        display_df = st.session_state.combined_df.fillna('')
        
        # Reorder columns for better display (without matching columns)
        column_order = [
            'Branch Code', 'Branch Name', 'Trans Date', 'Doctor Name', 'PTR No',
            'Address1', 'Address2', "Vendor's Name", 'Supplier Code', 'Supplier Name',
            'Item Code', 'Item Name', 'InnoGen Item Code', 'InnoGen Item Name', 'Standard Cost', 'Qty', 'Amount',
            'VAT Product Posting Group', 'DIVISION', 'file_loc'
        ]
        
        # Only include columns that exist
        display_columns = [col for col in column_order if col in display_df.columns]
        # Add any remaining columns that weren't in the order list
        remaining_cols = [col for col in display_df.columns if col not in display_columns]
        display_columns.extend(remaining_cols)
        display_df = display_df[display_columns]
        
        # Sort by Trans Date
        if 'Trans Date' in display_df.columns:
            try:
                # Create a temporary datetime column for sorting
                display_df['_temp_sort_date'] = pd.to_datetime(display_df['Trans Date'], errors='coerce')
                display_df = display_df.sort_values('_temp_sort_date', na_position='last')
                # Remove temporary column
                display_df = display_df.drop(columns=['_temp_sort_date'])
            except Exception:
                # If sorting fails, try string sort
                try:
                    display_df = display_df.sort_values('Trans Date', na_position='last')
                except Exception:
                    pass
        
        # Clean DataFrame for PyArrow compatibility before display
        display_df = clean_dataframe_for_display(display_df)
        
        # Use st.data_editor with built-in pagination (read-only mode)
        # Height parameter forces pagination when there are many rows
        st.data_editor(
            display_df,
            use_container_width=True,
            height=300,  # Height forces pagination if there are many rows
            disabled=True,  # Make it read-only
            hide_index=True,
            num_rows="fixed"  # Fixed number of rows per page (default is 10)
        )
        
        # Download button for original data
        @st.fragment
        def download_button_fragment():
            # Sort the dataframe for download
            download_df = st.session_state.combined_df.copy()
            if 'Trans Date' in download_df.columns:
                try:
                    # Create a temporary datetime column for sorting
                    download_df['_temp_sort_date'] = pd.to_datetime(download_df['Trans Date'], errors='coerce')
                    download_df = download_df.sort_values('_temp_sort_date', na_position='last')
                    # Remove temporary column
                    download_df = download_df.drop(columns=['_temp_sort_date'])
                except Exception:
                    try:
                        download_df = download_df.sort_values('Trans Date', na_position='last')
                    except Exception:
                        pass
            
            csv = download_df.to_csv(index=False)
            filename = generate_filename_from_trans_date(download_df, 'Summary_RXTracking_Report')
            st.download_button(
                label='📥 Download Combined Data as CSV',
                data=csv,
                file_name=filename,
                mime='text/csv'
            )
        
        download_button_fragment()
    
    # Doctor Name Matching Section - At the bottom
    if st.session_state.combined_df is not None and not st.session_state.combined_df.empty:
        st.divider()
        st.subheader('🔍 RX Tracking Full Process with AI Matching')
        
        # Initialize matched dataframe and matching settings in session state
        if 'matched_df' not in st.session_state:
            st.session_state.matched_df = None
        if 'matching_settings' not in st.session_state:
            st.session_state.matching_settings = {
                'matching_mode': 'Basic',
                'use_doctor_name_basic': True,
                'use_ptr_no_basic': True,
                'use_tfidf_matching': True,  # Enable TF-IDF by default
                'use_exact_address_matching': True,  # Enable exact address matching by default
                'use_quick_suggest_matching': True,  # Enable quick suggest matching by default
                'use_branch_filter': True,
                'use_product_filter': True,
                'use_name_matching': True,
                'use_ptr_matching': True,
                'similarity_threshold': 0.7,
                'ptr_threshold': 0.4,
                # AI Matching thresholds for Basic mode
                'name_threshold_both': 0.7,  # When both Doctor Name and PTR No are checked
                'name_threshold_name_only': 0.85,  # When only Doctor Name is checked
                'doctor_similarity_threshold': 0.7,  # When matching by PTR, validates doctor name similarity
                'doctor_similarity_threshold_exact_ptr': 0.7  # When PTR matches exactly
            }
        
        # Wrap the entire RX Tracking section in a fragment to prevent full page reloads
        @st.fragment
        def rx_tracking_fragment():
            # Include CSS for spinner animation
            st.markdown("""
            <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .spinner-large {
                display: inline-block;
                animation: spin 1s linear infinite;
                font-size: 60px;
                margin: 20px;
                text-align: center;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Matching Mode Selection (Advanced visible but disabled - Basic is default and only selectable)
            # Streamlit radio doesn't support per-option disabled; we force Basic when Advanced is selected
            if st.session_state.get('matching_mode_radio') == 'Advanced':
                st.session_state.matching_mode_radio = 'Basic'
            
            matching_mode = st.radio(
                "**Select Matching Mode:**",
                ['Basic', 'Advanced'],
                index=0,
                horizontal=True,
                format_func=lambda x: "Basic" if x == "Basic" else "Advanced (disabled)",
                help="Basic: Fast exact matching. Advanced: Currently disabled.",
                key='matching_mode_radio'
            )
            
            # Ensure Basic is always used (Advanced option is disabled)
            if matching_mode == 'Advanced':
                matching_mode = 'Basic'
            
            # Matching Configuration Panel
            if matching_mode == 'Advanced':
                with st.expander("⚙️ Advanced Matching Configuration", expanded=True):
                    col_config1, col_config2 = st.columns(2)
                    
                    with col_config1:
                        st.markdown("**Matching Criteria:**")
                        use_branch_filter = st.checkbox(
                            "Filter by Branch Code",
                            value=st.session_state.matching_settings.get('use_branch_filter', True),
                            help="Narrow down matches to doctors who work at the same branch",
                            key='adv_branch_filter'
                        )
                        use_product_filter = st.checkbox(
                            "Filter by Product Code",
                            value=st.session_state.matching_settings.get('use_product_filter', True),
                            help="Narrow down matches to doctors who officially carry the product",
                            key='adv_product_filter'
                        )
                        use_name_matching = st.checkbox(
                            "Use Doctor Name Fuzzy Matching",
                            value=st.session_state.matching_settings.get('use_name_matching', True),
                            help="Enable fuzzy matching on doctor names (required for name-based matches)",
                            key='adv_name_matching'
                        )
                        use_ptr_matching = st.checkbox(
                            "Use PTR No Matching",
                            value=st.session_state.matching_settings.get('use_ptr_matching', True),
                            help="Enable PTR No matching (exact match with high score boost)",
                            key='adv_ptr_matching'
                        )
                        use_quick_suggest_matching = st.checkbox(
                            "Enable Quick Suggest Matching (Word-Based)",
                            value=st.session_state.matching_settings.get('use_quick_suggest_matching', True),
                            help="Enable word-based quick suggest matching. Matches by word extraction from Doctor Name and validates with PTR No and Address1. This is the final matching step.",
                            key='adv_quick_suggest_matching'
                        )
                    
                    with col_config2:
                        st.markdown("**Sensitivity Settings:**")
                        similarity_threshold = st.slider(
                            "Name Similarity Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=st.session_state.matching_settings.get('similarity_threshold', 0.7),
                            step=0.05,
                            help="Minimum similarity score (0.0-1.0) for name matching. Higher = stricter matching. Default: 0.7 (70%)",
                            key='adv_similarity_threshold'
                        )
                        st.caption(f"Current threshold: {similarity_threshold:.0%}")
                        
                        ptr_threshold = st.slider(
                            "PTR Match Threshold",
                            min_value=0.0,
                            max_value=0.6,
                            value=st.session_state.matching_settings.get('ptr_threshold', 0.4),
                            step=0.05,
                            help="Minimum similarity score when PTR No matches exactly. Lower = more matches. Default: 0.4 (40%)",
                            key='adv_ptr_threshold'
                        )
                        st.caption(f"PTR match threshold: {ptr_threshold:.0%}")
                
                # Update session state with current settings
                st.session_state.matching_settings = {
                    'matching_mode': matching_mode,
                    'use_branch_filter': use_branch_filter,
                    'use_product_filter': use_product_filter,
                    'use_name_matching': use_name_matching,
                    'use_ptr_matching': use_ptr_matching,
                    'use_quick_suggest_matching': use_quick_suggest_matching,
                    'similarity_threshold': similarity_threshold,
                    'ptr_threshold': ptr_threshold
                }
            else:
                # Basic matching mode - show checkboxes for Doctor Name and PTR No
                with st.expander("⚙️ Basic Matching Configuration", expanded=False):
                    st.markdown("**Select Matching Criteria:** (Choose both for best results)")
                    use_doctor_name_basic = st.checkbox(
                        "Use Doctor Name",
                        value=st.session_state.matching_settings.get('use_doctor_name_basic', True),
                        help="Match using Doctor Name (exact match with md_suggest from masterlist)",
                        key='basic_doctor_name'
                    )
                    use_ptr_no_basic = st.checkbox(
                        "Use PTR No",
                        value=st.session_state.matching_settings.get('use_ptr_no_basic', True),
                        help="Match using PTR No (exact match with md_ptrs from masterlist)",
                        key='basic_ptr_no'
                    )
                    
                    st.divider()
                    st.markdown("**AI Matching (TF-IDF):**")
                    use_tfidf_matching = st.checkbox(
                        "Enable AI Matching (TF-IDF)",
                        value=st.session_state.matching_settings.get('use_tfidf_matching', True),
                        help="Enable TF-IDF AI matching for unmatched records. Disable this for low-memory systems to prevent crashes.",
                        key='basic_tfidf_matching'
                    )
                    if not use_tfidf_matching:
                        st.caption("⚠️ AI Matching disabled. Only exact matches will be performed. Recommended for systems with less than 2GB RAM.")
                                          
                    # AI Matching Threshold Controls in an expander (only show when TF-IDF is enabled)
                    if use_tfidf_matching:
                        with st.expander("⚙️ AI Matching Threshold Controls", expanded=False):
                            st.markdown("**Adjust similarity thresholds for AI matching:**")
                            
                            # Doctor Name Matching Thresholds
                            col_thresh1, col_thresh2 = st.columns(2)
                            
                            with col_thresh1:
                                st.markdown("**Doctor Name Matching:**")
                                name_threshold_both = st.slider(
                                    "When Both Doctor Name + PTR No checked",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=st.session_state.matching_settings.get('name_threshold_both', 0.7),
                                    step=0.05,
                                    help="Similarity threshold (0.0-1.0) for Doctor Name matching when both Doctor Name and PTR No are checked. Default: 0.7 (70%)",
                                    key='slider_name_threshold_both'
                                )
                                st.caption(f"Current: {name_threshold_both:.0%}")
                                
                                name_threshold_name_only = st.slider(
                                    "When Only Doctor Name checked",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=st.session_state.matching_settings.get('name_threshold_name_only', 0.85),
                                    step=0.05,
                                    help="Similarity threshold (0.0-1.0) for Doctor Name matching when only Doctor Name is checked (no PTR validation). Default: 0.85 (85%)",
                                    key='slider_name_threshold_name_only'
                                )
                                st.caption(f"Current: {name_threshold_name_only:.0%}")
                            
                            with col_thresh2:
                                st.markdown("**PTR Number Matching:**")
                                doctor_similarity_threshold = st.slider(
                                    "Doctor Name Similarity (PTR Matching)",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=st.session_state.matching_settings.get('doctor_similarity_threshold', 0.7),
                                    step=0.05,
                                    help="Validates doctor name similarity when matching by PTR number. Default: 0.7 (70%)",
                                    key='slider_doctor_similarity_threshold'
                                )
                                st.caption(f"Current: {doctor_similarity_threshold:.0%}")
                                
                                doctor_similarity_threshold_exact_ptr = st.slider(
                                    "Doctor Name Similarity (Exact PTR Match)",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=st.session_state.matching_settings.get('doctor_similarity_threshold_exact_ptr', 0.7),
                                    step=0.05,
                                    help="More lenient threshold when PTR matches exactly. Default: 0.7 (70%)",
                                    key='slider_doctor_similarity_threshold_exact_ptr'
                                )
                                st.caption(f"Current: {doctor_similarity_threshold_exact_ptr:.0%}")
                            
                            # Store threshold values in session state
                            st.session_state.matching_settings['name_threshold_both'] = name_threshold_both
                            st.session_state.matching_settings['name_threshold_name_only'] = name_threshold_name_only
                            st.session_state.matching_settings['doctor_similarity_threshold'] = doctor_similarity_threshold
                            st.session_state.matching_settings['doctor_similarity_threshold_exact_ptr'] = doctor_similarity_threshold_exact_ptr
                      
                    use_exact_address_matching = st.checkbox(
                        "Enable Exact Matching (MD Suggest + Address)",
                        value=st.session_state.matching_settings.get('use_exact_address_matching', True),
                        help="Enable exact matching of md_suggest + md_add_1 to Doctor Name + Address1. This step runs a simple AI matching.",
                        key='basic_exact_address_matching'
                    )
                    if not use_exact_address_matching:
                        st.caption("⚠️ Exact address matching disabled. This step will be skipped.")
                    
                    st.divider()
                    st.markdown("**Quick Suggest Matching:**")
                    use_quick_suggest_matching = st.checkbox(
                        "Enable Quick Suggest Matching (Word-Based)",
                        value=st.session_state.matching_settings.get('use_quick_suggest_matching', True),
                        help="Enable word-based quick suggest matching. Matches by word extraction from Doctor Name and validates with PTR No and Address1. This is the final matching step.",
                        key='basic_quick_suggest_matching'
                    )
                    if not use_quick_suggest_matching:
                        st.caption("⚠️ Quick Suggest matching disabled. This step will be skipped.")
                    
                    if not use_doctor_name_basic and not use_ptr_no_basic:
                        st.warning('⚠️ Please select at least one matching criterion (Doctor Name or PTR No).')
                    else:
                        if use_doctor_name_basic and use_ptr_no_basic:
                            st.info('✅ **Matching Mode**: Both Doctor Name + PTR No (combined)')
                        elif use_doctor_name_basic:
                            st.info('✅ **Matching Mode**: Doctor Name only')
                        else:
                            st.info('✅ **Matching Mode**: PTR No only')
                
                st.session_state.matching_settings['matching_mode'] = matching_mode
                st.session_state.matching_settings['use_doctor_name_basic'] = use_doctor_name_basic
                st.session_state.matching_settings['use_ptr_no_basic'] = use_ptr_no_basic
                st.session_state.matching_settings['use_quick_suggest_matching'] = use_quick_suggest_matching
                st.session_state.matching_settings['use_tfidf_matching'] = use_tfidf_matching
                st.session_state.matching_settings['use_exact_address_matching'] = use_exact_address_matching
            
            col_btn1, col_btn2 = st.columns([2, 1])
            with col_btn1:
                mode_info = 'Basic matching (fast exact match)' if matching_mode == 'Basic' else 'Advanced matching with configured options'
                st.info(f'Mode: {mode_info}. Click the button to process.')
            with col_btn2:
                # Initialize processing state if not exists
                if 'rx_processing' not in st.session_state:
                    st.session_state.rx_processing = False
                
                # Disable button if processing is in progress
                button_disabled = st.session_state.rx_processing
                
                if st.button('🚀 RX Tracking Full Process', type='primary', use_container_width=True, 
                            key='rx_process_button', disabled=button_disabled):
                    # Set processing flag to True to disable button
                    st.session_state.rx_processing = True
                    # Clear CSV cache when starting new matching (will regenerate after matching completes)
                    if 'matched_csv_cache' in st.session_state:
                        del st.session_state.matched_csv_cache
                    if 'matched_csv_filename' in st.session_state:
                        del st.session_state.matched_csv_filename
                    if 'matched_xlsx_cache' in st.session_state:
                        del st.session_state.matched_xlsx_cache
                    if 'matched_xlsx_filename' in st.session_state:
                        del st.session_state.matched_xlsx_filename
                    
                    # Start timing the process
                    start_time = time.time()
                    start_time_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Helper function to format elapsed time as hh:mm:ss
                    def format_time_hhmmss(elapsed_seconds):
                        """Format elapsed time as hh:mm:ss."""
                        hours = int(elapsed_seconds // 3600)
                        minutes = int((elapsed_seconds % 3600) // 60)
                        seconds = int(elapsed_seconds % 60)
                        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    # Helper function to format elapsed time in a human-readable format (for final messages)
                    def format_elapsed_time(elapsed_seconds):
                        """Format elapsed time in a human-readable format."""
                        if elapsed_seconds < 60:
                            return f"{elapsed_seconds:.1f} seconds"
                        elif elapsed_seconds < 3600:
                            minutes = int(elapsed_seconds // 60)
                            seconds = elapsed_seconds % 60
                            return f"{minutes} minute{'s' if minutes != 1 else ''} {seconds:.1f} second{'s' if seconds != 1 else ''}"
                        else:
                            hours = int(elapsed_seconds // 3600)
                            minutes = int((elapsed_seconds % 3600) // 60)
                            seconds = elapsed_seconds % 60
                            return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''} {seconds:.1f} second{'s' if seconds != 1 else ''}"
                    
                    # Create a custom spinner with larger animation (no live timer during processing)
                    spinner_container = st.empty()
                    
                    # Status message for display
                    current_status_message = {"message": "Processing doctor name matching..."}
                    
                    # Function to update spinner display only (no timer updates during processing)
                    def update_status_message(status_message):
                        """Update the status message."""
                        current_status_message['message'] = status_message
                        spinner_container.markdown(f"""
                        <div style="text-align: center; padding: 20px;">
                            <div class="spinner-large">⏳</div>
                            <div style="font-size: 20px; margin-top: 10px; font-weight: bold;">{current_status_message['message']}</div>
                            <div style="font-size: 14px; margin-top: 10px; color: #666;">Started at: {start_time_formatted}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Initial display
                    update_status_message("Processing doctor name matching...")
                    
                    try:
                        masterlist_df = load_masterlist_csv()
                        if masterlist_df is not None and not masterlist_df.empty:
                            # Create progress bar and status text
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # CRITICAL: Ensure Amount column is numeric BEFORE starting matching process
                            # This prevents data corruption from string concatenation or formatting issues
                            if 'Amount' in st.session_state.combined_df.columns:
                                # Unified Amount Pipeline: Standardize to float64
                                st.session_state.combined_df = standardize_amount_column(st.session_state.combined_df, 'Amount')
                            
                            # CRITICAL: Calculate ORIGINAL Total Amount from Combined Transaction Data BEFORE any matching
                            # This is the FOUNDATION - calculate directly from the dataframe (source of truth)
                            # The dataframe Amount column is what's displayed in the "Combined Transaction Data" table
                            original_total_amount = 0.0
                            
                            if 'Amount' in st.session_state.combined_df.columns:
                                # Amount is already standardized above
                                
                                # Calculate total from cleaned Amount column
                                original_total_amount = float(st.session_state.combined_df['Amount'].sum())
                                
                                # Also get summary total for comparison
                                summary_total = 0.0
                                summary_available = False
                                if 'combined_summary' in st.session_state and 'total_amount' in st.session_state.combined_summary:
                                    summary_total = st.session_state.combined_summary.get('total_amount', 0.0)
                                    if isinstance(summary_total, str):
                                        summary_total = float(str(summary_total).replace(',', '').replace('P', '').replace(' ', '').strip())
                                    if summary_total > 0:
                                        summary_available = True
                                
                                # CRITICAL: Validate Amount totals - HALT if discrepancy found
                                # This prevents processing corrupted data
                                # Only check if summary is available and non-zero
                                if summary_available and abs(original_total_amount - summary_total) > 0.01:
                                    # Calculate elapsed time even for error case
                                    elapsed_time = time.time() - start_time
                                    elapsed_time_str = format_elapsed_time(elapsed_time)
                                    
                                    error_msg = (
                                        f"❌ CRITICAL ERROR: Amount Discrepancy Detected!\n\n"
                                        f"   DataFrame Total: P {original_total_amount:,.2f}\n"
                                        f"   Summary Total:   P {summary_total:,.2f}\n"
                                        f"   Difference:      P {abs(original_total_amount - summary_total):,.2f}\n"
                                        f"   Records:         {len(st.session_state.combined_df):,}\n\n"
                                        f"   The Amount column in the Combined Transaction Data appears to be corrupted.\n"
                                        f"   This indicates a problem during file loading/combining phase.\n\n"
                                        f"   Processing has been HALTED to prevent further data corruption.\n\n"
                                        f"   Please:\n"
                                        f"   1. Clear all files and re-upload\n"
                                        f"   2. Check if files are properly formatted\n"
                                        f"   3. Verify Amount values in source files\n\n"
                                        f"⏱️ **Time elapsed:** {elapsed_time_str}"
                                    )
                                    spinner_container.empty()
                                    if progress_bar is not None:
                                        progress_bar.empty()
                                    if status_text is not None:
                                        status_text.empty()
                                    st.error(error_msg)
                                    # Re-enable button after error
                                    st.session_state.rx_processing = False
                                    # Cleanup before stopping
                                    cleanup_memory()
                                    st.stop()  # Halt processing immediately
                                else:
                                    if status_text is not None:
                                        status_text.text(f'✅ Original Total Amount (from Combined Transaction Data): P {original_total_amount:,.2f} | Records: {len(st.session_state.combined_df):,}')
                                    # Update status message
                                    update_status_message('Validating data...')
                            else:
                                # Calculate elapsed time even for error case
                                elapsed_time = time.time() - start_time
                                elapsed_time_str = format_elapsed_time(elapsed_time)
                                error_msg = f"❌ CRITICAL ERROR: Amount column not found in Combined Transaction Data! Processing halted.\n\n⏱️ **Time elapsed:** {elapsed_time_str}"
                                spinner_container.empty()
                                if progress_bar is not None:
                                    progress_bar.empty()
                                if status_text is not None:
                                    status_text.empty()
                                st.error(error_msg)
                                # Re-enable button after error
                                st.session_state.rx_processing = False
                                # Cleanup before stopping
                                cleanup_memory()
                                st.stop()  # Halt processing immediately
                            
                            # Group masterlist before matching to reduce rows and speed up processing
                            if masterlist_df is not None and not masterlist_df.empty:
                                if status_text is not None:
                                    original_masterlist_count = len(masterlist_df)
                                    status_text.text(f'Grouping masterlist to reduce rows (from {original_masterlist_count:,} records)...')
                                update_status_message('Grouping masterlist...')
                                masterlist_df = group_masterlist_for_matching(masterlist_df, matching_mode=matching_mode)
                                # Use copy with md_ptrs from md_ptrs_old when valid (else keep md_ptrs) for all matching
                                masterlist_df = prepare_masterlist_for_matching(masterlist_df)
                                if status_text is not None:
                                    grouped_masterlist_count = len(masterlist_df)
                                    reduction_pct = ((original_masterlist_count - grouped_masterlist_count) / original_masterlist_count * 100) if original_masterlist_count > 0 else 0
                                    status_text.text(f'✅ Masterlist grouped: {original_masterlist_count:,} → {grouped_masterlist_count:,} records ({reduction_pct:.1f}% reduction). Starting matching...')
                                update_status_message('Starting matching process...')
                            
                            # Process based on selected mode
                            if matching_mode == 'Basic':
                                use_doctor_name_basic = st.session_state.matching_settings.get('use_doctor_name_basic', True)
                                use_ptr_no_basic = st.session_state.matching_settings.get('use_ptr_no_basic', True)
                                
                                # Check if at least one criterion is selected
                                if not use_doctor_name_basic and not use_ptr_no_basic:
                                    spinner_container.empty()
                                    st.error('❌ Please select at least one matching criterion (Doctor Name or PTR No).')
                                    # Re-enable button after validation error
                                    st.session_state.rx_processing = False
                                    # Cleanup on validation error
                                    cleanup_memory()
                                else:
                                    update_status_message('Performing basic matching...')
                                    matched_df = process_doctor_matching_basic(
                                        st.session_state.combined_df.copy(),
                                        masterlist_df,
                                        use_doctor_name=use_doctor_name_basic,
                                        use_ptr_no=use_ptr_no_basic,
                                        progress_bar=progress_bar,
                                        status_text=status_text
                                    )
                                    
                                    # VALIDATION: Check Amount total after process_doctor_matching_basic
                                    validate_amount_total(matched_df, original_total_amount, "process_doctor_matching_basic")
                                    if status_text is not None:
                                        current_total = matched_df['Amount'].sum()
                                        status_text.text(f'✅ After Basic Matching: P {current_total:,.2f} (unchanged)')

                                    revalidator = DoctorNameMatchRevalidator()
                                    matched_df, reset_true_count = revalidator.revalidate_true_matches(
                                        matched_df,
                                        suggested_col='suggested_md',
                                        md_name_col='md_official_name'
                                    )
                                    if status_text is not None and reset_true_count > 0:
                                        status_text.text(f'🔁 Revalidated suggest_dn=TRUE: {reset_true_count} row(s) set to FALSE')
                                    
                                    # Perform exact matching (md_suggest + md_add_1) before TF-IDF matching (if enabled)
                                    use_exact_address_matching = st.session_state.matching_settings.get('use_exact_address_matching', True)
                                    exact_matched_count = 0
                                    if use_exact_address_matching:
                                        update_status_message('Performing exact address matching...')
                                        matched_df, exact_matched_count = exact_match_suggest_and_address(
                                            matched_df,
                                            masterlist_df,
                                            progress_bar=progress_bar,
                                            status_text=status_text
                                        )
                                        
                                        # VALIDATION: Check Amount total after exact_match_suggest_and_address
                                        validate_amount_total(matched_df, original_total_amount, "exact_match_suggest_and_address")
                                        if status_text is not None:
                                            current_total = matched_df['Amount'].sum()
                                            status_text.text(f'✅ After Exact Matching: P {current_total:,.2f} (unchanged)')
                                        update_status_message('Exact matching completed...')
                                    else:
                                        if status_text is not None:
                                            status_text.text('⏭️ Skipping exact address matching (disabled by user)')
                                        update_status_message('Skipping exact matching...')
                                    
                                    # Perform split matching for unmatched records (suggested_md = False)
                                    use_tfidf_matching = st.session_state.matching_settings.get('use_tfidf_matching', True)
                                    
                                    # Save checkpoint before TF-IDF (most memory-intensive step)
                                    if status_text is not None:
                                        status_text.text('Saving checkpoint before AI matching...')
                                    update_status_message('Saving checkpoint...')
                                    st.session_state.matched_df_checkpoint = matched_df.copy()
                                    cleanup_memory()
                                    
                                    if status_text is not None:
                                        status_text.text('Starting AI matching (TF-IDF). This may take 10-30 minutes for large datasets. Please keep this page open...')
                                    update_status_message('Starting AI matching (TF-IDF)...')
                                    
                                    use_quick_suggest_matching = st.session_state.matching_settings.get('use_quick_suggest_matching', True)
                                    matched_df, ai_matched_count, ai_completion_info = split_matching_for_unmatched(
                                        matched_df,
                                        masterlist_df,
                                        progress_bar=progress_bar,
                                        status_text=status_text,
                                        use_doctor_name=use_doctor_name_basic,
                                        use_ptr_no=use_ptr_no_basic,
                                        use_tfidf_matching=use_tfidf_matching,
                                        use_quick_suggest_matching=use_quick_suggest_matching
                                    )
                                    
                                    # VALIDATION: Check Amount total after split_matching_for_unmatched
                                    validate_amount_total(matched_df, original_total_amount, "split_matching_for_unmatched")
                                    if status_text is not None:
                                        current_total = matched_df['Amount'].sum()
                                        status_text.text(f'✅ After AI Matching: P {current_total:,.2f} (unchanged)')
                                    
                                    # Delete checkpoint immediately after TF-IDF succeeds (no longer needed)
                                    # This frees memory earlier rather than waiting until the end
                                    if 'matched_df_checkpoint' in st.session_state:
                                        del st.session_state.matched_df_checkpoint
                                        st.session_state.matched_df_checkpoint = None
                                        cleanup_memory()
                                    
                                    # Store total AI matched count (exact match + TF-IDF match) in session state for display
                                    total_ai_matched = exact_matched_count + ai_matched_count
                                    # Store completion info for display (include exact match count)
                                    ai_completion_info['md_address_matched'] = exact_matched_count
                                    st.session_state.ai_completion_info = ai_completion_info
                                    st.session_state.ai_matched_count = total_ai_matched
                                    # Add DOCTOR_CODE and CUSTOMER_CODE columns by matching md_official_name
                                    update_status_message('Adding MD codes...')
                                    matched_df = add_md_code_columns(matched_df)
                                    
                                    # VALIDATION: Check Amount total after add_md_code_columns
                                    validate_amount_total(matched_df, original_total_amount, "add_md_code_columns")
                                    if status_text is not None:
                                        current_total = matched_df['Amount'].sum()
                                        status_text.text(f'✅ After Adding MD Codes: P {current_total:,.2f} (unchanged)')
                                    update_status_message('Finalizing results...')
                                    
                                    # CRITICAL: Ensure Amount column remains numeric after matching operations
                                    # This prevents data corruption from string concatenation or formatting issues
                                    if 'Amount' in matched_df.columns:
                                        matched_df = standardize_amount_column(matched_df, 'Amount')
                                    
                                    # Add blank reserved columns for future use
                                    if 'quick_suggest_name' not in matched_df.columns:
                                        matched_df['quick_suggest_name'] = ''
                                    if 'suggested_name' not in matched_df.columns:
                                        matched_df['suggested_name'] = ''
                                    if 'OSCA DISC' not in matched_df.columns:
                                        matched_df['OSCA DISC'] = 0.0  # Initialize as numeric (float)
                                    if 'PERIOD' not in matched_df.columns:
                                        matched_df['PERIOD'] = ''
                                    
                                    # Ensure VAT Product Posting Group exists (from cross-reference, may be blank)
                                    if 'VAT Product Posting Group' not in matched_df.columns:
                                        matched_df['VAT Product Posting Group'] = ''
                                    
                                    # Auto-fill blank OSCA DISC based on VAT Product Posting Group
                                    if 'OSCA DISC' in matched_df.columns and 'VAT Product Posting Group' in matched_df.columns and 'Amount' in matched_df.columns:
                                        # Ensure Amount is standardized before calculation
                                        matched_df = standardize_amount_column(matched_df, 'Amount')
                                        
                                        def calculate_osca_disc(row):
                                            """Calculate OSCA DISC based on VAT Product Posting Group and Amount."""
                                            # Check if OSCA DISC already has a numeric value (not blank/zero)
                                            osca_disc = row['OSCA DISC']
                                            try:
                                                # Try to convert existing value to float
                                                if pd.notna(osca_disc):
                                                    osca_float = float(osca_disc) if not isinstance(osca_disc, (int, float)) else float(osca_disc)
                                                    # If it's a valid non-zero number, keep it
                                                    if osca_float != 0.0:
                                                        return osca_float
                                            except (ValueError, TypeError):
                                                pass  # If conversion fails, proceed to calculate
                                            
                                            vat_group = str(row['VAT Product Posting Group']).strip() if pd.notna(row['VAT Product Posting Group']) else ''
                                            # Amount is guaranteed to be float64 by standardize_amount_column
                                            amount = row['Amount']
                                            
                                            # Apply conditions
                                            if vat_group == 'EXEMPT':
                                                return float(amount * 0.14)
                                            elif vat_group == 'GOODS12':
                                                return float(amount * 0.2)
                                            else:
                                                return 0.0  # Return 0.0 if condition not met
                                        
                                        matched_df['OSCA DISC'] = matched_df.apply(calculate_osca_disc, axis=1)
                                        # Ensure OSCA DISC is numeric (float)
                                        matched_df['OSCA DISC'] = pd.to_numeric(matched_df['OSCA DISC'], errors='coerce').fillna(0.0).astype('float64')
                                    
                                    # Reorder columns to match requested order
                                    requested_column_order = [
                                        'suggested_md', 'md_official_name', 'quick_suggest_name', 'Doctor Name', 'md_ptrs',
                                        'Branch Code', 'Branch Name', 'Trans Date', 'Address1', 'Address2',
                                        'Supplier Code', 'Supplier Name', "Vendor's Name",
                                        'Item Code', 'Item Name', 'InnoGen Item Name', 'Standard Cost', 'Qty', 'Amount',
                                        'OSCA DISC', 'VAT Product Posting Group', 'PERIOD', 'PTR No',
                                        'DOCTOR_CODE', 'CUSTOMER_CODE', 'InnoGen Item Code', 'file_loc'
                                    ]
                                    
                                    # Get columns that exist in the dataframe
                                    existing_cols = [col for col in requested_column_order if col in matched_df.columns]
                                    # Add any remaining columns that weren't in the order list
                                    remaining_cols = [col for col in matched_df.columns if col not in existing_cols]
                                    final_column_order = existing_cols + remaining_cols
                                    matched_df = matched_df[final_column_order]
                                    
                                    # FINAL SAFEGUARD: Ensure Amount is numeric before storing in session state
                                    if 'Amount' in matched_df.columns:
                                        matched_df = standardize_amount_column(matched_df, 'Amount')
                                    
                                    # FINAL VALIDATION: Check Amount total before storing in session state
                                    validate_amount_total(matched_df, original_total_amount, "FINAL (before storing)")
                                    final_total = matched_df['Amount'].sum()
                                    if status_text is not None:
                                        status_text.text(f'✅ FINAL Total Amount: P {final_total:,.2f} (matches original: P {original_total_amount:,.2f})')
                                    
                                    # Calculate elapsed time
                                    elapsed_time = time.time() - start_time
                                    elapsed_time_str = format_elapsed_time(elapsed_time)
                                    
                                    # Stop timer thread
                                    
                                    # Clear spinner display
                                    spinner_container.empty()
                                    
                                    st.session_state.matched_df = matched_df
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                    # Display completion message with AI matching breakdown and elapsed time
                                    completion_msg = f'✅ Matching completed! Processed {len(matched_df)} records.\n\n⏱️ **Processing Time:** {elapsed_time_str}'
                                    if 'ai_completion_info' in st.session_state and st.session_state.ai_completion_info:
                                        info = st.session_state.ai_completion_info
                                        breakdown_parts = []
                                        if info.get('use_doctor_name') and info.get('doctor_name_matched', 0) > 0:
                                            breakdown_parts.append(f"Doctor Name: {info['doctor_name_matched']}")
                                        if info.get('use_ptr_no') and info.get('ptr_matched', 0) > 0:
                                            breakdown_parts.append(f"PTR Number: {info['ptr_matched']}")
                                        if info.get('md_address_matched', 0) > 0:
                                            breakdown_parts.append(f"MD+Address: {info['md_address_matched']}")
                                        if info.get('ppe_doctors_matched', 0) > 0:
                                            breakdown_parts.append(f"PPE Doctors: {info['ppe_doctors_matched']}")
                                        if breakdown_parts:
                                            completion_msg += f"\n\n🤖 AI Matching Breakdown: {' | '.join(breakdown_parts)}"
                                    st.success(completion_msg)
                                    # Re-enable button after successful completion
                                    st.session_state.rx_processing = False
                                    
                                    # Aggressive memory cleanup after successful completion
                                    # Delete checkpoint as it's no longer needed
                                    if 'matched_df_checkpoint' in st.session_state:
                                        del st.session_state.matched_df_checkpoint
                                        st.session_state.matched_df_checkpoint = None
                                    
                                    # Delete intermediate variables that are no longer needed
                                    if 'masterlist_df' in locals():
                                        del masterlist_df
                                    if 'matched_df' in locals():
                                        del matched_df
                                    
                                    # Force multiple garbage collection cycles
                                    for _ in range(3):
                                        cleanup_memory()
                                    
                                    # Results will be displayed below as the fragment continues executing
                            else:
                                # Advanced matching - get settings from session state
                                use_branch_filter = st.session_state.matching_settings.get('use_branch_filter', True)
                                use_product_filter = st.session_state.matching_settings.get('use_product_filter', True)
                                use_name_matching = st.session_state.matching_settings.get('use_name_matching', True)
                                use_ptr_matching = st.session_state.matching_settings.get('use_ptr_matching', True)
                                similarity_threshold = st.session_state.matching_settings.get('similarity_threshold', 0.7)
                                ptr_threshold = st.session_state.matching_settings.get('ptr_threshold', 0.4)
                                
                                matched_df = process_doctor_matching(
                                    st.session_state.combined_df.copy(),
                                    masterlist_df,
                                    use_branch_filter=use_branch_filter,
                                    use_product_filter=use_product_filter,
                                    use_name_matching=use_name_matching,
                                    use_ptr_matching=use_ptr_matching,
                                    similarity_threshold=similarity_threshold,
                                    ptr_threshold=ptr_threshold,
                                    progress_bar=progress_bar,
                                    status_text=status_text
                                )
                                
                                # VALIDATION: Check Amount total after process_doctor_matching
                                validate_amount_total(matched_df, original_total_amount, "process_doctor_matching (Advanced)")
                                if status_text is not None:
                                    matched_df = standardize_amount_column(matched_df, 'Amount')
                                    current_total = matched_df['Amount'].sum()
                                    status_text.text(f'✅ After Advanced Matching: P {current_total:,.2f} (unchanged)')
                                
                                # Perform exact matching (md_suggest + md_add_1) before TF-IDF matching (if enabled)
                                use_exact_address_matching = st.session_state.matching_settings.get('use_exact_address_matching', True)
                                exact_matched_count = 0
                                if use_exact_address_matching:
                                    update_status_message('Performing exact address matching (Advanced)...')
                                    matched_df, exact_matched_count = exact_match_suggest_and_address(
                                        matched_df,
                                        masterlist_df,
                                        progress_bar=progress_bar,
                                        status_text=status_text
                                    )
                                    
                                    # VALIDATION: Check Amount total after exact_match_suggest_and_address (Advanced)
                                    validate_amount_total(matched_df, original_total_amount, "exact_match_suggest_and_address (Advanced)")
                                    if status_text is not None:
                                        matched_df = standardize_amount_column(matched_df, 'Amount')
                                        current_total = matched_df['Amount'].sum()
                                        status_text.text(f'✅ After Exact Matching (Advanced): P {current_total:,.2f} (unchanged)')
                                    update_status_message('Exact matching completed (Advanced)...')
                                else:
                                    if status_text is not None:
                                        status_text.text('⏭️ Skipping exact address matching (disabled by user)')
                                    update_status_message('Skipping exact matching (Advanced)...')
                                
                                # Perform split matching for unmatched records (suggested_md = False)
                                # Use name matching and PTR matching from advanced settings
                                # For Advanced mode, TF-IDF is enabled by default (can be disabled via Basic mode)
                                use_tfidf_advanced = st.session_state.matching_settings.get('use_tfidf_matching', True)
                                
                                # Save checkpoint before TF-IDF (most memory-intensive step)
                                if status_text is not None:
                                    status_text.text('Saving checkpoint before AI matching...')
                                update_status_message('Saving checkpoint (Advanced)...')
                                st.session_state.matched_df_checkpoint = matched_df.copy()
                                cleanup_memory()
                                
                                if status_text is not None:
                                    status_text.text('Starting AI matching (TF-IDF). This may take 10-30 minutes for large datasets. Please keep this page open...')
                                update_status_message('Starting AI matching (TF-IDF - Advanced)...')
                                
                                use_quick_suggest_matching = st.session_state.matching_settings.get('use_quick_suggest_matching', True)
                                matched_df, ai_matched_count, ai_completion_info = split_matching_for_unmatched(
                                    matched_df,
                                    masterlist_df,
                                    progress_bar=progress_bar,
                                    status_text=status_text,
                                    use_doctor_name=use_name_matching,
                                    use_ptr_no=use_ptr_matching,
                                    use_tfidf_matching=use_tfidf_advanced,
                                    use_quick_suggest_matching=use_quick_suggest_matching
                                )
                                
                                # VALIDATION: Check Amount total after split_matching_for_unmatched (Advanced)
                                validate_amount_total(matched_df, original_total_amount, "split_matching_for_unmatched (Advanced)")
                                if status_text is not None:
                                    current_total = pd.to_numeric(matched_df['Amount'], errors='coerce').fillna(0.0).sum()
                                    status_text.text(f'✅ After AI Matching (Advanced): P {current_total:,.2f} (unchanged)')
                                
                                # Delete checkpoint immediately after TF-IDF succeeds (no longer needed)
                                # This frees memory earlier rather than waiting until the end
                                if 'matched_df_checkpoint' in st.session_state:
                                    del st.session_state.matched_df_checkpoint
                                    st.session_state.matched_df_checkpoint = None
                                    cleanup_memory()
                                
                                # Store total AI matched count (exact match + TF-IDF match) in session state for display
                                total_ai_matched = exact_matched_count + ai_matched_count
                                # Store completion info for display (include exact match count)
                                ai_completion_info['md_address_matched'] = exact_matched_count
                                st.session_state.ai_completion_info = ai_completion_info
                                st.session_state.ai_matched_count = total_ai_matched
                                # Add DOCTOR_CODE and CUSTOMER_CODE columns by matching md_official_name
                                update_status_message('Adding MD codes (Advanced)...')
                                matched_df = add_md_code_columns(matched_df)
                                
                                # VALIDATION: Check Amount total after add_md_code_columns (Advanced)
                                validate_amount_total(matched_df, original_total_amount, "add_md_code_columns (Advanced)")
                                if status_text is not None:
                                    current_total = pd.to_numeric(matched_df['Amount'], errors='coerce').fillna(0.0).sum()
                                    status_text.text(f'✅ After Adding MD Codes (Advanced): P {current_total:,.2f} (unchanged)')
                                update_status_message('Finalizing results (Advanced)...')
                                
                                # CRITICAL: Ensure Amount column remains numeric after matching operations
                                # This prevents data corruption from string concatenation or formatting issues
                                if 'Amount' in matched_df.columns:
                                    matched_df = standardize_amount_column(matched_df, 'Amount')
                                
                                # Add blank reserved columns for future use
                                if 'quick_suggest_name' not in matched_df.columns:
                                    matched_df['quick_suggest_name'] = ''
                                if 'OSCA DISC' not in matched_df.columns:
                                    matched_df['OSCA DISC'] = 0.0  # Initialize as numeric (float)
                                if 'PERIOD' not in matched_df.columns:
                                    matched_df['PERIOD'] = ''
                                
                                # Ensure VAT Product Posting Group exists (from cross-reference, may be blank)
                                if 'VAT Product Posting Group' not in matched_df.columns:
                                    matched_df['VAT Product Posting Group'] = ''
                                
                                # Auto-fill blank OSCA DISC based on VAT Product Posting Group
                                if 'OSCA DISC' in matched_df.columns and 'VAT Product Posting Group' in matched_df.columns and 'Amount' in matched_df.columns:
                                    def calculate_osca_disc(row):
                                        """Calculate OSCA DISC based on VAT Product Posting Group and Amount."""
                                        # Check if OSCA DISC already has a numeric value (not blank/zero)
                                        osca_disc = row['OSCA DISC']
                                        try:
                                            # Try to convert existing value to float
                                            if pd.notna(osca_disc):
                                                osca_float = float(osca_disc) if not isinstance(osca_disc, (int, float)) else float(osca_disc)
                                                # If it's a valid non-zero number, keep it
                                                if osca_float != 0.0:
                                                    return osca_float
                                        except (ValueError, TypeError):
                                            pass  # If conversion fails, proceed to calculate
                                        
                                        vat_group = str(row['VAT Product Posting Group']).strip() if pd.notna(row['VAT Product Posting Group']) else ''
                                        # Amount is guaranteed to be float64 by standardize_amount_column
                                        amount = row['Amount']
                                        
                                        # Apply conditions
                                        if vat_group == 'EXEMPT':
                                            return float(amount * 0.14)
                                        elif vat_group == 'GOODS12':
                                            return float(amount * 0.2)
                                        else:
                                            return 0.0  # Return 0.0 if condition not met
                                    
                                    matched_df['OSCA DISC'] = matched_df.apply(calculate_osca_disc, axis=1)
                                    # Ensure OSCA DISC is numeric (float)
                                    matched_df['OSCA DISC'] = pd.to_numeric(matched_df['OSCA DISC'], errors='coerce').fillna(0.0).astype('float64')
                                
                                # Reorder columns to match requested order
                                requested_column_order = [
                                    'suggested_md', 'md_official_name', 'quick_suggest_name', 'Doctor Name', 'md_ptrs',
                                    'Branch Code', 'Branch Name', 'Trans Date', 'Address1', 'Address2',
                                    'Supplier Code', 'Supplier Name', "Vendor's Name",
                                    'Item Code', 'Item Name', 'InnoGen Item Name', 'Qty', 'Amount',
                                    'OSCA DISC', 'VAT Product Posting Group', 'PERIOD', 'PTR No',
                                    'DOCTOR_CODE', 'CUSTOMER_CODE', 'InnoGen Item Code', 'file_loc'
                                ]
                                
                                # Get columns that exist in the dataframe
                                existing_cols = [col for col in requested_column_order if col in matched_df.columns]
                                # Add any remaining columns that weren't in the order list
                                remaining_cols = [col for col in matched_df.columns if col not in existing_cols]
                                final_column_order = existing_cols + remaining_cols
                                matched_df = matched_df[final_column_order]
                                
                                # FINAL SAFEGUARD: Ensure Amount and OSCA DISC are numeric before storing in session state
                                if 'Amount' in matched_df.columns:
                                    matched_df = standardize_amount_column(matched_df, 'Amount')
                                if 'OSCA DISC' in matched_df.columns:
                                    matched_df['OSCA DISC'] = pd.to_numeric(matched_df['OSCA DISC'], errors='coerce').fillna(0.0).astype('float64')
                                
                                # FINAL VALIDATION: Check Amount total before storing in session state (Advanced)
                                validate_amount_total(matched_df, original_total_amount, "FINAL (Advanced - before storing)")
                                final_total = matched_df['Amount'].sum()
                                if status_text is not None:
                                    status_text.text(f'✅ FINAL Total Amount (Advanced): P {final_total:,.2f} (matches original: P {original_total_amount:,.2f})')
                                
                                    # Calculate elapsed time
                                    elapsed_time = time.time() - start_time
                                    elapsed_time_str = format_elapsed_time(elapsed_time)
                                    
                                    # Clear spinner display
                                    spinner_container.empty()
                                    
                                    st.session_state.matched_df = matched_df
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                    # Display completion message with AI matching breakdown and elapsed time
                                    completion_msg = f'✅ Matching completed! Processed {len(matched_df)} records.\n\n⏱️ **Processing Time:** {elapsed_time_str}'
                                if 'ai_completion_info' in st.session_state and st.session_state.ai_completion_info:
                                    info = st.session_state.ai_completion_info
                                    breakdown_parts = []
                                    if info.get('use_doctor_name') and info.get('doctor_name_matched', 0) > 0:
                                        breakdown_parts.append(f"Doctor Name: {info['doctor_name_matched']}")
                                    if info.get('use_ptr_no') and info.get('ptr_matched', 0) > 0:
                                        breakdown_parts.append(f"PTR Number: {info['ptr_matched']}")
                                    if info.get('md_address_matched', 0) > 0:
                                        breakdown_parts.append(f"MD+Address: {info['md_address_matched']}")
                                    if info.get('ppe_doctors_matched', 0) > 0:
                                        breakdown_parts.append(f"PPE Doctors: {info['ppe_doctors_matched']}")
                                    if breakdown_parts:
                                        completion_msg += f"\n\n🤖 AI Matching Breakdown: {' | '.join(breakdown_parts)}"
                                st.success(completion_msg)
                                # Re-enable button after successful completion
                                st.session_state.rx_processing = False
                                
                                # Aggressive memory cleanup after successful completion
                                # Delete checkpoint as it's no longer needed
                                if 'matched_df_checkpoint' in st.session_state:
                                    del st.session_state.matched_df_checkpoint
                                    st.session_state.matched_df_checkpoint = None
                                
                                # Delete intermediate variables that are no longer needed
                                if 'masterlist_df' in locals():
                                    del masterlist_df
                                if 'matched_df' in locals():
                                    del matched_df
                                
                                # Force multiple garbage collection cycles
                                for _ in range(3):
                                    cleanup_memory()
                                
                                # Results will be displayed below as the fragment continues executing
                        else:
                            # Calculate elapsed time even for error case
                            elapsed_time = time.time() - start_time
                            elapsed_time_str = format_elapsed_time(elapsed_time)
                            spinner_container.empty()
                            st.error(f'❌ Masterlist CSV not found. Please update the masterlist first.\n\n⏱️ **Time elapsed:** {elapsed_time_str}')
                            # Re-enable button after error
                            st.session_state.rx_processing = False
                            
                            # Cleanup memory even on error
                            cleanup_memory()
                    
                    except MemoryError as e:
                        # Handle memory errors gracefully
                        # Calculate elapsed time even for error case
                        elapsed_time = time.time() - start_time
                        elapsed_time_str = format_elapsed_time(elapsed_time)
                        
                        spinner_container.empty()
                        if 'progress_bar' in locals():
                            progress_bar.empty()
                        if 'status_text' in locals():
                            status_text.empty()
                        cleanup_memory()
                        logger.error(f"Memory error during processing: {str(e)}")
                        
                        # Restore from checkpoint if available
                        if 'matched_df_checkpoint' in st.session_state and st.session_state.matched_df_checkpoint is not None:
                            st.session_state.matched_df = st.session_state.matched_df_checkpoint.copy()
                            st.warning(f'⚠️ **Memory Error**: Restored results from checkpoint (before AI matching). '
                                      f'AI matching was skipped due to insufficient memory.\n\n⏱️ **Time elapsed:** {elapsed_time_str}')
                            st.info('💡 **Suggestions**:\n'
                                   '1. Disable "Enable AI Matching (TF-IDF)" checkbox\n'
                                   '2. Process smaller batches of files\n'
                                   '3. Increase server RAM\n'
                                   '4. Check matched data table below for partial results')
                            # Re-enable button after error
                            st.session_state.rx_processing = False
                            # Cleanup: Delete checkpoint after restoring (keep only final result)
                            del st.session_state.matched_df_checkpoint
                            st.session_state.matched_df_checkpoint = None
                        else:
                            st.error(f'❌ **Memory Error**: The dataset is too large for available memory. '
                                    f'Please try:\n'
                                    f'1. Disable "Enable AI Matching (TF-IDF)" checkbox\n'
                                    f'2. Process smaller batches of files\n'
                                    f'3. Increase server RAM\n\n'
                                    f'Error details: {str(e)}\n\n'
                                    f'⏱️ **Time elapsed:** {elapsed_time_str}')
                            # Try to save partial results if available
                            if 'matched_df' in locals() and matched_df is not None:
                                st.warning('⚠️ Partial results may be available. Check the matched data table below.')
                        # Re-enable button after error
                        st.session_state.rx_processing = False
                        # Aggressive cleanup after error
                        if 'masterlist_df' in locals():
                            del masterlist_df
                        if 'matched_df' in locals():
                            del matched_df
                        for _ in range(3):
                            cleanup_memory()
                    
                    except Exception as e:
                        # Calculate elapsed time even for error case
                        elapsed_time = time.time() - start_time
                        elapsed_time_str = format_elapsed_time(elapsed_time)
                        
                        # Handle all other errors gracefully to prevent crashes
                        spinner_container.empty()
                        if 'progress_bar' in locals():
                            progress_bar.empty()
                        if 'status_text' in locals():
                            status_text.empty()
                        cleanup_memory()
                        error_msg = str(e)
                        error_trace = traceback.format_exc()
                        logger.error(f"Error during processing: {error_msg}\n{error_trace}")
                        
                        # Restore from checkpoint if available
                        if 'matched_df_checkpoint' in st.session_state and st.session_state.matched_df_checkpoint is not None:
                            st.session_state.matched_df = st.session_state.matched_df_checkpoint.copy()
                            st.warning(f'⚠️ **Processing Error**: Restored results from checkpoint. '
                                      f'An error occurred during AI matching, but previous results were saved.\n\n⏱️ **Time elapsed:** {elapsed_time_str}')
                            st.info(f'**Error**: {error_msg}\n\n'
                                   '**Suggestions**:\n'
                                   '1. Check if the data format is correct\n'
                                   '2. Try disabling "Enable AI Matching (TF-IDF)"\n'
                                   '3. Process smaller batches\n'
                                   '4. Check server logs for details')
                            # Re-enable button after error
                            st.session_state.rx_processing = False
                            # Cleanup: Delete checkpoint after restoring (keep only final result)
                            del st.session_state.matched_df_checkpoint
                            st.session_state.matched_df_checkpoint = None
                        else:
                            st.error(f'❌ **Processing Error**: An error occurred during matching. '
                                    f'The application will continue to run.\n\n'
                                    f'**Error**: {error_msg}\n\n'
                                    f'**Suggestions**:\n'
                                    f'1. Check if the data format is correct\n'
                                    f'2. Try disabling "Enable AI Matching (TF-IDF)"\n'
                                    f'3. Process smaller batches\n'
                                    f'4. Check server logs for details\n\n'
                                    f'⏱️ **Time elapsed:** {elapsed_time_str}')
                            # Try to save partial results if available
                            if 'matched_df' in locals() and matched_df is not None:
                                st.warning('⚠️ Partial results may be available. Check the matched data table below.')
                                st.session_state.matched_df = matched_df
                            # Re-enable button after error
                            st.session_state.rx_processing = False
                        
                        # Aggressive cleanup after error
                        if 'masterlist_df' in locals():
                            del masterlist_df
                        if 'matched_df' in locals():
                            del matched_df
                        for _ in range(3):
                            cleanup_memory()
            
            # Display matched results table if available (inside fragment to prevent full page reload)
            if st.session_state.matched_df is not None and not st.session_state.matched_df.empty:
                st.divider()
                st.subheader('📋 Matched Transaction Data (With AI Matching Results)')
                
                # Fill NaN values for display (but keep Amount and OSCA DISC numeric for calculations)
                matched_display_df = st.session_state.matched_df.copy()
                # Preserve numeric columns before fillna
                numeric_cols = ['Amount', 'OSCA DISC']
                numeric_data = {}
                for col in numeric_cols:
                    if col in matched_display_df.columns:
                        numeric_data[col] = matched_display_df[col].copy()
                
                matched_display_df = matched_display_df.fillna('')
                
                # Restore numeric columns after fillna
                for col in numeric_cols:
                    if col in numeric_data:
                        matched_display_df[col] = numeric_data[col]
                
                # Format Amount column for display only (keep original numeric in session state)
                if 'Amount' in matched_display_df.columns:
                    def format_amount_display(value):
                        """Format amount with commas for display only."""
                        if pd.isna(value) or value == '':
                            return ''
                        try:
                            # If already numeric, format it
                            if isinstance(value, (int, float)):
                                return f"{float(value):,.2f}"
                            # If string, try to convert to numeric first
                            amount_str = str(value).strip().replace(' ', '').replace(',', '')
                            amount_str = re.sub(r'[^\d.]', '', amount_str)
                            if amount_str == '' or amount_str == '.':
                                return ''
                            amount_float = float(amount_str)
                            return f"{amount_float:,.2f}"
                        except (ValueError, TypeError):
                            return str(value) if value != '' else ''
                    
                    # Create a display copy of Amount column (formatted)
                    matched_display_df['Amount'] = matched_display_df['Amount'].apply(format_amount_display)
                
                # Split Trans Date into YEAR, MONTH, DAYS
                if 'Trans Date' in matched_display_df.columns:
                    def split_trans_date(date_value):
                        """Split Trans Date into YEAR, MONTH, DAYS."""
                        if pd.isna(date_value) or date_value == '':
                            return pd.Series({'YEAR': '', 'MONTH': '', 'DAYS': ''})
                        try:
                            # Try to parse as datetime
                            if isinstance(date_value, str):
                                date_obj = pd.to_datetime(date_value, errors='coerce')
                            else:
                                date_obj = pd.to_datetime(date_value, errors='coerce')
                            
                            if pd.isna(date_obj):
                                return pd.Series({'YEAR': '', 'MONTH': '', 'DAYS': ''})
                            
                            return pd.Series({
                                'YEAR': str(date_obj.year),
                                'MONTH': str(date_obj.month).zfill(2),
                                'DAYS': str(date_obj.day).zfill(2)
                            })
                        except Exception:
                            return pd.Series({'YEAR': '', 'MONTH': '', 'DAYS': ''})
                    
                    # Apply splitting to create YEAR, MONTH, DAYS columns
                    date_split = matched_display_df['Trans Date'].apply(split_trans_date)
                    matched_display_df['YEAR'] = date_split['YEAR']
                    matched_display_df['MONTH'] = date_split['MONTH']
                    matched_display_df['DAYS'] = date_split['DAYS']
                
                # Rename columns for display
                rename_dict = {
                    'md_ptrs': 'PTR FINAL',
                    'suggested_md': 'suggest_dn',
                    'md_official_name': 'MD NAME FINAL'
                }
                matched_display_df = matched_display_df.rename(columns=rename_dict)
                
                # Reorder columns for better display (matching requested order)
                matched_column_order = [
                    'PTR FINAL', 'suggest_dn', 'CUSTOMER_CODE', 'MD NAME FINAL', 'suggested_name', 'quick_suggest_name', 
                    'Doctor Name', 'Branch Code', 'Branch Name', 'YEAR', 'MONTH', 'DAYS', 'Address1', 'Address2',
                    'Supplier Code', 'Supplier Name', "Vendor's Name",
                    'Item Code', 'InnoGen Item Code', 'Item Name', 'InnoGen Item Name', 'Standard Cost', 'Qty', 'Amount',
                    'OSCA DISC', 'DIVISION', 'VAT Product Posting Group', 'PERIOD', 'PTR No',
                    'DOCTOR_CODE', 'file_loc'
                ]
                
                # Only include columns that exist
                matched_display_columns = [col for col in matched_column_order if col in matched_display_df.columns]
                # Add any remaining columns that weren't in the order list
                remaining_cols = [col for col in matched_display_df.columns if col not in matched_display_columns]
                matched_display_columns.extend(remaining_cols)
                matched_display_df = matched_display_df[matched_display_columns]
                
                # Sort by suggest_dn (True first) and YEAR, MONTH, DAYS
                sort_columns = []
                sort_ascending = []
                
                if 'suggest_dn' in matched_display_df.columns:
                    sort_columns.append('suggest_dn')
                    sort_ascending.append(False)  # True values first (matched records)
                elif 'suggested_md' in matched_display_df.columns:
                    sort_columns.append('suggested_md')
                    sort_ascending.append(False)
                
                # Sort by YEAR, MONTH, DAYS if available, otherwise by Trans Date
                if all(col in matched_display_df.columns for col in ['YEAR', 'MONTH', 'DAYS']):
                    sort_columns.extend(['YEAR', 'MONTH', 'DAYS'])
                    sort_ascending.extend([True, True, True])  # Ascending order
                elif 'Trans Date' in matched_display_df.columns:
                    try:
                        # Create a temporary datetime column for sorting
                        matched_display_df['_temp_sort_date'] = pd.to_datetime(matched_display_df['Trans Date'], errors='coerce')
                        sort_columns.append('_temp_sort_date')
                        sort_ascending.append(True)  # Ascending date order (oldest first)
                    except Exception:
                        sort_columns.append('Trans Date')
                        sort_ascending.append(True)
                
                if sort_columns:
                    try:
                        matched_display_df = matched_display_df.sort_values(sort_columns, ascending=sort_ascending, na_position='last')
                        # Remove temporary column if it exists
                        if '_temp_sort_date' in matched_display_df.columns:
                            matched_display_df = matched_display_df.drop(columns=['_temp_sort_date'])
                    except Exception:
                        pass
                
                # Show matching statistics
                # Check for renamed column (suggest_dn) or original (suggested_md)
                stats_column = 'suggest_dn' if 'suggest_dn' in matched_display_df.columns else 'suggested_md'
                md_name_column = 'MD NAME FINAL' if 'MD NAME FINAL' in matched_display_df.columns else 'md_official_name'
                
                if stats_column in matched_display_df.columns:
                    total_records = len(matched_display_df)
                    # Count True values in suggest_dn/suggested_md column
                    if matched_display_df[stats_column].dtype == bool:
                        matched_count = matched_display_df[stats_column].sum()
                    else:
                        matched_count = (matched_display_df[stats_column].astype(str).str.lower() == 'true').sum()
                    match_rate = (matched_count / total_records * 100) if total_records > 0 else 0
                    
                    # Get AI matched count (TF-IDF matching) from session state
                    ai_matched_count = st.session_state.get('ai_matched_count', 0)
                    
                    # Count records with md_official_name/MD NAME FINAL populated but suggest_dn/suggested_md = False (AI matches)
                    # This includes:
                    #   - Exact Address Match (False + md_official_name populated)
                    #   - TF-IDF Masterlist matches (False + md_official_name populated)
                    #   - PPE Doctors matches (False + md_official_name populated)
                    # Excludes:
                    #   - Exact matches (True) - counted in Match Rate
                    #   - Quick Suggest matches (False + quick_suggest_name/suggested_name populated) - counted in Quick Suggest Match Rate
                    #   - Reference matches ('REF') - counted in Quick Suggest Match Rate
                    #   - Unmatched ('' blank) - not counted
                    ai_matched_count_actual = 0
                    if md_name_column in matched_display_df.columns:
                        # stats_column can be bool or str ('true'/'false'/'REF'/''); ~ only works on bool
                        if matched_display_df[stats_column].dtype == bool:
                            # For bool: False means AI match (if md_official_name is populated)
                            # But exclude Quick Suggest matches (which have quick_suggest_name or suggested_name populated)
                            not_matched = ~matched_display_df[stats_column]
                            # Exclude Quick Suggest matches (they have quick_suggest_name or suggested_name populated)
                            if 'quick_suggest_name' in matched_display_df.columns:
                                not_matched = not_matched & ((matched_display_df['quick_suggest_name'] == '') | (matched_display_df['quick_suggest_name'].isna()))
                            if 'suggested_name' in matched_display_df.columns:
                                not_matched = not_matched & ((matched_display_df['suggested_name'] == '') | (matched_display_df['suggested_name'].isna()))
                        else:
                            # Exclude 'REF' and '' (blank) from unmatched
                            # False/'false' with md_official_name populated = AI match (but exclude Quick Suggest)
                            # '' (blank) = truly unmatched, exclude from AI Match Rate
                            not_matched = (matched_display_df[stats_column].astype(str).str.lower() != 'true') & \
                                         (matched_display_df[stats_column].astype(str).str.lower() != 'ref') & \
                                         (matched_display_df[stats_column].astype(str).str.strip() != '')
                            # Exclude Quick Suggest matches (they have quick_suggest_name or suggested_name populated)
                            if 'quick_suggest_name' in matched_display_df.columns:
                                not_matched = not_matched & ((matched_display_df['quick_suggest_name'] == '') | (matched_display_df['quick_suggest_name'].isna()))
                            if 'suggested_name' in matched_display_df.columns:
                                not_matched = not_matched & ((matched_display_df['suggested_name'] == '') | (matched_display_df['suggested_name'].isna()))
                        ai_matched_mask = not_matched & (matched_display_df[md_name_column] != '') & (matched_display_df[md_name_column].notna())
                        ai_matched_count_actual = ai_matched_mask.sum()
                    
                    # Use actual count if available, otherwise use session state
                    ai_matched_display = ai_matched_count_actual if ai_matched_count_actual > 0 else ai_matched_count
                    
                    # Calculate AI Match Rate
                    ai_match_rate = (ai_matched_display / total_records * 100) if total_records > 0 else 0
                    
                    # Get Quick Suggest Matched Count from session state
                    quick_suggest_matched_count = 0
                    if 'ai_completion_info' in st.session_state and st.session_state.ai_completion_info:
                        quick_suggest_matched_count = st.session_state.ai_completion_info.get('quick_suggest_matched', 0)
                    
                    # Also count records with quick_suggest_name populated (as a validation)
                    quick_suggest_count_actual = 0
                    if 'quick_suggest_name' in matched_display_df.columns:
                        quick_suggest_mask = (matched_display_df['quick_suggest_name'] != '') & (matched_display_df['quick_suggest_name'].notna())
                        quick_suggest_count_actual = quick_suggest_mask.sum()
                    
                    # Use actual count if available, otherwise use session state
                    quick_suggest_display = quick_suggest_count_actual if quick_suggest_count_actual > 0 else quick_suggest_matched_count
                    
                    # Calculate Quick Suggest Match Rate
                    quick_suggest_match_rate = (quick_suggest_display / total_records * 100) if total_records > 0 else 0
                    
                    # Calculate Total Amount from matched data (use original numeric values, not display format)
                    total_amount = 0.0
                    if 'Amount' in st.session_state.matched_df.columns:
                        # Use the original matched_df (numeric) not the display_df (may have formatted strings)
                        amount_series = st.session_state.matched_df['Amount']
                        # CRITICAL: Ensure Amount is numeric before summing (prevent string concatenation)
                        # Convert to numeric, handling any string values
                        def ensure_numeric_for_sum(value):
                            """Ensure value is numeric for summation."""
                            if pd.isna(value):
                                return 0.0
                            try:
                                if isinstance(value, (int, float)):
                                    return float(value)
                                # If string, clean and convert
                                amount_str = str(value).strip().replace(' ', '').replace(',', '')
                                amount_str = re.sub(r'[^\d.]', '', amount_str)
                                if amount_str == '' or amount_str == '.':
                                    return 0.0
                                return float(amount_str)
                            except (ValueError, TypeError):
                                return 0.0
                        
                        # Apply numeric conversion to ensure all values are numeric
                        amount_series_numeric = amount_series.apply(ensure_numeric_for_sum)
                        # Sum the numeric values
                        total_amount = pd.to_numeric(amount_series_numeric, errors='coerce').fillna(0.0).sum()

                    process_validation = validate_matched_process_totals(
                        st.session_state.combined_df,
                        st.session_state.matched_df,
                        st.session_state.get('combined_summary'),
                    )
                    records_validated = process_validation.get('records_match') is True
                    amount_validated = process_validation.get('amount_match') is True
                    total_records_title = '✅ Total Records' if records_validated else 'Total Records'
                    total_amount_title = '✅ Total Amount' if amount_validated else 'Total Amount'
                    
                    # Display metrics in two rows with colored dashboard-style cards
                    # Define metric card style function (compact version)
                    def create_metric_card(title, value, gradient_color1, gradient_color2, icon="📊"):
                        """Create a compact styled metric card with gradient background."""
                        card_html = f"""
                        <div style="
                            background: linear-gradient(135deg, {gradient_color1} 0%, {gradient_color2} 100%);
                            padding: 12px 8px;
                            border-radius: 8px;
                            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
                            text-align: center;
                            color: white;
                            margin-bottom: 6px;
                            transition: transform 0.2s ease, box-shadow 0.2s ease;
                        ">
                            <div style="font-size: 16px; margin-bottom: 4px; line-height: 1;">{icon}</div>
                            <div style="font-size: 10px; font-weight: 600; opacity: 0.95; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.3px; line-height: 1.2;">
                                {title}
                            </div>
                            <div style="font-size: 18px; font-weight: bold; margin-top: 4px; line-height: 1.2;">
                                {value}
                            </div>
                        </div>
                        """
                        return card_html
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.markdown(create_metric_card(
                            total_records_title,
                            f"{total_records:,}", 
                            "#1e3c72", 
                            "#2a5298",
                            "📋"
                        ), unsafe_allow_html=True)
                    with col_stat2:
                        st.markdown(create_metric_card(
                            'Masterlist Matched', 
                            f"{matched_count:,}", 
                            "#0f5132", 
                            "#198754",
                            "✅"
                        ), unsafe_allow_html=True)
                    with col_stat3:
                        st.markdown(create_metric_card(
                            'AI Added Matched', 
                            f"{ai_matched_display:,}", 
                            "#6f42c1", 
                            "#8b5cf6",
                            "🤖"
                        ), unsafe_allow_html=True)
                    with col_stat4:
                        st.markdown(create_metric_card(
                            'Quick Suggest Matched', 
                            f"{quick_suggest_display:,}", 
                            "#0d6efd", 
                            "#3b82f6",
                            "⚡"
                        ), unsafe_allow_html=True)
                    
                    col_stat5, col_stat6, col_stat7, col_stat8 = st.columns(4)
                    with col_stat5:
                        st.markdown(create_metric_card(
                            'Masterlist Match Rate', 
                            f"{match_rate:.1f}%", 
                            "#0f5132", 
                            "#20c997",
                            "📈"
                        ), unsafe_allow_html=True)
                    with col_stat6:
                        st.markdown(create_metric_card(
                            'AI Match Rate', 
                            f"{ai_match_rate:.1f}%", 
                            "#6f42c1", 
                            "#a855f7",
                            "📊"
                        ), unsafe_allow_html=True)
                    with col_stat7:
                        st.markdown(create_metric_card(
                            'Quick Suggest Rate', 
                            f"{quick_suggest_match_rate:.1f}%", 
                            "#0d6efd", 
                            "#60a5fa",
                            "🎯"
                        ), unsafe_allow_html=True)
                    with col_stat8:
                        st.markdown(create_metric_card(
                            total_amount_title,
                            f"P {total_amount:,.2f}", 
                            "#f59e0b", 
                            "#fbbf24",
                            "💰"
                        ), unsafe_allow_html=True)

                    if records_validated and amount_validated:
                        st.success(
                            '✅ Process validation: Total Records and Total Amount match combined transaction data.'
                        )
                    else:
                        if process_validation.get('records_match') is False:
                            st.warning(
                                f"⚠️ Total Records mismatch after processing: "
                                f"combined {process_validation['combined_records']:,} "
                                f"vs matched {process_validation['matched_records']:,} "
                                f"(diff {process_validation['records_diff']:+,})."
                            )
                        if process_validation.get('amount_match') is False:
                            st.warning(
                                f"⚠️ Total Amount mismatch after processing: "
                                f"combined P {process_validation['combined_amount']:,.2f} "
                                f"vs matched P {process_validation['matched_amount']:,.2f} "
                                f"(diff P {process_validation['amount_diff']:+,.2f})."
                            )
                
                # Format OSCA DISC to show 2 decimal places for display
                if 'OSCA DISC' in matched_display_df.columns:
                    def format_osca_disc_display(value):
                        """Format OSCA DISC to show 2 decimal places for display."""
                        if pd.isna(value) or value == '':
                            return ''
                        try:
                            # Ensure it's numeric
                            val = float(value) if not isinstance(value, (int, float)) else float(value)
                            # Format with 2 decimal places
                            return f"{val:,.2f}"
                        except (ValueError, TypeError):
                            return ''
                    
                    # Create a display copy of OSCA DISC (formatted as string with 2 decimals)
                    matched_display_df['OSCA DISC'] = matched_display_df['OSCA DISC'].apply(format_osca_disc_display)
                
                # CRITICAL: Preserve suggest_dn column as string before cleaning
                # This column can have mixed types: True, False, 'REF', '' (blank)
                # We need to convert all values to string to preserve 'REF' and '' (blank)
                if 'suggest_dn' in matched_display_df.columns:
                    # Convert all values to string representation
                    def format_suggest_dn_display(value):
                        """Format suggest_dn for display, preserving 'REF' and blank values."""
                        if pd.isna(value) or value == '':
                            return ''  # Blank
                        if isinstance(value, bool):
                            return 'True' if value else 'False'
                        # Already string
                        value_str = str(value).strip()
                        if value_str.lower() == 'true':
                            return 'True'
                        elif value_str.lower() == 'false':
                            return 'False'
                        elif value_str.upper() == 'REF':
                            return 'REF'
                        else:
                            return value_str if value_str else ''  # Return blank if empty
                    
                    matched_display_df['suggest_dn'] = matched_display_df['suggest_dn'].apply(format_suggest_dn_display)
                
                # Clean DataFrame for PyArrow compatibility before display
                matched_display_df = clean_dataframe_for_display(matched_display_df)
                
                st.dataframe(
                    matched_display_df,
                    use_container_width=True,
                    height=600
                )
                
                # Download buttons for matched data (final report export only)
                csv_ready = 'matched_csv_cache' in st.session_state and 'matched_csv_filename' in st.session_state
                xlsx_ready = 'matched_xlsx_cache' in st.session_state and 'matched_xlsx_filename' in st.session_state
                
                export_placeholder = st.empty()
                
                if csv_ready and xlsx_ready:
                    export_placeholder.empty()
                else:
                    with export_placeholder.container():
                        st.info("⏳ Preparing final report download (this may take 30-60 seconds for large datasets)...")
                    
                    try:
                        download_df = prepare_matched_report_export_df(st.session_state.matched_df)
                        download_df_for_export = sanitize_matched_report_export_columns(download_df)

                        csv = download_df_for_export.to_csv(index=False)
                        st.session_state.matched_csv_cache = csv
                        st.session_state.matched_csv_filename = generate_filename_from_trans_date(
                            download_df, 'Summary_RXTracking_Report', extension='csv'
                        )

                        xlsx_output = BytesIO()
                        with pd.ExcelWriter(xlsx_output, engine='openpyxl') as writer:
                            download_df_for_export.to_excel(writer, index=False, sheet_name='RX Tracking Report')
                        st.session_state.matched_xlsx_cache = xlsx_output.getvalue()
                        st.session_state.matched_xlsx_filename = generate_filename_from_trans_date(
                            download_df, 'Summary_RXTracking_Report', extension='xlsx'
                        )

                        csv_ready = True
                        xlsx_ready = True
                        export_placeholder.empty()
                    except ImportError:
                        export_placeholder.warning("⚠️ openpyxl is required for Excel export. CSV download is still available.")
                        try:
                            if 'matched_csv_cache' not in st.session_state:
                                download_df = prepare_matched_report_export_df(st.session_state.matched_df)
                                download_df_for_export = sanitize_matched_report_export_columns(download_df)
                                st.session_state.matched_csv_cache = download_df_for_export.to_csv(index=False)
                                st.session_state.matched_csv_filename = generate_filename_from_trans_date(
                                    download_df, 'Summary_RXTracking_Report', extension='csv'
                                )
                            csv_ready = True
                            xlsx_ready = False
                            export_placeholder.empty()
                        except Exception as e:
                            export_placeholder.error(f"❌ Error preparing CSV: {str(e)}")
                            csv_ready = False
                            xlsx_ready = False
                    except Exception as e:
                        export_placeholder.error(f"❌ Error preparing final report: {str(e)}")
                        csv_ready = False
                        xlsx_ready = False
                
                if csv_ready or xlsx_ready:
                    col_csv_dl, col_xlsx_dl = st.columns(2)
                    if csv_ready:
                        with col_csv_dl:
                            st.download_button(
                                label='📥 Download Final Report (CSV)',
                                data=st.session_state.matched_csv_cache,
                                file_name=st.session_state.matched_csv_filename,
                                mime='text/csv',
                                key='download_matched_csv_btn',
                            )
                    if xlsx_ready:
                        with col_xlsx_dl:
                            st.download_button(
                                label='📥 Download Final Report (Excel)',
                                data=st.session_state.matched_xlsx_cache,
                                file_name=st.session_state.matched_xlsx_filename,
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                key='download_matched_xlsx_btn',
                            )
                
                # Download buttons for reference files
                st.divider()
                with st.expander('📚 Reference Files Download', expanded=False):
                    col_ref1, col_ref2, col_ref3 = st.columns(3)
                    
                    with col_ref1:
                        # Download button for masterlist
                        csv_dir = get_csv_dir()
                        masterlist_path = os.path.join(csv_dir, 'rx_md_masterlist.csv')
                        if os.path.exists(masterlist_path):
                            try:
                                with open(masterlist_path, 'rb') as f:
                                    masterlist_data = f.read()
                                st.download_button(
                                    label='📥 Download Masterlist (rx_md_masterlist.csv)',
                                    data=masterlist_data,
                                    file_name='rx_md_masterlist.csv',
                                    mime='text/csv',
                                    key='download_masterlist'
                                )
                            except Exception as e:
                                st.error(f'Error reading masterlist file: {str(e)}')
                        else:
                            st.warning('⚠️ Masterlist file not found: rx_md_masterlist.csv')
                    
                    with col_ref2:
                        # Download button for cross-reference
                        crossref_path = get_item_cross_ref_path()
                        if os.path.exists(crossref_path):
                            try:
                                with open(crossref_path, 'rb') as f:
                                    crossref_data = f.read()
                                st.download_button(
                                    label='📥 Download Cross-Reference (rx_item_cross_ref.csv)',
                                    data=crossref_data,
                                    file_name='rx_item_cross_ref.csv',
                                    mime='text/csv',
                                    key='download_crossref'
                                )
                            except Exception as e:
                                st.error(f'Error reading cross-reference file: {str(e)}')
                        else:
                            st.warning('⚠️ Cross-reference file not found: rx_item_cross_ref.csv')
                    
                    with col_ref3:
                        # Download button for ppe_doctors
                        csv_dir = get_reference_csv_dir()  # Use persistent location, not temp directory
                        ppe_path = os.path.join(csv_dir, 'ppe_doctors.csv')
                        if os.path.exists(ppe_path):
                            try:
                                with open(ppe_path, 'rb') as f:
                                    ppe_data = f.read()
                                st.download_button(
                                    label='📥 Download PPE Doctors (ppe_doctors.csv)',
                                    data=ppe_data,
                                    file_name='ppe_doctors.csv',
                                    mime='text/csv',
                                    key='download_ppe_doctors'
                                )
                            except Exception as e:
                                st.error(f'Error reading PPE Doctors file: {str(e)}')
                        else:
                            st.warning('⚠️ PPE Doctors file not found: ppe_doctors.csv')
        
        # Call the fragment
        rx_tracking_fragment()
