from pathlib import Path
import os

# Server configuration
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))

# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]
TEMP_DIR = BASE_DIR / "temp"
UPLOADS_DIR = TEMP_DIR / "uploads"
OUTPUTS_DIR = TEMP_DIR / "outputs"
FILTERS_DIR = TEMP_DIR / "filters"
LOGS_DIR = BASE_DIR / "logs"

# Marker CLI configuration
# Set to the marker CLI/binary you have installed, e.g. "marker_single" or full path
MARKER_CLI = os.environ.get("MARKER_CLI", "marker_single")
# Extra flags passed as a list, e.g. ["--force_ocr", "--language", "eng"]
# MARKER_FLAGS = os.environ.get("MARKER_FLAGS", f"--force_ocr --output_format markdown --output_dir {OUTPUTS_DIR}").split()
# REMOVED --output_dir to let marker_single use its default output directory behavior  
MARKER_FLAGS = os.environ.get("MARKER_FLAGS", "--force_ocr --output_format markdown").split() 
OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "markdown")

# Logging
LOG_FILE = LOGS_DIR / "app.log"

# GPU safety thresholds (degrees C and free memory in MB)
GPU_TEMP_THRESHOLD_C = int(os.environ.get("GPU_TEMP_THRESHOLD_C", 85))
GPU_MEM_FREE_MB = int(os.environ.get("GPU_MEM_FREE_MB", 500))
GPU_WAIT_TIMEOUT_SEC = int(os.environ.get("GPU_WAIT_TIMEOUT_SEC", 600))
GPU_POLL_INTERVAL_SEC = int(os.environ.get("GPU_POLL_INTERVAL_SEC", 5))

# Allowed upload extensions (include common image types)
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}

# Default marker output directory (can be overridden by Marker flags or env)
MARKER_OUTPUT_DIR = os.environ.get("MARKER_OUTPUT_DIR")
if MARKER_OUTPUT_DIR:
    MARKER_OUTPUT_DIR = Path(MARKER_OUTPUT_DIR)
else:
    MARKER_OUTPUT_DIR = OUTPUTS_DIR

# Ensure directories exist at runtime
def ensure_dirs():
    for p in (TEMP_DIR, UPLOADS_DIR, OUTPUTS_DIR, FILTERS_DIR, LOGS_DIR):
        p.mkdir(parents=True, exist_ok=True)
