# LSE Status Monitor Configuration

# LSE Website URL
LSE_URL = "https://www.lse.ac.uk/study-at-lse/Graduate/News/Current-processing-times"

# File paths
STATUS_FILE = "status.json"
HISTORY_FILE = "history.json"

# Regression settings
REGRESSION_MIN_POINTS = 2
CONFIDENCE_LEVEL = 1.96  # 95%-Konfidenzniveau

# Active stream configuration
import os
from datetime import datetime as dt
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# London timezone
LON = ZoneInfo("Europe/London")

# Active tracking stream
ACTIVE_STREAM = os.getenv("LSE_ACTIVE_STREAM", "pre_cas")  # default to pre_cas

# Single target date for Pre-CAS (tz-aware)
TARGET_DATE_PRE_CAS = dt(2025, 8, 13, 0, 0, tzinfo=LON)

# Target dates map (backward compatibility)
TARGET_DATES_MAP = {
    "pre_cas": TARGET_DATE_PRE_CAS,
    "all_other_applicants": ["25 July", "28 July"]  # Keep existing dual targets
}

# Legacy target dates for backward compatibility
TARGET_DATES = ["25 July", "28 July"]

# Request settings
REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Email settings
GMAIL_SMTP_SERVER = 'smtp.gmail.com'
GMAIL_SMTP_PORT = 587

# Telegram settings
TELEGRAM_API_BASE = "https://api.telegram.org/bot"

# Feste, zentral gepflegte UK-Feiertage (ISO, YYYY-MM-DD)
UK_HOLIDAYS = (
    "2025-08-25",  # Summer Bank Holiday 2025 (dein konkreter Wunsch)
)
