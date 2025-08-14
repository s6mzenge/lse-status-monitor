# LSE Status Monitor Configuration

# LSE Website URL
LSE_URL = "https://www.lse.ac.uk/study-at-lse/Graduate/News/Current-processing-times"

# File paths
STATUS_FILE = "status.json"
HISTORY_FILE = "history.json"

# Regression settings
REGRESSION_MIN_POINTS = 2
CONFIDENCE_LEVEL = 1.96  # 95%-Konfidenzniveau

# ============================================================================
# STREAM CONFIGURATION - NUR HIER ÄNDERN FÜR WECHSEL ZWISCHEN STREAMS
# ============================================================================

# Aktiver Stream - Optionen: "pre_cas", "cas", "all_other"
# Für Wechsel zu CAS: Einfach auf "cas" ändern
ACTIVE_STREAM = "pre_cas"

# Stream-spezifische Konfigurationen
STREAM_CONFIG = {
    "pre_cas": {
        "name": "Pre-CAS",
        "target_dates": ["13 August"],  # Zieldatum für Pre-CAS
        "email_subject_prefix": "LSE Pre-CAS Update",
        "telegram_emoji": "📘",
        "graph_color": "#ff7f0e",  # Orange
        "enabled": True,
        # Zusätzliche Konfigurationen für Zukunft
        "priority": 1,  # Priorität für Multi-Stream-Tracking
        "notification_level": "full"  # full, summary, oder none
    },
    "cas": {
        "name": "CAS", 
        "target_dates": ["20 August"],  # TODO: Hier dein CAS-Zieldatum eintragen!
        "email_subject_prefix": "LSE CAS Update",
        "telegram_emoji": "📗",
        "graph_color": "#2ca02c",  # Grün
        "enabled": True,
        "priority": 2,
        "notification_level": "full"
    },
    "all_other": {
        "name": "All Other Applicants",
        "target_dates": ["13 August"],  # Legacy für AOA (wird nicht mehr aktiv genutzt)
        "email_subject_prefix": "LSE AOA Update",
        "telegram_emoji": "📙",
        "graph_color": "#1f77b4",  # Blau
        "enabled": False,  # AOA ist dauerhaft passiv
        "priority": 3,
        "notification_level": "none"
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_active_config():
    """Gibt die Konfiguration für den aktiven Stream zurück"""
    return STREAM_CONFIG.get(ACTIVE_STREAM, STREAM_CONFIG["pre_cas"])

def get_target_dates():
    """Gibt die Zieldaten für den aktiven Stream zurück"""
    config = get_active_config()
    return config.get("target_dates", [])

def get_stream_name():
    """Gibt den Anzeigenamen für den aktiven Stream zurück"""
    config = get_active_config()
    return config.get("name", "Unknown")

def get_stream_emoji():
    """Gibt das Emoji für den aktiven Stream zurück"""
    config = get_active_config()
    return config.get("telegram_emoji", "📊")

def get_stream_config(stream_key):
    """Gibt die Konfiguration für einen spezifischen Stream zurück"""
    return STREAM_CONFIG.get(stream_key, {})

def is_stream_active(stream_key):
    """Prüft ob ein Stream der aktive Stream ist"""
    return stream_key == ACTIVE_STREAM

def get_all_enabled_streams():
    """Gibt alle aktivierten Streams zurück (für zukünftige Multi-Stream-Features)"""
    return {key: config for key, config in STREAM_CONFIG.items() if config.get("enabled", False)}

def get_stream_color(stream_key):
    """Gibt die Farbe für einen Stream zurück"""
    config = STREAM_CONFIG.get(stream_key, {})
    return config.get("graph_color", "#808080")  # Grau als Fallback

def get_email_subject(stream_key, suffix=""):
    """Generiert den E-Mail-Betreff für einen Stream"""
    config = STREAM_CONFIG.get(stream_key, {})
    prefix = config.get("email_subject_prefix", "LSE Update")
    return f"{prefix}{suffix}" if suffix else prefix

# Legacy-Kompatibilität (für bestehenden Code)
TARGET_DATES = get_target_dates()

# ============================================================================
# REQUEST SETTINGS
# ============================================================================

REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'no-cache'
}

# ============================================================================
# EMAIL SETTINGS
# ============================================================================

GMAIL_SMTP_SERVER = 'smtp.gmail.com'
GMAIL_SMTP_PORT = 587

# E-Mail-Empfänger-Kategorien (für zukünftige Konfiguration)
EMAIL_CATEGORIES = {
    "always": ["main", "secondary"],  # Immer benachrichtigen
    "on_target": ["conditional"],  # Nur bei Zieldatum
    "weekly": [],  # Für wöchentliche Zusammenfassungen
}

# ============================================================================
# TELEGRAM SETTINGS
# ============================================================================

TELEGRAM_API_BASE = "https://api.telegram.org/bot"

# Telegram-Bot-Konfiguration (für Dokumentation)
TELEGRAM_BOTS = {
    "main": {
        "env_token": "TELEGRAM_BOT_TOKEN",
        "env_chat": "TELEGRAM_CHAT_ID",
        "parse_mode": "HTML",
        "description": "Hauptbot mit vollständigen Updates"
    },
    "mama": {
        "env_token": "TELEGRAM_BOT_TOKEN_MAMA",
        "env_chat": "TELEGRAM_CHAT_ID_MAMA",
        "parse_mode": None,
        "description": "Einfache Updates für Mama"
    },
    "papa": {
        "env_token": "TELEGRAM_BOT_TOKEN_PAPA",
        "env_chat": "TELEGRAM_CHAT_ID_PAPA",
        "parse_mode": None,
        "description": "Einfache Updates für Papa"
    }
}

# ============================================================================
# UK HOLIDAYS & BUSINESS HOURS
# ============================================================================

# UK Bank Holidays 2025 (ISO Format: YYYY-MM-DD)
UK_HOLIDAYS = (
    "2025-01-01",  # New Year's Day
    "2025-04-18",  # Good Friday
    "2025-04-21",  # Easter Monday
    "2025-05-05",  # Early May Bank Holiday
    "2025-05-26",  # Spring Bank Holiday
    "2025-08-25",  # Summer Bank Holiday
    "2025-12-25",  # Christmas Day
    "2025-12-26",  # Boxing Day
)

# LSE Business Hours (London Time)
BUSINESS_HOURS = {
    "start": "10:00",  # 10:00 AM
    "end": "16:00",    # 4:00 PM
    "timezone": "Europe/London"
}

# ============================================================================
# ADVANCED FEATURES (Optional)
# ============================================================================

# Heartbeat-Konfiguration
HEARTBEAT_CONFIG = {
    "enabled": True,
    "min_interval_hours": 6,  # Minimum Zeit zwischen Heartbeats
    "max_heartbeats_per_date": 10,  # Maximum Heartbeats pro Datum
}

# Prognose-Konfiguration
FORECAST_CONFIG = {
    "min_data_points": 3,  # Minimum Datenpunkte für Prognose
    "confidence_bands": True,  # Konfidenzintervalle anzeigen
    "backtest_enabled": True,  # ETA-Backtest aktivieren
    "model_blend": True,  # ALT/NEU Modell-Blending
}

# Graph-Konfiguration
GRAPH_CONFIG = {
    "dpi": 110,
    "figsize": (12, 7),
    "show_heartbeats": True,
    "show_predictions": True,
    "max_forecast_days": 40,
    "style": "seaborn-v0_8-darkgrid"  # Matplotlib style
}

# ============================================================================
# DEBUG & MONITORING
# ============================================================================

# Debug-Level (0=off, 1=basic, 2=verbose)
DEBUG_LEVEL = 1

# Monitoring-Flags
MONITORING = {
    "log_api_calls": True,
    "save_raw_responses": False,
    "track_performance": True,
    "alert_on_parse_errors": True
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_config():
    """Validiert die Konfiguration beim Start"""
    errors = []
    
    # Prüfe ob aktiver Stream existiert
    if ACTIVE_STREAM not in STREAM_CONFIG:
        errors.append(f"ACTIVE_STREAM '{ACTIVE_STREAM}' nicht in STREAM_CONFIG definiert")
    
    # Prüfe ob mindestens ein Stream aktiviert ist
    if not any(config.get("enabled") for config in STREAM_CONFIG.values()):
        errors.append("Kein Stream ist aktiviert (enabled=True)")
    
    # Prüfe Zieldaten-Format
    for stream, config in STREAM_CONFIG.items():
        for date in config.get("target_dates", []):
            if not isinstance(date, str):
                errors.append(f"Zieldatum für {stream} ist kein String: {date}")
    
    return errors

# ============================================================================
# AUTO-CONFIGURATION (beim Import ausgeführt)
# ============================================================================

# Validiere Konfiguration beim Import
_validation_errors = validate_config()
if _validation_errors:
    print("⚠️ KONFIGURATIONSFEHLER:")
    for error in _validation_errors:
        print(f"   - {error}")

# Zeige aktive Konfiguration
if DEBUG_LEVEL > 0:
    print(f"📋 Konfiguration geladen:")
    print(f"   Aktiver Stream: {ACTIVE_STREAM}")
    print(f"   Stream-Name: {get_stream_name()}")
    print(f"   Zieldaten: {', '.join(get_target_dates())}")
    print(f"   Aktivierte Streams: {', '.join(get_all_enabled_streams().keys())}")
