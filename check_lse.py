import requests
from bs4 import BeautifulSoup
import json
import os
import re
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import math
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
from typing import Dict, List, Optional
from collections import defaultdict
import time
import sys
from io import BytesIO

# Import configuration
from config import (
    LSE_URL, STATUS_FILE, HISTORY_FILE, REGRESSION_MIN_POINTS, CONFIDENCE_LEVEL,
    TARGET_DATES, REQUEST_TIMEOUT, REQUEST_HEADERS, GMAIL_SMTP_SERVER, 
    GMAIL_SMTP_PORT, TELEGRAM_API_BASE, ACTIVE_STREAM, TARGET_DATE_PRE_CAS,
    TARGET_DATES_MAP, LON, UK_HOLIDAYS
)

# Lazy imports for heavy dependencies - only loaded when needed
_numpy = None
_matplotlib_plt = None
_matplotlib_dates = None
_warnings = None
_requests_session = None

def _get_numpy():
    global _numpy
    if _numpy is None:
        import numpy as np
        _numpy = np
    return _numpy

def _get_matplotlib():
    global _matplotlib_plt, _matplotlib_dates, _warnings
    if _matplotlib_plt is None:
        import warnings
        warnings.filterwarnings('ignore')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        _matplotlib_plt = plt
        _matplotlib_dates = mdates
        _warnings = warnings
    return _matplotlib_plt, _matplotlib_dates

def _get_requests_session():
    """Get a reusable requests session with optimized settings"""
    global _requests_session
    if _requests_session is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        _requests_session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]  # Updated from method_whitelist
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=1, pool_maxsize=1)
        _requests_session.mount("http://", adapter)
        _requests_session.mount("https://", adapter)
        _requests_session.headers.update(REQUEST_HEADERS)
    
    return _requests_session


# === Konfiguration und Konstanten (ergänzt) ===
# Constants now imported from config.py
# === Business-day helpers (x-axis skips weekends) ===

def business_days_elapsed(start_dt, end_dt):
    '''
    Zählt Arbeitstage (Mo–Fr) zwischen start_dt (inkl.) und end_dt (exkl.).
    Nutzt nur das Datum (keine Uhrzeiten). Für denselben Kalendertag -> 0.
    '''
    np = _get_numpy()
    s = np.datetime64(start_dt.date(), 'D')
    e = np.datetime64(end_dt.date(), 'D')
    # np.busday_count zählt Werktage im Intervall [s, e) - now with UK holidays
    return int(np.busday_count(s, e, holidays=UK_HOLIDAYS))

def add_business_days(start_dt, n):
    '''
    Addiert n Arbeitstage (Mo–Fr) auf start_dt und gibt das resultierende Datum zurück.
    n kann negativ sein. Bruchteile werden zur nächsten ganzen Zahl nach oben gerundet,
    da Vorhersagen in ganzen Kalendertagen kommuniziert werden.
    '''
    from math import ceil
    steps = int(ceil(n)) if n >= 0 else -int(ceil(abs(n)))
    current = start_dt
    step = 1 if steps >= 0 else -1
    remaining = abs(steps)
    while remaining > 0:
        current = current + timedelta(days=step)
        # Check if it's a weekday (Mo–Fr) and not a UK holiday
        if current.weekday() < 5 and current.strftime('%Y-%m-%d') not in UK_HOLIDAYS:
            remaining -= 1
    return current


# Lazy imports for optional advanced features
_scipy_stats = None
_sklearn_modules = None
_advanced_regression_available = None

def _check_advanced_regression_available():
    """Check if advanced regression libraries are available without importing them"""
    global _advanced_regression_available
    if _advanced_regression_available is None:
        try:
            import importlib.util
            scipy_spec = importlib.util.find_spec("scipy")
            sklearn_spec = importlib.util.find_spec("sklearn")
            _advanced_regression_available = scipy_spec is not None and sklearn_spec is not None
        except ImportError:
            _advanced_regression_available = False
    return _advanced_regression_available

def _get_advanced_regression():
    """Lazy load advanced regression modules only when needed"""
    global _scipy_stats, _sklearn_modules
    
    if not _check_advanced_regression_available():
        return False, (None, None, None, None, None)
    
    try:
        if _scipy_stats is None or _sklearn_modules is None:
            from scipy import stats
            from sklearn.linear_model import HuberRegressor, RANSACRegressor
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import make_pipeline
            
            _scipy_stats = stats
            _sklearn_modules = {
                'HuberRegressor': HuberRegressor,
                'RANSACRegressor': RANSACRegressor,
                'PolynomialFeatures': PolynomialFeatures,
                'make_pipeline': make_pipeline
            }
        
        return True, (_scipy_stats, 
                     _sklearn_modules['HuberRegressor'], 
                     _sklearn_modules['RANSACRegressor'], 
                     _sklearn_modules['PolynomialFeatures'], 
                     _sklearn_modules['make_pipeline'])
    except ImportError:
        return False, (None, None, None, None, None)

# File cache to reduce redundant I/O operations
_file_cache = {}
_cache_timestamps = {}

def _get_file_cache_key(filepath):
    """Get cache key for file operations"""
    try:
        import os
        mtime = os.path.getmtime(filepath)
        return f"{filepath}:{mtime}"
    except (OSError, IOError):
        return f"{filepath}:missing"

def _cached_json_load(filepath):
    """Load JSON file with caching to avoid redundant reads"""
    cache_key = _get_file_cache_key(filepath)
    
    if cache_key in _file_cache:
        return _file_cache[cache_key].copy()  # Return copy to avoid mutations
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            _file_cache[cache_key] = data
            return data.copy()
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def _cached_json_dump(data, filepath):
    """Save JSON file and update cache"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    # Update cache with new data
    cache_key = _get_file_cache_key(filepath)
    _file_cache[cache_key] = data.copy()

# Use lazy checking for advanced regression availability
def get_advanced_regression_status():
    return _check_advanced_regression_available()

URL = LSE_URL

from zoneinfo import ZoneInfo
def get_german_time():
    return datetime.now(ZoneInfo("Europe/Berlin"))

# ===== Compact forecast rendering (ALT vs. NEU) =====

def _now_berlin():
    dt = get_german_time()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ZoneInfo("Europe/Berlin"))
    return dt.astimezone(ZoneInfo("Europe/Berlin"))

def _short_date(d):  # "14 Aug"
    if d is None:
        return "—"
    if d.tzinfo is None:
        d = d.replace(tzinfo=ZoneInfo("Europe/Berlin"))
    return d.astimezone(ZoneInfo("Europe/Berlin")).strftime("%d %b").replace(".", "")

def _cal_days_until(dt, now=None):
    """Kalendertage (inkl. Wochenende) als Datumsdifferenz, ohne Uhrzeit-Rundung."""
    if not dt:
        return None
    if now is None:
        now = _now_berlin()
    tz = ZoneInfo("Europe/Berlin")
    # Falls dt naive ist, als Berlin interpretieren
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=tz)
    # Nur die Kalendertage vergleichen (Datum vs. Datum)
    return max(0, (dt.astimezone(tz).date() - now.date()).days)

def _diff_days(neu_dt, alt_dt):
    """Differenz NEU vs. ALT in Kalendertagen als kurzer Text."""
    if not neu_dt or not alt_dt:
        return ""
    d = (neu_dt.date() - alt_dt.date()).days
    if d == 0:
        return " (=)"
    sign = "+" if d > 0 else "−"
    return f" ({sign}{abs(d)} Tag{'e' if abs(d)!=1 else ''} ggü. 🔵)"

def render_compact_bullets(eta1_old_dt, eta1_new_dt,
                           eta2_old_dt, eta2_new_dt,
                           r2_old, pts_old, slope_old,
                           r2_new, pts_new, slope_new,
                           hb_count=None):
    """Kompakte, mobilfreundliche Prognose als Bullet-Liste."""
    now = _now_berlin()

    def line_for(dt):
        if not dt:
            return "—"
        return f"{_short_date(dt)} (in {_cal_days_until(dt, now)} Tagen)"

    hb_badge = f" HB×{hb_count}" if hb_count not in (None, 0) else ""

    parts = []
    parts.append("🎨 <b>Legende:</b> 🔵 ALT (linear) · 🟠 NEU (integriert)")
    parts.append("\n📌 <b>Kurzprognose</b>")

    # Dynamische Targets basierend auf aktuellem Stream
    if ACTIVE_STREAM == "pre_cas":
        target_name = TARGET_DATE_PRE_CAS.strftime('%d %B')
    else:
        target_name = "25 July"
    
    parts.append(f"🎯 <b>{target_name}</b>")
    parts.append(f"🔵 {line_for(eta1_old_dt)}")
    parts.append(f"🟠 {line_for(eta1_new_dt)}{_diff_days(eta1_new_dt, eta1_old_dt)}")

    # Nur zweites Target anzeigen, wenn nicht Pre-CAS
    if eta2_old_dt is not None and eta2_new_dt is not None:
        parts.append("\n🎯 <b>28 July</b>")
        parts.append(f"🔵 {line_for(eta2_old_dt)}")
        parts.append(f"🟠 {line_for(eta2_new_dt)}{_diff_days(eta2_new_dt, eta2_old_dt)}")

    parts.append("\n📐 <b>Modelle</b>")
    parts.append(f"R²: 🔵 {r2_old:.2f} ({pts_old}) · 🟠 {r2_new:.2f} ({pts_new}){hb_badge}")
    parts.append(f"Fortschritt: 🔵 {slope_old:.1f} d/Tag · 🟠 {slope_new:.1f} d/Tag")

    return "\n".join(parts)
# ====================================================


def _iter_observations_or_changes(history: Dict) -> List[Dict]:
    """
    Kombiniert observations und changes, normiert Timestamps auf ISO-UTC,
    validiert Daten, sortiert aufsteigend.
    Optimized for performance with early filtering and batch processing.
    """
    # Pre-allocate for better performance
    observations = history.get("observations", []) or []
    changes = history.get("changes", []) or []
    src = observations + changes
    
    if not src:
        return []
    
    # Pre-compile regex for better performance
    date_pattern = re.compile(r'^\d{1,2}\s+\w+$')
    
    out: List[Dict] = []
    utc_tz = ZoneInfo("UTC")
    
    for e in src:
        # Early type check
        if not isinstance(e, dict):
            continue
            
        ts = e.get("timestamp")
        dt = e.get("date")
        
        # Early validation
        if not ts or not dt or not date_pattern.match(str(dt)):
            continue
            
        try:
            # Optimized timestamp processing
            s = str(ts).replace('Z', '+00:00')
            dt_obj = datetime.fromisoformat(s)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=utc_tz)
            ts_iso = dt_obj.astimezone(utc_tz).isoformat()
        except Exception:
            continue
            
        kind = e.get("kind") or "change"
        out.append({"timestamp": ts_iso, "date": dt, "kind": kind})
    
    # Use key function once for sorting
    out.sort(key=lambda r: r["timestamp"])
    return out

def _iter_changes_only(history: Dict, changes_key: str = "changes") -> List[Dict]:
    """
    Gibt nur echte Änderungen zurück (keine Heartbeats), normiert Timestamps auf ISO-UTC,
    validiert Daten, sortiert aufsteigend.
    Optimized version with shared pattern and timezone objects.
    """
    src = history.get(changes_key, []) or []
    if not src:
        return []
    
    # Pre-compile regex and timezone for better performance
    date_pattern = re.compile(r'^\d{1,2}\s+\w+$')
    utc_tz = ZoneInfo("UTC")
    
    out: List[Dict] = []
    
    for e in src:
        # Early type and content validation
        if not isinstance(e, dict):
            continue
            
        ts = e.get("timestamp")
        dt = e.get("date")
        
        if not ts or not dt or not date_pattern.match(str(dt)):
            continue
            
        try:
            # Optimized timestamp processing
            s = str(ts).replace('Z', '+00:00')
            dt_obj = datetime.fromisoformat(s)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=utc_tz)
            ts_iso = dt_obj.astimezone(utc_tz).isoformat()
        except Exception:
            continue
            
        out.append({"timestamp": ts_iso, "date": dt, "kind": "change"})
    
    out.sort(key=lambda r: r["timestamp"])
    return out

# Regression result cache to avoid redundant calculations
_regression_cache = {}
_regression_cache_key = None

def _get_regression_cache_key(history):
    """Generate cache key for regression calculations"""
    import hashlib
    # Use the hash of changes for cache key
    changes_data = str(sorted(history.get('changes', []), key=lambda x: x.get('timestamp', '')))
    return hashlib.md5(changes_data.encode()).hexdigest()

def calculate_advanced_regression_forecast(history, current_date=None):
    """
    Erweiterte Regression mit mehreren Verbesserungen:
    - Outlier-resistente Regression
    - Polynomielle Regression 
    - Gewichtete Regression (neuere Daten wichtiger)
    - Konfidenzintervalle
    - Arbeitstagberechnung
    
    WICHTIG: Diese (alte) Regression verwendet nur echte Änderungen, keine Heartbeats!
    
    Optimized with caching to avoid redundant calculations.
    """
    # Stream-spezifische Datenauswahl
    if ACTIVE_STREAM == "pre_cas":
        changes_key = "pre_cas_changes"
    elif ACTIVE_STREAM == "cas":
        changes_key = "cas_changes"
    else:
        changes_key = "changes"
    
    # Check cache first
    cache_key = _get_regression_cache_key({changes_key: history.get(changes_key, [])})
    global _regression_cache, _regression_cache_key
    
    if cache_key == _regression_cache_key and _regression_cache:
        return _regression_cache.copy()  # Return cached result
    
    np = _get_numpy()  # Lazy load numpy
    
    # GEÄNDERT: Verwende stream-spezifische Daten
    source_data = _iter_changes_only(history, changes_key)
    if len(source_data) < REGRESSION_MIN_POINTS:
        return None
    
    # Extrahiere Datenpunkte (optimized)
    data_points = []
    first_timestamp = None
    
    # Pre-allocate list for better performance
    timestamps = []
    date_days_list = []
    
    for entry in source_data:  # GEÄNDERT: Verwende source_data (nur Änderungen)
        try:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            date_days = date_to_days(entry["date"])
            
            if date_days is None:
                continue
                
            if first_timestamp is None:
                first_timestamp = timestamp
            
            timestamps.append(timestamp)
            date_days_list.append(date_days)
                
        except (ValueError, KeyError) as e:
            continue
    
    # Calculate elapsed days in batch for better performance
    if not timestamps:
        return None
        
    for i, (timestamp, date_days) in enumerate(zip(timestamps, date_days_list)):
        try:
            days_elapsed = business_days_elapsed(first_timestamp, timestamp)
            data_points.append((days_elapsed, date_days, timestamp))
        except Exception as e:
            print(f"⚠️ Fehler beim Verarbeiten: {e}")
            continue
    
    if len(data_points) < REGRESSION_MIN_POINTS:
        return None
    
    x = np.array([p[0] for p in data_points]).reshape(-1, 1)
    y = np.array([p[1] for p in data_points])
    timestamps = [p[2] for p in data_points]
    
    # 1. STANDARD LINEARE REGRESSION (wie bisher)
    n = len(x)
    x_flat = x.flatten()
    slope_linear = (n * np.sum(x_flat * y) - np.sum(x_flat) * np.sum(y)) / (n * np.sum(x_flat**2) - np.sum(x_flat)**2)
    intercept_linear = (np.sum(y) - slope_linear * np.sum(x_flat)) / n
    y_pred_linear = slope_linear * x_flat + intercept_linear
    
    # R² für lineare Regression
    ss_res = np.sum((y - y_pred_linear)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2_linear = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Initialisiere Variablen für beste Modellauswahl
    best_slope = slope_linear
    best_intercept = intercept_linear
    best_r2 = r2_linear
    best_name = "Linear"
    
    models_comparison = {
        "linear": {"slope": slope_linear, "r2": r2_linear}
    }
    
    # Only try advanced regression if available
    advanced_available, (stats, HuberRegressor, RANSACRegressor, PolynomialFeatures, make_pipeline) = _get_advanced_regression()
    if advanced_available:
        # 2. ROBUST REGRESSION (Outlier-resistent)
        try:
            huber = HuberRegressor(epsilon=1.5)
            huber.fit(x, y)
            slope_robust = huber.coef_[0]
            intercept_robust = huber.intercept_
            y_pred_robust = huber.predict(x)
            r2_robust = 1 - np.sum((y - y_pred_robust)**2) / ss_tot if ss_tot > 0 else 0
            models_comparison["robust"] = {"slope": slope_robust, "r2": r2_robust}
            
            if r2_robust > best_r2:
                best_slope = slope_robust
                best_intercept = intercept_robust
                best_r2 = r2_robust
                best_name = "Robust"
        except (ValueError, TypeError, KeyError) as e:
                print(f"⚠️ Robust-Modell fehlgeschlagen: {e}")
                pass
        
        # 3. GEWICHTETE REGRESSION (neuere Daten wichtiger)
        try:
            weights = np.exp(x_flat / np.max(x_flat) * 2)  # Exponentiell steigende Gewichte
            weighted_slope, weighted_intercept = np.polyfit(x_flat, y, 1, w=weights)
            y_pred_weighted = weighted_slope * x_flat + weighted_intercept
            r2_weighted = 1 - np.sum((y - y_pred_weighted)**2) / ss_tot if ss_tot > 0 else 0
            models_comparison["weighted"] = {"slope": weighted_slope, "r2": r2_weighted}
            
            if r2_weighted > best_r2:
                best_slope = weighted_slope
                best_intercept = weighted_intercept
                best_r2 = r2_weighted
                best_name = "Gewichtet"
        except (ValueError, TypeError, KeyError) as e:
                print(f"⚠️ Gewichtetes Modell fehlgeschlagen: {e}")
                pass
        
        # 4. POLYNOMIELLE REGRESSION (Grad 2) - nur für Information
        if len(data_points) >= 4:
            try:
                poly_model = make_pipeline(PolynomialFeatures(2), HuberRegressor())
                poly_model.fit(x, y)
                y_pred_poly = poly_model.predict(x)
                r2_poly = 1 - np.sum((y - y_pred_poly)**2) / ss_tot if ss_tot > 0 else 0
                models_comparison["polynomial"] = {"r2": r2_poly}
            except (ValueError, TypeError, KeyError) as e:
                print(f"⚠️ Polynomielles Modell fehlgeschlagen: {e}")
                pass
    
    # 5. MOVING AVERAGE der Geschwindigkeit
    velocities = []
    if len(data_points) >= 3:
        for i in range(1, len(data_points)):
            dx = data_points[i][0] - data_points[i-1][0]
            dy = data_points[i][1] - data_points[i-1][1]
            if dx > 0:
                velocities.append(dy / dx)
        
        if velocities:
            # Verwende die letzten 3 Geschwindigkeiten für kurzfristigen Trend
            recent_velocities = velocities[-3:] if len(velocities) >= 3 else velocities
            recent_avg_velocity = np.mean(recent_velocities)
            
            if advanced_available:
                # Gewichteter Moving Average (neuere wichtiger)
                weights_ma = np.exp(np.linspace(0, 2, len(velocities)))
                weighted_avg_velocity = np.average(velocities, weights=weights_ma)
            else:
                weighted_avg_velocity = np.mean(velocities)
        else:
            recent_avg_velocity = best_slope
            weighted_avg_velocity = best_slope
    else:
        recent_avg_velocity = best_slope
        weighted_avg_velocity = best_slope
    
    # 6. KONFIDENZINTERVALLE berechnen
    residuals = y - (best_slope * x_flat + best_intercept)
    std_error = np.std(residuals)
    
    # 7. PROGNOSEN
    current_time = get_german_time()
    current_days_elapsed = business_days_elapsed(first_timestamp, current_time)
    current_predicted_days = best_slope * current_days_elapsed + best_intercept
    
    # Zieldaten
    target_25_days = date_to_days(TARGET_DATES[0])  # "25 July" 
    target_28_days = date_to_days(TARGET_DATES[1])  # "28 July"
    
    predictions = {}
    days_until_25 = None
    days_until_28 = None
    
    for target_name, target_days in [(TARGET_DATES[0], target_25_days), (TARGET_DATES[1], target_28_days)]:
        if target_days and best_slope > 0:
            days_until = (target_days - current_predicted_days) / best_slope
            
            # Konfidenzintervall (95%)
            confidence_margin = (CONFIDENCE_LEVEL * std_error / abs(best_slope)) if best_slope != 0 else 0
            days_until_lower = max(0, days_until - confidence_margin)
            days_until_upper = days_until + confidence_margin
            
            
            predictions[target_name] = {
                "days": days_until,
                "days_lower": days_until_lower,
                "days_upper": days_until_upper,
                # Termine werden auf Basis von Geschäftstagen (Mo–Fr) berechnet
                "date": add_business_days(current_time, days_until) if days_until > 0 else None,
                "date_lower": add_business_days(current_time, days_until_upper) if days_until_upper > 0 else None,
                "date_upper": add_business_days(current_time, days_until_lower) if days_until_lower > 0 else None,
            }

            
            # Für Legacy-Kompatibilität
            if target_name == TARGET_DATES[0]:  # "25 July"
                days_until_25 = days_until
            elif target_name == TARGET_DATES[1]:  # "28 July"
                days_until_28 = days_until
    
    # 8. TREND-ANALYSE
    trend_analysis = "unbekannt"
    if len(velocities) >= 2:
        # Ist der Trend beschleunigend oder verlangsamend?
        recent_acceleration = recent_avg_velocity - weighted_avg_velocity
        if recent_acceleration > 0.1:
            trend_analysis = "beschleunigend"
        elif recent_acceleration < -0.1:
            trend_analysis = "verlangsamend"
        else:
            trend_analysis = "konstant"
    
    result = {
        # Basis-Informationen (für Kompatibilität)
        "slope": best_slope,
        "r_squared": best_r2,
        "current_trend_days": current_predicted_days,
        "data_points": len(data_points),
        "days_until_25_july": days_until_25,
        "days_until_28_july": days_until_28,
        
        # Erweiterte Informationen
        "model_name": best_name,
        "std_error": std_error,
        "trend_analysis": trend_analysis,
        "recent_velocity": recent_avg_velocity,
        "models": models_comparison,
        "predictions": predictions,
    }
    
    # Cache the result for future use
    _regression_cache = result.copy()
    _regression_cache_key = cache_key
    
    return result

def calculate_regression_forecast(history):
    """Wrapper-Funktion für Rückwärtskompatibilität - ruft die erweiterte Version auf"""
    return calculate_advanced_regression_forecast(history)

def _fmt_eta(days, date_obj):
    """Gibt 'In X Tagen — Am DD. Monat YYYY' oder '—' zurück."""
    if days is None or (isinstance(days, (int, float)) and days <= 0):
        return "—"
    when = date_obj if date_obj else add_business_days(get_german_time(), days)
    return f"In {int(round(days))} Tagen — Am {when.strftime('%d. %B %Y')}"

def _old_regression_summary(forecast):
    """Zieht kompakte Kennzahlen aus der bestehenden (alten) Regression."""
    r2 = forecast.get('r_squared', 0.0)
    pts = forecast.get('data_points', 0)
    slope = forecast.get('slope', 0.0)
    trend = forecast.get('trend_analysis', None)
    # 25 July
    if 'predictions' in forecast and '25 July' in forecast['predictions']:
        p25 = forecast['predictions']['25 July']
        eta25 = _fmt_eta(p25.get('days'), p25.get('date'))
    else:
        d25 = forecast.get('days_until_25_july')
        eta25 = _fmt_eta(d25, None)
    # 28 July
    if 'predictions' in forecast and '28 July' in forecast['predictions']:
        p28 = forecast['predictions']['28 July']
        eta28 = _fmt_eta(p28.get('days'), p28.get('date'))
    else:
        d28 = forecast.get('days_until_28_july')
        eta28 = _fmt_eta(d28, None)
    return {
        "name": "ALT (linear)",
        "r2": r2, "points": pts,
        "speed": f"{slope:.1f} Tage/Tag" if slope > 0 else "—",
        "trend": (trend.upper() if trend and trend != "unbekannt" else "—"),
        "eta25": eta25, "eta28": eta28
    }

# ===== ETA-Backtest & Recency-Blend Helpers =====

def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("inf")
    mid = n // 2
    if n % 2:
        return xs[mid]
    return 0.5 * (xs[mid-1] + xs[mid])

def _to_aware_berlin(dt):
    """Nimmt datetime ODER ISO-String (auch mit 'Z') und gibt tz-aware Europe/Berlin zurück."""
    if dt is None:
        return None
    tz_berlin = ZoneInfo("Europe/Berlin")
    # Strings sauber nach datetime wandeln
    if isinstance(dt, str):
        s = dt.replace('Z', '+00:00')
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            # Fallback: behandle als naive UTC
            return datetime.now(tz_berlin)
    # Naive Zeitstempel als UTC interpretieren
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(tz_berlin)

def _hours_since(ts, now=None):
    if ts is None:
        return 0.0
    now = _now_berlin() if now is None else _to_aware_berlin(now)
    ts  = _to_aware_berlin(ts)
    return max(0.0, (now - ts).total_seconds() / 3600.0)

def _recency_weight(hours_since_change, tau_hours):
    # Smooth 0→1: kurz nach Änderung ~0 (mehr TS), lange Ruhe ~1 (mehr LOESS)
    # Center leicht unter tau_hours, damit Umschalten nicht zu spät kommt
    denom = max(1e-6, 0.25 * tau_hours)
    z = (hours_since_change - 0.75 * tau_hours) / denom
    return 1.0 / (1.0 + math.exp(-z))  # Sigmoid

def _subset_rows_until(rows_all, t_cut):
    """Alle Punkte (inkl. Heartbeats) bis < t_cut."""
    t_cut = _to_aware_berlin(t_cut)
    out = []
    for r in rows_all:
        ts = _to_aware_berlin(r["timestamp"])
        if ts < t_cut:
            out.append(r)
    return out

def _eta_backtest_score(rows_all, rows_changes, cal, loess_frac, tau_hours, tz_out):
    """
    Rolling-Origin-Backtest auf Änderungspunkten:
    Für jeden echten Change i: Trainiere auf < timestamp_i, predicte ETA für dessen 'date'.
    Score = Median der |ETA_pred - ETA_true| in Tagen.
    """
    from lse_integrated_model import IntegratedRegressor
    errors_days = []

    # Brauchen einige Splits; starte erst, wenn genug Punkte zum Fitten da sind
    MIN_TRAIN = 3
    if len(rows_changes) <= MIN_TRAIN:
        return float("inf")

    for i in range(MIN_TRAIN, len(rows_changes)):
        target_date_str = rows_changes[i]["date"]       # z.B. "15 July"
        true_when       = _to_aware_berlin(rows_changes[i]["timestamp"])

        train_rows = _subset_rows_until(rows_all, true_when)
        if len(train_rows) < MIN_TRAIN:
            continue

        try:
            im = IntegratedRegressor(cal=cal, loess_frac=loess_frac, tau_hours=tau_hours).fit(train_rows)
            pred = im.predict_datetime(target_date_str, tz_out=tz_out)
            if not pred or "when_point" not in pred or pred["when_point"] is None:
                continue
            pred_when = _to_aware_berlin(pred["when_point"])
            err_days = abs((pred_when - true_when).total_seconds()) / 86400.0
            errors_days.append(err_days)
        except Exception:
            # Bei Fehlern: diesen Split ignorieren
            continue

    if len(errors_days) == 0:
        return float("inf")
    return _median(errors_days)
# ================================================


def compute_integrated_model_metrics(history):
    """
    Integrierte Regression mit:
      (1) Hyperparameter-Tuning per ETA-Backtest
      (3) Recency-Blending (mehr LOESS nach längerer Ruhe, mehr TS direkt nach Change)
    """
    from lse_integrated_model import BusinessCalendar, IntegratedRegressor, LON, BER
    from datetime import time as _time
    _np = _get_numpy()  # Lazy load numpy

    # ---- Stream-spezifische Datenauswahl ----
    if ACTIVE_STREAM == "pre_cas":
        changes_key = "pre_cas_changes"
    elif ACTIVE_STREAM == "cas":
        changes_key = "cas_changes"
    else:
        changes_key = "changes"

    # ---- Daten aufbereiten: Changes & Heartbeats ----
    # rows_all: changes + filtered heartbeats (at most 1 heartbeat after last change)
    # rows_changes: nur echte Änderungen (für Backtest)
    rows_all = []
    rows_changes = []

    # 1) Build rows_changes from stream-specific data and sort them
    for ch in history.get(changes_key, []):
        rows_changes.append({"timestamp": ch["timestamp"], "date": ch["date"]})
        rows_all.append({"timestamp": ch["timestamp"], "date": ch["date"]})
    
    rows_changes.sort(key=lambda r: _to_aware_berlin(r["timestamp"]))

    # 2) If fewer than 3 changes exist, return None (unchanged logic)
    if len(rows_changes) < 3:
        return None  # zu wenig für integriertes Modell

    # 3) Determine last_change_ts from the last entry in rows_changes
    last_change_ts = _to_aware_berlin(rows_changes[-1]["timestamp"])

    # 4) Filter observations to only heartbeats with timestamp > last_change_ts
    # If any exist, pick the one with the latest timestamp and append it to rows_all
    heartbeats = 0
    latest_heartbeat = None
    
    for ob in history.get("observations", []):
        if ob.get("kind") == "heartbeat":
            ob_ts = _to_aware_berlin(ob["timestamp"])
            if ob_ts > last_change_ts:
                # This heartbeat is after the last change
                if latest_heartbeat is None or ob_ts > _to_aware_berlin(latest_heartbeat["timestamp"]):
                    latest_heartbeat = {"timestamp": ob["timestamp"], "date": ob["date"]}
    
    # Add only the latest heartbeat (if any) to rows_all
    if latest_heartbeat is not None:
        rows_all.append(latest_heartbeat)
        heartbeats = 1

    # 5) Sort rows_all and continue with existing logic
    rows_all.sort(key=lambda r: _to_aware_berlin(r["timestamp"]))

    # ---- Kalender / Konstanten ----
    cal = BusinessCalendar(tz=LON, start=_time(10, 0), end=_time(16, 0), holidays=UK_HOLIDAYS)
    tz_out = BER

    # ---- (1) ETA-Backtest: kleines Grid ----
    FRACS = [0.5, 0.6, 0.7]
    TAUS  = [9.0, 12.0, 18.0]

    best = {"score": float("inf"), "frac": 0.6, "tau": 12.0}
    for frac in FRACS:
        for tau in TAUS:
            score = _eta_backtest_score(rows_all, rows_changes, cal, frac, tau, tz_out)
            # Optional: leichte Regularisierung gegen zu hohe Glättung
            score_reg = score + 0.02 * (frac - 0.6)**2 + 0.001 * (tau - 12.0)**2
            if score_reg < best["score"]:
                best = {"score": score_reg, "frac": frac, "tau": tau}

    base_frac = best["frac"]
    base_tau  = best["tau"]

    # ---- (3) Recency-Blend: Parameter dynamisch anpassen ----
    last_change_ts = _to_aware_berlin(rows_changes[-1]["timestamp"])
    h_since = _hours_since(last_change_ts)
    w = _recency_weight(h_since, base_tau)  # 0..1

    # Bei frischer Änderung (w~0) → weniger LOESS / kürzere Tau,
    # bei langer Ruhe (w~1) → mehr LOESS / längere Tau
    eff_frac = max(0.35, min(0.85, base_frac * (0.85 + 0.30 * w)))
    eff_tau  = max(6.0,  min(24.0,  base_tau  * (0.80 + 0.50 * w)))

    # ---- Final fit mit effektiven Parametern ----
    imodel = IntegratedRegressor(cal=cal, loess_frac=eff_frac, tau_hours=eff_tau).fit(rows_all)

    # R² auf beobachteten Punkten (gegen die interne Blend-Vorhersage)
    x_obs = imodel.x_
    y_obs = imodel.y_
    y_hat = _np.array([imodel._blend_predict_scalar(float(xv)) for xv in x_obs])
    ss_res = float(_np.sum((y_obs - y_hat) ** 2))
    ss_tot = float(_np.sum((y_obs - _np.mean(y_obs)) ** 2))
    r2_new = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Geschwindigkeit in "Tage pro Geschäftstag"
    hours_per_day = (cal.end.hour - cal.start.hour) + (cal.end.minute - cal.start.minute) / 60.0
    avg_prog_new = imodel.ts_.b * hours_per_day  # y pro Business-Day

    def _fmt_eta(diff_days, when_dt):
        if when_dt is None:
            return "—"
        return f"{when_dt.strftime('%d. %B %Y')} (in {int(round(diff_days))} Tagen)"

    # Für "in X Tagen" verwenden wir Kalendertage (inkl. WE)
    now_de = _now_berlin()

    # Vorhersagen (echte Datumsobjekte) - Stream-spezifisch
    if ACTIVE_STREAM == "pre_cas":
        target_str = TARGET_DATE_PRE_CAS.strftime('%d %B')
        pred = imodel.predict_datetime(target_str, tz_out=tz_out)
        # Eine einzelne Vorhersage für Pre-CAS
        d_days = (pred["when_point"] - now_de).total_seconds() / 86400.0 if pred and pred.get("when_point") else None
        
        return {
            "name": "NEU (integriert)",
            "r2": r2_new,
            "points": len(rows_all),
            "speed": f"{avg_prog_new:.1f} Tage/Tag",
            "speed_val": float(avg_prog_new),
            "eta_pre_cas": _fmt_eta(d_days, pred["when_point"]) if d_days is not None else "—",
            "eta_pre_cas_dt": pred["when_point"] if pred else None,
            # Legacy compatibility
            "eta25": "—",
            "eta28": "—",
            "eta25_dt": None,
            "eta28_dt": None,
            "heartbeats": heartbeats,
            # Debug/Transparenz (optional):
            "params": {"base_frac": base_frac, "base_tau": base_tau, "eff_frac": eff_frac, "eff_tau": eff_tau, "recency_w": w, "h_since_change": h_since, "backtest_score_med_days": best["score"]},
        }
    else:
        # Existing code für AOA mit 25/28 July
        pred25 = imodel.predict_datetime("25 July", tz_out=tz_out)
        pred28 = imodel.predict_datetime("28 July", tz_out=tz_out)
        
        # Für "in X Tagen" verwenden wir Kalendertage (inkl. WE)
        d25 = (pred25["when_point"] - now_de).total_seconds() / 86400.0 if pred25 and pred25.get("when_point") else None
        d28 = (pred28["when_point"] - now_de).total_seconds() / 86400.0 if pred28 and pred28.get("when_point") else None

        return {
            "name": "NEU (integriert)",
            "r2": r2_new,
            "points": len(rows_all),
            "speed": f"{avg_prog_new:.1f} Tage/Tag",
            "speed_val": float(avg_prog_new),                 # numerisch
            "eta25": _fmt_eta(d25, pred25["when_point"]) if d25 is not None else "—",
            "eta28": _fmt_eta(d28, pred28["when_point"]) if d28 is not None else "—",
            "eta25_dt": pred25["when_point"] if pred25 else None,
            "eta28_dt": pred28["when_point"] if pred28 else None,
            "heartbeats": heartbeats,
            # Debug/Transparenz (optional):
            "params": {"base_frac": base_frac, "base_tau": base_tau, "eff_frac": eff_frac, "eff_tau": eff_tau, "recency_w": w, "h_since_change": h_since, "backtest_score_med_days": best["score"]},
        }


def create_enhanced_forecast_text(forecast):
    """Kompakte, mobilfreundliche Prognose (ALT vs. NEU) mit Legende und Kalendertagen."""
    if not forecast:
        return "\n📊 Prognose: Noch nicht genügend Daten für eine zuverlässige Vorhersage."

    # Basis (ALT)
    r2_old   = float(forecast.get("r_squared", 0.0))
    pts_old  = int(forecast.get("data_points", 0))
    slope_old = float(forecast.get("slope", 0.0))

    # Stream-spezifische Target-Logik
    if ACTIVE_STREAM == "pre_cas":
        target_date_str = TARGET_DATE_PRE_CAS.strftime('%d %B')
        # Für Pre-CAS nur ein Target verwenden
        target1_name = "Pre-CAS Target"
        def _old_eta_dt(frc, key):
            try:
                return frc.get("predictions", {}).get(target_date_str, {}).get("date")
            except Exception:
                return None
        eta1_old_dt = _old_eta_dt(forecast, target_date_str)
        eta2_old_dt = None
    else:
        # AOA: zwei Targets wie bisher
        target1_name = "25 July"
        target2_name = "28 July"
        def _old_eta_dt(frc, key):
            try:
                return frc.get("predictions", {}).get(key, {}).get("date")
            except Exception:
                return None
        eta1_old_dt = _old_eta_dt(forecast, "25 July")
        eta2_old_dt = _old_eta_dt(forecast, "28 July")

    # Neue (integrierte) Metriken + echte ETA-Datetimes
    hist = get_history()
    try:
        new_s = compute_integrated_model_metrics(hist)
    except Exception as e:
        print(f"⚠️ Integriertes Modell temporär deaktiviert: {e}")
        new_s = None

    if not new_s:
        # Fallback: nur ALT kompakt ausgeben
        parts = []
        parts.append("🎨 <b>Legende:</b> 🔵 ALT (linear) · 🟠 NEU (integriert)")
        parts.append("\n📌 <b>Kurzprognose</b>")
        parts.append(f"🎯 <b>{target1_name}</b>")
        parts.append(f"🔵 {_short_date(eta1_old_dt) if eta1_old_dt else '—'}"
                     f" (in {_cal_days_until(eta1_old_dt)} Tagen)" if eta1_old_dt else "🔵 —")
        if eta2_old_dt is not None:  # Only show second target for AOA
            parts.append(f"\n🎯 <b>{target2_name}</b>")
            parts.append(f"🔵 {_short_date(eta2_old_dt) if eta2_old_dt else '—'}"
                         f" (in {_cal_days_until(eta2_old_dt)} Tagen)" if eta2_old_dt else "🔵 —")
        parts.append("\n📐 <b>Modelle</b>")
        parts.append(f"R²: 🔵 {r2_old:.2f} ({pts_old})")
        parts.append(f"Fortschritt: 🔵 {slope_old:.1f} d/Tag")
        return "\n".join(parts)

    # NEU-Werte
    r2_new   = float(new_s["r2"])
    pts_new  = int(new_s["points"])
    slope_new = float(new_s.get("speed_val", 0.0))
    hb_count = int(new_s.get("heartbeats") or 0)
    
    if ACTIVE_STREAM == "pre_cas":
        eta1_new_dt = new_s.get("eta_pre_cas_dt")
        eta2_new_dt = None
    else:
        eta1_new_dt = new_s.get("eta25_dt")
        eta2_new_dt = new_s.get("eta28_dt")

    # Kompakte Bullets (Kalendertage!)
    text = render_compact_bullets(
        eta1_old_dt, eta1_new_dt,
        eta2_old_dt, eta2_new_dt,
        r2_old, pts_old, slope_old,
        r2_new, pts_new, slope_new,
        hb_count=hb_count
    )

    return "\n" + text

def create_forecast_text(forecast):
    """Verwendet immer die enhanced Version"""
    return create_enhanced_forecast_text(forecast)

def create_pre_cas_forecast_text(forecast):
    """Pre-CAS spezifische Prognose mit einzelnem Zieldatum"""
    if not forecast:
        return "\n📊 Prognose: Noch nicht genügend Daten für eine zuverlässige Pre-CAS Vorhersage."
    
    # Basis Metriken
    r2 = float(forecast.get("r_squared", 0.0))
    pts = int(forecast.get("data_points", 0))
    slope = float(forecast.get("slope", 0.0))
    
    parts = []
    parts.append("\n📊 <b>Pre-CAS Prognose</b>")
    parts.append(f"🎯 <b>Target:</b> {TARGET_DATE_PRE_CAS.strftime('%d %b %Y')} (LON)")
    
    # Versuche ETA zu berechnen für Pre-CAS Zieldatum
    try:
        target_date_str = TARGET_DATE_PRE_CAS.strftime('%d %B')
        target_days = date_to_days(target_date_str)
        
        if target_days and slope > 0:
            current_predicted_days = forecast.get("current_trend_days", 0)
            days_until = (target_days - current_predicted_days) / slope
            
            if days_until > 0:
                from datetime import datetime, timedelta
                eta_date = datetime.now() + timedelta(days=days_until)
                parts.append(f"📅 <b>ETA:</b> {eta_date.strftime('%d.%m.%Y')} (in ~{days_until:.0f} Tagen)")
            else:
                parts.append(f"📅 <b>ETA:</b> Ziel bereits erreicht oder überschritten")
        else:
            parts.append(f"📅 <b>ETA:</b> Nicht berechenbar (Trend zu schwach)")
    except Exception as e:
        parts.append(f"📅 <b>ETA:</b> Berechnung fehlgeschlagen")
    
    parts.append(f"\n📐 <b>Modell:</b> R²={r2:.2f} ({pts} Punkte)")
    parts.append(f"🚀 <b>Fortschritt:</b> {slope:.1f} Tage/Tag")
    
    return "\n".join(parts)

def create_progression_graph(history, current_date, forecast=None):
    """
    ALT vs. NEU mit:
      • exakten ETA-Schnittpunkten (25/28 July) auf den Regressionslinien,
      • Heartbeats als Kreuze auf dem jeweiligen y-Wert,
      • glatten Linien (fraktionale Business-Days) ohne Unsicherheitsflächen,
      • kompakten Achsen & Datumsformaten.

    Gibt BytesIO (PNG) zurück oder None.
    """
    # Lazy load matplotlib and numpy
    plt, mdates = _get_matplotlib()
    np = _get_numpy()
    
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    COL_ALT = "#1f77b4"   # Alt: blau
    COL_NEU = "#ff7f0e"   # Neu: orange
    COL_HB  = "#2ca02c"   # Heartbeats: grün

    # ---------- Helfer ----------
    def _to_naive_berlin(dt_like):
        if dt_like is None:
            return None
        if isinstance(dt_like, datetime):
            if dt_like.tzinfo is None:
                return dt_like.replace(tzinfo=ZoneInfo("Europe/Berlin")).astimezone(
                    ZoneInfo("Europe/Berlin")
                ).replace(tzinfo=None)
            return dt_like.astimezone(ZoneInfo("Europe/Berlin")).replace(tzinfo=None)
        try:
            dtx = datetime.fromisoformat(str(dt_like))
            return dtx.astimezone(ZoneInfo("Europe/Berlin")).replace(tzinfo=None)
        except Exception:
            return None

    # fraktionale Business-Days (werktags + Tagesanteil)
    def _bizdays_float(start: datetime, t: datetime) -> float:
        np = _get_numpy()  # Lazy load
        s0 = datetime(start.year, start.month, start.day)
        t0 = datetime(t.year, t.month, t.day)
        full = float(np.busday_count(np.datetime64(s0.date()), np.datetime64(t0.date()), holidays=UK_HOLIDAYS))
        def _frac(d: datetime) -> float:
            if not bool(np.is_busday(np.datetime64(d.date()), holidays=UK_HOLIDAYS)):
                return 0.0
            return (d - datetime(d.year, d.month, d.day)).total_seconds() / 86400.0
        return full + _frac(t) - _frac(s0)

    def _days_to_dt(year, doy):
        try:
            start = datetime(year, 1, 1)
            return (start + timedelta(days=float(doy) - 1)).replace(tzinfo=None)
        except Exception:
            return None

    def _fmt_day_of_year(v, pos=None, year=None):
        dt = _days_to_dt(year, v)
        return dt.strftime('%d %b') if dt else ''

    # exakter Schnittpunkt t* für Zielhöhe y_target auf y(t) = m * BD(t) + b
    def _solve_time_for_level(m, b, y_target, t0, t_hint, sign_positive=True):
        if m is None or m == 0:
            return None

        def f(t):  # y(t)
            return m * _bizdays_float(t0, t) + b

        # Bracketing um t_hint (heute) – expandieren bis Zielhöhe eingeschlossen ist
        low  = t_hint - timedelta(days=5)
        high = t_hint + timedelta(days=20)
        max_expand = 20
        if sign_positive:
            while f(low)  > y_target and max_expand > 0:
                low  -= timedelta(days=10); max_expand -= 1
            while f(high) < y_target and max_expand > 0:
                high += timedelta(days=10); max_expand -= 1
        else:
            while f(low)  < y_target and max_expand > 0:
                low  -= timedelta(days=10); max_expand -= 1
            while f(high) > y_target and max_expand > 0:
                high += timedelta(days=10); max_expand -= 1

        if (f(low) - y_target) * (f(high) - y_target) > 0:
            return None  # kein gültiges Intervall

        # Bisektion
        for _ in range(50):
            mid = low + (high - low) / 2
            val = f(mid)
            if abs(val - y_target) < 1e-6:
                return mid
            if (val < y_target) == sign_positive:
                low = mid
            else:
                high = mid
        return mid

    # ---------- Daten sammeln ----------
    entries = list(_iter_observations_or_changes(history))
    if len(entries) < REGRESSION_MIN_POINTS:
        return None

    change_ts, change_y, change_labels = [], [], []
    
    # Sammle alle echten Änderungen
    for e in entries:
        try:
            ts = _to_naive_berlin(datetime.fromisoformat(e["timestamp"]))
            yv = date_to_days(e["date"])
            if ts is None or yv is None:
                continue
            if e.get("kind") != "heartbeat":
                change_ts.append(ts); change_y.append(yv); change_labels.append(e["date"])
        except Exception:
            continue

    if not change_ts:
        return None

    ordered = sorted(zip(change_ts, change_y, change_labels), key=lambda r: r[0])
    change_ts, change_y, change_labels = [list(t) for t in zip(*ordered)]
    
    # GEÄNDERT: Filtere Heartbeats - nur den neuesten nach der letzten echten Änderung
    hb_ts, hb_y = [], []
    if change_ts:  # Wenn es echte Änderungen gibt
        last_change_ts = change_ts[-1]  # Letzter Change-Zeitpunkt
        
        # Sammle alle Heartbeats nach der letzten Änderung
        heartbeats_after_last_change = []
        for e in entries:
            try:
                if e.get("kind") == "heartbeat":
                    ts = _to_naive_berlin(datetime.fromisoformat(e["timestamp"]))
                    yv = date_to_days(e["date"])
                    if ts is None or yv is None:
                        continue
                    if ts > last_change_ts:  # Nur Heartbeats nach letzter Änderung
                        heartbeats_after_last_change.append((ts, yv))
            except Exception:
                continue
        
        # Wähle nur den neuesten Heartbeat (falls vorhanden)
        if heartbeats_after_last_change:
            heartbeats_after_last_change.sort(key=lambda x: x[0])  # Sortiere nach Zeitstempel
            latest_hb_ts, latest_hb_y = heartbeats_after_last_change[-1]  # Neuester
            hb_ts = [latest_hb_ts]
            hb_y = [latest_hb_y]

    now_de  = _to_naive_berlin(get_german_time())
    year_ref = now_de.year

    # ---------- Plot-Setup ----------
    try: plt.style.use("seaborn-v0_8-darkgrid")
    except Exception: pass
    fig, ax = plt.subplots(figsize=(12, 7))

    # Historische Punkte + Labels (nur die letzten 8 beschriften, versetzt)
    ax.scatter(change_ts, change_y, s=90, zorder=5, label="Änderungen (historisch)", alpha=0.9, color=COL_ALT)
    last_k = max(0, len(change_ts) - 8)
    for i, (ts, y, lbl) in enumerate(zip(change_ts[last_k:], change_y[last_k:], change_labels[last_k:])):
        dy = 10 if i % 2 == 0 else -14
        ax.annotate(lbl, (ts, y), xytext=(6, dy), textcoords="offset points",
                    fontsize=8.5, alpha=0.85, ha="left", va="center")

    # Heute & aktueller Punkt
    ax.axvline(now_de, linewidth=1.0, linestyle=":", alpha=0.8)
    ax.scatter([change_ts[-1]], [change_y[-1]], s=100, zorder=6, label="Aktuell", color=COL_NEU)

    # Zielhöhen (horizontale Linien) - Single Pre-CAS target
    if ACTIVE_STREAM == "pre_cas":
        target_date_str = TARGET_DATE_PRE_CAS.strftime('%d %B')
        target_days = date_to_days(target_date_str)
        if target_days is not None:
            ax.axhline(target_days, linestyle="--", linewidth=1.5, alpha=0.7, color='red')
            ax.text(change_ts[0], target_days, f" {target_date_str} (Pre-CAS)", va="center", ha="left", fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3))
    else:
        # Backward compatibility for AOA dual targets
        target_map = {TARGET_DATES[0]: date_to_days(TARGET_DATES[0]), TARGET_DATES[1]: date_to_days(TARGET_DATES[1])}
        for tname, ty in target_map.items():
            if ty is not None:
                ax.axhline(ty, linestyle=":", linewidth=1.0, alpha=0.5)
                ax.text(change_ts[0], ty, f" {tname}", va="center", ha="left", fontsize=9)

    # feines Zeitraster (6h) für glatte Linien
    first_ts = change_ts[0]
    left_bound  = min(change_ts[0], now_de - timedelta(days=3)) - timedelta(days=1)
    right_bound = now_de + timedelta(days=40)
    nsteps = int(max(1, (right_bound - left_bound).total_seconds() // (6 * 3600)))
    grid_ts = [left_bound + timedelta(hours=6 * i) for i in range(nsteps + 1)]
    grid_bd = [_bizdays_float(first_ts, t) for t in grid_ts]

    # ---------- ALT-Linie + ETAs ----------
    alt_eta = {}  # name -> (t*, y*)
    if forecast and float(forecast.get("slope", 0.0)) > 0.0:
        m_old = float(forecast["slope"])
        y_now = forecast.get("current_trend_days", change_y[-1])
        b_old = y_now - m_old * _bizdays_float(first_ts, now_de)

        y_old = np.array([m_old * bd + b_old for bd in grid_bd])
        ax.plot(
            grid_ts, y_old, linestyle=(0, (6, 4)), linewidth=2.0,
            label=f"ALT: {forecast.get('model_name','Linear')} (R²={float(forecast.get('r_squared',0.0)):.2f})",
            color=COL_ALT, alpha=0.95
        )

        # ETA for active stream target
        if ACTIVE_STREAM == "pre_cas":
            target_date_str = TARGET_DATE_PRE_CAS.strftime('%d %B')
            target_days = date_to_days(target_date_str)
            if target_days is not None:
                t_star = _solve_time_for_level(m_old, b_old, target_days, first_ts, now_de, sign_positive=(m_old > 0))
                if t_star:
                    y_star = m_old * _bizdays_float(first_ts, t_star) + b_old
                    alt_eta["pre_cas"] = (t_star, y_star)
        else:
            # Backward compatibility for dual AOA targets
            target_map = {TARGET_DATES[0]: date_to_days(TARGET_DATES[0]), TARGET_DATES[1]: date_to_days(TARGET_DATES[1])}
            for name, ty in target_map.items():
                if ty is None: continue
                t_star = _solve_time_for_level(m_old, b_old, ty, first_ts, now_de, sign_positive=(m_old > 0))
                if t_star:
                    y_star = m_old * _bizdays_float(first_ts, t_star) + b_old
                    alt_eta[name] = (t_star, y_star)

    # ---------- NEU-Linie + ETAs ----------
    neu_eta = {}
    try:
        from lse_integrated_model import BusinessCalendar, IntegratedRegressor, LON, BER
        from datetime import time as _time

        rows = [{"timestamp": e["timestamp"], "date": e["date"]} for e in entries]
        if len(rows) >= REGRESSION_MIN_POINTS:
            cal = BusinessCalendar(tz=LON, start=_time(10, 0), end=_time(16, 0), holidays=UK_HOLIDAYS)
            imodel = IntegratedRegressor(cal=cal, loess_frac=0.6, tau_hours=12.0).fit(rows)

            hours_per_day = (cal.end.hour - cal.start.hour) + (cal.end.minute - cal.start.minute) / 60.0
            m_new = float(imodel.ts_.b * hours_per_day)

            y_curr = date_to_days(current_date) or (change_y[-1] if change_y else None)
            if y_curr is not None and m_new is not None:
                b_new = y_curr - m_new * _bizdays_float(first_ts, now_de)
                y_new = np.array([m_new * bd + b_new for bd in grid_bd])
                ax.plot(grid_ts, y_new, linewidth=2.6, label="NEU: integrierte Regression",
                        color=COL_NEU, alpha=0.95)

                # ETA for active stream target
                if ACTIVE_STREAM == "pre_cas":
                    target_date_str = TARGET_DATE_PRE_CAS.strftime('%d %B')
                    target_days = date_to_days(target_date_str)
                    if target_days is not None:
                        t_star = _solve_time_for_level(m_new, b_new, target_days, first_ts, now_de, sign_positive=(m_new > 0))
                        if t_star:
                            y_star = m_new * _bizdays_float(first_ts, t_star) + b_new
                            neu_eta["pre_cas"] = (t_star, y_star)
                else:
                    # Backward compatibility for dual AOA targets
                    target_map = {TARGET_DATES[0]: date_to_days(TARGET_DATES[0]), TARGET_DATES[1]: date_to_days(TARGET_DATES[1])}
                    for name, ty in target_map.items():
                        if ty is None: continue
                        t_star = _solve_time_for_level(m_new, b_new, ty, first_ts, now_de, sign_positive=(m_new > 0))
                        if t_star:
                            y_star = m_new * _bizdays_float(first_ts, t_star) + b_new
                            neu_eta[name] = (t_star, y_star)
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ NEU-Regression konnte nicht gezeichnet werden: {e}")

    # ---------- Sterne & Labels (ALT oben, NEU unten) ----------
    def _plot_eta(name, alt_pt, neu_pt):
        if alt_pt:
            ax.plot([alt_pt[0]], [alt_pt[1]], marker="*", markersize=12, linestyle="None",
                    zorder=7, color=COL_ALT)
            ax.axvline(alt_pt[0], linestyle="--", linewidth=0.8, alpha=0.6, color=COL_ALT)
            ax.annotate(f"ALT ETA {name}", (alt_pt[0], alt_pt[1]),
                        xytext=(6, 9), textcoords="offset points", ha="left", va="bottom", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        if neu_pt:
            ax.plot([neu_pt[0]], [neu_pt[1]], marker="*", markersize=12, linestyle="None",
                    zorder=7, color=COL_NEU)
            ax.axvline(neu_pt[0], linestyle="--", linewidth=0.8, alpha=0.6, color=COL_NEU)
            ax.annotate(f"NEU ETA {name}", (neu_pt[0], neu_pt[1]),
                        xytext=(6, -12), textcoords="offset points", ha="left", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # Plot ETAs based on active stream
    if ACTIVE_STREAM == "pre_cas":
        _plot_eta("Pre-CAS", alt_eta.get("pre_cas"), neu_eta.get("pre_cas"))
    else:
        # Backward compatibility for dual AOA targets
        _plot_eta("25", alt_eta.get("25 July"), neu_eta.get("25 July"))
        _plot_eta("28", alt_eta.get("28 July"), neu_eta.get("28 July"))

    # ---------- Achsenbegrenzungen ----------
    eta_dates = [p[0] for p in list(alt_eta.values()) + list(neu_eta.values()) if p]
    right_edge = max(eta_dates) if eta_dates else change_ts[-1]
    left_edge  = min(change_ts[0], now_de - timedelta(days=3))
    ax.set_xlim(left_edge - timedelta(days=1), right_edge + timedelta(days=1))

    y_min = min(change_y)
    if ACTIVE_STREAM == "pre_cas":
        target_days = date_to_days(TARGET_DATE_PRE_CAS.strftime('%d %B'))
        y_max = max([*change_y, target_days] if target_days else change_y)
    else:
        # Backward compatibility
        target_map = {TARGET_DATES[0]: date_to_days(TARGET_DATES[0]), TARGET_DATES[1]: date_to_days(TARGET_DATES[1])}
        y_max = max([*change_y, *(v for v in target_map.values() if v is not None)])
    ax.set_ylim(y_min - 2, y_max + 2)

    # ---------- Heartbeats als Kreuze auf y ----------
    if hb_ts:
        ax.scatter(
            hb_ts, hb_y,
            marker='x', s=60, linewidths=1.8,
            color=COL_HB, alpha=0.9,
            label="Heartbeats (NEU)"
        )
        # Optionale, dezente Führungslinien:
        # ylo, yhi = ax.get_ylim()
        # ax.vlines(hb_ts, ylo + 0.01*(yhi-ylo), hb_y, colors=COL_HB, linewidth=0.8, alpha=0.35)

    # ---------- Achsenformat ----------
    if ACTIVE_STREAM == "pre_cas":
        ax.set_title(f"Forecast / ETA — {ACTIVE_STREAM.upper()} — Target: {TARGET_DATE_PRE_CAS.strftime('%d %b %Y')} (LON)")
    else:
        ax.set_title("Fortschritt & Prognose — ALT vs. NEU")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Verarbeitungsdatum")

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.tick_params(axis="x", which="major", labelsize=9)
    ax.tick_params(axis="x", which="minor", length=3)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 3, 5]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: _fmt_day_of_year(v, pos, year_ref)))
    ax.tick_params(axis="y", labelsize=9)

    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # ---------- Export ----------
    try:
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        print(f"⚠️ Diagramm konnte nicht erzeugt werden: {e}")
        try: plt.close()
        except Exception: pass
        return None

def send_telegram_message(message, chat_type='main', photo_buffer=None, caption=None, parse_mode='HTML'):
    """
    Unified Telegram sending function for all chat types.
    
    Args:
        message: Text message to send (ignored if photo_buffer provided)
        chat_type: 'main', 'mama', or 'papa'
        photo_buffer: Optional BytesIO buffer for photo
        caption: Caption for photo (used instead of message if photo provided)
        parse_mode: 'HTML' for main, None for mama/papa
    """
    # Determine bot credentials based on chat type
    if chat_type == 'mama':
        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN_MAMA')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID_MAMA')
        parse_mode = None  # Simple text for mama
    elif chat_type == 'papa':
        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN_PAPA')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID_PAPA')
        parse_mode = None  # Simple text for papa
    else:  # main
        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print(f"Telegram für {chat_type} nicht konfiguriert")
        return False
    
    try:
        if photo_buffer:
            # Send photo with caption
            url = f"{TELEGRAM_API_BASE}{bot_token}/sendPhoto"
            files = {'photo': ('graph.png', photo_buffer, 'image/png')}
            data = {
                'chat_id': chat_id,
                'caption': caption or '',
                'parse_mode': parse_mode
            }
            response = requests.post(url, files=files, data=data)
        else:
            # Send text message
            url = f"{TELEGRAM_API_BASE}{bot_token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            print(f"✅ Telegram-Nachricht an {chat_type} gesendet!")
            return True
        else:
            print(f"❌ Telegram-Fehler ({chat_type}): {response.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram-Fehler ({chat_type}): {e}")
        return False

# Wrapper functions for backwards compatibility
def send_telegram(message):
    """Sendet eine Nachricht über Telegram Bot"""
    return send_telegram_message(message, 'main')

def send_telegram_photo(photo_buffer, caption, parse_mode='HTML'):
    """Sendet ein Foto über Telegram Bot"""
    return send_telegram_message('', 'main', photo_buffer, caption, parse_mode)

def send_telegram_mama(old_date, new_date):
    """Sendet eine einfache Nachricht an Mama über separaten Bot"""
    message = f"LSE-Datums-Update!\n\nVom: {old_date}\nAuf: {new_date}"
    return send_telegram_message(message, 'mama')

def send_telegram_papa(old_date, new_date):
    """Sendet eine einfache Nachricht an Papa über separaten Bot"""
    message = f"LSE-Datums-Update!\n\nVom: {old_date}\nAuf: {new_date}"
    return send_telegram_message(message, 'papa')


def migrate_json_files():
    """Migriert die JSON-Dateien zur neuen Struktur ohne Datenverlust"""
    # Migriere status.json
    try:
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
        changed = False
        if 'pre_cas_date' not in status:
            status['pre_cas_date'] = None
            print("✅ Status-Migration: pre_cas_date hinzugefügt")
            changed = True
        if 'cas_date' not in status:
            status['cas_date'] = None
            print("✅ Status-Migration: cas_date hinzugefügt")
            changed = True
        if 'last_updated_seen_utc' not in status:
            status['last_updated_seen_utc'] = None
            print("✅ Status-Migration: last_updated_seen_utc hinzugefügt")
            changed = True
        if changed:
            with open(STATUS_FILE, 'w') as f:
                json.dump(status, f, indent=2)
    except Exception as e:
        print(f"⚠️ Fehler bei Status-Migration: {e}")
    
    # Migriere history.json
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        if 'pre_cas_changes' not in history:
            history['pre_cas_changes'] = []
        if 'cas_changes' not in history:
            history['cas_changes'] = []
        if 'observations' not in history:
            history['observations'] = []
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"⚠️ Fehler bei History-Migration: {e}")

def load_status():
    """Lädt Status mit Fehlerbehandlung und Validierung"""
    try:
        status = _cached_json_load(STATUS_FILE)
        if status is None:
            raise FileNotFoundError("Status file not found")
            
        if not isinstance(status, dict):
            print("⚠️ Status ist kein Dictionary, verwende Standardwerte")
            return {"last_date": "10 July", "last_check": None, "pre_cas_date": None, "cas_date": None, "last_updated_seen_utc": None}
        if 'last_date' not in status:
            print("⚠️ last_date fehlt in status.json, verwende Standardwert")
            status['last_date'] = "10 July"
        if 'last_check' not in status:
            status['last_check'] = None
        if 'pre_cas_date' not in status:
            status['pre_cas_date'] = None
        if 'cas_date' not in status:
            status['cas_date'] = None
        if 'last_updated_seen_utc' not in status:
            status['last_updated_seen_utc'] = None
        print(f"✅ Status geladen: {status['last_date']}")
        return status
    except FileNotFoundError:
        print("ℹ️ status.json nicht gefunden, erstelle neue Datei")
        return {"last_date": "10 July", "last_check": None, "pre_cas_date": None, "cas_date": None, "last_updated_seen_utc": None}
    except json.JSONDecodeError as e:
        print(f"❌ Fehler beim Parsen von status.json: {e}")
        print("Verwende Standardwerte")
        return {"last_date": "10 July", "last_check": None, "pre_cas_date": None, "cas_date": None, "last_updated_seen_utc": None}
    except Exception as e:
        print(f"❌ Unerwarteter Fehler beim Laden von status.json: {e}")
        return {"last_date": "10 July", "last_check": None, "pre_cas_date": None, "cas_date": None, "last_updated_seen_utc": None}
def save_status(status):
    """Speichert Status mit Validierung und Verifikation"""
    # Validiere dass last_date gesetzt ist
    if not status.get('last_date'):
        print("❌ Fehler: last_date ist leer, Status wird nicht gespeichert")
        return False
    
    # Erstelle Backup bevor wir speichern
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                backup = f.read()
            with open(STATUS_FILE + '.backup', 'w') as f:
                f.write(backup)
    except Exception as e:
        print(f"⚠️ Konnte kein Backup erstellen: {e}")
    
    # Speichere mit Fehlerbehandlung
    try:
        # Füge Zeitstempel hinzu wenn nicht vorhanden
        if 'last_check' not in status:
            status['last_check'] = datetime.now().astimezone(ZoneInfo('UTC')).isoformat()
        
        _cached_json_dump(status, STATUS_FILE)
        
        # Verifiziere dass es korrekt gespeichert wurde
        saved = _cached_json_load(STATUS_FILE)
        if saved and saved.get('last_date') == status['last_date']:
            print(f"✅ Status erfolgreich gespeichert: {status['last_date']}")
            return True
        else:
            print(f"❌ FEHLER: Status nicht korrekt gespeichert!")
            print(f"   Erwartet: {status['last_date']}")
            print(f"   Gespeichert: {saved.get('last_date') if saved else 'None'}")
            # Restore backup
            if os.path.exists(STATUS_FILE + '.backup'):
                os.rename(STATUS_FILE + '.backup', STATUS_FILE)
            return False
                
    except Exception as e:
        print(f"❌ Fehler beim Speichern von status.json: {e}")
        # Restore backup
        if os.path.exists(STATUS_FILE + '.backup'):
            os.rename(STATUS_FILE + '.backup', STATUS_FILE)
        return False

def load_history():
    """Lädt Historie mit Fehlerbehandlung"""
    try:
        history = _cached_json_load(HISTORY_FILE)
        if history is None:
            raise FileNotFoundError("History file not found")
            
        # Validiere die geladenen Daten
        if not isinstance(history, dict) or 'changes' not in history:
            print("⚠️ History ist ungültig, verwende leere Historie")
            return {"changes": [], "pre_cas_changes": [], "cas_changes": [], "observations": []}
            
        if not isinstance(history['changes'], list):
            print("⚠️ History changes ist keine Liste, verwende leere Historie")
            return {"changes": [], "pre_cas_changes": [], "cas_changes": [], "observations": []}
        
        # Stelle sicher dass neue Arrays existieren
        if 'pre_cas_changes' not in history:
            history['pre_cas_changes'] = []
        if 'cas_changes' not in history:
            history['cas_changes'] = []
        if 'observations' not in history:
            history['observations'] = []
            
        print(f"✅ Historie geladen: {len(history['changes'])} Änderungen, {len(history.get('observations', []))} Beobachtungen")
        return history
    except FileNotFoundError:
        print("ℹ️ history.json nicht gefunden, erstelle neue Datei")
        return {"changes": [], "pre_cas_changes": [], "cas_changes": [], "observations": []}
    except json.JSONDecodeError as e:
        print(f"❌ Fehler beim Parsen von history.json: {e}")
        return {"changes": [], "pre_cas_changes": [], "cas_changes": [], "observations": []}
    except Exception as e:
        print(f"❌ Unerwarteter Fehler beim Laden von history.json: {e}")
        return {"changes": [], "pre_cas_changes": [], "cas_changes": [], "observations": []}

def get_history():
    """Backward-compat wrapper to load history."""
    return load_history()


def save_history(history):
    """Speichert Historie mit Validierung"""
    try:
        # Validiere die Historie
        if not isinstance(history, dict) or 'changes' not in history:
            print("❌ Fehler: Historie ist ungültig")
            return False
            
        _cached_json_dump(history, HISTORY_FILE)
        print(f"✅ Historie gespeichert: {len(history['changes'])} Änderungen")
        return True
    except Exception as e:
        print(f"❌ Fehler beim Speichern von history.json: {e}")
        return False

def date_to_days(date_str):
    """Konvertiert ein Datum wie '10 July' in Tage seit dem 1. Januar"""
    try:
        # Füge das aktuelle Jahr hinzu
        current_year = get_german_time().year
        date_obj = datetime.strptime(f"{date_str} {current_year}", "%d %B %Y")
        jan_first = datetime(current_year, 1, 1)
        return (date_obj - jan_first).days
    except:
        return None

def days_to_date(days):
    """Konvertiert Tage seit 1. Januar zurück in ein Datum"""
    current_year = get_german_time().year
    jan_first = datetime(current_year, 1, 1)
    target_date = jan_first + timedelta(days=int(days))
    return target_date.strftime("%d %B").lstrip("0")

def extract_all_other_date(text):
    """Extrahiert nur das Datum für 'all other graduate applicants'"""
    text = ' '.join(text.split())
    
    date_pattern = r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))\b'
    all_dates = re.findall(date_pattern, text, re.IGNORECASE)
    
    if len(all_dates) >= 3:
        return all_dates[-1].strip()
    elif len(all_dates) > 0:
        return all_dates[-1].strip()
    
    return None

def extract_pre_cas_date(text):
    """Extrahiert das Datum für Pre-CAS"""
    # Pattern 1: Suche nach Pre-CAS mit Datum in der gleichen Zeile
    pattern1 = r'issuing\s+Pre-CAS.*?criteria\s+on:?\s*(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))'
    match = re.search(pattern1, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: Suche nach Pre-CAS gefolgt von Datum in der Nähe (für Tabellen)
    # Bereinige Text von mehrfachen Leerzeichen/Zeilenumbrüchen
    clean_text = ' '.join(text.split())
    
    # Suche nach Pre-CAS und dem nächsten Datum danach
    pattern2 = r'issuing\s+Pre-CAS\s+for\s+offer\s+holders.*?criteria\s+on:?\s*([^.]*?)(?:Please|We\s+are|$)'
    match = re.search(pattern2, clean_text, re.IGNORECASE)
    if match:
        potential_text = match.group(1)
        # Extrahiere das erste Datum aus diesem Bereich
        date_pattern = r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))'
        date_match = re.search(date_pattern, potential_text, re.IGNORECASE)
        if date_match:
            return date_match.group(1).strip()
    
    # Pattern 3: Allgemeinere Suche - Pre-CAS gefolgt von einem Datum innerhalb von ~100 Zeichen
    pattern3 = r'Pre-CAS[^0-9]{0,100}?(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))'
    match = re.search(pattern3, clean_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None

def extract_cas_date(text):
    """Extrahiert das Datum für CAS"""
    # Suche nach CAS Pattern (aber nicht Pre-CAS)
    pattern = r'issuing\s+CAS\s+to.*?Pre-CAS\s+on:?\s*(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
def extract_last_updated(text):
    """Extrahiert 'Last updated' als UTC-datetime; None wenn nicht gefunden."""
    try:
        m = re.search(r'Last\s+updated:\s*(\d{1,2}\s+\w+\s+\d{4}),\s*(\d{1,2}:\d{2})', text, re.IGNORECASE)
        if not m:
            return None
        date_part, time_part = m.group(1), m.group(2)
        naive = datetime.strptime(f"{date_part} {time_part}", "%d %B %Y %H:%M")
        dt_lon = naive.replace(tzinfo=ZoneInfo("Europe/London"))
        dt_utc = dt_lon.astimezone(ZoneInfo("UTC"))
        return dt_utc
    except Exception:
        return None


def fetch_processing_dates():
    """Holt alle Verarbeitungsdaten von der LSE-Webseite"""
    try:
        print("Rufe LSE-Webseite ab...")
        session = _get_requests_session()
        response = session.get(URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        full_text = soup.get_text()
        # Parse Last updated
        last_up_dt = extract_last_updated(full_text)
        
        # Initialisiere Rückgabewerte
        dates = {
            'all_other': None,
            'pre_cas': None,
            'cas': None,
            'last_updated_utc': (last_up_dt.astimezone(ZoneInfo('UTC')).isoformat() if last_up_dt else None)
        }
        
        # Suche nach "all other graduate applicants" (existierende Logik)
        for element in soup.find_all(string=re.compile(r'all other graduate applicants', re.IGNORECASE)):
            parent = element.parent
            while parent and parent.name not in ['td', 'th', 'tr']:
                parent = parent.parent
            
            if parent:
                if parent.name == 'tr':
                    cells = parent.find_all(['td', 'th'])
                    for cell in cells:
                        cell_text = cell.get_text()
                        date = extract_all_other_date(cell_text)
                        if date:
                            print(f"Gefundene Daten in Zelle: {cell_text.strip()}")
                            print(f"Extrahiertes Datum für 'all other graduate applicants': {date}")
                            dates['all_other'] = date
                            break
                else:
                    row = parent.find_parent('tr')
                    if row:
                        cells = row.find_all(['td', 'th'])
                        for cell in cells:
                            cell_text = cell.get_text()
                            date = extract_all_other_date(cell_text)
                            if date:
                                print(f"Gefundene Daten in Zelle: {cell_text.strip()}")
                                print(f"Extrahiertes Datum für 'all other graduate applicants': {date}")
                                dates['all_other'] = date
                                break
        
        # Fallback für all other
        if not dates['all_other']:
            pattern = r'all other graduate applicants[^0-9]*?((?:\d{1,2}\s+\w+\s*)+)'
            match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
            
            if match:
                dates_text = match.group(1)
                date = extract_all_other_date(dates_text)
                if date:
                    print(f"Datum durch Textsuche gefunden: {date}")
                    dates['all_other'] = date
        
        # Extrahiere Pre-CAS und CAS Daten
        dates['pre_cas'] = extract_pre_cas_date(full_text)
        dates['cas'] = extract_cas_date(full_text)
        
        print(f"\n📋 Gefundene Daten:")
        print(f"   All other applicants: {dates['all_other']}")
        print(f"   Pre-CAS: {dates['pre_cas'] or 'Nicht gefunden'}")
        print(f"   CAS: {dates['cas'] or 'Nicht gefunden'}")
        
        return dates
        
    except requests.exceptions.Timeout:
        print("❌ Timeout beim Abrufen der Webseite")
        return {'all_other': None, 'pre_cas': None, 'cas': None}
    except requests.exceptions.RequestException as e:
        print(f"❌ Netzwerkfehler beim Abrufen der Webseite: {e}")
        return {'all_other': None, 'pre_cas': None, 'cas': None}
    except Exception as e:
        print(f"❌ Unerwarteter Fehler beim Abrufen der Webseite: {e}")
        return {'all_other': None, 'pre_cas': None, 'cas': None}

def send_gmail(subject, body, recipients):
    """Sendet E-Mail über Gmail an spezifische Empfänger"""
    gmail_user = os.environ.get('GMAIL_USER')
    gmail_password = os.environ.get('GMAIL_APP_PASSWORD')
    
    if not gmail_user or not gmail_password:
        print("E-Mail-Konfiguration unvollständig!")
        return False
    
    if not recipients:
        print("Keine Empfänger angegeben!")
        return False
    
    print(f"Sende E-Mail an {len(recipients)} Empfänger: {', '.join(recipients)}")
    
    success_count = 0
    for recipient in recipients:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = gmail_user
        msg['To'] = recipient
        
        try:
            server = smtplib.SMTP(GMAIL_SMTP_SERVER, GMAIL_SMTP_PORT)
            server.starttls()
            server.login(gmail_user, gmail_password)
            server.send_message(msg)
            server.quit()
            print(f"✅ E-Mail erfolgreich gesendet an {recipient}")
            success_count += 1
        except Exception as e:
            print(f"❌ E-Mail-Fehler für {recipient}: {type(e).__name__}: {e}")
    
    return success_count > 0

def main():
    print("="*50)
    print(f"LSE Status Check - {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"ACTIVE_STREAM={ACTIVE_STREAM}")
    print(f"Target: {TARGET_DATE_PRE_CAS.strftime('%d %b %Y')} (LON)")
    
    # Zeige ob erweiterte Regression verfügbar ist
    if get_advanced_regression_status():
        print("✅ Erweiterte Regression aktiviert")
    else:
        print("⚠️ Erweiterte Regression nicht verfügbar (Standard-Regression wird verwendet)")
    
    # Migriere JSON-Dateien falls nötig
    migrate_json_files()
    
    # Prüfe ob manueller Run via Telegram
    IS_MANUAL = os.environ.get('GITHUB_EVENT_NAME') == 'repository_dispatch'
    if IS_MANUAL:
        print("🔄 MANUELLER CHECK VIA TELEGRAM")
    
    print("="*50)
    
    # Lade E-Mail-Adressen
    email_main = os.environ.get('EMAIL_TO', '')
    email_2 = os.environ.get('EMAIL_TO_2', '')
    email_3 = os.environ.get('EMAIL_TO_3', '')
    
    # Kategorisiere Empfänger
    always_notify = [email for email in [email_main, email_2] if email and 'engelquast' not in email.lower()]
    conditional_notify = [email for email in [email_main, email_2, email_3] if email and 'engelquast' in email.lower()]
    
    print(f"Immer benachrichtigen: {', '.join(always_notify)}")
    print(f"Nur bei 25/28 July: {', '.join(conditional_notify)}")
    
    # Lade Status und Historie mit Fehlerbehandlung
    status = load_status()
    history = load_history()
    print(f"Letztes bekanntes Datum: {status['last_date']}")
    print(f"Letztes Pre-CAS: {status.get('pre_cas_date') or 'Noch nicht getrackt'}")
    print(f"Letztes CAS: {status.get('cas_date') or 'Noch nicht getrackt'}")
    
    # Hole alle aktuellen Daten
    print("\nRufe LSE-Webseite ab...")
    current_dates = fetch_processing_dates()
    
    
    # Heartbeat-Beobachtung (Seite aktualisiert, Datum gleich)
    try:
        last_up_iso = current_dates.get('last_updated_utc')
        if last_up_iso:
            prev_seen = status.get('last_updated_seen_utc')
            is_new_update = (prev_seen != last_up_iso)
        else:
            is_new_update = False

        if is_new_update:
            status['last_updated_seen_utc'] = last_up_iso
            save_status(status)
            current_all_other = current_dates.get('all_other')
            if current_all_other == status.get('last_date'):
                history.setdefault('observations', [])
                if not any(o.get('timestamp') == last_up_iso for o in history['observations']):
                    history['observations'].append({
                        'timestamp': last_up_iso,
                        'date': current_all_other,
                        'kind': 'heartbeat'
                    })
                    save_history(history)
    except Exception as _e:
        print(f"⚠️ Heartbeat-Logik übersprungen: {_e}")

    # Stilles Tracking für AOA (keine primären Benachrichtigungen mehr)
    if current_dates['all_other'] and current_dates['all_other'] != status.get('last_date'):
        print(f"\n📝 AOA Änderung (stilles Tracking): {status.get('last_date') or 'Unbekannt'} → {current_dates['all_other']}")
        history['changes'].append({
            "timestamp": datetime.now().astimezone(ZoneInfo('UTC')).isoformat(),
            "date": current_dates['all_other'],
            "from": status.get('last_date')
        })
        status['last_date'] = current_dates['all_other']
        # Speichere sofort
        save_history(history)
    
    # Stilles Tracking für CAS (keine Benachrichtigungen)
    if current_dates['cas'] and current_dates['cas'] != status.get('cas_date'):
        print(f"\n📝 CAS Änderung (stilles Tracking): {status.get('cas_date') or 'Unbekannt'} → {current_dates['cas']}")
        history['cas_changes'].append({
            "timestamp": datetime.now().astimezone(ZoneInfo('UTC')).isoformat(),
            "date": current_dates['cas'],
            "from": status.get('cas_date')
        })
        status['cas_date'] = current_dates['cas']
        # Speichere sofort
        save_history(history)
    
    # Hauptlogik für Pre-CAS (mit primären Benachrichtigungen)
    current_date = current_dates['pre_cas']
    
    if current_date:
        print(f"Aktuelles Datum für '{ACTIVE_STREAM}': {current_date}")
        
        # Bei manuellem Check immer Status senden (für aktiven Pre-CAS Stream)
        if IS_MANUAL:
            # Berechne aktuellen Trend und erstelle vollständige Prognose basierend auf Pre-CAS
            active_history = {"changes": history.get('pre_cas_changes', [])}
            forecast = calculate_regression_forecast(active_history)
            forecast_text = create_forecast_text(forecast) or ""
            
            telegram_msg = f"""<b>📊 LSE Status Check Ergebnis</b>

<b>Active stream:</b> {ACTIVE_STREAM}
<b>Target:</b> {TARGET_DATE_PRE_CAS.strftime('%d %b %Y')} (LON)
<b>Aktuelles Datum:</b> {current_date}
<b>Letzter Stand:</b> {status.get('pre_cas_date')}
<b>Status:</b> {"🔔 ÄNDERUNG ERKANNT!" if current_date != status.get('pre_cas_date') else "✅ Keine Änderung"}

<b>Zeitpunkt:</b> {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
{forecast_text}

<a href="{LSE_URL}">📄 LSE Webseite öffnen</a>"""
            
            # Sende Text-Nachricht
            send_telegram(telegram_msg)
            
            # Erstelle und sende Graph
            graph_buffer = create_progression_graph(active_history, current_date, forecast)
            if graph_buffer:
                graph_caption = f"📈 Progression der LSE Verarbeitungsdaten (Pre-CAS)\nAktuell: {current_date}"
                send_telegram_photo(graph_buffer, graph_caption)
        
        # WICHTIG: Prüfe ob sich das Pre-CAS Datum wirklich geändert hat
        if current_date != status.get('pre_cas_date'):
            print("\n🔔 PRE-CAS ÄNDERUNG ERKANNT!")
            print(f"   Von: {status.get('pre_cas_date')}")
            print(f"   Auf: {current_date}")
            
            # Sende einfache Nachricht an Mama
            send_telegram_mama(status.get('pre_cas_date'), current_date)
            
            # Sende einfache Nachricht an Papa
            send_telegram_papa(status.get('pre_cas_date'), current_date)
            
            # Speichere in Pre-CAS Historie mit UTC Zeit (für Konsistenz)
            history.setdefault('pre_cas_changes', []).append({
                "timestamp": datetime.now().astimezone(ZoneInfo('UTC')).isoformat(),
                "date": current_date,
                "from": status.get('pre_cas_date')
            })
            
            # Speichere Historie sofort
            if not save_history(history):
                print("❌ Fehler beim Speichern der Historie!")
            
            # Berechne Prognose basierend auf Pre-CAS Historie
            active_history = {"changes": history.get('pre_cas_changes', [])}
            forecast = calculate_regression_forecast(active_history)
            
            # Use ALT/NEU enhanced forecast with fallback for compatibility
            try:
                forecast_text = create_enhanced_forecast_text(forecast) or ""
            except Exception:
                try:
                    forecast_text = create_forecast_text(forecast) or ""
                except Exception:
                    forecast_text = ""
            
            # Erstelle E-Mail-Inhalt
            subject = f"LSE Pre-CAS Status Update: Neues Datum {current_date}"
            
            # Bei manuellem Check: Hinweis in E-Mail
            manual_hint = "\n\n(Änderung durch manuellen Check via Telegram entdeckt)" if IS_MANUAL else ""
            
            # Basis-E-Mail für alle
            base_body = f"""Das Verarbeitungsdatum für Pre-CAS hat sich geändert!

ACTIVE STREAM: {ACTIVE_STREAM}
TARGET: {TARGET_DATE_PRE_CAS.strftime('%d %b %Y')} (LON)

ÄNDERUNG:
Von: {status.get('pre_cas_date')}
Auf: {current_date}

Zeitpunkt der Erkennung: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}

Link zur Seite: {LSE_URL}{manual_hint}"""
            
            # E-Mail mit Prognose für Hauptempfänger
            body_with_forecast = base_body + f"\n{forecast_text}\n\nDiese E-Mail wurde automatisch von deinem GitHub Actions Monitor generiert."
            
            # E-Mail ohne Prognose für bedingte Empfänger
            body_simple = base_body + "\n\nDiese E-Mail wurde automatisch von deinem GitHub Actions Monitor generiert."
            
            # Telegram-Nachricht formatieren
            if not IS_MANUAL:
                # Automatischer Check: Standard-Änderungsnachricht mit Graph
                telegram_msg = f"""<b>🔔 LSE Pre-CAS Update</b>

<b>PRE-CAS ÄNDERUNG ERKANNT!</b>
Target: {TARGET_DATE_PRE_CAS.strftime('%d %b %Y')} (LON)
Von: {status.get('pre_cas_date')}
Auf: <b>{current_date}</b>

Zeit: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
{forecast_text}

<a href="{LSE_URL}">📄 LSE Webseite öffnen</a>"""
                
                send_telegram(telegram_msg)
                
                # Sende Graph als separates Bild
                graph_buffer = create_progression_graph(active_history, current_date, forecast)
                if graph_buffer:
                    graph_caption = f"📈 Pre-CAS Progression Update\nNeues Datum: {current_date}"
                    send_telegram_photo(graph_buffer, graph_caption)
            else:
                # Manueller Check: Spezielle Nachricht bei Änderung mit Graph
                telegram_msg = f"""<b>🚨 PRE-CAS ÄNDERUNG GEFUNDEN!</b>

Dein manueller Check hat eine Pre-CAS Änderung entdeckt!

Target: {TARGET_DATE_PRE_CAS.strftime('%d %b %Y')} (LON)
Von: {status.get('pre_cas_date')}
Auf: <b>{current_date}</b>

Zeit: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
{forecast_text}

📧 E-Mails werden an die Hauptempfänger gesendet!

<a href="{LSE_URL}">📄 LSE Webseite öffnen</a>"""
                
                send_telegram(telegram_msg)
                
                # Sende Graph
                graph_buffer = create_progression_graph(active_history, current_date, forecast)
                if graph_buffer:
                    graph_caption = f"📈 Pre-CAS Änderung erkannt!\nVon {status.get('pre_cas_date')} auf {current_date}"
                    send_telegram_photo(graph_buffer, graph_caption)
            
            # Sende E-Mails
            emails_sent = False
            
            # Immer benachrichtigen (mit Prognose) - JETZT AUCH BEI MANUELLEN CHECKS
            if always_notify:
                if send_gmail(subject, body_with_forecast, always_notify):
                    emails_sent = True
            
            # Check if target date reached for Pre-CAS
            if conditional_notify and current_date == TARGET_DATE_PRE_CAS.strftime('%d %B'):
                print(f"\n🎯 Pre-CAS Zieldatum {current_date} erreicht! Benachrichtige zusätzliche Empfänger.")
                if send_gmail(subject, body_simple, conditional_notify):
                    emails_sent = True
                
                # Spezielle Telegram-Nachricht für Zieldatum mit Graph
                telegram_special = f"""<b>🎯 PRE-CAS ZIELDATUM ERREICHT!</b>

Das Pre-CAS Datum <b>{current_date}</b> wurde erreicht!

Target: {TARGET_DATE_PRE_CAS.strftime('%d %b %Y')} (LON)

Dies ist das wichtige Zieldatum für deine LSE Pre-CAS Bewerbung.

<a href="{LSE_URL}">📄 Jetzt zur LSE Webseite</a>"""
                send_telegram(telegram_special)
                
                # Sende speziellen Graph für Zieldatum
                graph_buffer = create_progression_graph(active_history, current_date, forecast)
                if graph_buffer:
                    graph_caption = f"🎯 PRE-CAS ZIELDATUM ERREICHT: {current_date}!"
                    send_telegram_photo(graph_buffer, graph_caption)
            
            if emails_sent or os.environ.get('TELEGRAM_BOT_TOKEN'):
                # KRITISCH: Update Pre-CAS Status IMMER nach einer erkannten Änderung
                # Update Status nur bei erfolgreicher Benachrichtigung
                status['pre_cas_date'] = current_date
                status['last_check'] = datetime.now().astimezone(ZoneInfo('UTC')).isoformat()
                
                # KRITISCH: Speichere Status mehrfach mit Verifikation
                print("\n🔄 Speichere aktualisierten Status...")
                save_attempts = 0
                save_success = False
                
                while save_attempts < 3 and not save_success:
                    save_attempts += 1
                    print(f"Speicherversuch {save_attempts}/3...")
                    
                    if save_status(status):
                        # Verifiziere durch erneutes Laden
                        verify_status = load_status()
                        if verify_status.get('pre_cas_date') == current_date:
                            print(f"✅ Pre-CAS Status erfolgreich gespeichert und verifiziert: {current_date}")
                            save_success = True
                        else:
                            print(f"❌ Verifikation fehlgeschlagen! Erwartet: {current_date}, Geladen: {verify_status.get('pre_cas_date')}")
                    else:
                        print(f"❌ Speichern fehlgeschlagen in Versuch {save_attempts}")
                    
                    if not save_success and save_attempts < 3:
                        time.sleep(1)
                
                if not save_success:
                    print("❌ KRITISCHER FEHLER: Status konnte nach 3 Versuchen nicht gespeichert werden!")
                    sys.exit(1)
                
                # Speichere auch die Historie nochmal zur Sicherheit
                if not save_history(history):
                    print("❌ Fehler beim erneuten Speichern der Historie!")
            else:
                print("⚠️  Status wurde NICHT aktualisiert (keine Benachrichtigung erfolgreich)")
        else:
            print("✅ Keine Pre-CAS Änderung - alles beim Alten.")
            status['last_check'] = datetime.now().astimezone(ZoneInfo('UTC')).isoformat()  # UTC für Konsistenz
            
            # Send structured manual no-change status message
            if IS_MANUAL:
                # Calculate forecast for status message (same as change branch)
                active_history = {"changes": history.get('pre_cas_changes', [])}
                forecast = calculate_regression_forecast(active_history)
                
                # Use ALT/NEU enhanced forecast with fallback for compatibility
                try:
                    forecast_text = create_enhanced_forecast_text(forecast) or ""
                except Exception:
                    try:
                        forecast_text = create_forecast_text(forecast) or ""
                    except Exception:
                        forecast_text = ""
                
                # Build structured status message
                telegram_msg = f"""<b>📊 LSE Status Check Ergebnis</b>

<b>Aktuelles Datum:</b> {status.get('pre_cas_date')}
<b>Letzter Stand:</b> {status.get('pre_cas_date')}
<b>Status:</b> ✅ Keine Änderung

<b>Zeitpunkt:</b> {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
{forecast_text}

<a href="{LSE_URL}">📄 LSE Webseite öffnen</a>"""
                
                send_telegram(telegram_msg)
            
            # Speichere auch bei keiner Änderung den aktualisierten Timestamp
            save_status(status)
    else:
        print(f"\n⚠️  WARNUNG: Konnte das Pre-CAS Datum nicht von der Webseite extrahieren!")
        
        # Bei manueller Ausführung auch Fehler melden
        if IS_MANUAL:
            telegram_error = f"""<b>❌ Manueller Pre-CAS Check fehlgeschlagen</b>

Konnte das Pre-CAS Datum nicht von der Webseite extrahieren!

<b>Active stream:</b> {ACTIVE_STREAM}
<b>Zeitpunkt:</b> {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
<b>Letztes bekanntes Pre-CAS Datum:</b> {status.get('pre_cas_date')}

Bitte prüfe die Webseite manuell.

<a href="{LSE_URL}">📄 LSE Webseite öffnen</a>"""
            
            send_telegram(telegram_error)
        
        # Sende Warnung per E-Mail
        subject = "LSE Pre-CAS Monitor WARNUNG: Datum nicht gefunden"
        body = f"""WARNUNG: Der LSE Pre-CAS Monitor konnte das Datum nicht von der Webseite extrahieren!

Active stream: {ACTIVE_STREAM}
Target: {TARGET_DATE_PRE_CAS.strftime('%d %b %Y')} (LON)

Zeitpunkt: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
Letztes bekanntes Pre-CAS Datum: {status.get('pre_cas_date')}

Bitte überprüfe:
1. Ist die Webseite erreichbar? {LSE_URL}
2. Hat sich die Struktur der Seite geändert?

Der Monitor wird weiterhin prüfen."""
        
        if always_notify:
            send_gmail(subject, body, always_notify)
        
        # Telegram-Warnung (nur bei automatischer Ausführung)
        if not IS_MANUAL:
            telegram_warning = f"""<b>⚠️ LSE Pre-CAS Monitor WARNUNG</b>

Konnte das Pre-CAS Datum nicht von der Webseite extrahieren!

Active stream: <b>{ACTIVE_STREAM}</b>
Letztes bekanntes Pre-CAS Datum: <b>{status.get('pre_cas_date')}</b>
Zeit: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}

Mögliche Gründe:
• Webseite nicht erreichbar
• Struktur hat sich geändert
• Netzwerkfehler

<a href="{LSE_URL}">📄 Webseite manuell prüfen</a>"""
            
            send_telegram(telegram_warning)
        
        # Speichere trotzdem den Status (mit last_check Update)
        status['last_check'] = datetime.now().astimezone(ZoneInfo('UTC')).isoformat()
        save_status(status)
    
    print("\n" + "="*50)
    
    # Debug: Zeige finale Dateien (optimized - avoid expensive shell commands)
    print("\n📁 FINALE DATEIEN:")
    print("=== status.json ===")
    try:
        with open("status.json", 'r') as f:
            print(f.read())
    except Exception as e:
        print(f"Fehler beim Lesen von status.json: {e}")
        
    print("\n=== history.json (letzte 3 Einträge) ===")
    try:
        with open("history.json", 'r') as f:
            lines = f.readlines()
            # Show last 20 lines efficiently
            print(''.join(lines[-20:]))
    except Exception as e:
        print(f"Fehler beim Lesen von history.json: {e}")
    
    # Finaler Status-Output für Debugging (use cached version)
    print("\n📊 FINALER STATUS:")
    try:
        final_status = _cached_json_load(STATUS_FILE)
        if final_status:
            print(f"   last_date: {final_status.get('last_date')}")
            print(f"   last_check: {final_status.get('last_check')}")
            print(f"   pre_cas_date: {final_status.get('pre_cas_date') or 'Nicht getrackt'}")
            print(f"   cas_date: {final_status.get('cas_date') or 'Nicht getrackt'}")
        else:
            print("   Status file not found or invalid")
    except Exception as e:
        print(f"   Fehler beim Lesen des finalen Status: {e}")

if __name__ == "__main__":
    main()
