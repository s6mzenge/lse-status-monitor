import requests
from bs4 import BeautifulSoup
import json
import os
import re
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta, date  # date hinzugefügt
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
    GMAIL_SMTP_PORT, TELEGRAM_API_BASE, ACTIVE_STREAM
)

# Advanced optimization imports
try:
    from advanced_optimizer import AdvancedBayesianOptimizer
    from lse_integrated_model import ProfessionalAutoCalibratedRegressor
    PROFESSIONAL_CALIBRATION_AVAILABLE = True
    print("✅ Professional Auto-Calibration verfügbar")
except ImportError as e:
    PROFESSIONAL_CALIBRATION_AVAILABLE = False
    print(f"⚠️ Professional Auto-Calibration nicht verfügbar: {e}")

# Lazy imports for heavy dependencies - only loaded when needed
_numpy = None
_matplotlib_plt = None
_matplotlib_dates = None
_warnings = None

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

# === Business-day helpers (x-axis skips weekends) ===

def business_days_elapsed(start_dt, end_dt):
    '''
    Zählt Arbeitstage (Mo–Fr) zwischen start_dt (inkl.) und end_dt (exkl.).
    Nutzt nur das Datum (keine Uhrzeiten). Für denselben Kalendertag -> 0.
    '''
    np = _get_numpy()
    s = np.datetime64(start_dt.date(), 'D')
    e = np.datetime64(end_dt.date(), 'D')
    # np.busday_count zählt Werktage im Intervall [s, e)
    return int(np.busday_count(s, e))

def add_business_days(start_dt, n):
    '''
    Addiert n Arbeitstage (Mo–Fr) auf start_dt und gibt das resultierende Datum zurück.
    n kann negativ sein. 
    
    Rundungslogik:
    - Runde nur auf, wenn es sehr spät am Tag wäre (nach ~15:30 London Zeit)
    - Das entspricht einem Bruchteil >= 0.9 (da 0.9 * 6h = 5.4h → 15:24)
    - Ansonsten verwende den ganzzahligen Teil (abrunden)
    '''
    from math import floor, ceil
    
    if n >= 0:
        fraction = n - int(n)
        # Nur aufrunden wenn sehr spät am Tag (0.9 = ca. 15:24 Uhr)
        # Sie können diesen Wert anpassen: 0.9 für ~15:30, 0.95 für ~15:45
        if fraction >= 0.9:
            steps = int(ceil(n))
        else:
            steps = int(floor(n))
    else:
        # Für negative Werte
        steps = -int(floor(abs(n)))
    
    current = start_dt
    step = 1 if steps >= 0 else -1
    remaining = abs(steps)
    while remaining > 0:
        current = current + timedelta(days=step)
        if current.weekday() < 5:  # Mo–Fr
            remaining -= 1
    return current


# Lazy imports for optional advanced features
_scipy_stats = None
_sklearn_HuberRegressor = None
_sklearn_RANSACRegressor = None
_sklearn_PolynomialFeatures = None
_sklearn_make_pipeline = None

def _get_advanced_regression():
    global _scipy_stats, _sklearn_HuberRegressor, _sklearn_RANSACRegressor, _sklearn_PolynomialFeatures, _sklearn_make_pipeline
    try:
        if _scipy_stats is None:
            from scipy import stats
            from sklearn.linear_model import HuberRegressor, RANSACRegressor
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import make_pipeline
            _scipy_stats = stats
            _sklearn_HuberRegressor = HuberRegressor
            _sklearn_RANSACRegressor = RANSACRegressor
            _sklearn_PolynomialFeatures = PolynomialFeatures
            _sklearn_make_pipeline = make_pipeline
        return True, (_scipy_stats, _sklearn_HuberRegressor, _sklearn_RANSACRegressor, _sklearn_PolynomialFeatures, _sklearn_make_pipeline)
    except ImportError:
        return False, (None, None, None, None, None)

# Check if advanced regression is available (but don't import yet)
ADVANCED_REGRESSION = True
try:
    import scipy
    import sklearn
except ImportError:
    ADVANCED_REGRESSION = False

URL = LSE_URL

from zoneinfo import ZoneInfo
def get_german_time():
    return datetime.now(ZoneInfo("Europe/Berlin"))

# ===== Compact forecast rendering (ALT vs. NEU) =====
import math
from zoneinfo import ZoneInfo

def _now_berlin():
    from zoneinfo import ZoneInfo
    dt = get_german_time()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ZoneInfo("Europe/Berlin"))
    return dt.astimezone(ZoneInfo("Europe/Berlin"))

def _short_date(d):  # "14 Aug"
    if d is None:
        return "—"
    from zoneinfo import ZoneInfo
    if d.tzinfo is None:
        d = d.replace(tzinfo=ZoneInfo("Europe/Berlin"))
    return d.astimezone(ZoneInfo("Europe/Berlin")).strftime("%d %b").replace(".", "")

def _cal_days_until(dt, now=None):
    """Kalendertage bis zum Datum (nur ganze Tage zählen)."""
    if not dt:
        return None
    if now is None:
        now = _now_berlin()
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=ZoneInfo("Europe/Berlin"))
    
    # Vergleiche nur die Daten, nicht die Uhrzeiten
    date_target = dt.date()
    date_now = now.date()
    
    return max(0, (date_target - date_now).days)

def _diff_days(neu_dt, alt_dt):
    """Differenz NEU vs. ALT in Kalendertagen als kurzer Text."""
    if not neu_dt or not alt_dt:
        return ""
    d = (neu_dt.date() - alt_dt.date()).days
    if d == 0:
        return " (=)"
    sign = "+" if d > 0 else "−"
    return f" ({sign}{abs(d)} Tag{'e' if abs(d)!=1 else ''} ggü. 🔵)"

def render_compact_bullets_precas(eta_old_dt, eta_new_dt,
                           r2_old, pts_old, slope_old,
                           r2_new, pts_new, slope_new,
                           hb_count=None):
    """Kompakte, mobilfreundliche Prognose für Pre-CAS mit nur einem Zieldatum."""
    now = _now_berlin()

    def line_for(dt):
        if not dt:
            return "—"
        return f"{_short_date(dt)} (in {_cal_days_until(dt, now)} Tagen)"

    hb_badge = f" HB×{hb_count}" if hb_count not in (None, 0) else ""

    parts = []
    parts.append("🎨 <b>Legende:</b> 🔵 ALT (linear) · 🟠 NEU (integriert)")
    parts.append("\n📌 <b>Pre-CAS Kurzprognose</b>")

    parts.append("🎯 <b>13 August</b>")
    parts.append(f"🔵 {line_for(eta_old_dt)}")
    parts.append(f"🟠 {line_for(eta_new_dt)}{_diff_days(eta_new_dt, eta_old_dt)}")

    parts.append("\n📐 <b>Modelle</b>")
    parts.append(f"R²: 🔵 {r2_old:.2f} ({pts_old}) · 🟠 {r2_new:.2f} ({pts_new}){hb_badge}")
    parts.append(f"Fortschritt: 🔵 {slope_old:.1f} d/Tag · 🟠 {slope_new:.1f} d/Tag")

    return "\n".join(parts)


def _iter_observations_or_changes(history: Dict, stream: str = "all_other") -> List[Dict]:
    """
    Kombiniert observations und changes für den angegebenen Stream.
    stream: "all_other", "pre_cas", oder "cas"
    """
    out: List[Dict] = []
    
    if stream == "pre_cas":
        src = (history.get("pre_cas_observations", []) or []) + (history.get("pre_cas_changes", []) or [])
    elif stream == "cas":
        src = (history.get("cas_observations", []) or []) + (history.get("cas_changes", []) or [])
    else:  # all_other (default)
        src = (history.get("observations", []) or []) + (history.get("changes", []) or [])
    
    for e in src:
        if not isinstance(e, dict):
            continue
        ts = e.get("timestamp")
        dt = e.get("date")
        if not ts or not dt:
            continue
        if not re.match(r'^\d{1,2}\s+\w+$', str(dt)):
            continue
        try:
            s = str(ts).replace('Z', '+00:00')
            dt_obj = datetime.fromisoformat(s)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=ZoneInfo("UTC"))
            ts_iso = dt_obj.astimezone(ZoneInfo("UTC")).isoformat()
        except Exception:
            continue
        kind = e.get("kind") or "change"
        out.append({"timestamp": ts_iso, "date": dt, "kind": kind})
    out.sort(key=lambda r: r["timestamp"])
    return out

def _iter_changes_only(history: Dict, stream: str = "all_other") -> List[Dict]:
    """
    Gibt nur echte Änderungen zurück (keine Heartbeats) für den angegebenen Stream.
    """
    out: List[Dict] = []
    
    if stream == "pre_cas":
        src = history.get("pre_cas_changes", []) or []
    elif stream == "cas":
        src = history.get("cas_changes", []) or []
    else:  # all_other
        src = history.get("changes", []) or []
    
    for e in src:
        if not isinstance(e, dict):
            continue
        ts = e.get("timestamp")
        dt = e.get("date")
        if not ts or not dt:
            continue
        if not re.match(r'^\d{1,2}\s+\w+$', str(dt)):
            continue
        try:
            s = str(ts).replace('Z', '+00:00')
            dt_obj = datetime.fromisoformat(s)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=ZoneInfo("UTC"))
            ts_iso = dt_obj.astimezone(ZoneInfo("UTC")).isoformat()
        except Exception:
            continue
        out.append({"timestamp": ts_iso, "date": dt, "kind": "change"})
    out.sort(key=lambda r: r["timestamp"])
    return out

def calculate_advanced_regression_forecast(history, current_date=None, stream="all_other"):
    """
    Erweiterte Regression für den angegebenen Stream.
    stream: "all_other", "pre_cas", oder "cas"
    """
    np = _get_numpy()  # Lazy load numpy
    
    # Verwende nur echte Änderungen für den angegebenen Stream
    source_data = _iter_changes_only(history, stream)
    if len(source_data) < REGRESSION_MIN_POINTS:
        return None
    
    # Extrahiere Datenpunkte
    data_points = []
    first_timestamp = None
    for entry in source_data:
        try:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            date_days = date_to_days(entry["date"])
            
            if date_days is None:
                continue
                
            if first_timestamp is None:
                first_timestamp = timestamp
                
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
            # In calculate_advanced_regression_forecast()
            def optimize_huber_epsilon(x, y, cv_folds=3):
                """Optimiere Epsilon durch Cross-Validation"""
                epsilons = np.linspace(1.0, 2.5, 10)
                best_epsilon = 1.5
                best_score = float('inf')
                
                for epsilon in epsilons:
                    scores = []
                    for train_idx, val_idx in time_series_split(len(x), cv_folds):
                        huber = HuberRegressor(epsilon=epsilon)
                        huber.fit(x[train_idx], y[train_idx])
                        pred = huber.predict(x[val_idx])
                        scores.append(np.mean(np.abs(y[val_idx] - pred)))
                    
                    if np.mean(scores) < best_score:
                        best_score = np.mean(scores)
                        best_epsilon = epsilon
                
                return best_epsilon
            
            # Verwendung:
            optimal_epsilon = optimize_huber_epsilon(x, y)
            huber = HuberRegressor(epsilon=optimal_epsilon)
            huber.fit(x, y)
            print(f"   Optimales Huber Epsilon: {optimal_epsilon:.2f}")
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
            # Adaptiver Exponent basierend auf Datenvariabilität
            def calculate_weight_exponent(x_flat, y):
                """Berechne optimalen Exponenten für Gewichtung"""
                # Analysiere wie stark sich die Steigung über Zeit ändert
                window = max(3, len(x_flat) // 4)
                local_slopes = []
                
                for i in range(len(x_flat) - window):
                    local_slope = np.polyfit(x_flat[i:i+window], y[i:i+window], 1)[0]
                    local_slopes.append(local_slope)
                
                if local_slopes:
                    slope_variability = np.std(local_slopes) / (np.mean(np.abs(local_slopes)) + 1e-6)
                    # Höhere Variabilität → stärkere Gewichtung neuerer Daten
                    weight_exp = 1.0 + 3.0 * np.tanh(slope_variability)
                else:
                    weight_exp = 2.0  # Fallback
                
                return weight_exp
            
            # Verwendung:
            weight_exp = calculate_weight_exponent(x_flat, y)
            weights = np.exp(x_flat / np.max(x_flat) * weight_exp)
            weighted_slope, weighted_intercept = np.polyfit(x_flat, y, 1, w=weights)
            print(f"   Adaptiver Gewichtungsexponent: {weight_exp:.2f}")
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
            def adaptive_window_size(velocities, data_points):
                """Berechne optimale Fenstergröße basierend auf Datenmenge und Volatilität"""
                n = len(velocities)
                if n < 2:
                    return n
                
                # Basis: Wurzel aus Anzahl (klassische Heuristik)
                base_window = int(np.sqrt(n))
                
                # Anpassung basierend auf Volatilität
                velocity_std = np.std(velocities)
                velocity_mean = np.mean(np.abs(velocities))
                cv = velocity_std / (velocity_mean + 1e-6)  # Variationskoeffizient
                
                # Höhere Volatilität → größeres Fenster für Stabilität
                window = int(base_window * (1 + 0.5 * np.tanh(cv - 0.5)))
                
                # Begrenzen auf sinnvollen Bereich
                return max(2, min(window, min(10, n)))
            
            # Verwendung:
            window_size = adaptive_window_size(velocities, len(data_points))
            recent_velocities = velocities[-window_size:] if len(velocities) >= window_size else velocities
            recent_avg_velocity = np.mean(recent_velocities)
            print(f"   Adaptive Fenstergröße: {window_size}")
            
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
    
    # Zieldaten - für Pre-CAS nur ein Ziel
    if stream == "pre_cas":
        target_dates_list = TARGET_DATES  # ["13 August"]
    else:
        target_dates_list = TARGET_DATES  # Für andere Streams
    
    predictions = {}
    days_until_target = None
    
    for target_name in target_dates_list:
        target_days = date_to_days(target_name)
        if target_days and best_slope > 0:
            days_until = (target_days - current_predicted_days) / best_slope
            
            def bootstrap_confidence_intervals(data_points, target_days, n_bootstrap=500):
                """Bootstrap-basierte Konfidenzintervalle"""
                predictions = []
                
                for _ in range(n_bootstrap):
                    # Resample mit Replacement
                    n = len(data_points)
                    indices = np.random.choice(n, n, replace=True)
                    resampled = [data_points[i] for i in indices]
                    
                    # Fit model auf resampled data
                    x_resamp = np.array([p[0] for p in resampled]).reshape(-1, 1)
                    y_resamp = np.array([p[1] for p in resampled])
                    
                    if len(np.unique(x_resamp)) > 1:  # Vermeid degenerierte Samples
                        slope, intercept = np.polyfit(x_resamp.flatten(), y_resamp, 1)
                        
                        # Vorhersage
                        current_x = data_points[-1][0]
                        if slope > 0:
                            days_until = (target_days - (slope * current_x + intercept)) / slope
                            predictions.append(days_until)
                
                if predictions:
                    # Quantile für 95% Konfidenzintervall
                    days_until_lower = np.percentile(predictions, 2.5)
                    days_until_upper = np.percentile(predictions, 97.5)
                    days_until_median = np.median(predictions)
                    
                    return {
                        'median': days_until_median,
                        'lower': days_until_lower,
                        'upper': days_until_upper,
                        'std': np.std(predictions)
                    }
                else:
                    # Fallback auf alte Methode
                    return {
                        'median': days_until,
                        'lower': days_until_lower,
                        'upper': days_until_upper,
                        'std': std_error / abs(best_slope) if best_slope != 0 else 0
                    }
            
            # Verwendung:
            ci = bootstrap_confidence_intervals(data_points, target_days)
            days_until_lower = max(0, ci['lower'])
            days_until_upper = ci['upper']
            days_until = ci['median']
            
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
            days_until_target = days_until
    
    # 8. TREND-ANALYSE
    trend_analysis = "unbekannt"
    if len(velocities) >= 2:
        def calculate_trend_threshold(velocities, sensitivity=1.5):
            """Dynamische Trendschwelle basierend auf Geschwindigkeitsvariabilität"""
            if len(velocities) < 3:
                return 0.1  # Fallback
            
            # Berechne Änderungen zwischen aufeinanderfolgenden Geschwindigkeiten
            velocity_changes = np.diff(velocities)
            
            # Robuste Streuungsschätzung (Median Absolute Deviation)
            mad = np.median(np.abs(velocity_changes - np.median(velocity_changes)))
            robust_std = 1.4826 * mad  # Konversion zu Standardabweichung
            
            # Schwelle als Vielfaches der robusten Streuung
            threshold = sensitivity * robust_std
            
            # Sinnvolle Grenzen
            return np.clip(threshold, 0.05, 0.3)
        
        # Verwendung:
        trend_threshold = calculate_trend_threshold(velocities)
        recent_acceleration = recent_avg_velocity - weighted_avg_velocity
        
        if recent_acceleration > trend_threshold:
            trend_analysis = "beschleunigend"
        elif recent_acceleration < -trend_threshold:
            trend_analysis = "verlangsamend"
        else:
            trend_analysis = "konstant"
        print(f"   Adaptive Trendschwelle: {trend_threshold:.3f}")
    
    return {
        # Basis-Informationen (für Kompatibilität)
        "slope": best_slope,
        "r_squared": best_r2,
        "current_trend_days": current_predicted_days,
        "data_points": len(data_points),
        "days_until_target": days_until_target,
        
        # Erweiterte Informationen
        "model_name": best_name,
        "std_error": std_error,
        "trend_analysis": trend_analysis,
        "recent_velocity": recent_avg_velocity,
        "models": models_comparison,
        "predictions": predictions,
    }

def calculate_regression_forecast(history, stream="all_other"):
    """Wrapper-Funktion für Rückwärtskompatibilität - ruft die erweiterte Version auf"""
    return calculate_advanced_regression_forecast(history, stream=stream)

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
    
    # Für Pre-CAS nur ein Zieldatum
    if 'predictions' in forecast and '13 August' in forecast['predictions']:
        p_target = forecast['predictions']['13 August']
        eta_target = _fmt_eta(p_target.get('days'), p_target.get('date'))
    else:
        d_target = forecast.get('days_until_target')
        eta_target = _fmt_eta(d_target, None)
    
    return {
        "name": "ALT (linear)",
        "r2": r2, "points": pts,
        "speed": f"{slope:.1f} Tage/Tag" if slope > 0 else "—",
        "trend": (trend.upper() if trend and trend != "unbekannt" else "—"),
        "eta_target": eta_target
    }


# ===== ETA-Backtest & Recency-Blend Helpers (unchanged) =====
import math
from datetime import datetime
from zoneinfo import ZoneInfo

def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("inf")
    mid = n // 2
    if n % 2:
        return xs[mid]
    return 0.5 * (xs[mid-1] + xs[mid])

from datetime import datetime
from zoneinfo import ZoneInfo

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
    # Sanfterer Übergang
    z = (hours_since_change - tau_hours) / (0.5 * tau_hours)
    return 1.0 / (1.0 + math.exp(-z))

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


def compute_integrated_model_metrics(history, stream="all_other", return_model=False,
                                    use_professional_calibration=None):
    """
    Integrierte Regression für den angegebenen Stream mit optionaler Professional Calibration.
    
    Args:
        history: Historie-Dictionary
        stream: "all_other", "pre_cas", oder "cas"
        return_model: Wenn True, gibt auch das trainierte Modell und Calendar zurück
        use_professional_calibration: Wenn True, nutzt professionelle Auto-Kalibrierung
    """
    from lse_integrated_model import BusinessCalendar, IntegratedRegressor, LON, BER
    from datetime import time as _time
    _np = _get_numpy()  # Lazy load numpy
    
    # Auto-detect if professional calibration should be used
    if use_professional_calibration is None:
        use_professional_calibration = PROFESSIONAL_CALIBRATION_AVAILABLE

    # ---- Daten aufbereiten: Changes & Heartbeats für den gewählten Stream ----
    rows_all = []
    rows_changes = []

    # Verwende stream-spezifische Daten
    if stream == "pre_cas":
        changes_key = "pre_cas_changes"
        obs_key = "pre_cas_observations"
    elif stream == "cas":
        changes_key = "cas_changes"
        obs_key = "cas_observations"
    else:  # all_other
        changes_key = "changes"
        obs_key = "observations"

    # 1) Build rows_changes from history changes and sort them
    for ch in history.get(changes_key, []):
        rows_changes.append({"timestamp": ch["timestamp"], "date": ch["date"]})
        rows_all.append({"timestamp": ch["timestamp"], "date": ch["date"]})
    
    rows_changes.sort(key=lambda r: _to_aware_berlin(r["timestamp"]))

    # 2) If fewer than 3 changes exist, return None
    if len(rows_changes) < 3:
        if return_model:
            return None, None, None
        return None  # zu wenig für integriertes Modell

    # 3) Determine last_change_ts from the last entry in rows_changes
    last_change_ts = _to_aware_berlin(rows_changes[-1]["timestamp"])

    # 4) Filter observations to only heartbeats with timestamp > last_change_ts
    heartbeats = 0
    latest_heartbeat = None
    
    for ob in history.get(obs_key, []):
        if ob.get("kind") == "heartbeat":
            ob_ts = _to_aware_berlin(ob["timestamp"])
            if ob_ts > last_change_ts:
                # This heartbeat is after the last change
                if latest_heartbeat is None or ob_ts > _to_aware_berlin(latest_heartbeat["timestamp"]):
                    latest_heartbeat = {"timestamp": ob["timestamp"], "date": ob["date"]}
    
    # Alle Heartbeats nach letzter Änderung einbeziehen
    for ob in history.get(obs_key, []):
        if ob.get("kind") == "heartbeat":
            ob_ts = _to_aware_berlin(ob["timestamp"])
            if ob_ts > last_change_ts:
                rows_all.append({"timestamp": ob["timestamp"], "date": ob["date"]})
                heartbeats += 1

    # 5) Sort rows_all and continue with existing logic
    rows_all.sort(key=lambda r: _to_aware_berlin(r["timestamp"]))

    from config import UK_HOLIDAYS
    # ---- Kalender / Konstanten ----
    cal = BusinessCalendar(tz=LON, start=_time(10, 0), end=_time(16, 0), holidays=UK_HOLIDAYS)
    tz_out = BER

    # ---- Professional Calibration oder Standard ----
    if use_professional_calibration and PROFESSIONAL_CALIBRATION_AVAILABLE and len(rows_all) >= 5:
        print("\n    🚀 Verwende Professional Auto-Calibration...")
        
        # Wähle Strategie basierend auf Datenmenge
        if len(rows_all) < 10:
            strategy = 'skopt'
            budget = 20
            print(f"    📊 Kleine Datenmenge: Nutze Scikit-Optimize (Budget: {budget})")
        elif len(rows_all) < 20:
            strategy = 'hybrid'
            budget = 30
            print(f"    📊 Mittlere Datenmenge: Nutze Hybrid-Ansatz (Budget: {budget})")
        else:
            strategy = 'ensemble'
            budget = 40
            print(f"    📊 Große Datenmenge: Nutze Ensemble-Strategie (Budget: {budget})")
        
        # Erstelle professionell kalibriertes Modell
        try:
            imodel = ProfessionalAutoCalibratedRegressor(
                cal=cal,
                base_date=date(2025, 1, 1),
                optimization_strategy=strategy,
                optimization_budget=budget,
                use_parallel=False  # GitHub Actions hat meist nur 2 Cores
            ).fit(rows_all)
            
            # Extrahiere die verwendeten Parameter für Reporting
            eff_frac = imodel.loess_frac
            eff_tau = imodel.tau_hours
            
            print(f"    ✅ Optimale Parameter gefunden:")
            print(f"       - LOESS Fraction: {eff_frac:.3f}")
            print(f"       - Tau (hours): {eff_tau:.1f}")
            
            if hasattr(imodel, 'performance_metrics') and imodel.performance_metrics:
                metrics = imodel.performance_metrics
                if 'cv_score' in metrics:
                    print(f"    📈 Performance:")
                    print(f"       - Cross-Validation Error: {metrics['cv_score']:.2f} Stunden")
                if 'feature_importance' in metrics:
                    print(f"       - Parameter-Wichtigkeit:")
                    for param, importance in metrics['feature_importance'].items():
                        print(f"         • {param}: {importance:.1%}")
        
        except Exception as e:
            print(f"    ❌ Professional Calibration fehlgeschlagen: {e}")
            print("    ↩️ Fallback auf Standard-Methode...")
            use_professional_calibration = False
            # Set flag to continue with standard method
            imodel = None
    else:
        use_professional_calibration = False
        imodel = None
    
    # Standard-Methode wenn Professional Calibration nicht verwendet wird
    if imodel is None:
        # ---- (1) ETA-Backtest: kleines Grid ----
        def generate_adaptive_grid(n_data_points, data_variance=None):
            """Erstelle adaptives Grid basierend auf Datenmenge und Varianz"""
            # Mehr Punkte bei mehr Daten
            if n_data_points < 10:
                grid_density = 3
                frac_range = (0.5, 0.7)
                tau_range = (9.0, 18.0)
            elif n_data_points < 20:
                grid_density = 5
                frac_range = (0.45, 0.75)
                tau_range = (8.0, 20.0)
            else:
                grid_density = 7
                frac_range = (0.4, 0.8)
                tau_range = (6.0, 24.0)
            
            # Optional: Anpassung basierend auf Varianz
            if data_variance is not None and data_variance > np.median(data_variance):
                # Erweitere Suchraum bei hoher Varianz
                frac_range = (frac_range[0] - 0.05, frac_range[1] + 0.05)
                tau_range = (tau_range[0] - 2, tau_range[1] + 4)
            
            FRACS = np.linspace(frac_range[0], frac_range[1], grid_density)
            TAUS = np.linspace(tau_range[0], tau_range[1], grid_density)
            
            return FRACS, TAUS
        
        # Verwendung:
        FRACS, TAUS = generate_adaptive_grid(len(rows_all))
        print(f"   Adaptives Grid: {len(FRACS)}×{len(TAUS)} Punkte")
        
        # Oder mit Bayesscher Optimierung statt Grid:
        from skopt import gp_minimize
        
        def objective(params):
            frac, tau = params
            return _eta_backtest_score(rows_all, rows_changes, cal, frac, tau, tz_out)
        
        result = gp_minimize(
            func=objective,
            dimensions=[(0.4, 0.8), (6.0, 24.0)],  # Suchraum
            n_calls=20,  # Weniger Evaluierungen als Grid
            random_state=42
        )
        base_frac, base_tau = result.x

        # ---- (3) Recency-Blend: Parameter dynamisch anpassen ----
        last_change_ts = _to_aware_berlin(rows_changes[-1]["timestamp"])
        h_since = _hours_since(last_change_ts)
        w = _recency_weight(h_since, base_tau)  # 0..1

        def learn_blend_parameters(history_data, base_frac, base_tau):
            """Lerne optimale Blend-Parameter aus historischen Daten"""
            
            # Sammle Fehler für verschiedene Parameterkombinationen
            param_combinations = []
            
            for frac_base_mult in np.linspace(0.7, 1.0, 5):
                for frac_w_mult in np.linspace(0.2, 0.5, 5):
                    for tau_base_mult in np.linspace(0.6, 1.0, 5):
                        for tau_w_mult in np.linspace(0.3, 0.7, 5):
                            # Backtest mit diesen Parametern
                            errors = []
                            for split in rolling_splits(history_data):
                                w = calculate_recency_weight(split['hours_since'])
                                test_frac = base_frac * (frac_base_mult + frac_w_mult * w)
                                test_tau = base_tau * (tau_base_mult + tau_w_mult * w)
                                
                                error = evaluate_prediction(split['train'], split['test'], test_frac, test_tau)
                                errors.append(error)
                            
                            param_combinations.append({
                                'params': (frac_base_mult, frac_w_mult, tau_base_mult, tau_w_mult),
                                'error': np.mean(errors)
                            })
            
            # Beste Parameter
            best = min(param_combinations, key=lambda x: x['error'])
            return best['params']
        
        # Wenn keine Historie für Lernen vorhanden, nutze adaptive Defaults:
        def get_adaptive_blend_params(n_points, hours_since_change):
            """Adaptive Default-Parameter basierend auf Datensituation"""
            # Grenzen basierend auf Datenmenge
            if n_points < 10:
                frac_bounds = (0.4, 0.75)
                tau_bounds = (8.0, 20.0)
            else:
                frac_bounds = (0.35, 0.85)
                tau_bounds = (6.0, 24.0)
            
            # Faktoren basierend auf Zeit seit letzter Änderung
            if hours_since_change < 24:  # Sehr frisch
                frac_mults = (0.9, 0.2)
                tau_mults = (0.85, 0.3)
            elif hours_since_change < 72:  # Mittel
                frac_mults = (0.85, 0.3)
                tau_mults = (0.8, 0.5)
            else:  # Älter
                frac_mults = (0.8, 0.4)
                tau_mults = (0.75, 0.6)
            
            return frac_bounds, tau_bounds, frac_mults, tau_mults
        
        # Verwendung:
        if len(history_data) > 20:  # Genug Daten zum Lernen
            frac_base_mult, frac_w_mult, tau_base_mult, tau_w_mult = learn_blend_parameters(
                history_data, base_frac, base_tau
            )
        else:  # Adaptive Defaults
            frac_bounds, tau_bounds, frac_mults, tau_mults = get_adaptive_blend_params(
                len(rows_all), h_since
            )
            frac_base_mult, frac_w_mult = frac_mults
            tau_base_mult, tau_w_mult = tau_mults
        
        eff_frac = max(frac_bounds[0], min(frac_bounds[1], 
                        base_frac * (frac_base_mult + frac_w_mult * w)))
        eff_tau = max(tau_bounds[0], min(tau_bounds[1], 
                       base_tau * (tau_base_mult + tau_w_mult * w)))

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

    # Vorhersagen - für Pre-CAS nur ein Zieldatum
    if stream == "pre_cas":
        pred_target = imodel.predict_datetime("13 August", tz_out=tz_out)
    else:
        # Für andere Streams könnten mehrere Ziele sein
        pred_target = imodel.predict_datetime(TARGET_DATES[0], tz_out=tz_out) if TARGET_DATES else None

    def _fmt_eta(diff_days, when_dt):
        if when_dt is None:
            return "—"
        return f"{when_dt.strftime('%d. %B %Y')} (in {int(round(diff_days))} Tagen)"

    # Für "in X Tagen" verwenden wir Kalendertage (inkl. WE)
    now_de = _now_berlin()
    d_target = (pred_target["when_point"] - now_de).total_seconds() / 86400.0 if pred_target and pred_target.get("when_point") else None

    result = {
        "name": "NEU (integriert)",
        "r2": r2_new,
        "points": len(rows_all),
        "speed": f"{avg_prog_new:.1f} Tage/Tag",
        "speed_val": float(avg_prog_new),
        "eta_target": _fmt_eta(d_target, pred_target["when_point"]) if d_target is not None else "—",
        "eta_target_dt": pred_target["when_point"] if pred_target else None,
        "heartbeats": heartbeats,
        # Debug/Transparenz (optional):
        "params": {
            "loess_frac": eff_frac, 
            "tau_hours": eff_tau,
            "calibration_method": "professional" if use_professional_calibration else "standard"
        },
    }
    
    if return_model:
        return result, imodel, cal  # Gebe auch das trainierte Modell und Calendar zurück
    return result


def create_enhanced_forecast_text_precas(forecast):
    """Kompakte, mobilfreundliche Prognose für Pre-CAS (ALT vs. NEU) mit nur einem Zieldatum."""
    if not forecast:
        return "\n📊 Pre-CAS Prognose: Noch nicht genügend Daten für eine zuverlässige Vorhersage."

    # Basis (ALT)
    r2_old   = float(forecast.get("r_squared", 0.0))
    pts_old  = int(forecast.get("data_points", 0))
    slope_old = float(forecast.get("slope", 0.0))

    # Alte ETA
    def _old_eta_dt(frc):
        try:
            return frc.get("predictions", {}).get("13 August", {}).get("date")
        except Exception:
            return None

    eta_old_dt = _old_eta_dt(forecast)

# Neue (integrierte) Metriken + echte ETA-Datetime
    hist = get_history()
    try:
        new_s = compute_integrated_model_metrics(
            hist, 
            stream="pre_cas",
            use_professional_calibration=True  # Aktiviere Professional Calibration
        )
    except Exception as e:
        print(f"⚠️ Integriertes Modell temporär deaktiviert: {e}")
        new_s = None

    if not new_s:
        # Fallback: nur ALT kompakt ausgeben
        parts = []
        parts.append("🎨 <b>Legende:</b> 🔵 ALT (linear) · 🟠 NEU (integriert)")
        parts.append("\n📌 <b>Pre-CAS Kurzprognose</b>")
        parts.append("🎯 <b>13 August</b>")
        parts.append(f"🔵 {_short_date(eta_old_dt) if eta_old_dt else '—'}"
                     f" (in {_cal_days_until(eta_old_dt)} Tagen)" if eta_old_dt else "🔵 —")
        parts.append("\n📐 <b>Modelle</b>")
        parts.append(f"R²: 🔵 {r2_old:.2f} ({pts_old})")
        parts.append(f"Fortschritt: 🔵 {slope_old:.1f} d/Tag")
        return "\n".join(parts)

    # NEU-Werte
    r2_new   = float(new_s["r2"])
    pts_new  = int(new_s["points"])
    slope_new = float(new_s.get("speed_val", 0.0))
    hb_count = int(new_s.get("heartbeats") or 0)
    eta_new_dt = new_s.get("eta_target_dt")

    # Kompakte Bullets für Pre-CAS
    text = render_compact_bullets_precas(
        eta_old_dt, eta_new_dt,
        r2_old, pts_old, slope_old,
        r2_new, pts_new, slope_new,
        hb_count=hb_count
    )

    return "\n" + text

def create_forecast_text(forecast, stream="all_other"):
    """Wrapper für Rückwärtskompatibilität"""
    if stream == "pre_cas":
        return create_enhanced_forecast_text_precas(forecast)
    else:
        # Für andere Streams könnte man die alte Funktion verwenden
        return create_enhanced_forecast_text_precas(forecast)

def create_progression_graph(history, current_date, forecast=None, stream="all_other"):
    """
    Graph für den angegebenen Stream erstellen.
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

    # Stream-spezifischer Titel
    if stream == "pre_cas":
        title = "Pre-CAS Fortschritt & Prognose — ALT vs. NEU"
        ylabel = "Pre-CAS Verarbeitungsdatum"
    elif stream == "cas":
        title = "CAS Fortschritt & Prognose — ALT vs. NEU"
        ylabel = "CAS Verarbeitungsdatum"
    else:
        title = "Fortschritt & Prognose — ALT vs. NEU"
        ylabel = "Verarbeitungsdatum"

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
        full = float(np.busday_count(np.datetime64(s0.date()), np.datetime64(t0.date())))
        def _frac(d: datetime) -> float:
            if not bool(np.is_busday(np.datetime64(d.date()))):
                return 0.0
            return (d - datetime(d.year, d.month, d.day)).total_seconds() / 86400.0
        return full + _frac(t) - _frac(s0)

    def _days_to_dt(year, doy):
        try:
            start = datetime(year, 1, 1)
            return (start + timedelta(days=float(doy))).replace(tzinfo=None)
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
    entries = list(_iter_observations_or_changes(history, stream))
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
    
    # Filtere Heartbeats - nur den neuesten nach der letzten echten Änderung
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

    # Zielhöhen (horizontale Linien) - für Pre-CAS nur ein Ziel
    if stream == "pre_cas":
        target_map = {"13 August": date_to_days("13 August")}
    else:
        # Für andere Streams
        target_map = {name: date_to_days(name) for name in TARGET_DATES}
    
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

        for name, ty in target_map.items():
            if ty is None: continue
            t_star = _solve_time_for_level(m_old, b_old, ty, first_ts, now_de, sign_positive=(m_old > 0))
            if t_star:
                y_star = m_old * _bizdays_float(first_ts, t_star) + b_old
                alt_eta[name] = (t_star, y_star)

    # ---------- NEU-Linie + ETAs (VERWENDET DAS GLEICHE MODELL WIE DIE PROGNOSE) ----------
    neu_eta = {}
    try:
        # WICHTIG: Hole das GLEICHE trainierte Modell wie für die Prognose
        new_metrics_result = compute_integrated_model_metrics(
            history, 
            stream=stream, 
            return_model=True,
            use_professional_calibration=True  # Aktiviere Professional Calibration
        )
        
        if new_metrics_result is not None:
            new_metrics, imodel, cal = new_metrics_result
            
            if new_metrics and imodel:
                from zoneinfo import ZoneInfo
                
                # Verwende das bereits trainierte Modell zum Plotten
                t0_aware = imodel.t0_  # Startzeitpunkt aus dem Training
                
                y_new = []
                for t in grid_ts:
                    # Konvertiere zu London Zeit (konsistent mit Training)
                    if t.tzinfo is None:
                        t_berlin = t.replace(tzinfo=ZoneInfo("Europe/Berlin"))
                    else:
                        t_berlin = t.astimezone(ZoneInfo("Europe/Berlin"))
                    t_london = t_berlin.astimezone(cal.tz)
                    
                    # Berechne business minutes GENAU wie beim Training
                    business_mins = cal.business_minutes_between(t0_aware, t_london)
                    x_hours = business_mins / 60.0
                    
                    # Verwende das trainierte Modell
                    try:
                        if x_hours >= 0:
                            # Innerhalb oder nach den Trainingsdaten
                            y_pred = imodel._blend_predict_scalar(x_hours)
                        else:
                            # Extrapolation vor dem ersten Datenpunkt
                            y_pred = float(imodel.ts_.predict(x_hours))
                    except Exception as e:
                        # Fallback auf Theil-Sen
                        y_pred = float(imodel.ts_.predict(x_hours))
                    
                    y_new.append(y_pred)
                
                # Plotte die Kurve
                if y_new:
                    y_new = np.array(y_new)
                    # Label anpassen je nach Kalibrierungsmethode
                    calibration_info = new_metrics.get('params', {}).get('calibration_method', 'standard')
                    if calibration_info == 'professional':
                        label_suffix = " [Auto-Kalibriert]"
                    else:
                        label_suffix = ""
                    
                    ax.plot(grid_ts, y_new, linewidth=2.6,
                            label=f"NEU: integrierte Regression (R²={new_metrics['r2']:.2f}){label_suffix}",
                            color=COL_NEU, alpha=0.95)
                    
                    # ETAs mit dem GLEICHEN Modell
                    for name, ty in target_map.items():
                        if ty is None:
                            continue
                        try:
                            # Verwende die predict_datetime Methode des trainierten Modells
                            from lse_integrated_model import BER
                            pred = imodel.predict_datetime(name, tz_out=BER)
                            if pred and pred.get("when_point"):
                                t_star = _to_naive_berlin(pred["when_point"])
                                
                                # Berechne y-Wert an dieser Stelle (konsistent)
                                t_star_berlin = t_star.replace(tzinfo=ZoneInfo("Europe/Berlin"))
                                t_star_london = t_star_berlin.astimezone(cal.tz)
                                business_mins_star = cal.business_minutes_between(t0_aware, t_star_london)
                                x_hours_star = business_mins_star / 60.0
                                
                                try:
                                    y_star = imodel._blend_predict_scalar(x_hours_star)
                                except:
                                    y_star = float(imodel.ts_.predict(x_hours_star))
                                
                                neu_eta[name] = (t_star, y_star)
                        except Exception as e:
                            print(f"⚠️ Konnte ETA für {name} nicht berechnen: {e}")
                else:
                    print("⚠️ Keine Punkte für NEU-Linie berechnet")
            else:
                print("⚠️ NEU-Modell konnte nicht trainiert werden")
        else:
            print("⚠️ compute_integrated_model_metrics gab None zurück")
    
    except Exception as e:
        print(f"⚠️ NEU-Regression konnte nicht gezeichnet werden: {e}")
        import traceback
        traceback.print_exc()

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
        
        # Datumsbeschriftungen - OBERHALB der X-Achse im ersten Quadranten
        y_range = ax.get_ylim()
        y_min = y_range[0]
        
        # Basis Y-Position etwa 8% oberhalb des unteren Rands
        y_text_pos = y_min + (y_range[1] - y_min) * 0.08
        
        if alt_pt and neu_pt:
            # Prüfe ob GLEICHES DATUM
            same_date = (alt_pt[0].date() == neu_pt[0].date())
            
            if same_date:
                # GLEICHES DATUM: Eine gemeinsame Beschriftung
                date_str = alt_pt[0].strftime('%d. %B')
                
                # Position: Direkt an der ersten Linie (minimaler Abstand)
                x_pos = alt_pt[0] - timedelta(days=0.02)  # Nur 0.15 Tage links
                
                # Eine Beschriftung für beide
                ax.text(x_pos, y_text_pos, date_str,
                        ha='right', va='bottom', fontsize=8, color='gray',
                        rotation=90, alpha=0.9, weight='semibold')
            
            else:
                # UNTERSCHIEDLICHE DATEN
                days_diff = abs((neu_pt[0] - alt_pt[0]).days)
                
                if days_diff < 2:  # Zu nah für normale Platzierung
                    # Versetze vertikal, aber nah an den Linien
                    ax.text(alt_pt[0] - timedelta(days=0.02), y_text_pos + 1.5,  # ALT höher
                            alt_pt[0].strftime('%d. %B'),
                            ha='right', va='bottom', fontsize=8, color=COL_ALT,
                            rotation=90, alpha=0.9)
                    
                    ax.text(neu_pt[0] - timedelta(days=0.02), y_text_pos,  # NEU tiefer
                            neu_pt[0].strftime('%d. %B'),
                            ha='right', va='bottom', fontsize=8, color=COL_NEU,
                            rotation=90, alpha=0.9)
                else:
                    # Weit genug auseinander: normale Platzierung, aber näher
                    ax.text(alt_pt[0] - timedelta(days=0.02), y_text_pos, 
                            alt_pt[0].strftime('%d. %B'),
                            ha='right', va='bottom', fontsize=8, color=COL_ALT,
                            rotation=90, alpha=0.9)
                    
                    ax.text(neu_pt[0] - timedelta(days=0.02), y_text_pos,
                            neu_pt[0].strftime('%d. %B'),
                            ha='right', va='bottom', fontsize=8, color=COL_NEU,
                            rotation=90, alpha=0.9)
        
        elif alt_pt:
            # Nur ALT vorhanden
            ax.text(alt_pt[0] - timedelta(days=0.02), y_text_pos,
                    alt_pt[0].strftime('%d. %B'),
                    ha='right', va='bottom', fontsize=8, color=COL_ALT,
                    rotation=90, alpha=0.9)
        
        elif neu_pt:
            # Nur NEU vorhanden
            ax.text(neu_pt[0] - timedelta(days=0.02), y_text_pos,
                    neu_pt[0].strftime('%d. %B'),
                    ha='right', va='bottom', fontsize=8, color=COL_NEU,
                    rotation=90, alpha=0.9)

    # Für Pre-CAS nur ein Ziel
    if stream == "pre_cas":
        _plot_eta("13 Aug", alt_eta.get("13 August"), neu_eta.get("13 August"))
    else:
        # Für andere Streams könnten mehrere Ziele sein
        for name in target_map:
            _plot_eta(name.split()[0], alt_eta.get(name), neu_eta.get(name))

    # ---------- Achsenbegrenzungen ----------
    eta_dates = [p[0] for p in list(alt_eta.values()) + list(neu_eta.values()) if p]
    right_edge = max(eta_dates) if eta_dates else change_ts[-1]
    left_edge  = min(change_ts[0], now_de - timedelta(days=3))
    ax.set_xlim(left_edge - timedelta(days=1), right_edge + timedelta(days=1))

    y_min = min(change_y)
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

    # ---------- Achsenformat ----------
    ax.set_title(title)
    ax.set_xlabel("Datum")
    ax.set_ylabel(ylabel)

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
                "text": message
            }
            if parse_mode:
                data["parse_mode"] = parse_mode
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

def send_telegram_mama(stream_name, old_date, new_date):
    """Sendet eine einfache Nachricht an Mama über separaten Bot"""
    message = f"LSE {stream_name}-Update!\n\nVom: {old_date}\nAuf: {new_date}"
    return send_telegram_message(message, 'mama')

def send_telegram_papa(stream_name, old_date, new_date):
    """Sendet eine einfache Nachricht an Papa über separaten Bot"""
    message = f"LSE {stream_name}-Update!\n\nVom: {old_date}\nAuf: {new_date}"
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
        if 'pre_cas_observations' not in history:
            history['pre_cas_observations'] = []
        if 'cas_observations' not in history:
            history['cas_observations'] = []
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"⚠️ Fehler bei History-Migration: {e}")

def load_status():
    """Lädt Status mit Fehlerbehandlung und Validierung"""
    try:
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
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
        print(f"✅ Status geladen: AOA={status['last_date']}, Pre-CAS={status['pre_cas_date']}, CAS={status['cas_date']}")
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
    # Validiere dass wichtige Felder gesetzt sind
    if not status.get('pre_cas_date'):
        print("⚠️ Warnung: pre_cas_date ist leer")
    
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
            status['last_check'] = datetime.utcnow().isoformat()
        
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
        
        # Verifiziere dass es korrekt gespeichert wurde
        with open(STATUS_FILE, 'r') as f:
            saved = json.load(f)
            print(f"✅ Status erfolgreich gespeichert: Pre-CAS={status['pre_cas_date']}")
            return True
                
    except Exception as e:
        print(f"❌ Fehler beim Speichern von status.json: {e}")
        # Restore backup
        if os.path.exists(STATUS_FILE + '.backup'):
            os.rename(STATUS_FILE + '.backup', STATUS_FILE)
        return False

def load_history():
    """Lädt Historie mit Fehlerbehandlung"""
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
            
        # Validiere die geladenen Daten
        if not isinstance(history, dict):
            print("⚠️ History ist ungültig, verwende leere Historie")
            return {"changes": [], "pre_cas_changes": [], "cas_changes": [], 
                   "observations": [], "pre_cas_observations": [], "cas_observations": []}
            
        # Stelle sicher dass alle Arrays existieren
        for key in ['changes', 'pre_cas_changes', 'cas_changes', 
                   'observations', 'pre_cas_observations', 'cas_observations']:
            if key not in history:
                history[key] = []
            
        print(f"✅ Historie geladen: AOA={len(history['changes'])} Änderungen, "
              f"Pre-CAS={len(history['pre_cas_changes'])} Änderungen, "
              f"CAS={len(history['cas_changes'])} Änderungen")
        return history
    except FileNotFoundError:
        print("ℹ️ history.json nicht gefunden, erstelle neue Datei")
        return {"changes": [], "pre_cas_changes": [], "cas_changes": [], 
               "observations": [], "pre_cas_observations": [], "cas_observations": []}
    except json.JSONDecodeError as e:
        print(f"❌ Fehler beim Parsen von history.json: {e}")
        return {"changes": [], "pre_cas_changes": [], "cas_changes": [], 
               "observations": [], "pre_cas_observations": [], "cas_observations": []}
    except Exception as e:
        print(f"❌ Unerwarteter Fehler beim Laden von history.json: {e}")
        return {"changes": [], "pre_cas_changes": [], "cas_changes": [], 
               "observations": [], "pre_cas_observations": [], "cas_observations": []}

def get_history():
    """Backward-compat wrapper to load history."""
    return load_history()


def save_history(history):
    """Speichert Historie mit Validierung"""
    try:
        # Validiere die Historie
        if not isinstance(history, dict):
            print("❌ Fehler: Historie ist ungültig")
            return False
            
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✅ Historie gespeichert")
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
        response = requests.get(URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
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
    print(f"AKTIVES TRACKING: {ACTIVE_STREAM.upper()}")
    
    # Zeige ob erweiterte Regression verfügbar ist
    if ADVANCED_REGRESSION:
        print("✅ Erweiterte Regression aktiviert")
    else:
        print("⚠️ Erweiterte Regression nicht verfügbar (Standard-Regression wird verwendet)")
    
    if PROFESSIONAL_CALIBRATION_AVAILABLE:
        print("✅ Professional Auto-Calibration aktiviert")
    else:
        print("ℹ️ Professional Auto-Calibration nicht verfügbar (Standard-Kalibrierung wird verwendet)")
        
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
    print(f"Nur bei Zieldatum: {', '.join(conditional_notify)}")
    
    # Lade Status und Historie mit Fehlerbehandlung
    status = load_status()
    history = load_history()
    
    # Hole alle aktuellen Daten
    print("\nRufe LSE-Webseite ab...")
    current_dates = fetch_processing_dates()
    
    # Heartbeat-Beobachtung für alle Streams
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
            
            # Heartbeat für Pre-CAS (aktiv)
            current_pre_cas = current_dates.get('pre_cas')
            if current_pre_cas == status.get('pre_cas_date'):
                history.setdefault('pre_cas_observations', [])
                if not any(o.get('timestamp') == last_up_iso for o in history['pre_cas_observations']):
                    history['pre_cas_observations'].append({
                        'timestamp': last_up_iso,
                        'date': current_pre_cas,
                        'kind': 'heartbeat'
                    })
                    save_history(history)
            
            # Heartbeat für AOA (passiv)
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
            
            # Heartbeat für CAS (passiv)
            current_cas = current_dates.get('cas')
            if current_cas == status.get('cas_date'):
                history.setdefault('cas_observations', [])
                if not any(o.get('timestamp') == last_up_iso for o in history['cas_observations']):
                    history['cas_observations'].append({
                        'timestamp': last_up_iso,
                        'date': current_cas,
                        'kind': 'heartbeat'
                    })
                    save_history(history)
    except Exception as _e:
        print(f"⚠️ Heartbeat-Logik übersprungen: {_e}")

    # PASSIVE TRACKING FÜR AOA (stilles Tracking, keine Benachrichtigungen)
    if current_dates['all_other'] and current_dates['all_other'] != status.get('last_date'):
        print(f"\n📝 AOA Änderung (passives Tracking): {status.get('last_date') or 'Unbekannt'} → {current_dates['all_other']}")
        history['changes'].append({
            "timestamp": datetime.utcnow().isoformat(),
            "date": current_dates['all_other'],
            "from": status.get('last_date')
        })
        status['last_date'] = current_dates['all_other']
        save_history(history)
        save_status(status)
    
    # PASSIVE TRACKING FÜR CAS (stilles Tracking, keine Benachrichtigungen)
    if current_dates['cas'] and current_dates['cas'] != status.get('cas_date'):
        print(f"\n📝 CAS Änderung (passives Tracking): {status.get('cas_date') or 'Unbekannt'} → {current_dates['cas']}")
        history['cas_changes'].append({
            "timestamp": datetime.utcnow().isoformat(),
            "date": current_dates['cas'],
            "from": status.get('cas_date')
        })
        status['cas_date'] = current_dates['cas']
        save_history(history)
        save_status(status)
    
    # HAUPTLOGIK FÜR PRE-CAS (AKTIVES TRACKING MIT BENACHRICHTIGUNGEN)
    current_date = current_dates['pre_cas']
    
    if current_date:
        print(f"Aktuelles Pre-CAS Datum: {current_date}")
        
        # Bei manuellem Check immer Status senden
        if IS_MANUAL:
            # Berechne aktuellen Trend und erstelle vollständige Prognose
            forecast = calculate_regression_forecast(history, stream="pre_cas")
            forecast_text = create_forecast_text(forecast, stream="pre_cas") or ""
            
            telegram_msg = f"""<b>📊 LSE Pre-CAS Status Check Ergebnis</b>

<b>Aktuelles Pre-CAS Datum:</b> {current_date}
<b>Letzter Stand:</b> {status.get('pre_cas_date', 'Unbekannt')}
<b>Status:</b> {"🔔 ÄNDERUNG ERKANNT!" if current_date != status.get('pre_cas_date') else "✅ Keine Änderung"}

<b>Zeitpunkt:</b> {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
{forecast_text}

<a href="{URL}">📄 LSE Webseite öffnen</a>"""
            
            # Sende Text-Nachricht
            send_telegram(telegram_msg)
            
            # Erstelle und sende Graph
            graph_buffer = create_progression_graph(history, current_date, forecast, stream="pre_cas")
            if graph_buffer:
                graph_caption = f"📈 Progression der Pre-CAS Verarbeitungsdaten\nAktuell: {current_date}"
                send_telegram_photo(graph_buffer, graph_caption)
        
        # WICHTIG: Prüfe ob sich das Pre-CAS Datum wirklich geändert hat
        if current_date != status.get('pre_cas_date'):
            print("\n🔔 PRE-CAS ÄNDERUNG ERKANNT!")
            print(f"   Von: {status.get('pre_cas_date', 'Unbekannt')}")
            print(f"   Auf: {current_date}")
            
            # Sende einfache Nachricht an Mama
            mama_sent = send_telegram_mama("Pre-CAS", status.get('pre_cas_date', 'Unbekannt'), current_date)
            
            # Sende einfache Nachricht an Papa
            papa_sent = send_telegram_papa("Pre-CAS", status.get('pre_cas_date', 'Unbekannt'), current_date)
            
            # NEU: Wenn Eltern-Benachrichtigungen fehlschlagen, informiere den Hauptbot
            if not mama_sent or not papa_sent:
                warning_msg = f"""<b>⚠️ WARNUNG: Eltern-Benachrichtigung fehlgeschlagen!</b>
            
            Pre-CAS Änderung: {status.get('pre_cas_date', 'Unbekannt')} → {current_date}
            
            Status der Benachrichtigungen:
            - Mama: {'✅ Erfolgreich' if mama_sent else '❌ FEHLGESCHLAGEN'}
            - Papa: {'✅ Erfolgreich' if papa_sent else '❌ FEHLGESCHLAGEN'}
            
            <b>Bitte informiere deine Eltern manuell!</b>
            
            Zeit: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}"""
                
                send_telegram(warning_msg)
                print("⚠️ Warnung an Hauptbot gesendet wegen fehlgeschlagener Eltern-Benachrichtigung")
            
            # Speichere in Historie mit UTC Zeit (für Konsistenz)
            history["pre_cas_changes"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "date": current_date,
                "from": status.get('pre_cas_date')
            })
            
            # Speichere Historie sofort
            if not save_history(history):
                print("❌ Fehler beim Speichern der Historie!")
            
            # Berechne Prognose
            forecast = calculate_regression_forecast(history, stream="pre_cas")
            forecast_text = create_forecast_text(forecast, stream="pre_cas") or ""
            
            # Erstelle E-Mail-Inhalt
            subject = f"LSE Pre-CAS Update: Neues Datum {current_date}"
            
            # Bei manuellem Check: Hinweis in E-Mail
            manual_hint = "\n\n(Änderung durch manuellen Check via Telegram entdeckt)" if IS_MANUAL else ""
            
            # Basis-E-Mail für alle
            base_body = f"""Das Pre-CAS Verarbeitungsdatum hat sich geändert!

ÄNDERUNG:
Von: {status.get('pre_cas_date', 'Unbekannt')}
Auf: {current_date}

Zeitpunkt der Erkennung: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}

Link zur Seite: {URL}{manual_hint}"""
            
            # E-Mail mit Prognose für Hauptempfänger
            body_with_forecast = base_body + f"\n{re.sub('<.*?>', '', forecast_text)}\n\nDiese E-Mail wurde automatisch von deinem GitHub Actions Monitor generiert."
            
            # E-Mail ohne Prognose für bedingte Empfänger
            body_simple = base_body + "\n\nDiese E-Mail wurde automatisch von deinem GitHub Actions Monitor generiert."
            
            # Telegram-Nachricht formatieren
            if not IS_MANUAL:
                # Automatischer Check: Standard-Änderungsnachricht mit Graph
                telegram_msg = f"""<b>🔔 LSE Pre-CAS Update</b>

<b>ÄNDERUNG ERKANNT!</b>
Von: {status.get('pre_cas_date', 'Unbekannt')}
Auf: <b>{current_date}</b>

Zeit: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
{forecast_text}

<a href="{URL}">📄 LSE Webseite öffnen</a>"""
                
                send_telegram(telegram_msg)
                
                # Sende Graph als separates Bild
                graph_buffer = create_progression_graph(history, current_date, forecast, stream="pre_cas")
                if graph_buffer:
                    graph_caption = f"📈 Pre-CAS Progression Update\nNeues Datum: {current_date}"
                    send_telegram_photo(graph_buffer, graph_caption)
            else:
                # Manueller Check: Spezielle Nachricht bei Änderung mit Graph
                telegram_msg = f"""<b>🚨 PRE-CAS ÄNDERUNG GEFUNDEN!</b>

Dein manueller Check hat eine Pre-CAS Änderung entdeckt!

Von: {status.get('pre_cas_date', 'Unbekannt')}
Auf: <b>{current_date}</b>

Zeit: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
{forecast_text}

📧 E-Mails werden an die Hauptempfänger gesendet!

<a href="{URL}">📄 LSE Webseite öffnen</a>"""
                
                send_telegram(telegram_msg)
                
                # Sende Graph
                graph_buffer = create_progression_graph(history, current_date, forecast, stream="pre_cas")
                if graph_buffer:
                    graph_caption = f"📈 Pre-CAS Änderung erkannt!\nVon {status.get('pre_cas_date', 'Unbekannt')} auf {current_date}"
                    send_telegram_photo(graph_buffer, graph_caption)
            
            # Sende E-Mails
            emails_sent = False
            
            # Immer benachrichtigen (mit Prognose) - JETZT AUCH BEI MANUELLEN CHECKS
            if always_notify:
                if send_gmail(subject, body_with_forecast, always_notify):
                    emails_sent = True
            
            # Bedingt benachrichtigen (nur bei Zieldatum 13 August)
            if conditional_notify and current_date == "13 August":
                print(f"\n🎯 Pre-CAS Zieldatum 13 August erreicht! Benachrichtige zusätzliche Empfänger.")
                if send_gmail(subject, body_simple, conditional_notify):
                    emails_sent = True
                
                # Spezielle Telegram-Nachricht für Zieldatum mit Graph
                telegram_special = f"""<b>🎯 PRE-CAS ZIELDATUM ERREICHT!</b>

Das Pre-CAS Datum <b>13 August</b> wurde erreicht!

Dies ist das wichtige Zieldatum für deine Pre-CAS Verarbeitung.

<a href="{URL}">📄 Jetzt zur LSE Webseite</a>"""
                send_telegram(telegram_special)
                
                # Sende speziellen Graph für Zieldatum
                graph_buffer = create_progression_graph(history, current_date, forecast, stream="pre_cas")
                if graph_buffer:
                    graph_caption = f"🎯 PRE-CAS ZIELDATUM ERREICHT: 13 August!"
                    send_telegram_photo(graph_buffer, graph_caption)
            
            if emails_sent or os.environ.get('TELEGRAM_BOT_TOKEN'):
                # Update Status nach erfolgreicher Benachrichtigung
                status['pre_cas_date'] = current_date
                status['last_check'] = datetime.utcnow().isoformat()
                
                # Speichere Status mehrfach mit Verifikation
                print("\n🔄 Speichere aktualisierten Pre-CAS Status...")
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
                    print("❌ KRITISCHER FEHLER: Pre-CAS Status konnte nach 3 Versuchen nicht gespeichert werden!")
                    sys.exit(1)
                
                # Speichere auch die Historie nochmal zur Sicherheit
                if not save_history(history):
                    print("❌ Fehler beim erneuten Speichern der Historie!")
            else:
                print("⚠️  Pre-CAS Status wurde NICHT aktualisiert (keine Benachrichtigung erfolgreich)")
        else:
            print("✅ Keine Pre-CAS Änderung - alles beim Alten.")
            status['last_check'] = datetime.utcnow().isoformat()  # UTC für Konsistenz
            # Speichere auch bei keiner Änderung den aktualisierten Timestamp
            save_status(status)
    else:
        print("\n⚠️  WARNUNG: Konnte das Pre-CAS Datum nicht von der Webseite extrahieren!")
        
        # Bei manueller Ausführung auch Fehler melden
        if IS_MANUAL:
            telegram_error = f"""<b>❌ Manueller Pre-CAS Check fehlgeschlagen</b>

Konnte das Pre-CAS Datum nicht von der Webseite extrahieren!

<b>Zeitpunkt:</b> {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
<b>Letztes bekanntes Pre-CAS Datum:</b> {status.get('pre_cas_date', 'Unbekannt')}

Bitte prüfe die Webseite manuell.

<a href="{URL}">📄 LSE Webseite öffnen</a>"""
            
            send_telegram(telegram_error)
        
        # Sende Warnung per E-Mail
        subject = "LSE Monitor WARNUNG: Pre-CAS Datum nicht gefunden"
        body = f"""WARNUNG: Der LSE Monitor konnte das Pre-CAS Datum nicht von der Webseite extrahieren!

Zeitpunkt: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
Letztes bekanntes Pre-CAS Datum: {status.get('pre_cas_date', 'Unbekannt')}

Bitte überprüfe:
1. Ist die Webseite erreichbar? {URL}
2. Hat sich die Struktur der Seite geändert?

Der Monitor wird weiterhin prüfen."""
        
        if always_notify:
            send_gmail(subject, body, always_notify)
        
        # Telegram-Warnung (nur bei automatischer Ausführung)
        if not IS_MANUAL:
            telegram_warning = f"""<b>⚠️ LSE Monitor WARNUNG</b>

Konnte das Pre-CAS Datum nicht von der Webseite extrahieren!

Letztes bekanntes Pre-CAS Datum: <b>{status.get('pre_cas_date', 'Unbekannt')}</b>
Zeit: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}

Mögliche Gründe:
• Webseite nicht erreichbar
• Struktur hat sich geändert
• Netzwerkfehler

<a href="{URL}">📄 Webseite manuell prüfen</a>"""
            
            send_telegram(telegram_warning)
        
        # Speichere trotzdem den Status (mit last_check Update)
        status['last_check'] = datetime.utcnow().isoformat()
        save_status(status)
    
    print("\n" + "="*50)
    
    # Debug: Zeige finale Dateien
    print("\n📁 FINALE DATEIEN:")
    print("=== status.json ===")
    os.system("cat status.json")
    print("\n=== history.json (letzte 3 Pre-CAS Einträge) ===")
    os.system("grep -A3 pre_cas_changes history.json | tail -n 10")
    
    # Finaler Status-Output für Debugging
    print("\n📊 FINALER STATUS:")
    try:
        with open(STATUS_FILE, 'r') as f:
            final_status = json.load(f)
            print(f"   last_date (AOA): {final_status.get('last_date')}")
            print(f"   pre_cas_date: {final_status.get('pre_cas_date')}")
            print(f"   cas_date: {final_status.get('cas_date')}")
            print(f"   last_check: {final_status.get('last_check')}")
    except Exception as e:
        print(f"   Fehler beim Lesen des finalen Status: {e}")

if __name__ == "__main__":
    main()
