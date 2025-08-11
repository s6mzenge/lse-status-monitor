import requests
from bs4 import BeautifulSoup
import json
import os
import re
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')


# === Konfiguration und Konstanten (ergänzt) ===
REGRESSION_MIN_POINTS: int = 2
CONFIDENCE_LEVEL: float = 1.96  # 95%-Konfidenzniveau
# === Business-day helpers (x-axis skips weekends) ===

def business_days_elapsed(start_dt, end_dt):
    '''
    Zählt Arbeitstage (Mo–Fr) zwischen start_dt (inkl.) und end_dt (exkl.).
    Nutzt nur das Datum (keine Uhrzeiten). Für denselben Kalendertag -> 0.
    '''
    s = np.datetime64(start_dt.date(), 'D')
    e = np.datetime64(end_dt.date(), 'D')
    # np.busday_count zählt Werktage im Intervall [s, e)
    return int(np.busday_count(s, e))

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
        if current.weekday() < 5:  # Mo–Fr
            remaining -= 1
    return current


# Versuche erweiterte Bibliotheken zu importieren
try:
    from scipy import stats
    from sklearn.linear_model import HuberRegressor, RANSACRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    ADVANCED_REGRESSION = True
except ImportError:
    print("⚠️ Erweiterte Regression nicht verfügbar. Installiere: pip install scikit-learn scipy")
    ADVANCED_REGRESSION = False

URL = "https://www.lse.ac.uk/study-at-lse/Graduate/News/Current-processing-times"
STATUS_FILE = "status.json"
HISTORY_FILE = "history.json"

def get_german_time():
    """Gibt die aktuelle Zeit in deutscher Zeitzone zurück (UTC+2 für Sommerzeit)"""
    utc_time = datetime.utcnow()
    # Deutschland ist UTC+1 (Winter) oder UTC+2 (Sommer)
    # Hier verwenden wir UTC+2 für Sommerzeit
    # Im Winter auf hours=1 ändern
    german_time = utc_time + timedelta(hours=2)
    return german_time



def _iter_observations_or_changes(history: Dict) -> List[Dict]:
    """
    Kombiniert observations und changes, normiert Timestamps auf ISO-UTC,
    validiert Daten, sortiert aufsteigend.
    """
    out: List[Dict] = []
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

def calculate_advanced_regression_forecast(history, current_date=None):
    """
    Erweiterte Regression mit mehreren Verbesserungen:
    - Outlier-resistente Regression
    - Polynomielle Regression 
    - Gewichtete Regression (neuere Daten wichtiger)
    - Konfidenzintervalle
    - Arbeitstagberechnung
    """
    source_data = _iter_observations_or_changes(history)
    if len(source_data) < REGRESSION_MIN_POINTS:
        return None
    
    # Extrahiere Datenpunkte
    data_points = []
    first_timestamp = None
    for entry in _iter_observations_or_changes(history):
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
    
    if ADVANCED_REGRESSION:
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
                print(f"⚠️ Modellberechnung fehlgeschlagen: {e}")
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
                print(f"⚠️ Modellberechnung fehlgeschlagen: {e}")
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
                print(f"⚠️ Modellberechnung fehlgeschlagen: {e}")
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
            
            if ADVANCED_REGRESSION:
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
    target_25_days = date_to_days("25 July")
    target_28_days = date_to_days("28 July")
    
    predictions = {}
    days_until_25 = None
    days_until_28 = None
    
    for target_name, target_days in [("25 July", target_25_days), ("28 July", target_28_days)]:
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
            if target_name == "25 July":
                days_until_25 = days_until
            elif target_name == "28 July":
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
    
    return {
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

def compute_integrated_model_metrics(history):
    """
    Berechnet Kennzahlen der neuen integrierten Regression (Theil–Sen ⊕ LOESS)
    inkl. Heartbeats. Gibt None zurück, wenn das Modul fehlt.
    """
    try:
        from lse_integrated_model import BusinessCalendar, IntegratedRegressor, LON, BER
        from datetime import time as _time
        import numpy as _np
    except ImportError:
        return None

    # Alle Punkte (Änderungen + Beobachtungen/Heartbeats)
    rows = [{"timestamp": e["timestamp"], "date": e["date"]}
            for e in _iter_observations_or_changes(history)]
    if len(rows) < REGRESSION_MIN_POINTS:
        return None

    # Kalender/Modell
    cal = BusinessCalendar(tz=LON, start=_time(10, 0), end=_time(16, 0), holidays=tuple([]))
    imodel = IntegratedRegressor(cal=cal, loess_frac=0.6, tau_hours=12.0).fit(rows)

    # R² auf beobachteten Punkten
    x_obs = imodel.x_
    y_obs = imodel.y_
    y_hat = _np.array([imodel._blend_predict_scalar(float(v)) for v in x_obs])
    ss_res = float(_np.sum((y_obs - y_hat) ** 2))
    ss_tot = float(_np.sum((y_obs - _np.mean(y_obs)) ** 2))
    r2_new = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Durchschnittlicher Fortschritt (Tage pro Business-Tag)
    hours_per_day = (cal.end.hour - cal.start.hour) + (cal.end.minute - cal.start.minute) / 60.0
    avg_prog_new = imodel.ts_.b * hours_per_day

    # Vorhersagen
    pred25 = imodel.predict_datetime("25 July", tz_out=BER)
    pred28 = imodel.predict_datetime("28 July", tz_out=BER)
    now_de = get_german_time()
    d25 = business_days_elapsed(now_de, pred25["when_point"])
    d28 = business_days_elapsed(now_de, pred28["when_point"])

    # Heartbeats zählen
    heartbeats = sum(1 for o in history.get("observations", []) if o.get("kind") == "heartbeat")

    return {
        "name": "NEU (integriert)",
        "r2": r2_new,
        "points": len(rows),
        "speed": f"{avg_prog_new:.1f} Tage/Tag",
        "eta25": _fmt_eta(d25, pred25["when_point"]),
        "eta28": _fmt_eta(d28, pred28["when_point"]),
        "heartbeats": heartbeats
    }

def format_regression_comparison_table_mono(old_s, new_s):
    """
    Baut eine Monospace-Tabelle mit drei Spalten: Metrik | ALT | NEU.
    Wird als <pre>...</pre> in Telegram (HTML parse_mode) gesendet.
    """
    if not old_s or not new_s:
        return ""

    rows = [
        ("R² (Datenpunkte)", f"{old_s['r2']:.2f} ({old_s['points']})",       f"{new_s['r2']:.2f} ({new_s['points']})"),
        ("Fortschritt",       old_s['speed'],                                new_s['speed']),
        ("Trend",             old_s.get('trend', '—'),                       "—"),
        ("25 July ETA",       old_s['eta25'],                                new_s['eta25']),
        ("28 July ETA",       old_s['eta28'],                                new_s['eta28']),
    ]

    h1, h2, h3 = "Metrik", old_s["name"], new_s["name"]
    w1 = max(len(h1), *(len(r[0]) for r in rows))
    w2 = max(len(h2), *(len(r[1]) for r in rows))
    w3 = max(len(h3), *(len(r[2]) for r in rows))

    def fmt(a, b, c): return f"{a:<{w1}}  |  {b:<{w2}}  |  {c:<{w3}}"
    line = "-" * (w1 + w2 + w3 + 6)

    out = []
    out.append(fmt(h1, h2, h3))
    out.append(line)
    for r in rows:
        out.append(fmt(*r))
    out.append(line)
    out.append(f"* Neue Regression nutzt Heartbeats: {new_s['heartbeats']}")
    return "\n".join(out)

def create_enhanced_forecast_text(forecast):
    """Erstellt erweiterten Prognosetext mit mehr Details wenn verfügbar"""
    if not forecast:
        return "\n📊 Prognose: Noch nicht genügend Daten für eine zuverlässige Vorhersage."
    
    text = "\n📊 PROGNOSEN basierend auf bisherigen Änderungen:\n"
    
    # Zeige verwendetes Modell wenn erweiterte Regression verfügbar
    if 'model_name' in forecast and ADVANCED_REGRESSION:
        text += f"📈 Bestes Modell: {forecast['model_name']} "
    
    text += f"(R²={forecast['r_squared']:.2f}, {forecast['data_points']} Datenpunkte)\n\n"

    # === ALT vs. NEU: Monospace-Vergleichstabelle ===
    try:
        hist = get_history()
        old_s = _old_regression_summary(forecast)
        new_s = compute_integrated_model_metrics(hist)
        if new_s:
            table = format_regression_comparison_table_mono(old_s, new_s)
            if table:
                text += "<b>ALT vs. NEU (kompakt)</b>\n<pre>" + table + "</pre>\n"
    except Exception as _e:
        print(f"❌ Tabelle (ALT vs. NEU) konnte nicht erzeugt werden: {_e}")

    
    if forecast['slope'] <= 0:
        text += "⚠️ Die Daten zeigen keinen Fortschritt oder sogar Rückschritte.\n"
    else:
        text += f"📈 Durchschnittlicher Fortschritt: {forecast['slope']:.1f} Tage pro Tag\n"
        
        # Zeige Trend-Analyse wenn verfügbar
        if 'trend_analysis' in forecast and forecast['trend_analysis'] != "unbekannt":
            emoji = {"beschleunigend": "🚀", "verlangsamend": "🐌", "konstant": "➡️"}
            text += f"{emoji.get(forecast['trend_analysis'], '❓')} Trend: {forecast['trend_analysis'].upper()}\n"
        
        text += "\n"
        
        # Verwende erweiterte Prognosen wenn verfügbar
        if 'predictions' in forecast and forecast['predictions']:
            for target_name in ["25 July", "28 July"]:
                if target_name in forecast['predictions']:
                    pred = forecast['predictions'][target_name]
                    if pred['days'] and pred['days'] > 0:
                        date_pred = pred.get('date', get_german_time() + timedelta(days=pred['days']))
                        text += f"📅 {target_name} wird voraussichtlich erreicht:\n"
                        text += f"   • In {pred['days']:.0f} Tagen"
                        
                        # Zeige Konfidenzintervall wenn verfügbar
                        if 'days_lower' in pred and 'days_upper' in pred and ADVANCED_REGRESSION:
                            text += f" ({pred['days_lower']:.0f}-{pred['days_upper']:.0f} Tage)\n"
                        else:
                            text += "\n"
                        
                        text += f"   • Am {date_pred.strftime('%d. %B %Y')}\n\n"
        else:
            # Fallback auf alte Methode
            if forecast.get('days_until_25_july') is not None and forecast['days_until_25_july'] > 0:
                date_25 = get_german_time() + timedelta(days=forecast['days_until_25_july'])
                text += f"📅 25 July wird voraussichtlich erreicht:\n"
                text += f"   • In {forecast['days_until_25_july']:.0f} Tagen\n"
                text += f"   • Am {date_25.strftime('%d. %B %Y')}\n\n"
            
            if forecast.get('days_until_28_july') is not None and forecast['days_until_28_july'] > 0:
                date_28 = get_german_time() + timedelta(days=forecast['days_until_28_july'])
                text += f"📅 28 July wird voraussichtlich erreicht:\n"
                text += f"   • In {forecast['days_until_28_july']:.0f} Tagen\n"
                text += f"   • Am {date_28.strftime('%d. %B %Y')}\n\n"
        
        # Qualitätshinweis
        if forecast['r_squared'] < 0.5:
            text += "⚠️ Hinweis: Die Vorhersage ist unsicher (niedrige Korrelation).\n"
        elif forecast['r_squared'] > 0.8:
            text += "✅ Die Vorhersage basiert auf einem stabilen Trend.\n"
        
        # Zeige Modellvergleich wenn mehrere Modelle verfügbar
        if ADVANCED_REGRESSION and 'models' in forecast and len(forecast['models']) > 1:
            text += "\n🔬 Modell-Vergleich:\n"
            for name, data in sorted(forecast['models'].items(), key=lambda x: x[1].get('r2', 0), reverse=True):
                if 'r2' in data:
                    text += f"   • {name.capitalize()}: R²={data['r2']:.3f}\n"

    return text

def create_forecast_text(forecast):
    """Wrapper für Rückwärtskompatibilität - ruft erweiterte Version auf"""
    return create_enhanced_forecast_text(forecast)

def create_progression_graph(history, current_date, forecast=None):
    """
    ALT vs. NEU mit echten Schnittpunkt-ETAs (25/28 July), gut sichtbaren Heartbeats,
    glatten Regressionslinien (fraktionale Business-Days) und kompakten Achsen.
    Gibt BytesIO (PNG) zurück oder None.
    """
    from io import BytesIO
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    import numpy as np
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    COL_ALT = "#1f77b4"   # blau
    COL_NEU = "#ff7f0e"   # orange
    COL_HB  = "#2ca02c"   # grün

    # ----- Zeit-/Datum-Helfer -----
    def _to_naive_berlin(dt_like):
        if dt_like is None: return None
        if isinstance(dt_like, datetime):
            if dt_like.tzinfo is None:
                return dt_like.replace(tzinfo=ZoneInfo("Europe/Berlin")).astimezone(
                    ZoneInfo("Europe/Berlin")).replace(tzinfo=None)
            return dt_like.astimezone(ZoneInfo("Europe/Berlin")).replace(tzinfo=None)
        try:
            dtx = datetime.fromisoformat(str(dt_like))
            return dtx.astimezone(ZoneInfo("Europe/Berlin")).replace(tzinfo=None)
        except Exception:
            return None

    # kontinuierliche (fraktionale) Business-Days
    def _bizdays_float(start: datetime, t: datetime) -> float:
        import numpy as _np
        s0 = datetime(start.year, start.month, start.day)
        t0 = datetime(t.year, t.month, t.day)
        full = float(_np.busday_count(_np.datetime64(s0.date()), _np.datetime64(t0.date())))
        def _frac(d: datetime) -> float:
            if not bool(_np.is_busday(_np.datetime64(d.date()))):
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

    # invertiert y(t)=m*BD(t)+b -> t*, per Bisektion (garantiert Schnittpunkt)
    def _solve_time_for_level(slope, intercept, y_target, t0, t_hint, sign_positive=True):
        if slope is None or slope == 0:
            return None
        # Startintervall: rund um hint (heute), dann nach außen expandieren bis wir y_target bracketen
        low  = t_hint - timedelta(days=5)
        high = t_hint + timedelta(days=20)

        def f(t):  # y(t)
            return slope * _bizdays_float(t0, t) + intercept

        # Bracketing – monotone Funktion
        # Expandieren bis f(low) <= y <= f(high) (bei positiver Steigung)
        max_expand = 20
        if sign_positive:
            while f(low)  > y_target and max_expand > 0: low  -= timedelta(days=10); max_expand -= 1
            while f(high) < y_target and max_expand > 0: high += timedelta(days=10); max_expand -= 1
        else:
            while f(low)  < y_target and max_expand > 0: low  -= timedelta(days=10); max_expand -= 1
            while f(high) > y_target and max_expand > 0: high += timedelta(days=10); max_expand -= 1

        # Falls nicht gebracketed: Abbruch
        if (f(low) - y_target) * (f(high) - y_target) > 0:
            return None

        # Bisektion
        for _ in range(50):
            mid = low + (high - low) / 2
            val = f(mid)
            if abs(val - y_target) < 1e-6:  # genügend genau
                return mid
            if (val < y_target) == sign_positive:
                low = mid
            else:
                high = mid
        return mid

    # ----- Daten sammeln -----
    data = list(_iter_observations_or_changes(history))
    if len(data) < REGRESSION_MIN_POINTS:
        return None

    change_ts, change_y, change_labels, hb_ts = [], [], [], []
    for e in data:
        try:
            ts = _to_naive_berlin(datetime.fromisoformat(e["timestamp"]))
            yv = date_to_days(e["date"])
            if ts is None or yv is None:
                continue
            if e.get("kind") == "heartbeat":
                hb_ts.append(ts)
            else:
                change_ts.append(ts); change_y.append(yv); change_labels.append(e["date"])
        except Exception:
            continue
    if not change_ts:
        return None

    ordered = sorted(zip(change_ts, change_y, change_labels), key=lambda r: r[0])
    change_ts, change_y, change_labels = [list(t) for t in zip(*ordered)]

    now_de  = _to_naive_berlin(get_german_time())
    year_ref = now_de.year

    # ----- Plot-Grundgerüst -----
    try: plt.style.use("seaborn-v0_8-darkgrid")
    except Exception: pass
    fig, ax = plt.subplots(figsize=(12, 7))

    # Historische Punkte + Labels (nur die letzten 8)
    ax.scatter(change_ts, change_y, s=90, zorder=5, label="Änderungen (historisch)", alpha=0.9, color=COL_ALT)
    last_k = max(0, len(change_ts) - 8)
    for i, (ts, y, lbl) in enumerate(zip(change_ts[last_k:], change_y[last_k:], change_labels[last_k:])):
        dy = 10 if i % 2 == 0 else -14
        ax.annotate(lbl, (ts, y), xytext=(6, dy), textcoords="offset points",
                    fontsize=8.5, alpha=0.85, ha="left", va="center")

    # Heute + aktueller Punkt
    ax.axvline(now_de, linewidth=1.0, linestyle=":", alpha=0.8)
    ax.scatter([change_ts[-1]], [change_y[-1]], s=100, zorder=6, label="Aktuell", color=COL_NEU)

    # Zielhöhen
    target_map = {"25 July": date_to_days("25 July"), "28 July": date_to_days("28 July")}
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

    # ----- ALT-Linie -----
    alt_eta_pts = {}  # name -> (t,y)
    if forecast and float(forecast.get("slope", 0.0)) > 0.0:
        m_old = float(forecast["slope"])
        y_now = forecast.get("current_trend_days", change_y[-1])
        b_old = y_now - m_old * _bizdays_float(first_ts, now_de)
        y_old = np.array([m_old * bd + b_old for bd in grid_bd])
        ax.plot(grid_ts, y_old, linestyle=(0, (6, 4)), linewidth=2.0,
                label=f"ALT: {forecast.get('model_name','Linear')} (R²={float(forecast.get('r_squared',0.0)):.2f})",
                color=COL_ALT, alpha=0.95)

        # echte Schnittpunkte ALT ∩ (y = target)
        for name, ty in target_map.items():
            if ty is None: continue
            t_star = _solve_time_for_level(m_old, b_old, ty, first_ts, now_de, sign_positive=(m_old>0))
            if t_star:
                y_star = m_old * _bizdays_float(first_ts, t_star) + b_old
                alt_eta_pts[name] = (t_star, y_star)

    # ----- NEU-Linie -----
    neu_eta_pts = {}
    try:
        from lse_integrated_model import BusinessCalendar, IntegratedRegressor, LON, BER
        from datetime import time as _time

        rows = [{"timestamp": e["timestamp"], "date": e["date"]} for e in data]
        if len(rows) >= REGRESSION_MIN_POINTS:
            cal = BusinessCalendar(tz=LON, start=_time(10, 0), end=_time(16, 0), holidays=tuple([]))
            imodel = IntegratedRegressor(cal=cal, loess_frac=0.6, tau_hours=12.0).fit(rows)

            hours_per_day = (cal.end.hour - cal.start.hour) + (cal.end.minute - cal.start.minute) / 60.0
            m_new = float(imodel.ts_.b * hours_per_day)

            y_curr = date_to_days(current_date) or (change_y[-1] if change_y else None)
            if y_curr is not None and m_new is not None:
                b_new = y_curr - m_new * _bizdays_float(first_ts, now_de)
                y_new = np.array([m_new * bd + b_new for bd in grid_bd])
                ax.plot(grid_ts, y_new, linewidth=2.6, label="NEU: integrierte Regression",
                        color=COL_NEU, alpha=0.95)

                # echte Schnittpunkte NEU ∩ (y = target) – gleiche BD-Funktion wie gezeichnete Linie
                for name, ty in target_map.items():
                    if ty is None: continue
                    t_star = _solve_time_for_level(m_new, b_new, ty, first_ts, now_de, sign_positive=(m_new>0))
                    if t_star:
                        y_star = m_new * _bizdays_float(first_ts, t_star) + b_new
                        neu_eta_pts[name] = (t_star, y_star)
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ NEU-Regression konnte nicht gezeichnet werden: {e}")

    # ----- Sterne + Labels (nicht überlappend: ALT oben, NEU unten) -----
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

    _plot_eta("25", alt_eta_pts.get("25 July"), neu_eta_pts.get("25 July"))
    _plot_eta("28", alt_eta_pts.get("28 July"), neu_eta_pts.get("28 July"))

    # ----- Achsen & Heartbeats -----
    # X bis zur spätesten ETA
    eta_all = [p[0] for p in list(alt_eta_pts.values()) + list(neu_eta_pts.values()) if p]
    right_edge = max(eta_all) if eta_all else change_ts[-1]
    left_edge  = min(change_ts[0], now_de - timedelta(days=3))
    ax.set_xlim(left_edge - timedelta(days=1), right_edge + timedelta(days=1))

    # Y eng führen
    y_min = min(change_y)
    y_max = max([*change_y, *(v for v in target_map.values() if v is not None)])
    ax.set_ylim(y_min - 2, y_max + 2)

    # Heartbeats als kurze vertikale Striche unten (jetzt nach set_ylim, damit sichtbar)
    if hb_ts:
        ylo, yhi = ax.get_ylim()
        y0 = ylo + 0.02 * (yhi - ylo)
        y1 = ylo + 0.10 * (yhi - ylo)
        ax.vlines(hb_ts, y0, y1, colors=COL_HB, linewidth=1.4, label="Heartbeats (NEU)", alpha=0.9)

    # Achsenformat
    ax.set_title("Fortschritt & Prognose — ALT vs. NEU")
    ax.set_xlabel("Datum"); ax.set_ylabel("Verarbeitungsdatum")
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

    # Export
    try:
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        buf.seek(0); plt.close()
        return buf
    except Exception as e:
        print(f"⚠️ Diagramm konnte nicht erzeugt werden: {e}")
        try: plt.close()
        except Exception: pass
        return None

def send_telegram_photo(photo_buffer, caption, parse_mode='HTML'):
    """Sendet ein Foto über Telegram Bot"""
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("Telegram nicht konfiguriert")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        files = {'photo': ('graph.png', photo_buffer, 'image/png')}
        data = {
            'chat_id': chat_id,
            'caption': caption,
            'parse_mode': parse_mode
        }
        
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            print("✅ Telegram-Foto gesendet!")
            return True
        else:
            print(f"❌ Telegram-Fehler beim Foto senden: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram-Fehler beim Foto senden: {e}")
        return False

def send_telegram(message):
    """Sendet eine Nachricht über Telegram Bot"""
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("Telegram nicht konfiguriert")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("✅ Telegram-Nachricht gesendet!")
            return True
        else:
            print(f"❌ Telegram-Fehler: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram-Fehler: {e}")
        return False

def send_telegram_mama(old_date, new_date):
    """Sendet eine einfache Nachricht an Mama über separaten Bot"""
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN_MAMA')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID_MAMA')
    
    if not bot_token or not chat_id:
        print("Telegram für Mama nicht konfiguriert")
        return False
    
    try:
        # Einfache Nachricht ohne HTML-Formatierung
        message = f"LSE-Datums-Update!\n\nVom: {old_date}\nAuf: {new_date}"
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("✅ Telegram-Nachricht an Mama gesendet!")
            return True
        else:
            print(f"❌ Telegram-Fehler (Mama): {response.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram-Fehler (Mama): {e}")
        return False

def send_telegram_papa(old_date, new_date):
    """Sendet eine einfache Nachricht an Papa über separaten Bot"""
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN_PAPA')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID_PAPA')
    
    if not bot_token or not chat_id:
        print("Telegram für Papa nicht konfiguriert")
        return False
    
    try:
        # Einfache Nachricht ohne HTML-Formatierung
        message = f"LSE-Datums-Update!\n\nVom: {old_date}\nAuf: {new_date}"
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("✅ Telegram-Nachricht an Papa gesendet!")
            return True
        else:
            print(f"❌ Telegram-Fehler (Papa): {response.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram-Fehler (Papa): {e}")
        return False


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
            status['last_check'] = datetime.utcnow().isoformat()
        
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
        
        # Verifiziere dass es korrekt gespeichert wurde
        with open(STATUS_FILE, 'r') as f:
            saved = json.load(f)
            if saved.get('last_date') == status['last_date']:
                print(f"✅ Status erfolgreich gespeichert: {status['last_date']}")
                return True
            else:
                print(f"❌ FEHLER: Status nicht korrekt gespeichert!")
                print(f"   Erwartet: {status['last_date']}")
                print(f"   Gespeichert: {saved.get('last_date')}")
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
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
            
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
            
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
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
        response = requests.get(URL, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }, timeout=30)
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
            server = smtplib.SMTP('smtp.gmail.com', 587)
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
    
    # Zeige ob erweiterte Regression verfügbar ist
    if ADVANCED_REGRESSION:
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
# Stilles Tracking für Pre-CAS (keine Benachrichtigungen)
    if current_dates['pre_cas'] and current_dates['pre_cas'] != status.get('pre_cas_date'):
        print(f"\n📝 Pre-CAS Änderung (stilles Tracking): {status.get('pre_cas_date') or 'Unbekannt'} → {current_dates['pre_cas']}")
        history['pre_cas_changes'].append({
            "timestamp": datetime.utcnow().isoformat(),
            "date": current_dates['pre_cas'],
            "from": status.get('pre_cas_date')
        })
        status['pre_cas_date'] = current_dates['pre_cas']
        # Speichere sofort
        save_history(history)
    
    # Stilles Tracking für CAS (keine Benachrichtigungen)
    if current_dates['cas'] and current_dates['cas'] != status.get('cas_date'):
        print(f"\n📝 CAS Änderung (stilles Tracking): {status.get('cas_date') or 'Unbekannt'} → {current_dates['cas']}")
        history['cas_changes'].append({
            "timestamp": datetime.utcnow().isoformat(),
            "date": current_dates['cas'],
            "from": status.get('cas_date')
        })
        status['cas_date'] = current_dates['cas']
        # Speichere sofort
        save_history(history)
    
    # Hauptlogik für "all other applicants" (mit Benachrichtigungen wie bisher)
    current_date = current_dates['all_other']
    
    if current_date:
        print(f"Aktuelles Datum für 'all other graduate applicants': {current_date}")
        
        # Bei manuellem Check immer Status senden (NUR für all other applicants)
        if IS_MANUAL:
            # Berechne aktuellen Trend und erstelle vollständige Prognose
            forecast = calculate_regression_forecast(history)
            forecast_text = create_forecast_text(forecast) or ""
            
            telegram_msg = f"""<b>📊 LSE Status Check Ergebnis</b>

<b>Aktuelles Datum:</b> {current_date}
<b>Letzter Stand:</b> {status['last_date']}
<b>Status:</b> {"🔔 ÄNDERUNG ERKANNT!" if current_date != status['last_date'] else "✅ Keine Änderung"}

<b>Zeitpunkt:</b> {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
{forecast_text}

<a href="{URL}">📄 LSE Webseite öffnen</a>"""
            
            # Sende Text-Nachricht
            send_telegram(telegram_msg)
            
            # Erstelle und sende Graph
            graph_buffer = create_progression_graph(history, current_date, forecast)
            if graph_buffer:
                graph_caption = f"📈 Progression der LSE Verarbeitungsdaten\nAktuell: {current_date}"
                send_telegram_photo(graph_buffer, graph_caption)
        
        # WICHTIG: Prüfe ob sich das Datum wirklich geändert hat
        if current_date != status['last_date']:
            print("\n🔔 ÄNDERUNG ERKANNT!")
            print(f"   Von: {status['last_date']}")
            print(f"   Auf: {current_date}")
            
            # Sende einfache Nachricht an Mama
            send_telegram_mama(status['last_date'], current_date)
            
            # Sende einfache Nachricht an Papa
            send_telegram_papa(status['last_date'], current_date)
            
            # Speichere in Historie mit UTC Zeit (für Konsistenz)
            history["changes"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "date": current_date,
                "from": status['last_date']
            })
            
            # Speichere Historie sofort
            if not save_history(history):
                print("❌ Fehler beim Speichern der Historie!")
            
            # Berechne Prognose
            forecast = calculate_regression_forecast(history)
            forecast_text = create_forecast_text(forecast) or ""
            
            # Erstelle E-Mail-Inhalt
            subject = f"LSE Status Update: Neues Datum {current_date}"
            
            # Bei manuellem Check: Hinweis in E-Mail
            manual_hint = "\n\n(Änderung durch manuellen Check via Telegram entdeckt)" if IS_MANUAL else ""
            
            # Basis-E-Mail für alle
            base_body = f"""Das Verarbeitungsdatum für "all other graduate applicants" hat sich geändert!

ÄNDERUNG:
Von: {status['last_date']}
Auf: {current_date}

Zeitpunkt der Erkennung: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}

Link zur Seite: {URL}{manual_hint}"""
            
            # E-Mail mit Prognose für Hauptempfänger
            body_with_forecast = base_body + f"\n{forecast_text}\n\nDiese E-Mail wurde automatisch von deinem GitHub Actions Monitor generiert."
            
            # E-Mail ohne Prognose für bedingte Empfänger
            body_simple = base_body + "\n\nDiese E-Mail wurde automatisch von deinem GitHub Actions Monitor generiert."
            
            # Telegram-Nachricht formatieren
            if not IS_MANUAL:
                # Automatischer Check: Standard-Änderungsnachricht mit Graph
                telegram_msg = f"""<b>🔔 LSE Status Update</b>

<b>ÄNDERUNG ERKANNT!</b>
Von: {status['last_date']}
Auf: <b>{current_date}</b>

Zeit: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
{forecast_text}

<a href="{URL}">📄 LSE Webseite öffnen</a>"""
                
                send_telegram(telegram_msg)
                
                # Sende Graph als separates Bild
                graph_buffer = create_progression_graph(history, current_date, forecast)
                if graph_buffer:
                    graph_caption = f"📈 Progression Update\nNeues Datum: {current_date}"
                    send_telegram_photo(graph_buffer, graph_caption)
            else:
                # Manueller Check: Spezielle Nachricht bei Änderung mit Graph
                telegram_msg = f"""<b>🚨 ÄNDERUNG GEFUNDEN!</b>

Dein manueller Check hat eine Änderung entdeckt!

Von: {status['last_date']}
Auf: <b>{current_date}</b>

Zeit: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
{forecast_text}

📧 E-Mails werden an die Hauptempfänger gesendet!

<a href="{URL}">📄 LSE Webseite öffnen</a>"""
                
                send_telegram(telegram_msg)
                
                # Sende Graph
                graph_buffer = create_progression_graph(history, current_date, forecast)
                if graph_buffer:
                    graph_caption = f"📈 Änderung erkannt!\nVon {status['last_date']} auf {current_date}"
                    send_telegram_photo(graph_buffer, graph_caption)
            
            # Sende E-Mails
            emails_sent = False
            
            # Immer benachrichtigen (mit Prognose) - JETZT AUCH BEI MANUELLEN CHECKS
            if always_notify:
                if send_gmail(subject, body_with_forecast, always_notify):
                    emails_sent = True
            
            # Bedingt benachrichtigen (nur bei 25 oder 28 July)
            if conditional_notify and current_date in ["25 July", "28 July"]:
                print(f"\n🎯 Zieldatum {current_date} erreicht! Benachrichtige zusätzliche Empfänger.")
                if send_gmail(subject, body_simple, conditional_notify):
                    emails_sent = True
                
                # Spezielle Telegram-Nachricht für Zieldatum mit Graph
                telegram_special = f"""<b>🎯 ZIELDATUM ERREICHT!</b>

Das Datum <b>{current_date}</b> wurde erreicht!

Dies ist eines der wichtigen Zieldaten für deine LSE-Bewerbung.

<a href="{URL}">📄 Jetzt zur LSE Webseite</a>"""
                send_telegram(telegram_special)
                
                # Sende speziellen Graph für Zieldatum
                graph_buffer = create_progression_graph(history, current_date, forecast)
                if graph_buffer:
                    graph_caption = f"🎯 ZIELDATUM ERREICHT: {current_date}!"
                    send_telegram_photo(graph_buffer, graph_caption)
            
            if emails_sent or os.environ.get('TELEGRAM_BOT_TOKEN'):
                # KRITISCH: Update Status IMMER nach einer erkannten Änderung
                # Update Status nur bei erfolgreicher Benachrichtigung
                status['last_date'] = current_date
                status['last_check'] = datetime.utcnow().isoformat()
                
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
                        if verify_status.get('last_date') == current_date:
                            print(f"✅ Status erfolgreich gespeichert und verifiziert: {current_date}")
                            save_success = True
                        else:
                            print(f"❌ Verifikation fehlgeschlagen! Erwartet: {current_date}, Geladen: {verify_status.get('last_date')}")
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
            print("✅ Keine Änderung - alles beim Alten.")
            status['last_check'] = datetime.utcnow().isoformat()  # UTC für Konsistenz
            # Speichere auch bei keiner Änderung den aktualisierten Timestamp
            save_status(status)
    else:
        print("\n⚠️  WARNUNG: Konnte das Datum nicht von der Webseite extrahieren!")
        
        # Bei manueller Ausführung auch Fehler melden
        if IS_MANUAL:
            telegram_error = f"""<b>❌ Manueller Check fehlgeschlagen</b>

Konnte das Datum nicht von der Webseite extrahieren!

<b>Zeitpunkt:</b> {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
<b>Letztes bekanntes Datum:</b> {status['last_date']}

Bitte prüfe die Webseite manuell.

<a href="{URL}">📄 LSE Webseite öffnen</a>"""
            
            send_telegram(telegram_error)
        
        # Sende Warnung per E-Mail
        subject = "LSE Monitor WARNUNG: Datum nicht gefunden"
        body = f"""WARNUNG: Der LSE Monitor konnte das Datum nicht von der Webseite extrahieren!

Zeitpunkt: {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}
Letztes bekanntes Datum: {status['last_date']}

Bitte überprüfe:
1. Ist die Webseite erreichbar? {URL}
2. Hat sich die Struktur der Seite geändert?

Der Monitor wird weiterhin prüfen."""
        
        if always_notify:
            send_gmail(subject, body, always_notify)
        
        # Telegram-Warnung (nur bei automatischer Ausführung)
        if not IS_MANUAL:
            telegram_warning = f"""<b>⚠️ LSE Monitor WARNUNG</b>

Konnte das Datum nicht von der Webseite extrahieren!

Letztes bekanntes Datum: <b>{status['last_date']}</b>
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
    print("\n=== history.json (letzte 3 Einträge) ===")
    os.system("tail -n 20 history.json | head -n 20")
    
    # Finaler Status-Output für Debugging
    print("\n📊 FINALER STATUS:")
    try:
        with open(STATUS_FILE, 'r') as f:
            final_status = json.load(f)
            print(f"   last_date: {final_status.get('last_date')}")
            print(f"   last_check: {final_status.get('last_check')}")
            print(f"   pre_cas_date: {final_status.get('pre_cas_date') or 'Nicht getrackt'}")
            print(f"   cas_date: {final_status.get('cas_date') or 'Nicht getrackt'}")
    except Exception as e:
        print(f"   Fehler beim Lesen des finalen Status: {e}")

if __name__ == "__main__":
    main()
