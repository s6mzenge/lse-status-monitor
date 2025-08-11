import requests
from bs4 import BeautifulSoup
import json
import os
import re
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import numpy as np
# --- Business-day helpers (Mon–Fri, optional UK_HOLIDAYS env) ---
_UK_HOLIDAYS = [s.strip() for s in os.getenv('UK_HOLIDAYS','').split(',') if s.strip()] + ['2025-08-25']
def _to_npday(x):
    if hasattr(x, 'date'):
        x = x.date()
    return np.datetime64(x, 'D')
def _busday_count(start_dt, end_dt):
    try:
        s = _to_npday(start_dt); e = _to_npday(end_dt)
        holidays_np = np.array(_UK_HOLIDAYS, dtype='datetime64[D]') if _UK_HOLIDAYS else None
        return int(np.busday_count(s, e, holidays=holidays_np))
    except Exception:
        try:
            sd = start_dt.date() if hasattr(start_dt, 'date') else start_dt
            ed = end_dt.date() if hasattr(end_dt, 'date') else end_dt
            return max(0, (ed - sd).days)
        except Exception:
            return 0
def _busday_offset(start_dt, n):
    def _to_npday(x):
        if hasattr(x, 'date'):
            x = x.date()
        return np.datetime64(x, 'D')
    s = _to_npday(start_dt)
    holidays_np = np.array(_UK_HOLIDAYS, dtype='datetime64[D]') if _UK_HOLIDAYS else None
    return np.busday_offset(s, int(np.ceil(n)), roll='forward', holidays=holidays_np).astype(object)
# --- Advanced forecasting helpers ---
def _monotone_non_decreasing(dates_int):
    # Legacy name kept; see _monotonic_smooth below

    out = dates_int.copy()
    for i in range(1, len(out)):
        if out[i] < out[i-1]:
            out[i] = out[i-1]
    return out


# Backwards-compatible alias used by robust regression

def _theil_sen_slope(x, y):
    slopes = []
    n = len(x)
    for i in range(n):
        for j in range(i+1, n):
            if x[j] != x[i]:
                slopes.append((y[j]-y[i])/(x[j]-x[i]))
    if not slopes:
        return None
    b = float(np.median(slopes))
    a = float(np.median(y - b*x))
    return a, b

def _ols_weighted(x, y, w=None):
    X = np.vstack([np.ones_like(x, dtype=float), x.astype(float)]).T
    if w is not None:
        W = np.diag(w.astype(float))
        Xw = W @ X
        yw = W @ y.astype(float)
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    else:
        beta, *_ = np.linalg.lstsq(X, y.astype(float), rcond=None)
    a, b = float(beta[0]), float(beta[1])
    yhat = a + b*x
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res/ss_tot) if ss_tot > 0 else 0.0
    return a, b, float(r2)

def _weekday_mean_gains(obs_dates, proc_days):
    if len(obs_dates) < 2:
        return {i:0.0 for i in range(5)}
    gains = {i: [] for i in range(5)}
    for i in range(1, len(obs_dates)):
        bd = _busday_count(obs_dates[i-1], obs_dates[i])
        gain = (proc_days[i]-proc_days[i-1])
        if bd <= 0:
            continue
        per_bd = gain / bd
        weekday = obs_dates[i].weekday()
        if weekday < 5:
            gains[weekday].append(per_bd)
    return {k: (float(np.mean(v)) if v else 0.0) for k,v in gains.items()}

def _simulate_weekday_path(current_date, current_gap_days, weekday_means, max_steps=300):
    steps = 0
    d = current_date
    gap = float(current_gap_days)
    while steps < max_steps and gap > 0.0:
        d = _busday_offset(d, 1)
        wd = d.weekday()
        if wd > 4:
            continue
        gain = weekday_means.get(wd, 0.0)
        if gain <= 0:
            gain = max(0.1, gain)
        gap -= gain
        steps += 1
    return d, steps



def _fit_multivariate(x, features, target):
    Xcols = [np.ones_like(x, dtype=float), x.astype(float)]
    names = ["intercept", "x"]
    for name, arr in features.items():
        if arr is None or len(arr) != len(x):
            continue
        Xcols.append(arr.astype(float))
        names.append(name)
    X = np.vstack(Xcols).T
    beta, *_ = np.linalg.lstsq(X, target.astype(float), rcond=None)
    yhat = X @ beta
    ss_res = np.sum((target - yhat)**2)
    ss_tot = np.sum((target - np.mean(target))**2)
    r2 = 1 - (ss_res/ss_tot) if ss_tot > 0 else 0.0
    return beta, names, float(r2)

from collections import defaultdict
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64

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


def create_progression_graph(history, current_date, forecast=None):
    """Erstellt ein Diagramm mit der Progression (x=Beobachtungszeit, y=Verarbeitungsdatum in Tagen seit 1. Jan).
    Gibt einen BytesIO-Puffer (PNG) zur Weitergabe an Telegram zurück.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from io import BytesIO

        changes = history.get("changes", [])
        if len(changes) < 2:
            return None

        # Daten extrahieren
        obs_ts = []
        y_days = []
        for e in changes:
            try:
                ts = datetime.fromisoformat(e["timestamp"])
                y = date_to_days(e.get("date") or e.get("main") or "")
                if y is None:
                    continue
                obs_ts.append(ts)
                y_days.append(y)
            except Exception:
                continue

        if len(obs_ts) < 2:
            return None

        # Sortieren
        order = sorted(range(len(obs_ts)), key=lambda i: obs_ts[i])
        obs_ts = [obs_ts[i] for i in order]
        y_days = [y_days[i] for i in order]

        # Plot
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(obs_ts, y_days, s=36, label='Beobachtungen')

        # y-Achse als Datums-Labels (aus Tage-seit-1.Jan)
        y_ticks = ax.get_yticks()
        y_labels = []
        for y in y_ticks:
            try:
                y_labels.append(days_to_date(int(y)))
            except Exception:
                y_labels.append("")
        ax.set_yticklabels(y_labels)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.xticks(rotation=45)

        ax.set_xlabel('Beobachtungsdatum')
        ax.set_ylabel('Verarbeitungsdatum (Tage seit 1. Jan)')
        ax.set_title('LSE Verarbeitungsdatum - Progression und Prognose')

        # Trendlinie aus EWLS-Modell (falls vorhanden)
        if forecast and isinstance(forecast, dict):
            model = (forecast.get("models") or {}).get("ewls") or {}
            a = model.get("a"); b = model.get("b")
            if a is not None and b is not None and b > 0:
                first_ts = obs_ts[0]
                last_ts = obs_ts[-1]
                # Business-Day Gitter von 0 bis etwas in die Zukunft
                extra = 10
                import numpy as _np
                bd_span = int(_np.busday_count(first_ts.date(), last_ts.date())) + extra
                bd_x = _np.arange(0, max(2, bd_span))
                # y = a + b*x
                y_fit = a + b*bd_x
                # x (Kalender) aus Business Days zurueckrechnen
                x_fit = [ _busday_offset(first_ts, int(n)) for n in bd_x ]
                # Zu datetimes konvertieren
                x_fit_dt = [ datetime(x.year, x.month, x.day, 12, 0) if hasattr(x, 'year') else first_ts for x in x_fit ]
                ax.plot(x_fit_dt, y_fit, 'r--', linewidth=2, alpha=0.7, label=f"Business-Day Fit (R²={forecast.get('r_squared',0):.2f})")

            # Zielmarken einzeichnen, wenn ETA vorhanden
            def _plot_target(label, dstr):
                if not dstr:
                    return
                y_target = date_to_days(label)
                try:
                    eta_date = datetime.fromisoformat(dstr).date()
                    ax.scatter([eta_date], [y_target], marker='X', s=80, label=f"ETA {label}")
                    ax.axvline(eta_date, color='gray', linestyle=':', alpha=0.5)
                except Exception:
                    pass

            bands = forecast.get("eta_bands") or {}
            b25 = bands.get("25_july") or {}
            b28 = bands.get("28_july") or {}

            _plot_target("25 July", b25.get("p50"))
            _plot_target("28 July", b28.get("p50"))

        ax.legend(loc='best', framealpha=0.9)
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print("Fehler beim Erstellen des Diagramms:", e)
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
        
        # Füge neue Felder hinzu wenn nicht vorhanden
        if 'pre_cas_date' not in status:
            status['pre_cas_date'] = None
            print("✅ Status-Migration: pre_cas_date hinzugefügt")
        if 'cas_date' not in status:
            status['cas_date'] = None
            print("✅ Status-Migration: cas_date hinzugefügt")
            
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        print(f"⚠️ Fehler bei Status-Migration: {e}")
    
    # Migriere history.json
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        
        # Füge neue Arrays hinzu wenn nicht vorhanden
        if 'pre_cas_changes' not in history:
            history['pre_cas_changes'] = []
            print("✅ History-Migration: pre_cas_changes hinzugefügt")
        if 'cas_changes' not in history:
            history['cas_changes'] = []
            print("✅ History-Migration: cas_changes hinzugefügt")
            
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"⚠️ Fehler bei History-Migration: {e}")

def load_status():
    """Lädt Status mit Fehlerbehandlung und Validierung"""
    try:
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
            
        # Validiere die geladenen Daten
        if not isinstance(status, dict):
            print("⚠️ Status ist kein Dictionary, verwende Standardwerte")
            return {"last_date": "10 July", "last_check": None, "pre_cas_date": None, "cas_date": None}
            
        if 'last_date' not in status:
            print("⚠️ last_date fehlt in status.json, verwende Standardwert")
            status['last_date'] = "10 July"
        
        # Stelle sicher dass neue Felder existieren
        if 'pre_cas_date' not in status:
            status['pre_cas_date'] = None
        if 'cas_date' not in status:
            status['cas_date'] = None
            
        print(f"✅ Status geladen: {status['last_date']}")
        return status
    except FileNotFoundError:
        print("ℹ️ status.json nicht gefunden, erstelle neue Datei")
        return {"last_date": "10 July", "last_check": None, "pre_cas_date": None, "cas_date": None}
    except json.JSONDecodeError as e:
        print(f"❌ Fehler beim Parsen von status.json: {e}")
        print("Verwende Standardwerte")
        return {"last_date": "10 July", "last_check": None, "pre_cas_date": None, "cas_date": None}
    except Exception as e:
        print(f"❌ Unerwarteter Fehler beim Laden von status.json: {e}")
        return {"last_date": "10 July", "last_check": None, "pre_cas_date": None, "cas_date": None}

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
            return {"changes": [], "pre_cas_changes": [], "cas_changes": []}
            
        if not isinstance(history['changes'], list):
            print("⚠️ History changes ist keine Liste, verwende leere Historie")
            return {"changes": [], "pre_cas_changes": [], "cas_changes": []}
        
        # Stelle sicher dass neue Arrays existieren
        if 'pre_cas_changes' not in history:
            history['pre_cas_changes'] = []
        if 'cas_changes' not in history:
            history['cas_changes'] = []
            
        print(f"✅ Historie geladen: {len(history['changes'])} Änderungen")
        return history
    except FileNotFoundError:
        print("ℹ️ history.json nicht gefunden, erstelle neue Datei")
        return {"changes": [], "pre_cas_changes": [], "cas_changes": []}
    except json.JSONDecodeError as e:
        print(f"❌ Fehler beim Parsen von history.json: {e}")
        return {"changes": [], "pre_cas_changes": [], "cas_changes": []}
    except Exception as e:
        print(f"❌ Unerwarteter Fehler beim Laden von history.json: {e}")
        return {"changes": [], "pre_cas_changes": [], "cas_changes": []}

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


def calculate_regression_forecast(history):

    """Berechnet robuste Regression (Business Days) + Bootstrap-Intervalle fuer die 'all other applicants'-Schlange"""
    half_life = float(os.getenv("REG_HALFLIFE_BDAYS", "10"))
    recent_k  = int(os.getenv("REG_RECENT_K", "10"))
    max_delta = float(os.getenv("REG_MAX_DELTA_PER_BDAY", "7"))
    n_boot    = int(os.getenv("REG_BOOT_N", "300"))

    if len(history.get("changes", [])) < 2:
        return None

    data = []
    first_ts = None
    for entry in history["changes"]:
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            y_val = date_to_days(entry["date"])
            if y_val is None:
                continue
            if first_ts is None:
                first_ts = ts
            x_val = _busday_count(first_ts, ts)
            data.append((x_val, y_val))
        except Exception as e:
            print(f"Fehler beim Verarbeiten von Historie-Eintrag: {e}")
            continue

    if len(data) < 2:
        return None

    data.sort(key=lambda t: t[0])
    x = np.array([d[0] for d in data], dtype=float)
    y = np.array([d[1] for d in data], dtype=float)

    y = _monotonic_smooth(y)

    good_mask = np.ones_like(y, dtype=bool)
    if len(x) >= 2:
        dx = np.diff(x)
        dy = np.diff(y)
        with np.errstate(divide='ignore', invalid='ignore'):
            rate = np.where(dx > 0, dy / dx, 0.0)
        bad_phys = (rate < 0) | (rate > max_delta)
        hw = _hampel_weights(rate, k=3.0)
        robust_w_point = np.ones_like(y, dtype=float)
        robust_w_point[1:] = hw
        good_mask[1:] &= ~bad_phys
        if not np.any(good_mask):
            good_mask[:] = True
            robust_w_point[:] = 1.0
    else:
        robust_w_point = np.ones_like(y, dtype=float)

    xg = x[good_mask]; yg = y[good_mask]; wg = robust_w_point[good_mask]

    if len(xg) < 2:
        return None

    a_ew, b_ew = _ewls(xg, yg, half_life_bdays=half_life, w_robust=wg)
    a_ts, b_ts = _theil_sen(xg, yg)
    a_rw, b_rw = _recent_window_ols(xg, yg, k=recent_k)

    a_arr = np.array([a_ew, a_ts, a_rw], dtype=float)
    b_arr = np.array([b_ew, b_ts, b_rw], dtype=float)

    y_hat = a_ew + b_ew * xg
    ss_res = np.sum((yg - y_hat)**2)
    ss_tot = np.sum((yg - yg.mean())**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    target_25_days = date_to_days("25 July")
    target_28_days = date_to_days("28 July")
    if target_25_days is None or target_28_days is None:
        return None

    current_time = get_german_time()
    def _x_star(a,b,t):
        return (t - a)/b if b > 0 else np.nan
    x25_models = np.array([_x_star(a_arr[i], b_arr[i], target_25_days) for i in range(3)])
    x28_models = np.array([_x_star(a_arr[i], b_arr[i], target_28_days) for i in range(3)])
    x25 = np.nanmedian(x25_models); x28 = np.nanmedian(x28_models)

    if np.isfinite(x25):
        eta25_date = _busday_offset(current_time, x25)
        days_until_25 = (eta25_date - current_time.date()).days
    else:
        eta25_date = None
        days_until_25 = None

    if np.isfinite(x28):
        eta28_date = _busday_offset(current_time, x28)
        days_until_28 = (eta28_date - current_time.date()).days
    else:
        eta28_date = None
        days_until_28 = None

    def _ewls_solver(xb, yb):
        ae, be = _ewls(xb, yb, half_life_bdays=half_life)
        return ae, be

    boot = _bootstrap_eta(xg, yg, _ewls_solver, [target_25_days, target_28_days], b=n_boot)
    def _mk_band(boot_tuple):
        p10, p50, p90 = boot_tuple
        if not np.isfinite(p50):
            return None
        d10 = _busday_offset(current_time, p10) if np.isfinite(p10) else None
        d50 = _busday_offset(current_time, p50) if np.isfinite(p50) else None
        d90 = _busday_offset(current_time, p90) if np.isfinite(p90) else None
        return {"p10": d10.isoformat() if d10 else None,
                "p50": d50.isoformat() if d50 else None,
                "p90": d90.isoformat() if d90 else None}

    bands_25 = _mk_band(boot.get(target_25_days, (np.nan, np.nan, np.nan)))
    bands_28 = _mk_band(boot.get(target_28_days, (np.nan, np.nan, np.nan)))

    slope_rep = float(np.nanmedian(b_arr))
    current_elapsed_bd = _busday_count(first_ts, current_time)
    current_predicted_days = float(a_ew + b_ew * current_elapsed_bd)

    return {
        "slope": slope_rep,
        "r_squared": float(r_squared),
        "current_trend_days": current_predicted_days,
        "days_until_25_july": days_until_25,
        "days_until_28_july": days_until_28,
        "data_points": int(len(xg)),
        "models": {
            "ewls": {"a": float(a_ew), "b": float(b_ew)},
            "theil_sen": {"a": float(a_ts), "b": float(b_ts)},
            "recent_window": {"a": float(a_rw), "b": float(b_rw), "k": int(recent_k)},
        },
        "eta_bands": {
            "25_july": bands_25,
            "28_july": bands_28
        }
    }
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


def create_forecast_text(forecast):

    """Erstellt einen Prognosetext basierend auf der Regression"""
    if not forecast:
        return "\n📊 Prognose: Noch nicht genügend Daten für eine zuverlässige Vorhersage."

    text = "\n📊 PROGNOSE basierend auf bisherigen Änderungen:\n"
    text += f"(Analyse von {forecast['data_points']} Datenpunkten, R²={forecast['r_squared']:.2f})\n\n"

    if forecast['slope'] <= 0:
        text += "⚠️ Die Daten zeigen keinen Fortschritt oder sogar Rückschritte.\n"
    else:
        text += f"📈 Durchschnittlicher Fortschritt (robust): {forecast['slope']:.2f} Tage pro Geschäftstag\n\n"

        if forecast.get('days_until_25_july') is not None and forecast['days_until_25_july'] > 0:
            date_25 = get_german_time() + timedelta(days=forecast['days_until_25_july'])
            text += f"📅 25 July wird voraussichtlich erreicht:\n"
            text += f"   • In {forecast['days_until_25_july']:.0f} Tagen\n"
            text += f"   • Am {date_25.strftime('%a, %d. %B %Y')}\n"
            bands = (forecast.get('eta_bands') or {}).get('25_july')
            if bands and bands.get('p10') and bands.get('p90'):
                d10 = datetime.fromisoformat(bands['p10']).date()
                d90 = datetime.fromisoformat(bands['p90']).date()
                text += f"   • 80%-Intervall: {d10.strftime('%a, %d.%m')} – {d90.strftime('%a, %d.%m')}\n"
            text += "\n"

        if forecast.get('days_until_28_july') is not None and forecast['days_until_28_july'] > 0:
            date_28 = get_german_time() + timedelta(days=forecast['days_until_28_july'])
            text += f"📅 28 July wird voraussichtlich erreicht:\n"
            text += f"   • In {forecast['days_until_28_july']:.0f} Tagen\n"
            text += f"   • Am {date_28.strftime('%a, %d. %B %Y')}\n"
            bands = (forecast.get('eta_bands') or {}).get('28_july')
            if bands and bands.get('p10') and bands.get('p90'):
                d10 = datetime.fromisoformat(bands['p10']).date()
                d90 = datetime.fromisoformat(bands['p90']).date()
                text += f"   • 80%-Intervall: {d10.strftime('%a, %d.%m')} – {d90.strftime('%a, %d.%m')}\n"
            text += "\n"

        if forecast['r_squared'] < 0.5:
            text += "⚠️ Hinweis: Die Vorhersage ist unsicher (niedrige Korrelation).\n"
        elif forecast['r_squared'] > 0.8:
            text += "✅ Die Vorhersage basiert auf einem stabilen Trend.\n"

    return text
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
        
        # Initialisiere Rückgabewerte
        dates = {
            'all_other': None,
            'pre_cas': None,
            'cas': None
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
            # Berechne aktuellen Trend wenn möglich
            forecast = calculate_regression_forecast(history)
            trend_text = ""
            if forecast and forecast['slope'] > 0:
                if forecast['days_until_25_july'] and forecast['days_until_25_july'] > 0:
                    trend_text = f"\n\n📈 <b>Prognose:</b> 25 July in ~{forecast['days_until_25_july']:.0f} Tagen"
            
            telegram_msg = f"""<b>📊 LSE Status Check Ergebnis</b>

<b>Aktuelles Datum:</b> {current_date}
<b>Letzter Stand:</b> {status['last_date']}
<b>Status:</b> {"🔔 ÄNDERUNG ERKANNT!" if current_date != status['last_date'] else "✅ Keine Änderung"}

<b>Zeitpunkt:</b> {get_german_time().strftime('%d.%m.%Y %H:%M:%S')}{trend_text}

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
            forecast_text = create_forecast_text(forecast)
            
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
        print(f"   Fehler beim Lesen des finalen Status: {e}")# ===== Robust regression helpers (appended) =====
def _monotonic_smooth(y):
    y = np.array(y, dtype=float)
    for i in range(1, len(y)):
        if y[i] < y[i-1]:
            y[i] = y[i-1]
    return y

def _pairwise_slopes(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x); sl = []
    for i in range(n):
        for j in range(i+1, n):
            dx = x[j] - x[i]
            if dx != 0:
                sl.append((y[j]-y[i]) / dx)
    return np.array(sl, dtype=float) if sl else np.array([0.0])

def _theil_sen(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    b = np.median(_pairwise_slopes(x, y))
    a = np.median(y - b*x)
    return a, b

def _ols(x, y, w=None):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if w is None:
        n = len(x); Sx = x.sum(); Sy = y.sum()
        Sxx = (x*x).sum(); Sxy = (x*y).sum()
        den = n*Sxx - Sx*Sx
        if den == 0:
            return y.mean(), 0.0
        b = (n*Sxy - Sx*Sy) / den
        a = (Sy - b*Sx) / n
        return a, b
    else:
        w = np.asarray(w, dtype=float); W = w.sum()
        if W == 0:
            return y.mean(), 0.0
        xw = (w*x).sum()/W; yw = (w*y).sum()/W
        Sxx = (w*(x-xw)*(x-xw)).sum()
        if Sxx == 0:
            return yw, 0.0
        b = (w*(x-xw)*(y-yw)).sum()/Sxx
        a = yw - b*xw
        return a, b

def _ewls(x, y, half_life_bdays=10, w_robust=None):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return (y.mean() if len(y) else 0.0), 0.0
    age = x.max() - x
    lam = np.log(2.0) / max(1e-9, half_life_bdays)
    w = np.exp(-lam * age)
    if w_robust is not None:
        w = w * np.asarray(w_robust, dtype=float)
    return _ols(x, y, w=w)

def _recent_window_ols(x, y, k=10):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) <= 1:
        return (y.mean() if len(y) else 0.0), 0.0
    idx = max(0, len(x)-int(k))
    return _ols(x[idx:], y[idx:])

def _hampel_weights(increments, scale=None, k=3.0):
    inc = np.asarray(increments, dtype=float)
    med = np.median(inc)
    mad = np.median(np.abs(inc - med)) if scale is None else scale
    if mad == 0:
        return np.ones_like(inc)
    z = np.abs(inc - med) / (1.4826*mad)
    w = np.ones_like(inc)
    w[z > k] = 0.3
    return w

def _bootstrap_eta(x, y, solver_fn, y_targets, b=300, rng=None):
    rng = np.random.default_rng(rng)
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return {t: (np.nan, np.nan, np.nan) for t in y_targets}
    samples = {t: [] for t in y_targets}
    for _ in range(int(b)):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        order = np.argsort(xb)
        xb, yb = xb[order], yb[order]
        a1,b1 = solver_fn(xb, yb)
        for t in y_targets:
            if b1 <= 0:
                est = np.nan
            else:
                est = (t - a1)/b1
            samples[t].append(est)
    out = {}
    for t, arr in samples.items():
        arr = np.array([v for v in arr if np.isfinite(v)])
        if len(arr) == 0:
            out[t] = (np.nan, np.nan, np.nan)
        else:
            out[t] = (np.percentile(arr, 10), np.percentile(arr, 50), np.percentile(arr, 90))
    return out

if __name__ == '__main__':
    main()
