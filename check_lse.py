import requests
from bs4 import BeautifulSoup
import json
import os
import re
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import numpy as np
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
    """Erstellt ein Diagramm mit der Progression der Verarbeitungsdaten"""
    if len(history["changes"]) < 2:
        return None
    
    try:
        # Matplotlib Style für bessere Optik
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Extrahiere Datenpunkte
        timestamps = []
        dates_numeric = []
        date_labels = []
        
        for entry in history["changes"]:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            date_days = date_to_days(entry["date"])
            if date_days is not None:
                timestamps.append(timestamp)
                dates_numeric.append(date_days)
                date_labels.append(entry["date"])
        
        # Plot historische Datenpunkte
        ax.scatter(timestamps, dates_numeric, color='darkblue', s=100, zorder=5, label='Historische Daten', alpha=0.8)
        
        # Füge Labels für jeden Punkt hinzu
        for i, (ts, days, label) in enumerate(zip(timestamps, dates_numeric, date_labels)):
            ax.annotate(label, (ts, days), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7)
        
        # Wenn Regression verfügbar ist, zeichne Trendlinie
        if forecast and forecast['slope'] > 0:
            # Erstelle Trendlinie
            first_timestamp = timestamps[0]
            x_days = [(ts - first_timestamp).total_seconds() / 86400 for ts in timestamps]
            
            # Erweitere die Linie in die Zukunft
            future_days = 30  # 30 Tage in die Zukunft projizieren
            last_x = x_days[-1]
            future_x = np.linspace(0, last_x + future_days, 100)
            future_y = forecast['slope'] * future_x + (forecast['current_trend_days'] - forecast['slope'] * ((get_german_time() - first_timestamp).total_seconds() / 86400))
            
            # Konvertiere zurück zu Timestamps für x-Achse
            future_timestamps = [first_timestamp + timedelta(days=d) for d in future_x]
            
            # Plot Trendlinie
            ax.plot(future_timestamps, future_y, 'r--', alpha=0.5, linewidth=2, 
                   label=f'Trend (R²={forecast["r_squared"]:.2f})')
            
            # Markiere Zielpunkte (25. und 28. Juli)
            target_dates = {"25 July": date_to_days("25 July"), "28 July": date_to_days("28 July")}
            
            for target_name, target_days in target_dates.items():
                if target_days:
                    # Finde wann dieses Datum erreicht wird
                    days_until = (target_days - forecast['current_trend_days']) / forecast['slope'] if forecast['slope'] > 0 else None
                    if days_until and days_until > 0:
                        target_timestamp = get_german_time() + timedelta(days=days_until)
                        ax.axhline(y=target_days, color='green', linestyle=':', alpha=0.3)
                        ax.scatter([target_timestamp], [target_days], color='green', s=150, marker='*', 
                                  zorder=6, alpha=0.7)
                        ax.annotate(f'{target_name}\n(~{days_until:.0f} Tage)', 
                                  (target_timestamp, target_days), 
                                  xytext=(10, 0), textcoords='offset points',
                                  fontsize=9, color='green', weight='bold')
        
        # Markiere aktuelles Datum
        current_days = date_to_days(current_date)
        if current_days:
            ax.axhline(y=current_days, color='blue', linestyle='-', alpha=0.3, linewidth=1)
            ax.scatter([get_german_time()], [current_days], color='red', s=150, marker='o', 
                      zorder=7, label=f'Aktuell: {current_date}')
        
        # Formatierung
        ax.set_xlabel('Datum', fontsize=12)
        ax.set_ylabel('Verarbeitungsdatum (Tage seit 1. Januar)', fontsize=12)
        ax.set_title('LSE Verarbeitungsdatum - Progression und Prognose', fontsize=14, weight='bold')
        
        # X-Achse formatieren
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.xticks(rotation=45)
        
        # Y-Achse mit Datum-Labels
        y_ticks = ax.get_yticks()
        y_labels = []
        for y in y_ticks:
            if y >= 0:
                try:
                    y_labels.append(days_to_date(int(y)))
                except:
                    y_labels.append(str(int(y)))
            else:
                y_labels.append('')
        ax.set_yticklabels(y_labels)
        
        # Legende
        ax.legend(loc='upper left', framealpha=0.9)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Layout anpassen
        plt.tight_layout()
        
        # In BytesIO speichern
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"❌ Fehler beim Erstellen des Graphen: {e}")
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

def load_status():
    """Lädt Status mit Fehlerbehandlung und Validierung"""
    try:
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
            
        # Validiere die geladenen Daten
        if not isinstance(status, dict):
            print("⚠️ Status ist kein Dictionary, verwende Standardwerte")
            return {"last_date": "10 July", "last_check": None}
            
        if 'last_date' not in status:
            print("⚠️ last_date fehlt in status.json, verwende Standardwert")
            status['last_date'] = "10 July"
            
        print(f"✅ Status geladen: {status['last_date']}")
        return status
    except FileNotFoundError:
        print("ℹ️ status.json nicht gefunden, erstelle neue Datei")
        return {"last_date": "10 July", "last_check": None}
    except json.JSONDecodeError as e:
        print(f"❌ Fehler beim Parsen von status.json: {e}")
        print("Verwende Standardwerte")
        return {"last_date": "10 July", "last_check": None}
    except Exception as e:
        print(f"❌ Unerwarteter Fehler beim Laden von status.json: {e}")
        return {"last_date": "10 July", "last_check": None}

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
            return {"changes": []}
            
        if not isinstance(history['changes'], list):
            print("⚠️ History changes ist keine Liste, verwende leere Historie")
            return {"changes": []}
            
        print(f"✅ Historie geladen: {len(history['changes'])} Änderungen")
        return history
    except FileNotFoundError:
        print("ℹ️ history.json nicht gefunden, erstelle neue Datei")
        return {"changes": []}
    except json.JSONDecodeError as e:
        print(f"❌ Fehler beim Parsen von history.json: {e}")
        return {"changes": []}
    except Exception as e:
        print(f"❌ Unerwarteter Fehler beim Laden von history.json: {e}")
        return {"changes": []}

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
    """Berechnet eine lineare Regression und Prognose basierend auf der Historie"""
    if len(history["changes"]) < 2:
        return None
    
    # Extrahiere Datenpunkte (Zeit in Tagen seit erstem Eintrag, Datum in Tagen seit 1. Januar)
    data_points = []
    first_timestamp = None
    
    for entry in history["changes"]:
        try:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            date_days = date_to_days(entry["date"])
            
            if date_days is None:
                continue
                
            if first_timestamp is None:
                first_timestamp = timestamp
                
            days_elapsed = (timestamp - first_timestamp).total_seconds() / 86400  # Tage seit erstem Eintrag
            data_points.append((days_elapsed, date_days))
        except Exception as e:
            print(f"⚠️ Fehler beim Verarbeiten von Historie-Eintrag: {e}")
            continue
    
    if len(data_points) < 2:
        return None
    
    try:
        # Lineare Regression
        x = np.array([p[0] for p in data_points])
        y = np.array([p[1] for p in data_points])
        
        # Berechne Steigung und y-Achsenabschnitt
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        intercept = (np.sum(y) - slope * np.sum(x)) / n
        
        # Berechne R² für Qualität der Regression
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Prognose für July 25 und July 28
        target_25_days = date_to_days("25 July")
        target_28_days = date_to_days("28 July")
        
        if target_25_days is None or target_28_days is None:
            return None
        
        # Berechne wann diese Daten erreicht werden
        current_time = get_german_time()
        current_days_elapsed = (current_time - first_timestamp).total_seconds() / 86400
        current_predicted_days = slope * current_days_elapsed + intercept
        
        days_until_25 = (target_25_days - current_predicted_days) / slope if slope > 0 else None
        days_until_28 = (target_28_days - current_predicted_days) / slope if slope > 0 else None
        
        return {
            "slope": slope,
            "r_squared": r_squared,
            "current_trend_days": current_predicted_days,
            "days_until_25_july": days_until_25,
            "days_until_28_july": days_until_28,
            "data_points": len(data_points)
        }
    except Exception as e:
        print(f"❌ Fehler bei der Prognoseberechnung: {e}")
        return None

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

def create_forecast_text(forecast):
    """Erstellt einen Prognosetext basierend auf der Regression"""
    if not forecast:
        return "\n📊 Prognose: Noch nicht genügend Daten für eine zuverlässige Vorhersage."
    
    text = "\n📊 PROGNOSE basierend auf bisherigen Änderungen:\n"
    text += f"(Analyse von {forecast['data_points']} Datenpunkten, R²={forecast['r_squared']:.2f})\n\n"
    
    if forecast['slope'] <= 0:
        text += "⚠️ Die Daten zeigen keinen Fortschritt oder sogar Rückschritte.\n"
    else:
        text += f"📈 Durchschnittlicher Fortschritt: {forecast['slope']:.1f} Tage pro Tag\n\n"
        
        if forecast['days_until_25_july'] is not None and forecast['days_until_25_july'] > 0:
            date_25 = get_german_time() + timedelta(days=forecast['days_until_25_july'])
            text += f"📅 25 July wird voraussichtlich erreicht:\n"
            text += f"   • In {forecast['days_until_25_july']:.0f} Tagen\n"
            text += f"   • Am {date_25.strftime('%d. %B %Y')}\n\n"
        
        if forecast['days_until_28_july'] is not None and forecast['days_until_28_july'] > 0:
            date_28 = get_german_time() + timedelta(days=forecast['days_until_28_july'])
            text += f"📅 28 July wird voraussichtlich erreicht:\n"
            text += f"   • In {forecast['days_until_28_july']:.0f} Tagen\n"
            text += f"   • Am {date_28.strftime('%d. %B %Y')}\n\n"
        
        if forecast['r_squared'] < 0.5:
            text += "⚠️ Hinweis: Die Vorhersage ist unsicher (niedrige Korrelation).\n"
        elif forecast['r_squared'] > 0.8:
            text += "✅ Die Vorhersage basiert auf einem stabilen Trend.\n"
    
    return text

def fetch_processing_date():
    """Holt das aktuelle Verarbeitungsdatum von der LSE-Webseite"""
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
        
        # Suche nach "all other graduate applicants"
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
                            return date
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
                                return date
        
        # Fallback: Textsuche
        full_text = soup.get_text()
        pattern = r'all other graduate applicants[^0-9]*?((?:\d{1,2}\s+\w+\s*)+)'
        match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        
        if match:
            dates_text = match.group(1)
            date = extract_all_other_date(dates_text)
            if date:
                print(f"Datum durch Textsuche gefunden: {date}")
                return date
        
        print("WARNUNG: Konnte das Datum nicht finden!")
        return None
        
    except requests.exceptions.Timeout:
        print("❌ Timeout beim Abrufen der Webseite")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Netzwerkfehler beim Abrufen der Webseite: {e}")
        return None
    except Exception as e:
        print(f"❌ Unerwarteter Fehler beim Abrufen der Webseite: {e}")
        return None

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
    
    # Hole aktuelles Datum
    print("\nRufe LSE-Webseite ab...")
    current_date = fetch_processing_date()
    
    if current_date:
        print(f"Aktuelles Datum für 'all other graduate applicants': {current_date}")
        
        # Bei manuellem Check immer Status senden
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
    except Exception as e:
        print(f"   Fehler beim Lesen des finalen Status: {e}")

if __name__ == "__main__":
    main()
