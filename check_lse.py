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

URL = "https://www.lse.ac.uk/study-at-lse/Graduate/News/Current-processing-times"
STATUS_FILE = "status.json"
HISTORY_FILE = "history.json"

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

def load_status():
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"last_date": "10 July", "last_check": None}

def save_status(status):
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)

def load_history():
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"changes": []}

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def date_to_days(date_str):
    """Konvertiert ein Datum wie '10 July' in Tage seit dem 1. Januar"""
    try:
        # Füge das aktuelle Jahr hinzu
        current_year = datetime.now().year
        date_obj = datetime.strptime(f"{date_str} {current_year}", "%d %B %Y")
        jan_first = datetime(current_year, 1, 1)
        return (date_obj - jan_first).days
    except:
        return None

def days_to_date(days):
    """Konvertiert Tage seit 1. Januar zurück in ein Datum"""
    current_year = datetime.now().year
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
        timestamp = datetime.fromisoformat(entry["timestamp"])
        date_days = date_to_days(entry["date"])
        
        if date_days is None:
            continue
            
        if first_timestamp is None:
            first_timestamp = timestamp
            
        days_elapsed = (timestamp - first_timestamp).total_seconds() / 86400  # Tage seit erstem Eintrag
        data_points.append((days_elapsed, date_days))
    
    if len(data_points) < 2:
        return None
    
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
    current_days_elapsed = (datetime.now() - first_timestamp).total_seconds() / 86400
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

def fetch_processing_date():
    try:
        response = requests.get(URL, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for element in soup.find_all(text=re.compile(r'all other graduate applicants', re.IGNORECASE)):
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
        
    except Exception as e:
        print(f"Fehler beim Abrufen der Webseite: {e}")
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
            date_25 = datetime.now() + timedelta(days=forecast['days_until_25_july'])
            text += f"📅 25 July wird voraussichtlich erreicht:\n"
            text += f"   • In {forecast['days_until_25_july']:.0f} Tagen\n"
            text += f"   • Am {date_25.strftime('%d. %B %Y')}\n\n"
        
        if forecast['days_until_28_july'] is not None and forecast['days_until_28_july'] > 0:
            date_28 = datetime.now() + timedelta(days=forecast['days_until_28_july'])
            text += f"📅 28 July wird voraussichtlich erreicht:\n"
            text += f"   • In {forecast['days_until_28_july']:.0f} Tagen\n"
            text += f"   • Am {date_28.strftime('%d. %B %Y')}\n\n"
        
        if forecast['r_squared'] < 0.5:
            text += "⚠️ Hinweis: Die Vorhersage ist unsicher (niedrige Korrelation).\n"
        elif forecast['r_squared'] > 0.8:
            text += "✅ Die Vorhersage basiert auf einem stabilen Trend.\n"
    
    return text

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
    print(f"LSE Status Check - {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
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
    
    # Lade Status und Historie
    status = load_status()
    history = load_history()
    print(f"Letztes bekanntes Datum: {status['last_date']}")
    
    # Hole aktuelles Datum
    print("\nRufe LSE-Webseite ab...")
    current_date = fetch_processing_date()
    
    if current_date:
        print(f"Aktuelles Datum für 'all other graduate applicants': {current_date}")
        
        if current_date != status['last_date']:
            print("\n🔔 ÄNDERUNG ERKANNT!")
            
            # Speichere in Historie
            history["changes"].append({
                "timestamp": datetime.now().isoformat(),
                "date": current_date,
                "from": status['last_date']
            })
            save_history(history)
            
            # Berechne Prognose
            forecast = calculate_regression_forecast(history)
            forecast_text = create_forecast_text(forecast)
            
            # Erstelle E-Mail-Inhalt
            subject = f"LSE Status Update: Neues Datum {current_date}"
            
            # Basis-E-Mail für alle
            base_body = f"""Das Verarbeitungsdatum für "all other graduate applicants" hat sich geändert!

ÄNDERUNG:
Von: {status['last_date']}
Auf: {current_date}

Zeitpunkt der Erkennung: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}

Link zur Seite: {URL}"""
            
            # E-Mail mit Prognose für Hauptempfänger
            body_with_forecast = base_body + f"\n{forecast_text}\n\nDiese E-Mail wurde automatisch von deinem GitHub Actions Monitor generiert."
            
            # E-Mail ohne Prognose für bedingte Empfänger
            body_simple = base_body + "\n\nDiese E-Mail wurde automatisch von deinem GitHub Actions Monitor generiert."
            
            # Telegram-Nachricht formatieren
            telegram_msg = f"""<b>🔔 LSE Status Update</b>

<b>ÄNDERUNG ERKANNT!</b>
Von: {status['last_date']}
Auf: <b>{current_date}</b>

Zeit: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
{forecast_text}

<a href="{URL}">📄 LSE Webseite öffnen</a>"""
            
            # Sende E-Mails
            emails_sent = False
            
            # Immer benachrichtigen (mit Prognose)
            if always_notify:
                if send_gmail(subject, body_with_forecast, always_notify):
                    emails_sent = True
            
            # Telegram-Nachricht senden
            send_telegram(telegram_msg)
            
            # Bedingt benachrichtigen (nur bei 25 oder 28 July)
            if conditional_notify and current_date in ["25 July", "28 July"]:
                print(f"\n🎯 Zieldatum {current_date} erreicht! Benachrichtige zusätzliche Empfänger.")
                if send_gmail(subject, body_simple, conditional_notify):
                    emails_sent = True
                
                # Spezielle Telegram-Nachricht für Zieldatum
                telegram_special = f"""<b>🎯 ZIELDATUM ERREICHT!</b>

Das Datum <b>{current_date}</b> wurde erreicht!

Dies ist eines der wichtigen Zieldaten für deine LSE-Bewerbung.

<a href="{URL}">📄 Jetzt zur LSE Webseite</a>"""
                send_telegram(telegram_special)
            
            if emails_sent or os.environ.get('TELEGRAM_BOT_TOKEN'):
                # Update Status nur bei erfolgreicher Benachrichtigung
                status['last_date'] = current_date
                status['last_check'] = datetime.now().isoformat()
                save_status(status)
                print("✅ Status wurde aktualisiert.")
            else:
                print("⚠️  Status wurde NICHT aktualisiert (keine Benachrichtigung erfolgreich)")
        else:
            print("✅ Keine Änderung - alles beim Alten.")
            status['last_check'] = datetime.now().isoformat()
            save_status(status)
    else:
        print("\n⚠️  WARNUNG: Konnte das Datum nicht von der Webseite extrahieren!")
        
        # Sende Warnung per E-Mail
        subject = "LSE Monitor WARNUNG: Datum nicht gefunden"
        body = f"""WARNUNG: Der LSE Monitor konnte das Datum nicht von der Webseite extrahieren!

Zeitpunkt: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
Letztes bekanntes Datum: {status['last_date']}

Bitte überprüfe:
1. Ist die Webseite erreichbar? {URL}
2. Hat sich die Struktur der Seite geändert?

Der Monitor wird weiterhin prüfen."""
        
        if always_notify:
            send_gmail(subject, body, always_notify)
        
        # Telegram-Warnung
        telegram_warning = f"""<b>⚠️ LSE Monitor WARNUNG</b>

Konnte das Datum nicht von der Webseite extrahieren!

Letztes bekanntes Datum: <b>{status['last_date']}</b>
Zeit: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}

Mögliche Gründe:
• Webseite nicht erreichbar
• Struktur hat sich geändert
• Netzwerkfehler

<a href="{URL}">📄 Webseite manuell prüfen</a>"""
        
        send_telegram(telegram_warning)
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
