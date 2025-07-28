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
        return {
            "last_date": "10 July",
            "last_check": None,
            "phase": "tracking_applications",
            "phase_2_start_date": None,  # Datum an dem Phase 2 gestartet wurde
            "phase_2_target_date": None,  # Datum das Pre-CAS erreichen muss
            "last_precas_date": None,
            "phase_3_start_date": None,  # Datum an dem Phase 3 gestartet wurde
            "phase_3_target_date": None,  # Datum das CAS erreichen muss
            "last_cas_date": None
        }

def save_status(status):
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)

def load_history():
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"changes": [], "precas_changes": [], "cas_changes": []}

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def date_to_days(date_str):
    """Konvertiert ein Datum wie '10 July' in Tage seit dem 1. Januar"""
    try:
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

def date_str_to_datetime(date_str):
    """Konvertiert '12 August' zu einem datetime Objekt"""
    try:
        current_year = datetime.now().year
        return datetime.strptime(f"{date_str} {current_year}", "%d %B %Y")
    except:
        return None

def calculate_regression_forecast(history, target_type="application"):
    """Berechnet eine lineare Regression und Prognose"""
    changes_key = {
        "application": "changes",
        "precas": "precas_changes", 
        "cas": "cas_changes"
    }[target_type]
    
    if len(history.get(changes_key, [])) < 2:
        return None
    
    data_points = []
    first_timestamp = None
    
    for entry in history[changes_key]:
        timestamp = datetime.fromisoformat(entry["timestamp"])
        date_days = date_to_days(entry["date"])
        
        if date_days is None:
            continue
            
        if first_timestamp is None:
            first_timestamp = timestamp
            
        days_elapsed = (timestamp - first_timestamp).total_seconds() / 86400
        data_points.append((days_elapsed, date_days))
    
    if len(data_points) < 2:
        return None
    
    x = np.array([p[0] for p in data_points])
    y = np.array([p[1] for p in data_points])
    
    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    intercept = (np.sum(y) - slope * np.sum(x)) / n
    
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Berechne aktuelle Position und Prognosen
    current_days_elapsed = (datetime.now() - first_timestamp).total_seconds() / 86400
    current_predicted_days = slope * current_days_elapsed + intercept
    
    if target_type == "application":
        target_25_days = date_to_days("25 July")
        target_28_days = date_to_days("28 July")
        
        days_until_25 = (target_25_days - current_predicted_days) / slope if slope > 0 else None
        days_until_28 = (target_28_days - current_predicted_days) / slope if slope > 0 else None
        
        return {
            "slope": slope,
            "r_squared": r_squared,
            "days_until_25_july": days_until_25,
            "days_until_28_july": days_until_28,
            "data_points": len(data_points)
        }
    else:
        return {
            "slope": slope,
            "r_squared": r_squared,
            "current_predicted_days": current_predicted_days,
            "data_points": len(data_points),
            "days_per_day": slope
        }

def calculate_target_forecast(history, target_date, changes_key):
    """Berechnet wann ein bestimmtes Zieldatum erreicht wird"""
    if len(history.get(changes_key, [])) < 2:
        return None
    
    # Nutze die Regression
    forecast_type = {
        "precas_changes": "precas",
        "cas_changes": "cas"
    }[changes_key]
    
    regression = calculate_regression_forecast(history, forecast_type)
    if not regression or regression['slope'] <= 0:
        return None
    
    target_days = date_to_days(target_date)
    if not target_days:
        return None
    
    days_until_target = (target_days - regression['current_predicted_days']) / regression['slope']
    
    return {
        "target_date": target_date,
        "days_until": days_until_target,
        "expected_date": datetime.now() + timedelta(days=days_until_target) if days_until_target > 0 else None,
        "progress_rate": regression['slope'],
        "r_squared": regression['r_squared']
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

def extract_date_from_cell(text):
    """Extrahiert ein einzelnes Datum aus Text"""
    date_pattern = r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))\b'
    match = re.search(date_pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def fetch_all_dates(soup):
    """Extrahiert alle relevanten Daten aus der Webseite - VERBESSERTE VERSION"""
    dates = {
        "all_other_date": None,
        "precas_date": None,
        "cas_date": None
    }
    
    # Hole den gesamten Text und bereinige ihn
    page_text = soup.get_text()
    # Entferne mehrfache Leerzeichen und Zeilenumbrüche
    page_text = ' '.join(page_text.split())
    
    # Debug-Ausgabe
    print("\nDEBUG: Suche nach Datumsmustern...")
    
    # Pattern 1: Suche nach "all other graduate applicants"
    # Das Pattern berücksichtigt jetzt auch mögliche Sternchen und Leerzeichen
    all_other_pattern = r'all\s+other\s+graduate\s+applicants\s+on\*?\s*\*?\s*(?:\([^)]*\))?\s*\*?\s*(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))'
    match = re.search(all_other_pattern, page_text, re.IGNORECASE)
    if match:
        dates["all_other_date"] = match.group(1).strip()
        print(f"DEBUG: Gefunden all_other_date = {dates['all_other_date']}")
    else:
        # Alternativer Pattern für den Fall, dass das Datum in einer separaten Zelle steht
        # Suche zuerst nach dem Text und dann nach dem nächsten Datum
        if 'all other graduate applicants' in page_text.lower():
            # Finde die Position des Textes
            pos = page_text.lower().find('all other graduate applicants')
            # Suche nach dem nächsten Datum nach dieser Position
            remaining_text = page_text[pos:]
            date_pattern = r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))'
            date_matches = re.finditer(date_pattern, remaining_text, re.IGNORECASE)
            
            # Nimm das erste Datum nach "all other graduate applicants"
            for i, match in enumerate(date_matches):
                if i == 0:  # Erstes Datum nach dem Text
                    dates["all_other_date"] = match.group(1).strip()
                    print(f"DEBUG: Gefunden all_other_date (Alternative) = {dates['all_other_date']}")
                    break
    
    # Pattern 2: Suche nach Pre-CAS
    precas_pattern = r'issuing\s+\*?\*?Pre-CAS\*?\*?\s+[^:]*:\s*(?:[^:]*:)?\s*(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))'
    match = re.search(precas_pattern, page_text, re.IGNORECASE)
    if match:
        dates["precas_date"] = match.group(1).strip()
        print(f"DEBUG: Gefunden precas_date = {dates['precas_date']}")
    else:
        # Alternative: Suche nach "issuing Pre-CAS" und dann dem nächsten Datum
        if 'issuing' in page_text.lower() and 'pre-cas' in page_text.lower():
            # Finde die Position von "Pre-CAS"
            pos = page_text.lower().find('pre-cas')
            if pos != -1:
                remaining_text = page_text[pos:]
                date_pattern = r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))'
                match = re.search(date_pattern, remaining_text, re.IGNORECASE)
                if match:
                    dates["precas_date"] = match.group(1).strip()
                    print(f"DEBUG: Gefunden precas_date (Alternative) = {dates['precas_date']}")
    
    # Pattern 3: Suche nach CAS (aber nicht Pre-CAS)
    cas_pattern = r'issuing\s+\*?\*?CAS\*?\*?\s+to\s+[^:]*:\s*(?:[^:]*:)?\s*(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))'
    match = re.search(cas_pattern, page_text, re.IGNORECASE)
    if match:
        dates["cas_date"] = match.group(1).strip()
        print(f"DEBUG: Gefunden cas_date = {dates['cas_date']}")
    else:
        # Alternative: Suche nach "issuing CAS to" und dann dem nächsten Datum
        cas_text_pattern = r'issuing\s+\*?\*?CAS\*?\*?\s+to\s+offer\s+holders'
        match = re.search(cas_text_pattern, page_text, re.IGNORECASE)
        if match:
            # Position nach dem gefundenen Text
            pos = match.end()
            remaining_text = page_text[pos:]
            date_pattern = r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))'
            date_match = re.search(date_pattern, remaining_text, re.IGNORECASE)
            if date_match:
                dates["cas_date"] = date_match.group(1).strip()
                print(f"DEBUG: Gefunden cas_date (Alternative) = {dates['cas_date']}")
    
    # Zusätzliche Strategie: Suche in Tabellen
    tables = soup.find_all('table')
    if tables and not all(dates.values()):
        print(f"\nDEBUG: Durchsuche {len(tables)} Tabellen...")
        
        for table in tables:
            # Hole alle Zeilen
            rows = table.find_all('tr')
            
            for i in range(len(rows)):
                row_text = rows[i].get_text(separator=' ', strip=True)
                
                # Prüfe ob diese Zeile einen der gesuchten Texte enthält
                if 'all other graduate applicants' in row_text.lower() and not dates["all_other_date"]:
                    # Schaue in der gleichen Zeile und den nächsten Zeilen nach einem Datum
                    for j in range(i, min(i+3, len(rows))):
                        check_text = rows[j].get_text(separator=' ', strip=True)
                        date_pattern = r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))'
                        match = re.search(date_pattern, check_text, re.IGNORECASE)
                        if match:
                            dates["all_other_date"] = match.group(1).strip()
                            print(f"DEBUG: Gefunden in Tabelle all_other_date = {dates['all_other_date']}")
                            break
                
                if 'issuing' in row_text.lower() and 'pre-cas' in row_text.lower() and not dates["precas_date"]:
                    for j in range(i, min(i+3, len(rows))):
                        check_text = rows[j].get_text(separator=' ', strip=True)
                        date_pattern = r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))'
                        match = re.search(date_pattern, check_text, re.IGNORECASE)
                        if match:
                            dates["precas_date"] = match.group(1).strip()
                            print(f"DEBUG: Gefunden in Tabelle precas_date = {dates['precas_date']}")
                            break
                
                if 'issuing' in row_text.lower() and 'cas' in row_text.lower() and 'pre-cas' not in row_text.lower() and not dates["cas_date"]:
                    for j in range(i, min(i+3, len(rows))):
                        check_text = rows[j].get_text(separator=' ', strip=True)
                        date_pattern = r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))'
                        match = re.search(date_pattern, check_text, re.IGNORECASE)
                        if match:
                            dates["cas_date"] = match.group(1).strip()
                            print(f"DEBUG: Gefunden in Tabelle cas_date = {dates['cas_date']}")
                            break
    
    print(f"\nDEBUG: Finale Ergebnisse: {dates}")
    return dates
    
def create_forecast_text(forecast):
    """Erstellt einen Prognosetext für Phase 1"""
    if not forecast:
        return "\n📊 Prognose: Noch nicht genügend Daten für eine zuverlässige Vorhersage."
    
    text = "\n📊 PROGNOSE basierend auf bisherigen Änderungen:\n"
    text += f"(Analyse von {forecast['data_points']} Datenpunkten, R²={forecast['r_squared']:.2f})\n\n"
    
    if forecast['slope'] <= 0:
        text += "⚠️ Die Daten zeigen keinen Fortschritt oder sogar Rückschritte.\n"
    else:
        text += f"📈 Durchschnittlicher Fortschritt: {forecast['slope']:.1f} Tage pro Tag\n\n"
        
        if forecast.get('days_until_25_july') and forecast['days_until_25_july'] > 0:
            date_25 = datetime.now() + timedelta(days=forecast['days_until_25_july'])
            text += f"📅 25 July wird voraussichtlich erreicht:\n"
            text += f"   • In {forecast['days_until_25_july']:.0f} Tagen\n"
            text += f"   • Am {date_25.strftime('%d. %B %Y')}\n\n"
        
        if forecast.get('days_until_28_july') and forecast['days_until_28_july'] > 0:
            date_28 = datetime.now() + timedelta(days=forecast['days_until_28_july'])
            text += f"📅 28 July wird voraussichtlich erreicht:\n"
            text += f"   • In {forecast['days_until_28_july']:.0f} Tagen\n"
            text += f"   • Am {date_28.strftime('%d. %B %Y')}\n\n"
            text += "⚠️ Bei 28 July wechselt der Monitor zu Pre-CAS Tracking!\n"
        
        if forecast['r_squared'] < 0.5:
            text += "\n⚠️ Hinweis: Die Vorhersage ist unsicher (niedrige Korrelation).\n"
        elif forecast['r_squared'] > 0.8:
            text += "\n✅ Die Vorhersage basiert auf einem stabilen Trend.\n"
    
    return text

def create_target_forecast_text(forecast, phase):
    """Erstellt Prognosetext für Phase 2 oder 3"""
    if not forecast:
        return "\n📊 Prognose: Noch nicht genügend Daten."
    
    phase_name = "Pre-CAS" if phase == 2 else "CAS"
    
    text = f"\n📊 {phase_name.upper()} PROGNOSE:\n"
    text += f"Zieldatum: {forecast['target_date']}\n"
    
    if forecast['days_until'] and forecast['days_until'] > 0:
        text += f"\n📈 Fortschrittsrate: {forecast['progress_rate']:.1f} Tage/Tag\n"
        text += f"📅 {phase_name} für dich wird voraussichtlich erreicht:\n"
        text += f"   • In {forecast['days_until']:.0f} Tagen\n"
        text += f"   • Am {forecast['expected_date'].strftime('%d. %B %Y')}\n"
        
        if forecast['r_squared'] < 0.5:
            text += "\n⚠️ Vorhersage unsicher (R²={:.2f})".format(forecast['r_squared'])
        elif forecast['r_squared'] > 0.8:
            text += "\n✅ Stabile Vorhersage (R²={:.2f})".format(forecast['r_squared'])
    else:
        text += f"\n✅ {phase_name} sollte bereits für dich ausgestellt sein!"
    
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
    print(f"Nur bei Phasenwechsel: {', '.join(conditional_notify)}")
    
    # Lade Status und Historie
    status = load_status()
    history = load_history()
    
    # Stelle sicher, dass neue Felder existieren
    if "phase" not in status:
        status["phase"] = "tracking_applications"
    if "precas_changes" not in history:
        history["precas_changes"] = []
    if "cas_changes" not in history:
        history["cas_changes"] = []
    
    print(f"\nAktuelle Phase: {status['phase']}")
    print("="*30)
    
    # Hole Daten von der Webseite
    print("\nRufe LSE-Webseite ab...")
    try:
        response = requests.get(URL, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        dates = fetch_all_dates(soup)
        
        print(f"Gefundene Daten:")
        print(f"  All other graduates: {dates['all_other_date']}")
        print(f"  Pre-CAS: {dates['precas_date']}")
        print(f"  CAS: {dates['cas_date']}")
        
    except Exception as e:
        print(f"Fehler beim Abrufen der Webseite: {e}")
        
        # Sende Warnung
        subject = "LSE Monitor WARNUNG: Webseite nicht erreichbar"
        body = f"""WARNUNG: Der LSE Monitor konnte die Webseite nicht abrufen!

Fehler: {str(e)}
Zeit: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}

Link: {URL}"""
        
        if always_notify:
            send_gmail(subject, body, always_notify)
        
        telegram_warning = f"""<b>⚠️ LSE Monitor WARNUNG</b>

Webseite nicht erreichbar!
Fehler: {str(e)}

<a href="{URL}">📄 Webseite manuell prüfen</a>"""
        
        send_telegram(telegram_warning)
        return
    
    # PHASE 1: Tracking Applications
    if status["phase"] == "tracking_applications":
        print(f"\n--- PHASE 1: Application Tracking ---")
        print(f"Letztes bekanntes Datum: {status['last_date']}")
        print(f"Aktuelles Datum: {dates['all_other_date']}")
        
        if dates['all_other_date'] and dates['all_other_date'] != status['last_date']:
            print("\n🔔 ÄNDERUNG ERKANNT!")
            
            # Speichere in Historie
            history["changes"].append({
                "timestamp": datetime.now().isoformat(),
                "date": dates['all_other_date'],
                "from": status['last_date']
            })
            save_history(history)
            
            # Berechne Prognose
            forecast = calculate_regression_forecast(history, "application")
            forecast_text = create_forecast_text(forecast)
            
            # E-Mail an Hauptempfänger (immer)
            subject = f"LSE Status Update: Neues Datum {dates['all_other_date']}"
            base_body = f"""Das Verarbeitungsdatum für "all other graduate applicants" hat sich geändert!

ÄNDERUNG:
Von: {status['last_date']}
Auf: {dates['all_other_date']}

Zeitpunkt: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}

Link: {URL}"""
            
            body_with_forecast = base_body + f"\n{forecast_text}"
            
            if always_notify:
                send_gmail(subject, body_with_forecast, always_notify)
            
            # Telegram immer
            telegram_msg = f"""<b>🔔 LSE Status Update</b>

<b>ÄNDERUNG ERKANNT!</b>
Von: {status['last_date']}
Auf: <b>{dates['all_other_date']}</b>

{forecast_text}

<a href="{URL}">📄 LSE Webseite</a>"""
            
            send_telegram(telegram_msg)
            
            # Spezialbehandlung für 25/28 July für Ulli
            if conditional_notify and dates['all_other_date'] in ["25 July", "28 July"]:
                special_subject = f"LSE: Wichtiges Datum {dates['all_other_date']} erreicht"
                special_body = base_body + "\n\nDies ist eines der wichtigen Zieldaten."
                send_gmail(special_subject, special_body, conditional_notify)
            
            # Prüfe ob wir zu Phase 2 wechseln müssen
            # Wenn morgen 29 July erreicht wird, starte Phase 2 heute
            if dates['all_other_date'] == "28 July":
                print("\n🎯 28 JULY ERREICHT! Phase 2 startet HEUTE!")
                status["phase"] = "tracking_precas"
                status["phase_2_start_date"] = datetime.now().strftime("%d %B")
                # Das Zieldatum für Pre-CAS ist das Datum an dem 29 July erscheinen wird (morgen)
                tomorrow = datetime.now() + timedelta(days=1)
                status["phase_2_target_date"] = tomorrow.strftime("%d %B").lstrip("0")
                status["last_precas_date"] = dates['precas_date']
                
                phase_msg = f"""<b>🎉 PHASE 2 BEGINNT!</b>

28 July wurde erreicht. Morgen wird 29 July angezeigt.

Ab jetzt trackt der Monitor das <b>Pre-CAS</b> Datum.

Ziel: Pre-CAS muss <b>{status['phase_2_target_date']}</b> erreichen.
Aktueller Pre-CAS Stand: <b>{dates['precas_date']}</b>

Dies bedeutet, dass deine Dokumente morgen fertig bearbeitet sein werden und du dann für Pre-CAS berechtigt bist."""
                
                send_telegram(phase_msg)
                
                # Benachrichtige auch Ulli über Phasenwechsel
                phase_email_subject = "LSE Monitor: Phase 2 (Pre-CAS Tracking) beginnt"
                phase_email_body = f"""Der LSE Monitor wechselt zu Phase 2!

28 July wurde bei "all other graduate applicants" erreicht.
Morgen (29 July) werden deine Dokumente fertig bearbeitet sein.

Der Monitor trackt nun den Pre-CAS Fortschritt.
Zieldatum: {status['phase_2_target_date']}
Aktueller Stand: {dates['precas_date']}

Link: {URL}"""
                
                all_recipients = list(set(always_notify + conditional_notify))
                if all_recipients:
                    send_gmail(phase_email_subject, phase_email_body, all_recipients)
            
            # Update Status
            status['last_date'] = dates['all_other_date']
            status['last_check'] = datetime.now().isoformat()
            save_status(status)
            
        else:
            print("✅ Keine Änderung.")
            status['last_check'] = datetime.now().isoformat()
            save_status(status)
    
    # PHASE 2: Tracking Pre-CAS
    elif status["phase"] == "tracking_precas":
        print(f"\n--- PHASE 2: Pre-CAS Tracking ---")
        print(f"Phase 2 gestartet am: {status['phase_2_start_date']}")
        print(f"Pre-CAS Zieldatum: {status['phase_2_target_date']}")
        print(f"Letztes Pre-CAS: {status.get('last_precas_date', 'Unbekannt')}")
        print(f"Aktuelles Pre-CAS: {dates['precas_date']}")
        
        if dates['precas_date'] and dates['precas_date'] != status.get('last_precas_date'):
            print("\n🔔 PRE-CAS ÄNDERUNG!")
            
            # Speichere Änderung
            history["precas_changes"].append({
                "timestamp": datetime.now().isoformat(),
                "date": dates['precas_date'],
                "from": status.get('last_precas_date', 'Unbekannt')
            })
            save_history(history)
            
            # Berechne Prognose
            precas_forecast = calculate_target_forecast(history, status['phase_2_target_date'], "precas_changes")
            forecast_text = create_target_forecast_text(precas_forecast, 2)
            
            # Benachrichtigungen an Hauptempfänger
            subject = f"LSE Pre-CAS Update: {dates['precas_date']}"
            body = f"""Pre-CAS Fortschritt!

ÄNDERUNG:
Von: {status.get('last_precas_date', 'Unbekannt')}
Auf: {dates['precas_date']}

Zieldatum: {status['phase_2_target_date']} (dann bist du Pre-CAS berechtigt)
{forecast_text}

Link: {URL}"""
            
            if always_notify:
                send_gmail(subject, body, always_notify)
            
            # Telegram
            telegram_msg = f"""<b>📋 Pre-CAS Update</b>

<b>Fortschritt!</b>
Von: {status.get('last_precas_date', 'Unbekannt')}
Auf: <b>{dates['precas_date']}</b>

Ziel: <b>{status['phase_2_target_date']}</b>
{forecast_text}

<a href="{URL}">📄 LSE Webseite</a>"""
            
            send_telegram(telegram_msg)
            
            # Prüfe ob wir das Zieldatum fast erreicht haben
            target_date_obj = date_str_to_datetime(status['phase_2_target_date'])
            current_precas_obj = date_str_to_datetime(dates['precas_date'])
            
            if target_date_obj and current_precas_obj:
                days_diff = (target_date_obj - current_precas_obj).days
                
                if days_diff == 1:
                    # Morgen wird das Zieldatum erreicht, starte Phase 3 heute
                    print("\n🎯 PRE-CAS FAST AM ZIEL! Phase 3 startet HEUTE!")
                    status["phase"] = "tracking_cas"
                    status["phase_3_start_date"] = datetime.now().strftime("%d %B")
                    status["phase_3_target_date"] = status["phase_3_start_date"]  # CAS muss heute's Datum erreichen
                    status["last_cas_date"] = dates['cas_date']
                    
                    phase_msg = f"""<b>🎊 PHASE 3 BEGINNT!</b>

Pre-CAS wird morgen dein Zieldatum ({status['phase_2_target_date']}) erreichen!

Das bedeutet: <b>Dein Pre-CAS wird morgen ausgestellt!</b>

Ab jetzt trackt der Monitor das <b>CAS</b> Datum.

Ziel: CAS muss <b>{status['phase_3_target_date']}</b> erreichen.
Aktueller CAS Stand: <b>{dates['cas_date']}</b>"""
                    
                    send_telegram(phase_msg)
                    
                    # Benachrichtige alle über Phasenwechsel
                    phase_email_subject = "LSE Monitor: Phase 3 (CAS Tracking) beginnt - Pre-CAS kommt morgen!"
                    phase_email_body = f"""Großartige Neuigkeiten!

Dein Pre-CAS wird morgen ausgestellt!

Der Monitor wechselt zu Phase 3 (CAS Tracking).
CAS Zieldatum: {status['phase_3_target_date']}
Aktueller Stand: {dates['cas_date']}

Link: {URL}"""
                    
                    all_recipients = list(set(always_notify + conditional_notify))
                    if all_recipients:
                        send_gmail(phase_email_subject, phase_email_body, all_recipients)
            
            # Update Status
            status['last_precas_date'] = dates['precas_date']
            status['last_check'] = datetime.now().isoformat()
            save_status(status)
            
        else:
            print("✅ Keine Pre-CAS Änderung.")
            status['last_check'] = datetime.now().isoformat()
            save_status(status)
    
    # PHASE 3: Tracking CAS
    elif status["phase"] == "tracking_cas":
        print(f"\n--- PHASE 3: CAS Tracking ---")
        print(f"Phase 3 gestartet am: {status['phase_3_start_date']}")
        print(f"CAS Zieldatum: {status['phase_3_target_date']}")
        print(f"Letztes CAS: {status.get('last_cas_date', 'Unbekannt')}")
        print(f"Aktuelles CAS: {dates['cas_date']}")
        
        if dates['cas_date'] and dates['cas_date'] != status.get('last_cas_date'):
            print("\n🔔 CAS ÄNDERUNG!")
            
            # Speichere Änderung
            history["cas_changes"].append({
                "timestamp": datetime.now().isoformat(),
                "date": dates['cas_date'],
                "from": status.get('last_cas_date', 'Unbekannt')
            })
            save_history(history)
            
            # Berechne Prognose
            cas_forecast = calculate_target_forecast(history, status['phase_3_target_date'], "cas_changes")
            forecast_text = create_target_forecast_text(cas_forecast, 3)
            
            # Prüfe ob CAS das Zieldatum erreicht hat
            if dates['cas_date'] == status['phase_3_target_date']:
                print("\n🎉🎉🎉 DEIN CAS IST DA! 🎉🎉🎉")
                
                celebration_msg = f"""<b>🎉🎉🎉 DEIN CAS IST DA! 🎉🎉🎉</b>

CAS hat dein Zieldatum erreicht: <b>{status['phase_3_target_date']}</b>

<b>HERZLICHEN GLÜCKWUNSCH!</b>

Du kannst jetzt dein Visum beantragen!

Der gesamte Prozess ist abgeschlossen:
✅ Dokumente bearbeitet (29 July)
✅ Pre-CAS erhalten
✅ CAS erhalten

<a href="{URL}">📄 LSE Webseite</a>"""
                
                send_telegram(celebration_msg)
                
                # Sende an ALLE
                celebration_subject = "🎉 LSE CAS IST DA! 🎉"
                celebration_body = f"""FANTASTISCHE NEUIGKEITEN!

Dein CAS wurde ausgestellt!

Der komplette Prozess ist abgeschlossen:
✅ Dokumente bearbeitet
✅ Pre-CAS erhalten  
✅ CAS erhalten

Du kannst jetzt dein Visum beantragen!

HERZLICHEN GLÜCKWUNSCH! 🎊

Link: {URL}"""
                
                all_recipients = list(set(always_notify + conditional_notify))
                if all_recipients:
                    send_gmail(celebration_subject, celebration_body, all_recipients)
            else:
                # Normale Update-Benachrichtigung
                subject = f"LSE CAS Update: {dates['cas_date']}"
                body = f"""CAS Fortschritt!

ÄNDERUNG:
Von: {status.get('last_cas_date', 'Unbekannt')}
Auf: {dates['cas_date']}

Zieldatum: {status['phase_3_target_date']} (dann erhältst du dein CAS)
{forecast_text}

Link: {URL}"""
                
                if always_notify:
                    send_gmail(subject, body, always_notify)
                
                telegram_msg = f"""<b>🎫 CAS Update</b>

<b>Fortschritt!</b>
Von: {status.get('last_cas_date', 'Unbekannt')}
Auf: <b>{dates['cas_date']}</b>

Ziel: <b>{status['phase_3_target_date']}</b>
{forecast_text}

<a href="{URL}">📄 LSE Webseite</a>"""
                
                send_telegram(telegram_msg)
            
            # Update Status
            status['last_cas_date'] = dates['cas_date']
            status['last_check'] = datetime.now().isoformat()
            save_status(status)
            
        else:
            print("✅ Keine CAS Änderung.")
            status['last_check'] = datetime.now().isoformat()
            save_status(status)
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
