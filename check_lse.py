import requests
from bs4 import BeautifulSoup
import json
import os
import re
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

URL = "https://www.lse.ac.uk/study-at-lse/Graduate/News/Current-processing-times"
STATUS_FILE = "status.json"

def load_status():
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"last_date": "10 July", "last_check": None}

def save_status(status):
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)

def extract_all_other_date(text):
    """Extrahiert nur das Datum für 'all other graduate applicants'"""
    # Entferne überflüssige Whitespaces und Zeilenumbrüche
    text = ' '.join(text.split())
    
    # Suche nach allen Datumsangaben in der Form "DD Month"
    date_pattern = r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))\b'
    all_dates = re.findall(date_pattern, text, re.IGNORECASE)
    
    # Wenn wir genau 3 Daten haben, nehmen wir das letzte (für "all other graduate applicants")
    if len(all_dates) >= 3:
        return all_dates[-1].strip()
    elif len(all_dates) > 0:
        # Falls weniger als 3 Daten, nehme das letzte verfügbare
        return all_dates[-1].strip()
    
    return None

def fetch_processing_date():
    try:
        response = requests.get(URL, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Suche nach der relevanten Tabellenzelle
        # Methode 1: Suche nach dem Text "all other graduate applicants"
        for element in soup.find_all(text=re.compile(r'all other graduate applicants', re.IGNORECASE)):
            # Finde das übergeordnete TD/TH Element
            parent = element.parent
            while parent and parent.name not in ['td', 'th', 'tr']:
                parent = parent.parent
            
            if parent:
                # Suche in der gleichen Zeile nach der Datumszelle
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
                    # Schaue in Geschwister-Elementen
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
        
        # Methode 2: Falls Methode 1 fehlschlägt, suche im gesamten Text
        full_text = soup.get_text()
        # Suche nach dem Muster "all other graduate applicants" gefolgt von Daten
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

def send_gmail(subject, body):
    """Sendet E-Mail über Gmail"""
    gmail_user = os.environ.get('GMAIL_USER')
    gmail_password = os.environ.get('GMAIL_APP_PASSWORD')
    to_email = os.environ.get('EMAIL_TO')
    
    if not all([gmail_user, gmail_password, to_email]):
        print("E-Mail-Konfiguration unvollständig!")
        print(f"GMAIL_USER vorhanden: {'Ja' if gmail_user else 'Nein'}")
        print(f"GMAIL_APP_PASSWORD vorhanden: {'Ja' if gmail_password else 'Nein'}")
        print(f"EMAIL_TO vorhanden: {'Ja' if to_email else 'Nein'}")
        return False
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = gmail_user
    msg['To'] = to_email
    
    try:
        print(f"Verbinde mit Gmail SMTP Server...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        print(f"Anmeldung als {gmail_user}...")
        server.login(gmail_user, gmail_password)
        print(f"Sende E-Mail an {to_email}...")
        server.send_message(msg)
        server.quit()
        print("✅ E-Mail erfolgreich gesendet!")
        return True
    except smtplib.SMTPAuthenticationError:
        print("❌ FEHLER: Gmail Login fehlgeschlagen. Überprüfe das App-Passwort!")
        return False
    except Exception as e:
        print(f"❌ E-Mail-Fehler: {type(e).__name__}: {e}")
        return False

def main():
    print("="*50)
    print(f"LSE Status Check - {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print("="*50)
    
    # Lade gespeicherten Status
    status = load_status()
    print(f"Letztes bekanntes Datum: {status['last_date']}")
    
    # Hole aktuelles Datum von der Webseite
    print("\nRufe LSE-Webseite ab...")
    current_date = fetch_processing_date()
    
    if current_date:
        print(f"Aktuelles Datum für 'all other graduate applicants': {current_date}")
        
        if current_date != status['last_date']:
            print("\n🔔 ÄNDERUNG ERKANNT!")
            
            subject = f"LSE Status Update: Neues Datum {current_date}"
            body = f"""Das Verarbeitungsdatum für "all other graduate applicants" hat sich geändert!

ÄNDERUNG:
Von: {status['last_date']}
Auf: {current_date}

Zeitpunkt der Erkennung: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}

Link zur Seite: {URL}

Diese E-Mail wurde automatisch von deinem GitHub Actions Monitor generiert.
Der Monitor überprüft die Seite alle 5 Minuten und erfasst nur das Datum für "all other graduate applicants"."""
            
            # Sende E-Mail
            if send_gmail(subject, body):
                # Update Status nur bei erfolgreicher E-Mail
                status['last_date'] = current_date
                status['last_check'] = datetime.now().isoformat()
                save_status(status)
                print("✅ Status wurde aktualisiert.")
            else:
                print("⚠️  Status wurde NICHT aktualisiert (E-Mail fehlgeschlagen)")
        else:
            print("✅ Keine Änderung - alles beim Alten.")
            status['last_check'] = datetime.now().isoformat()
            save_status(status)
    else:
        print("\n⚠️  WARNUNG: Konnte das Datum nicht von der Webseite extrahieren!")
        print("Mögliche Gründe:")
        print("- Die Webseite ist nicht erreichbar")
        print("- Die Struktur der Webseite hat sich geändert")
        print("- Netzwerkfehler")
        
        # Sende Warnung per E-Mail
        subject = "LSE Monitor WARNUNG: Datum nicht gefunden"
        body = f"""WARNUNG: Der LSE Monitor konnte das Datum nicht von der Webseite extrahieren!

Zeitpunkt: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
Letztes bekanntes Datum: {status['last_date']}

Bitte überprüfe:
1. Ist die Webseite erreichbar? {URL}
2. Hat sich die Struktur der Seite geändert?

Der Monitor wird weiterhin alle 5 Minuten prüfen."""
        
        send_gmail(subject, body)
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
