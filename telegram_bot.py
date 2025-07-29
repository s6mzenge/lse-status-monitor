import os
import json
import requests
from datetime import datetime
import time

# Konstanten
STATUS_FILE = "status.json"
HISTORY_FILE = "history.json"
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
LAST_UPDATE_ID_FILE = "last_update_id.json"

def load_json_file(filename):
    """Lädt JSON-Datei und gibt Inhalt zurück"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return None

def save_last_update_id(update_id):
    """Speichert die letzte verarbeitete Update-ID"""
    with open(LAST_UPDATE_ID_FILE, 'w') as f:
        json.dump({"last_update_id": update_id}, f)

def get_last_update_id():
    """Holt die letzte verarbeitete Update-ID"""
    data = load_json_file(LAST_UPDATE_ID_FILE)
    return data.get("last_update_id", 0) if data else 0

def send_telegram_message(text):
    """Sendet eine Nachricht über Telegram Bot"""
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram nicht konfiguriert")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=data)
        return response.status_code == 200
    except Exception as e:
        print(f"Fehler beim Senden der Telegram-Nachricht: {e}")
        return False

def get_telegram_updates(offset=None):
    """Holt neue Nachrichten vom Telegram Bot"""
    if not BOT_TOKEN:
        return None
    
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params = {"timeout": 1}  # Kurzes Timeout für GitHub Actions
        if offset:
            params["offset"] = offset
            
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Fehler beim Abrufen der Updates: {e}")
        return None

def format_status_message(status, history):
    """Formatiert die Status-Nachricht"""
    if not status:
        return "❌ Konnte Status-Datei nicht lesen!"
    
    last_date = status.get('last_date', 'Unbekannt')
    last_check = status.get('last_check', 'Nie')
    
    # Zeitstempel formatieren
    if last_check != 'Nie':
        try:
            check_time = datetime.fromisoformat(last_check)
            time_diff = datetime.now() - check_time
            
            # Zeitdifferenz in lesbarem Format
            if time_diff.total_seconds() < 60:
                time_ago = f"{int(time_diff.total_seconds())} Sekunden"
            elif time_diff.total_seconds() < 3600:
                time_ago = f"{int(time_diff.total_seconds() / 60)} Minuten"
            elif time_diff.total_seconds() < 86400:
                time_ago = f"{int(time_diff.total_seconds() / 3600)} Stunden"
            else:
                time_ago = f"{int(time_diff.total_seconds() / 86400)} Tage"
                
            formatted_time = check_time.strftime('%d.%m.%Y %H:%M:%S')
        except:
            time_ago = "Unbekannt"
            formatted_time = last_check
    else:
        time_ago = "Noch nie"
        formatted_time = "Noch nie"
    
    # Historie analysieren
    change_count = len(history.get('changes', [])) if history else 0
    last_change = None
    if history and history.get('changes'):
        last_change = history['changes'][-1]
    
    # Nachricht erstellen
    message = f"""<b>📊 LSE Status Update</b>

<b>Aktuelles Datum:</b> {last_date}
<b>Letzte Prüfung:</b> vor {time_ago}
<i>({formatted_time})</i>

<b>📈 Statistik:</b>
• Änderungen erkannt: {change_count}"""
    
    if last_change:
        change_time = datetime.fromisoformat(last_change['timestamp'])
        days_since_change = (datetime.now() - change_time).days
        message += f"\n• Letzte Änderung: vor {days_since_change} Tagen"
        message += f"\n  Von: {last_change['from']} → {last_change['date']}"
    
    # Fortschrittsindikator
    if last_date == "10 July":
        message += "\n\n⏳ <i>Noch keine Bewegung erkannt</i>"
    elif last_date in ["25 July", "28 July"]:
        message += f"\n\n🎯 <b>Zieldatum {last_date} erreicht!</b>"
    else:
        # Fortschrittsbalken berechnen (10 July = 0%, 28 July = 100%)
        date_mapping = {
            "10 July": 0, "11 July": 6, "12 July": 11, "13 July": 17,
            "14 July": 22, "15 July": 28, "16 July": 33, "17 July": 39,
            "18 July": 44, "19 July": 50, "20 July": 56, "21 July": 61,
            "22 July": 67, "23 July": 72, "24 July": 78, "25 July": 83,
            "26 July": 89, "27 July": 94, "28 July": 100
        }
        progress = date_mapping.get(last_date, 0)
        filled = int(progress / 10)
        empty = 10 - filled
        progress_bar = "▓" * filled + "░" * empty
        message += f"\n\n📊 Fortschritt: [{progress_bar}] {progress}%"
    
    message += "\n\n<a href='https://www.lse.ac.uk/study-at-lse/Graduate/News/Current-processing-times'>📄 LSE Webseite</a>"
    
    return message

def process_update_command():
    """Verarbeitet den Update-Befehl"""
    # Lade Status und Historie
    status = load_json_file(STATUS_FILE)
    history = load_json_file(HISTORY_FILE)
    
    # Erstelle und sende Nachricht
    message = format_status_message(status, history)
    return send_telegram_message(message)

def main():
    """Hauptfunktion - prüft auf neue Telegram-Nachrichten"""
    print(f"=== Telegram Bot Check - {datetime.now().strftime('%d.%m.%Y %H:%M:%S')} ===")
    
    if not BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN nicht konfiguriert!")
        return
    
    # Hole letzte Update-ID
    last_update_id = get_last_update_id()
    print(f"Letzte verarbeitete Update-ID: {last_update_id}")
    
    # Hole neue Updates
    updates = get_telegram_updates(offset=last_update_id + 1)
    
    if not updates or not updates.get('ok'):
        print("Keine neuen Updates oder Fehler beim Abrufen")
        return
    
    new_updates = updates.get('result', [])
    print(f"Neue Updates gefunden: {len(new_updates)}")
    
    # Verarbeite jedes Update
    for update in new_updates:
        update_id = update.get('update_id')
        message = update.get('message', {})
        text = message.get('text', '').strip()
        chat_id = message.get('chat', {}).get('id')
        
        print(f"Update {update_id}: '{text}' von Chat {chat_id}")
        
        # Prüfe ob die Nachricht von unserem konfigurierten Chat kommt
        if str(chat_id) == CHAT_ID:
            # Prüfe auf "Update" Befehl (case-insensitive)
            if text.lower() in ['update', '/update', 'status', '/status']:
                print("Update-Befehl erkannt! Sende Status...")
                if process_update_command():
                    print("✅ Status erfolgreich gesendet!")
                else:
                    print("❌ Fehler beim Senden des Status!")
            elif text.lower() in ['help', '/help', 'hilfe', '/hilfe']:
                help_message = """<b>🤖 LSE Monitor Bot Befehle:</b>

<b>/update</b> oder <b>Update</b>
Zeigt den aktuellen Status der LSE-Bearbeitung

<b>/help</b> oder <b>Help</b>
Zeigt diese Hilfe an

<i>Der Bot prüft automatisch alle 10 Minuten die LSE-Webseite und benachrichtigt dich bei Änderungen.</i>"""
                send_telegram_message(help_message)
                print("✅ Hilfe-Nachricht gesendet!")
        
        # Speichere die Update-ID
        save_last_update_id(update_id)
    
    print("=== Bot-Check abgeschlossen ===\n")

if __name__ == "__main__":
    main()
