# lse-status-monitor

# LSE Status Monitor

Ein automatisierter Monitor, der die Bearbeitungszeiten für Graduate-Bewerbungen an der London School of Economics (LSE) überwacht und bei Änderungen Benachrichtigungen versendet.

## 🎯 Funktionsweise

Der Monitor durchläuft drei Phasen:

### Phase 1: Application Tracking
- Überwacht das Verarbeitungsdatum für "all other graduate applicants"
- Ziel: Warten bis das Datum "28 July" erreicht wird
- Bei Erreichen von "28 July" → Automatischer Wechsel zu Phase 2

### Phase 2: Pre-CAS Tracking  
- Startet automatisch wenn "28 July" erreicht wird
- Überwacht das Pre-CAS Ausgabedatum
- Ziel: Pre-CAS muss das Datum erreichen, an dem die Bewerbung fertig bearbeitet wurde

### Phase 3: CAS Tracking
- Startet automatisch kurz bevor Pre-CAS das Zieldatum erreicht
- Überwacht das CAS Ausgabedatum
- Ziel: CAS muss das aktuelle Datum erreichen (dann wird es ausgestellt)

## 📊 Features

- **Automatische Phasenwechsel**: Der Monitor wechselt intelligent zwischen den Tracking-Phasen
- **Prognosen**: Berechnet basierend auf historischen Daten, wann wichtige Meilensteine erreicht werden
- **Multi-Channel Benachrichtigungen**: 
  - Telegram-Nachrichten für alle Updates
  - E-Mail-Benachrichtigungen mit konfigurierbaren Empfängern
- **Intelligente Benachrichtigungslogik**: Unterscheidet zwischen Haupt- und bedingten Empfängern
- **Fehlerbehandlung**: Warnt bei Problemen mit der Webseite

## 📁 Dateistruktur

### `check_lse.py`
Hauptskript mit folgenden Funktionen:
- Web-Scraping der LSE-Webseite
- Datums-Extraktion und -Vergleich
- Benachrichtigungsversand (Telegram & E-Mail)
- Phasenverwaltung und automatische Übergänge
- Lineare Regression für Prognosen
- Historienführung aller Änderungen

### `status.json`
Speichert den aktuellen Zustand:
```json
{
  "last_date": "10 July",
  "last_check": null,
  "phase": "tracking_applications",
  "phase_2_start_date": null,
  "phase_2_target_date": null,
  "last_precas_date": null,
  "phase_3_start_date": null,
  "phase_3_target_date": null,
  "last_cas_date": null
}
```

### `history.json`
Protokolliert alle Änderungen:
```json
{
  "changes": [],        // Phase 1 Änderungen
  "precas_changes": [], // Phase 2 Änderungen
  "cas_changes": []     // Phase 3 Änderungen
}
```

### `monitor.yml`
GitHub Actions Workflow für automatische Ausführung:
- Kann manuell oder per API getriggert werden
- Installiert Abhängigkeiten
- Führt Check aus
- Committed Änderungen zurück ins Repository

## 🔧 Konfiguration

### Erforderliche GitHub Secrets:
- `GMAIL_USER`: Gmail-Adresse für E-Mail-Versand
- `GMAIL_APP_PASSWORD`: App-spezifisches Passwort
- `EMAIL_TO`: Haupt-E-Mail-Empfänger
- `EMAIL_TO_2`: Zweiter E-Mail-Empfänger (optional)
- `EMAIL_TO_3`: Dritter E-Mail-Empfänger (optional)
- `TELEGRAM_BOT_TOKEN`: Token für Telegram Bot
- `TELEGRAM_CHAT_ID`: Ziel-Chat für Telegram-Nachrichten

## 📈 Prognose-Feature

Der Monitor berechnet basierend auf vergangenen Änderungen:
- Durchschnittliche Fortschrittsrate (Tage pro Tag)
- Voraussichtliches Datum für Zielerreichung
- Konfidenzwert (R²) für die Vorhersagegenauigkeit

## 🚀 Verwendung

### Manueller Start:
```bash
python check_lse.py
```

### Automatisierung via GitHub Actions:
- Workflow kann manuell über GitHub UI getriggert werden
- Oder per API-Call an `/repos/{owner}/{repo}/dispatches`

### Automatisierung via Cron-Job:
Der Workflow kann mit einem externen Cron-Service (z.B. cron-job.org) regelmäßig getriggert werden.

## 📋 Anforderungen

- Python 3.10+
- requests
- beautifulsoup4
- numpy

## 🔗 Überwachte URL

https://www.lse.ac.uk/study-at-lse/Graduate/News/Current-processing-times
