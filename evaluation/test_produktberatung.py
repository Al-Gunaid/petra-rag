#!/usr/bin/env python3
# ============================================================
# test_produktberatung.py — Abnahmetest der Produktberatung (Stand Schritt 7)
#
# Prüft jeden Fragetyp (Q1–Q5) und die zugehörigen Use Cases / Anforderungen
# über denselben Weg wie der Chat: POST /query (FastAPI) -> n8n -> Antwort,
# mit den Standardwerten der Einstellungsseite. Referenzwerte stammen aus dem
# Testset (testfaelle.json, Fall-Nr. in Klammern) bzw. aus belegten
# Systembefunden (Preisliste).
#
#   Q1 Produktspezifikation  UC-01, F-13     P01–P04
#   Q2 Produktvergleich      UC-02, UC-03    P05–P07
#   Q3 Produktsuche          Kap. 4.4.1 Q3   P08–P09
#   Q4 Preisanfrage          UC-04, UC-11    P10–P12
#   Q5 Allgemeine Frage      Kap. 4.4.1 Q5   P13–P14
#   Negativfälle             F-13            P15–P16
#   Sprache                  NF-08, F-08, UC-01 A5   P17–P19, P21
#   Web-Agent                UC-11           P22
#   Cache / Latenz           Kap. 4.2 [1c], NF-03   P20
#   Quellenangaben           UC-09, F-11     über alle Antworten
#
# Ändert nichts am System außer dem CAG-Cache (wird am Anfang geleert).
# Dauer: etwa 6–9 Minuten.
#
# AUFRUF:
#   docker cp test_produktberatung.py petra_frontend:/tmp/
#   docker exec petra_frontend python3 /tmp/test_produktberatung.py
# Ergebnis: /data/eval/test_produktberatung_<Zeit>.xlsx
#           (auf dem Server: /mnt/petra-rag/eval/)
# ============================================================
from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

API = os.getenv("PETRA_API_URL", "http://ingestion-api:8001").rstrip("/")
OUT_DIR = Path(os.getenv("PETRA_EVAL_DIR", "/data/eval"))

# Genau das, was der Chat mit den Standardwerten der Einstellungsseite sendet
# (ab Schritt 8: Antwortsprache "auto" = Sprache der Frage, NF-08).
STANDARD = {"language": "auto", "top_k": 6, "score_threshold": 0.65,
            "temperature": 0.0, "model": "llama3.1:8b", "web_agent_enabled": False}

ABLEHNUNG = ("nicht belegt", "ich weiß es nicht", "ich weiss es nicht", "nicht eindeutig",
             "keine angabe", "nicht enthalten", "nicht verfügbar", "nicht verfuegbar",
             "liegen keine", "liegt keine", "nicht angegeben", "keine informationen",
             "keine technischen daten", "nicht aufgeführt", "nicht aufgefuehrt",
             "nicht dokumentiert", "nicht bestimmbar", "keine preisangabe", "kein preis",
             "nicht vorhanden", "nicht in der lokalen", "nicht gefunden", "keine daten")
EUR_RE = re.compile(r"(\d{1,3}(?:[.\s]\d{3})*,\d{2}|\d+\.\d{2})\s*(?:€|eur)", re.IGNORECASE)
DE_MARKER = {"der", "die", "das", "und", "ist", "hat", "mit", "von", "für", "bei", "eine", "einen",
             "nicht", "beträgt", "wird", "sind", "oder", "zu", "den", "dem", "des"}
EN_MARKER = {"the", "and", "is", "has", "with", "of", "for", "at", "a", "an", "not", "are",
             "rated", "current", "which", "to", "in", "this", "it", "be"}

# ── Testfälle ────────────────────────────────────────────────────────
# klasse: erwartete query_class (Menge) · dok: Artikelnummern, die in den Quellen
# stehen müssen (alle; Tupel = eine davon genügt) · werte: Zahlenwerte (alle) ·
# begriffe: Gruppen, je Gruppe muss ein Begriff vorkommen · ablehnung: True/False ·
# verboten: Regex, die nicht vorkommen darf · sprache: erwartete Antwortsprache.
FAELLE = [
    # Q1 – Produktspezifikation (UC-01)
    dict(id="P01", typ="Q1 Produktspezifikation", anf="UC-01, F-13 (Fall 1)",
         frage="Wie hoch ist der Nennstrom der WDU 2.5?",
         klasse={"produktspezifikation"}, dok=["1020000000"], werte=["24 A"], ablehnung=False),
    dict(id="P02", typ="Q1 Eignungsfrage", anf="UC-01 (Fall 33, Produkt ausgeschrieben)",
         frage="Welchen Leiterquerschnitt darf ich an den PUSH-IN-Klemmen des UR20-FBC-PN-IRT-V2 "
               "anschließen? Reichen 2,5 mm²?",
         klasse={"produktspezifikation"}, dok=["2566380000"], werte=["1,5 mm²"], ablehnung=False,
         begriffe=[["nein", "nicht zulässig", "nicht ausreichend", "reichen nicht", "nicht geeignet",
                    "nicht möglich", "überschreitet", "zu groß"]]),
    dict(id="P03", typ="Q1 Artikelnummer → Produkt", anf="UC-01 (Fall 10)",
         frage="Was ist Artikel 1527690000 und wie breit ist das Teil?",
         klasse={"produktspezifikation"}, dok=[("1527690000", "1020000000")], werte=["48,7 mm"],
         begriffe=[["zqv 2.5n/10", "zqv 2,5n/10", "zqv2.5n/10"]], ablehnung=False),
    dict(id="P04", typ="Q1 Regression K7", anf="F-07 Klassifikation (Fall 37: 'Steuerspannung' ≠ Preis)",
         frage="Welchen Dauerstrom kann das Relaismodul TRS 24VDC 1CO schalten und welche "
               "Steuerspannung braucht es?",
         klasse={"produktspezifikation"}, dok=["1122770000"], werte=["6 A", "24 V"], ablehnung=False),
    # Q2 – Produktvergleich (UC-02 / UC-03)
    dict(id="P05", typ="Q2 Produktvergleich", anf="UC-02, UC-03 (Fall 38, als Vergleich formuliert)",
         frage="Vergleiche TRS 24VDC 1CO und TRS 24VDC 2CO hinsichtlich Kontaktanzahl und Dauerstrom.",
         klasse={"produktvergleich"}, dok=["1122770000", "1123490000"], werte=["6 A", "8 A"],
         tabelle=True, ablehnung=False),
    dict(id="P06", typ="Q2 Produktvergleich", anf="UC-02 (Fall 13)",
         frage="Was ist der Unterschied zwischen WDU 2.5 und A2C 2.5?",
         klasse={"produktvergleich"}, dok=["1020000000", "1521850000"],
         begriffe=[["push in", "push-in"], ["schraub"]], ablehnung=False),
    dict(id="P07", typ="Q2 nur ein Produkt", anf="UC-02 A1 (Hinweis: zweites Produkt nennen)",
         frage="Vergleiche die WDU 2.5.",
         begriffe=[["zwei produkte", "zweites produkt", "beide produkt", "weiteres produkt",
                    "mit welchem", "zweite produkt", "vergleichsprodukt", "produkte nennen"]]),
    # Q3 – Produktsuche
    dict(id="P08", typ="Q3 Produktsuche", anf="Kap. 4.4.1 Q3, F-13 (Fall 45)",
         frage="Ich brauche eine 4-polige M12-Sensorleitung mit Zustandsanzeige (LED) am Stecker. "
               "Gibt es die bei Weidmüller und ist die geschirmt?",
         klasse={"produktsuche"}, dok=["1094190500"],
         begriffe=[["sail-m12bg-4-2l5.0u", "1094190500"],
                   ["nicht geschirmt", "ungeschirmt", "geschirmt: nein", "keine schirmung",
                    "nicht abgeschirmt", "ohne schirm"]], ablehnung=False),
    dict(id="P09", typ="Q3 Produktsuche offen", anf="Kap. 4.4.1 Q3",
         frage="Welche Reihenklemmen mit PUSH-IN-Anschluss gibt es für 2,5 mm² Leiter?",
         klasse={"produktsuche"}, ablehnung=False, min_quellen=1),
    # Q4 – Preisanfrage (UC-04, Web-Agent aus)
    dict(id="P10", typ="Q4 Preis lokal", anf="UC-04 (Preisliste, Wert belegt 07.09.2026)",
         frage="Preis für BL 3.50/05/180F SN OR BX",
         klasse={"preisanfrage"}, werte=["415,36"], preis_beleg=True,
         begriffe=[["stand", "datum", "veraltet", "aktuell", "aktualität", "undatiert", "abweichen", "gültig"]]),
    dict(id="P11", typ="Q4 Preis", anf="UC-04: Preis nur mit Beleg (Fall 49)",
         frage="Was kostet das PRO ECO3 240W 24V 10A II aktuell?",
         klasse={"preisanfrage"}, preis_ehrlich=True),
    dict(id="P12", typ="Q4 kein Preis vorhanden", anf="UC-04 A1, NF-01 (Web-Agent aus)",
         frage="Was kostet der ifm O1D302?",
         klasse={"preisanfrage"}, ablehnung=True, verboten=[EUR_RE.pattern]),
    # Q5 – Allgemeine technische Frage
    dict(id="P13", typ="Q5 Allgemeine Frage", anf="Kap. 4.4.1 Q5 (nur aus Kontext, keine Empfehlung)",
         frage="Was bedeutet die Anschlusstechnik PUSH IN?",
         klasse={"allgemeine_technische_frage"}, belegt_oder_ablehnung=True),
    dict(id="P14", typ="Q5 Allgemeine Frage", anf="Kap. 4.4.1 Q5",
         frage="Wofür steht der Verschmutzungsgrad in einem Datenblatt?",
         klasse={"allgemeine_technische_frage"}, belegt_oder_ablehnung=True),
    # Negativfälle (F-13)
    dict(id="P15", typ="Negativfall", anf="F-13 keine Halluzination (Fall 48)",
         frage="Welche IP-Schutzart hat die Reihenklemme WDU 2.5?",
         ablehnung=True, verboten=[r"\bip\s?\d{2}"]),
    dict(id="P16", typ="Negativfall", anf="F-13 keine Halluzination (Fall 75)",
         frage="Wie viele Steckzyklen hält das HDC-KIT-HA 10.110 aus und wie hoch ist der Kontaktwiderstand?",
         ablehnung=True),
    # Sprache (NF-08, UC-01 A5)
    dict(id="P17", typ="Sprache EN, Einstellung Automatisch", anf="NF-08 Antwortsprache = Anfragesprache",
         frage="What is the rated current of the WDU 2.5?", sprache="Englisch", werte=["24 A"]),
    dict(id="P18", typ="Sprache EN, Einstellung Englisch", anf="NF-08, F-08",
         frage="What is the rated current of the WDU 2.5?", sprache="Englisch", werte=["24 A"],
         einstellungen={"language": "Englisch"}),
    dict(id="P21", typ="Sprache DE, Einstellung Englisch", anf="F-08 Vorgabe der Antwortsprache (E6)",
         frage="Wie hoch ist der Nennstrom der WDU 2.5?", sprache="Englisch", werte=["24 A"],
         einstellungen={"language": "Englisch"}),
    dict(id="P19", typ="Nicht unterstützte Sprache", anf="UC-01 A5",
         frage="Какой номинальный ток у клеммы WDU 2.5?", klasse={"nicht_unterstuetzte_sprache"}),
    # Web-Agent (UC-11) – braucht Internet; Ergebnis darf "kein Preis" sein, aber nichts Erfundenes
    dict(id="P22", typ="Q4 Web-Agent an", anf="UC-11, UC-04 (Agent ≤ 15 s, Knoten 20 s)",
         frage="Was kostet der ifm O1D302?", klasse={"preisanfrage"}, preis_ehrlich=True, latenz_max=25.0,
         einstellungen={"web_agent_enabled": True}),
    # Cache / Latenz
    dict(id="P20", typ="Cache-Wiederholung von P01", anf="Kap. 4.2 [1c], NF-03 (≤ 5 s)",
         frage="Wie hoch ist der Nennstrom der WDU 2.5?", cache=True, gleich_wie="P01", latenz_max=5.0),
]


# ── Hilfsfunktionen ──────────────────────────────────────────────────
def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower().replace(",", ".").replace("mm2", "mm²")


def ist_ablehnung(text: str) -> bool:
    t = str(text or "").lower()
    return any(p in t for p in ABLEHNUNG)


def sprache(text: str) -> str:
    tok = re.findall(r"[a-zäöüß]+", re.sub(r"\[Quelle:[^\]]*\]", " ", str(text or "")).lower())
    de = sum(t in DE_MARKER for t in tok) + 2 * bool(re.search(r"[äöüß]", " ".join(tok)))
    en = sum(t in EN_MARKER for t in tok)
    return "Englisch" if en > de else "Deutsch"


def quellen_text(d: dict) -> str:
    return " ".join(f"{q.get('dateiname', '')} {q.get('artikelnummer') or ''}" for q in d.get("sources") or [])


def pruefe(f: dict, d: dict, frueher: dict[str, dict]) -> list[tuple[str, bool, str]]:
    """Liefert (Kriterium, bestanden, Detail) je automatisch prüfbarem Kriterium."""
    a = str(d.get("answer") or "")
    al, an = a.lower(), norm(a)
    qt = quellen_text(d)
    erg: list[tuple[str, bool, str]] = []
    if d.get("_error"):
        return [("Anfrage", False, d["_error"])]
    platzhalter = re.findall(r"<(?:produkt|attribut)>", al)
    erg.append(("Keine Platzhalter", not platzhalter, ", ".join(sorted(set(platzhalter))) or "ok"))
    if "klasse" in f:
        ist = d.get("query_class") or "?"
        erg.append(("Klasse", ist in f["klasse"], f"ist {ist}, soll {'/'.join(sorted(f['klasse']))}"))
    for dok in f.get("dok", []):
        alt = dok if isinstance(dok, tuple) else (dok,)
        erg.append(("Datenblatt in Quellen", any(x in qt for x in alt), " oder ".join(alt)))
    if f.get("werte"):
        fehlt = [w for w in f["werte"] if norm(w) not in an]
        erg.append(("Werte", not fehlt, "fehlt: " + ", ".join(fehlt) if fehlt else ", ".join(f["werte"])))
    for gruppe in f.get("begriffe", []):
        treffer = [g for g in gruppe if g in al]
        erg.append(("Aussage", bool(treffer), treffer[0] if treffer else "keiner von: " + " | ".join(gruppe[:4])))
    if f.get("ablehnung") is True:
        erg.append(("Ablehnung statt Erfindung", ist_ablehnung(a), "erwartet: 'nicht belegt' o. ä."))
    elif f.get("ablehnung") is False:
        erg.append(("Keine Ablehnung", not ist_ablehnung(a), "Antwort erwartet"))
    for muster in f.get("verboten", []):
        m = re.search(muster, al, re.IGNORECASE)
        erg.append(("Nichts erfunden", m is None, f"gefunden: {m.group(0)!r}" if m else "ok"))
    if f.get("tabelle"):
        erg.append(("Vergleichstabelle", a.count("|") >= 6, "Markdown-Tabelle"))
    if f.get("min_quellen"):
        n = len(d.get("sources") or [])
        erg.append(("Quellen vorhanden", n >= f["min_quellen"], f"{n} Quellen"))
    if f.get("preis_beleg") or f.get("preis_ehrlich"):
        betraege = EUR_RE.findall(a) or re.findall(r"\d+,\d{2}", a)
        belege = norm(" ".join(str(q.get("textauszug") or "") for q in d.get("sources") or []))
        if betraege:
            unbelegt = [b for b in betraege if norm(b) not in belege]
            erg.append(("Preis belegt (Quelle)", not unbelegt,
                        "unbelegt: " + ", ".join(unbelegt) if unbelegt else ", ".join(betraege)))
        else:
            erg.append(("Preis belegt (Quelle)", bool(f.get("preis_ehrlich")) and ist_ablehnung(a),
                        "kein Betrag genannt" + (" (ehrliche Ablehnung)" if ist_ablehnung(a) else "")))
    if f.get("belegt_oder_ablehnung"):
        n = len(d.get("sources") or [])
        erg.append(("Belegt oder ehrlich abgelehnt", n > 0 or ist_ablehnung(a), f"{n} Quellen"))
    if f.get("sprache"):
        ist = sprache(a)
        erg.append(("Antwortsprache", ist == f["sprache"], f"ist {ist}, soll {f['sprache']}"))
    if f.get("cache"):
        erg.append(("Cache-Treffer", bool(d.get("cache_hit")), f"cache_hit={d.get('cache_hit')}"))
    if f.get("gleich_wie"):
        ref = frueher.get(f["gleich_wie"], {})
        erg.append(("Antwort identisch", bool(a) and a == str(ref.get("answer") or ""), f"wie {f['gleich_wie']}"))
    if f.get("latenz_max"):
        s = d.get("_latency_s", 0)
        erg.append(("Latenz", s <= f["latenz_max"], f"{s:.1f} s (≤ {f['latenz_max']:.0f} s)"))
    return erg


def main() -> None:
    import httpx

    c = httpx.Client(timeout=240.0)

    def frage(f: dict) -> dict:
        body = {"query": f["frage"], **STANDARD, **f.get("einstellungen", {})}
        for versuch in range(19):                         # bis 3 min auf eine laufende Anfrage warten
            t0 = time.time()
            try:
                r = c.post(f"{API}/query", json=body)
                if r.status_code == 503 and versuch < 18:  # vorherige Anfrage läuft noch
                    time.sleep(10)
                    continue
                d = r.json() if r.status_code < 400 else {"_error": f"HTTP {r.status_code}: {r.text[:200]}"}
            except Exception as exc:  # noqa: BLE001
                d = {"_error": str(exc)}
            d["_latency_s"] = time.time() - t0
            return d
        return {"_error": "503 nach 3 min Wartezeit", "_latency_s": 0.0}

    print(f"API: {API}")
    # Aufwärmen: wartet, bis keine andere Anfrage mehr läuft (z. B. aus einem abgebrochenen
    # Lauf), und lädt das Modell – geht nicht in die Bewertung ein.
    t0 = time.time()
    w = frage({"frage": "Welche Schutzart hat der Switch IE-SW-BL05-5TX?"})
    print(f"Aufwärmen: {time.time() - t0:.1f} s" + (f" – {w['_error']}" if w.get("_error") else ""))
    try:
        c.post(f"{API}/cache/invalidate", json={})
        print("CAG-Cache geleert (kalte Messung).\n")
    except Exception as exc:  # noqa: BLE001
        print(f"Hinweis: Cache nicht geleert: {exc}\n")

    ergebnisse: dict[str, dict] = {}
    zeilen = []
    for f in FAELLE:
        print(f"{f['id']} {f['typ']}: {f['frage'][:70]}")
        d = frage(f)
        ergebnisse[f["id"]] = d
        checks = pruefe(f, d, ergebnisse)
        ok = all(b for _, b, _ in checks) if checks else None
        for name, b, det in checks:
            print(f"   {'✓' if b else '✗'} {name:28s} {det}")
        print(f"   → {'BESTANDEN' if ok else 'NICHT BESTANDEN' if ok is False else 'nur Sichtprüfung'}"
              f" · {d.get('_latency_s', 0):.1f} s · Klasse {d.get('query_class')} · "
              f"verified {d.get('verified')} · Cache {d.get('cache_hit')}")
        if ok is False or ok is None:
            print("   Antwort: " + re.sub(r"\s+", " ", str(d.get("answer") or d.get("_error") or ""))[:260])
        print()
        zeilen.append({
            "ID": f["id"], "Fragetyp": f["typ"], "Anforderung": f["anf"], "Frage": f["frage"],
            "Automatisch": "bestanden" if ok else "nicht bestanden" if ok is False else "Sichtprüfung",
            "Prüfungen": "\n".join(f"{'✓' if b else '✗'} {n}: {det}" for n, b, det in checks),
            "Klasse (ist)": d.get("query_class"), "Latenz s": round(d.get("_latency_s", 0), 1),
            "Cache-Treffer": d.get("cache_hit"), "verified": d.get("verified"),
            "Faithfulness": d.get("faithfulness_score"),
            "Quellen": "\n".join(f"{q.get('dateiname')} S. {q.get('seite')}" for q in d.get("sources") or []),
            "Antwort": d.get("answer") or d.get("_error"),
            "Manuelle Bewertung": "", "Bemerkung": "",
        })

    # UC-09 / F-11: Metadatenvollständigkeit der Quellen (Ziel ≥ 95 %)
    alle = [q for d in ergebnisse.values() for q in d.get("sources") or []]
    voll = [q for q in alle if q.get("dateiname") not in (None, "", "unbekannt") and q.get("seite") is not None]
    f11 = len(voll) / len(alle) if alle else 0.0
    ohne_cache = [d["_latency_s"] for d in ergebnisse.values() if not d.get("cache_hit") and "_error" not in d]
    bestanden = sum(z["Automatisch"] == "bestanden" for z in zeilen)
    print("=" * 72)
    print(f"Automatisch bestanden: {bestanden} von {len(zeilen)}")
    print(f"F-11 Quellen mit Dateiname und Seite: {len(voll)}/{len(alle)} = {f11:.0%} (Ziel ≥ 95 %)")
    if ohne_cache:
        print(f"NF-03 Latenz ohne Cache: Median {sorted(ohne_cache)[len(ohne_cache) // 2]:.1f} s, "
              f"≤ 5 s: {sum(s <= 5 for s in ohne_cache)}/{len(ohne_cache)}")
    for z in zeilen:
        if z["Automatisch"] == "nicht bestanden":
            print(f"  ✗ {z['ID']} {z['Fragetyp']} – " + "; ".join(
                l[2:] for l in z["Prüfungen"].splitlines() if l.startswith("✗")))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    try:
        import pandas as pd
        out = OUT_DIR / f"test_produktberatung_{stamp}.xlsx"
        with pd.ExcelWriter(out) as xw:
            pd.DataFrame(zeilen).to_excel(xw, sheet_name="Ergebnisse", index=False)
            pd.DataFrame([
                {"Kennzahl": "Automatisch bestanden", "Wert": f"{bestanden}/{len(zeilen)}"},
                {"Kennzahl": "F-11 Quellen vollständig (Dateiname + Seite)", "Wert": f"{f11:.0%}"},
                {"Kennzahl": "NF-03 Anteil ≤ 5 s (ohne Cache)",
                 "Wert": f"{sum(s <= 5 for s in ohne_cache)}/{len(ohne_cache)}" if ohne_cache else "–"},
                {"Kennzahl": "Einstellungen", "Wert": str(STANDARD)},
                {"Kennzahl": "Zeitpunkt", "Wert": datetime.now().strftime("%d.%m.%Y %H:%M")},
            ]).to_excel(xw, sheet_name="Zusammenfassung", index=False)
    except Exception as exc:  # noqa: BLE001 – ohne pandas/openpyxl als CSV
        import csv
        out = OUT_DIR / f"test_produktberatung_{stamp}.csv"
        with out.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(zeilen[0]), delimiter=";")
            w.writeheader()
            w.writerows(zeilen)
        print(f"(Excel nicht möglich: {exc})")
    print(f"\nErgebnis gespeichert: {out}  (Server: /mnt/petra-rag/eval/{out.name})")
    try:
        c.post(f"{API}/cache/invalidate", json={})
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    sys.exit(main())
