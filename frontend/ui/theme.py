"""
ui/theme.py — Ergänzende CSS-Regeln zum Basistheme (.streamlit/config.toml).

Betrifft Ingestion-Stufenleiste, Quellentitel, Markdown-Tabellen im Chat
und die Chat-Eingabezeile. Bewusst minimal, um bei Streamlit-Updates
stabil zu bleiben.

Die Klassen `.st-key-<key>` erzeugt Streamlit ab Version 1.36 für Elemente
mit `key`. In älteren Versionen fehlen sie; die Buttons funktionieren dann
weiterhin, erscheinen aber im Standardstil.

In der Eingabezeile sind Textfeld-Enter, Mikrofon und Senden jeweils
`st.form_submit_button` (in einem Formular sind nur Submit-Buttons
erlaubt). Sie werden deshalb einzeln über ihren Key gestylt
(siehe views/chat.py).
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root { --petra-muted:#64748B; --petra-accent:#0F766E; }
.block-container { max-width: 1100px; padding-top: 2.2rem; }
h1, h2, h3 { letter-spacing: -0.01em; }
[data-testid="stSidebar"] { border-right: 1px solid #E2E8F0; }

/* Ingestion-Stufenleiste */
.petra-stage { text-align:center; padding:0.6rem 0.2rem; border-radius:10px;
  border:1px solid #E2E8F0; background:#FFFFFF; font-size:0.78rem;
  min-height:5.4rem; }
.petra-stage-detail { color:var(--petra-muted); font-size:0.72rem; }
.petra-running { border-color:var(--petra-accent);
  box-shadow:0 0 0 2px rgba(15,118,110,.15); }
.petra-done    { border-color:#16A34A22; background:#F0FDF4; }
.petra-error   { border-color:#DC262655; background:#FEF2F2; }
.petra-skipped { opacity:.55; }

/* Quellenbereich */
.petra-sources-title { font-weight:600; color:var(--petra-accent);
  margin:.4rem 0 .2rem 0; font-size:.9rem; }

/* ── Produktvergleichs-/Markdown-Tabellen im Chat (ui/markdown.py) ────
   `.petra-table-wrap` ist der horizontal scrollbare Rahmen: breite
   Vergleichstabellen laufen dadurch NICHT mehr aus dem Chat-Container
   heraus, sondern bekommen eine eigene Scrollleiste. `width:max-content`
   auf der Tabelle verhindert, dass Spalten zusammengequetscht werden —
   erst dadurch wird das Scrollen überhaupt wirksam. Die Kopfzeile bleibt
   beim vertikalen Scrollen langer Tabellen sichtbar (sticky). */
.petra-table-wrap {
  overflow-x: auto;
  overflow-y: auto;
  max-height: 32rem;
  margin: .5rem 0 .9rem 0;
  border: 1px solid #E2E8F0;
  border-radius: 10px;
  background: #FFFFFF;
  -webkit-overflow-scrolling: touch;
}
table.petra-table {
  border-collapse: collapse;
  width: max-content;
  min-width: 100%;
  font-size: .88rem;
  line-height: 1.45;
}
table.petra-table th, table.petra-table td {
  padding: .5rem .8rem;
  border-bottom: 1px solid #EEF2F6;
  vertical-align: top;
  white-space: normal;
  word-break: normal;
  overflow-wrap: anywhere;
  max-width: 28rem;
}
table.petra-table thead th {
  position: sticky; top: 0; z-index: 1;
  background: #EEF2F6;
  color: #0F172A;
  font-weight: 600;
  white-space: nowrap;
  border-bottom: 2px solid #CBD5E1;
}
table.petra-table tbody tr:nth-child(even) { background: #F8FAFC; }
table.petra-table tbody tr:hover { background: #F0FDFA; }
table.petra-table tbody tr:last-child td { border-bottom: none; }
table.petra-table code {
  background: #EEF2F6; padding: .05rem .3rem; border-radius: 4px;
  font-size: .85em;
}
/* Erste Spalte (i. d. R. der Produktname) hervorheben und beim
   horizontalen Scrollen stehen lassen. */
table.petra-table tbody td:first-child { font-weight: 600; white-space: nowrap; }

/* ── Composer (Chat-Eingabezeile) — Pillenform nach Vorlage ──────────
   Nur stTextInput/stFormSubmitButton betroffen — davon gibt es im
   gesamten Frontend ausschließlich in views/chat.py Vorkommen, daher
   ist eine globale Regel hier unkritisch (kein Seiteneffekt auf andere
   Ansichten). Die eigentlichen Aufnahme-Widgets (streamlit-webrtc /
   streamlit-mic-recorder) laufen in einem eigenen, isolierten Frame und
   sind von HIER AUS NICHT stylebar — deshalb das eigene, klickbare
   Mikrofon-Icon (siehe .petra-mic-btn unten), das dieses Fenster nur
   bei Bedarf einblendet, statt es direkt zu ersetzen. */
div[data-testid="stTextInput"] input {
  border-radius: 999px !important;
  border: 1px solid #E2E8F0 !important;
  background: #FFFFFF !important;
  padding: 0.7rem 1.2rem !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--petra-accent) !important;
  box-shadow: 0 0 0 2px rgba(15,118,110,.15) !important;
}
/* Unsichtbarer erster Submit-Button: fängt ausschließlich die
   Enter-Taste ab (siehe views/chat.py). Er MUSS im DOM bleiben, damit
   Streamlit ihn als ersten Submit-Button des Formulars kennt — deshalb
   wird er nur ausgeblendet, nicht entfernt. */
.st-key-petra_enter_submit { display: none !important; }

/* Senden-Button (rundes Pfeil-Icon). Bewusst NICHT global auf
   [data-testid="stFormSubmitButton"], denn das Mikrofon ist technisch
   ebenfalls ein Submit-Button (siehe views/chat.py) und würde sonst
   genauso aussehen. */
.st-key-petra_send_btn button {
  border-radius: 999px !important;
  aspect-ratio: 1 / 1;
  width: 2.7rem !important;
  min-width: 2.7rem !important;
  padding: 0 !important;
  color: transparent !important;
  background-color: var(--petra-accent) !important;
  border: none !important;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><line x1='22' y1='2' x2='11' y2='13'/><polygon points='22 2 15 22 11 13 2 9 22 2'/></svg>");
  background-repeat: no-repeat;
  background-position: center;
  background-size: 1.15rem 1.15rem;
}

/* Mikrofon-Icon-Button — öffnet/schließt das eigentliche Aufnahme-Widget
   darunter. Zwei Container-Keys je nach Öffnungszustand (siehe
   views/chat.py) — Basis-Look für beide, zusätzlicher Akzentrahmen nur
   für den "geöffnet"-Zustand.

   KORREKTUR (siehe DOKUMENTATION.md, "Bugfix-Runde 4 — Icon unsichtbar"):
   Frühere Version setzte `color: transparent` und verließ sich komplett
   auf ein per data-URI eingebettetes SVG als `background-image`, um
   trotzdem ein Icon zu zeigen. Schlägt dieses SVG aus irgendeinem Grund
   fehl zu rendern (nicht in einer echten Browser-Umgebung überprüfbar),
   bleibt ein UNSICHTBARER Button übrig (weißer Kreis auf weißem
   Hintergrund) — genau das gemeldete Symptom. Diese Regel verlässt sich
   jetzt NICHT MEHR auf das SVG: Das 🎤-Emoji bleibt als normaler,
   garantiert sichtbarer Button-Text erhalten (exakt wie bei allen
   anderen Emoji-Buttons in dieser App, z. B. "🗑️ Verlauf löschen"), nur
   Form/Größe/Rahmen werden gestylt. Funktioniert dadurch selbst dann
   korrekt (nur optisch weniger speziell), wenn die `.st-key-*`-Klasse in
   einer älteren Streamlit-Version gar nicht erzeugt wird. */
.st-key-petra_mic_toggle button,
.st-key-petra_mic_toggle_open button {
  border-radius: 999px !important;
  width: 2.7rem !important; height: 2.7rem !important;
  padding: 0 !important;
  border: 1px solid #E2E8F0 !important;
  background: #FFFFFF !important;
  font-size: 1.3rem !important;
  line-height: 1 !important;
}
.st-key-petra_mic_toggle_open button {
  border-color: var(--petra-accent) !important;
  box-shadow: 0 0 0 2px rgba(15,118,110,.15) !important;
  background: #F0FDFA !important;
}
</style>
"""


def apply_theme() -> None:
  """Fügt die CSS-Regeln in die Seite ein."""
  st.markdown(_CSS, unsafe_allow_html=True)
