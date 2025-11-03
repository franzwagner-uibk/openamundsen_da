# openAMUNDSEN-Style – Coding Guide

**Ziel:** Einheitlicher Code-Stil für `openamundsen_da`, kompatibel zu openAMUNDSEN.

---

## 1) Docstrings & Kommentare

- **NumPy-Style Docstrings** mit `Parameters`, `Returns`, optional `Raises`.
- Kurze, präzise Kommentare – nur dort, wo nötig.
- Modul-/Datei-Header enthalten: **Titel, Autor, Datum, Kurzbeschreibung**.

**Beispiel:**

```python
"""
module_name.py
Author: Franz Wagner
Date: 2025-10-30
Description:
    Kurze, präzise Modulbeschreibung.
"""
def foo(bar, baz):
    """
    Do X given bar and baz.

    Parameters
    ----------
    bar : int
        ...
    baz : str
        ...

    Returns
    -------
    out : float
        ...
    """
    ...

2) Logging

    loguru logger (kein print).

    Zentrales Setup via configure_logger() (einmal pro Prozess).

    Level: info, debug, warning, success, error.

    Drittanbieter-Logs filtern/beruhigen, schlankes Format.

Beispiel (Gerüst):

from loguru import logger

def configure_logger(log_path=None):
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")  # console
    if log_path:
        logger.add(log_path, level="DEBUG", rotation="10 MB", encoding="utf-8")

3) Struktur & Benennung

    Funktionen/Klassen modular, private Helfer mit _-Prefix.

    snake_case für Funktionen/Variablen, UPPER_CASE für Konstanten.

    Datei-/Paketstruktur klar nach Verantwortlichkeiten (z. B. core/, io/, methods/, observers/, util/).

4) Konfiguration

    YAML (ruamel round-trip) + Validierung/Normalisierung (z. B. Cerberus).

    Ergebnis in bequeme Struktur (z. B. Munch).

    Nur deinen Namespace lesen (z. B. data_assimilation), defensiv parsen.

5) Daten-Stack & Konventionen

    numpy / pandas / xarray als Kern.

    CF-artige Metadaten, Einheiten prüfen/konvertieren.

    Niederschlags-Raten → Summen je Zeitschritt korrekt behandeln.

    Koordinaten/Projektionen via pyproj/rasterio; Helper in util.

6) Zeit & Resampling

    Frequenzen über util.to_offset / util.offset_to_timedelta.

    Downsampling nur als exakte Teilmenge (kein Upsampling).

    Tägliche Aggregation: konsistente Regeln/Warnings.

    Performance: pandas.resample in Hot-Paths (xarray-Resample meiden, wenn langsam).

7) Fehlerbehandlung

    Früh validieren, klar scheitern mit openamundsen.errors.* (z. B. MeteoDataError, ConfigurationError).

    Sonst Standard-Errors (FileNotFoundError, NotImplementedError) – aussagekräftige Messages.

8) Typisierung (pragmatisch)

    Leichte, gezielte Type Hints (z. B. List[Dict], xr.Dataset) – kein Over-Engineering.

    Signaturen lesbar halten; Typen dort, wo sie Klarheit schaffen.

9) I/O-Prinzipien

    pathlib.Path überall; keine hart kodierten Strings.

    Lesen/Schreiben über dedizierte fileio-Helper.

    Kein direktes xarray-Resampling in engen Schleifen.

    Ordentliche Exceptions bei I/O-Problemen, keine stummen Fehler.

10) Keine unicode characters in log statements verwenden -> ausschliesslich ascii! -> bei Pfeilen aufpassen
```
