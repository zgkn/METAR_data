#!/usr/bin/env python3
"""Fetch METAR observations for Southeast Asia from the Iowa Environmental Mesonet.

The day is specified in Singapore time (SGT, Asia/Singapore). IEM works in UTC, so
the SGT calendar day is converted to a UTC window, requested with tz=Etc/UTC, and a
local-time column is added to the output. Needs only `requests` and `pandas`.

Outputs (in --outdir):
  metar_sea_<date>.csv           all stations combined
  metar_sea_<date>_summary.txt   station / observation counts and gaps
"""
import argparse
import datetime as dt
import io
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
NETWORK_URL = "https://mesonet.agron.iastate.edu/geojson/network/{net}.geojson"
LOCAL_TZ = ZoneInfo("Asia/Singapore")

COUNTRIES = {
    "SG": "Singapore",
    "MY": "Malaysia",
    "ID": "Indonesia",
    "TH": "Thailand",
    "VN": "Vietnam",
    "PH": "Philippines",
    "KH": "Cambodia",
    "LA": "Laos",
    "MM": "Myanmar",
    "BN": "Brunei",
    "TL": "Timor-Leste",
}

# IEM network listings are not always country-pure (e.g. Vietnamese VVCR/VVCT/VVPQ sit in
# LA__ASOS, Brunei's WBSB sits in MY__ASOS), so country comes from the ICAO prefix when
# known; the `network` column keeps the IEM listing. Longest prefix wins.
ICAO_PREFIX_COUNTRY = {
    "WBSB": "BN", "WS": "SG", "WM": "MY", "WB": "MY", "WI": "ID", "WA": "ID", "WP": "TL",
    "VT": "TH", "VV": "VN", "RP": "PH", "VD": "KH", "VL": "LA", "VY": "MM",
}


def country_code(station, network):
    for n in (4, 2):
        code = ICAO_PREFIX_COUNTRY.get(station[:n])
        if code:
            return code
    return network[:2]


# Sumatra / Kalimantan airports that matter for haze analysis. They are always
# requested, even if a network listing omits them, and reported separately.
HAZE_WATCHLIST = {
    "WIPP": "Palembang",
    "WIBB": "Pekanbaru",
    "WIOO": "Pontianak",
    "WAOO": "Banjarmasin",
    "WIMM": "Medan",
    "WIEE": "Padang (Minangkabau)",
    "WIJJ": "Jambi",
    "WIPL": "Bengkulu",
    "WIPK": "Pangkal Pinang",
    "WALL": "Balikpapan",
    "WAOP": "Palangkaraya",
    "WAQQ": "Tarakan",
}

# Columns the analysis relies on; warn if the download lacks any of them.
EXPECTED_COLS = ["metar", "drct", "sknt", "gust", "vsby", "tmpf", "dwpf", "alti", "mslp", "wxcodes"]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


class Fetcher:
    def __init__(self, sleep, tries=4, timeout=180):
        self.sleep = sleep
        self.tries = tries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "metar-sea-fetch (GitHub Actions; requests)"
        self.last_request = 0.0

    def get(self, url, params=None):
        """GET with politeness delay and retry on network errors / 429 / 5xx.

        Other 4xx responses are returned to the caller without retrying.
        """
        last_err = None
        for attempt in range(1, self.tries + 1):
            wait = self.sleep - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                self.last_request = time.monotonic()
                if r.status_code == 429 or r.status_code >= 500:
                    last_err = f"HTTP {r.status_code}"
                else:
                    return r
            except requests.RequestException as e:
                self.last_request = time.monotonic()
                last_err = f"{type(e).__name__}: {e}"
            if attempt < self.tries:
                backoff = 5 * 2 ** (attempt - 1)
                log(f"  retry {attempt}/{self.tries - 1} after {backoff}s ({last_err})")
                time.sleep(backoff)
        raise RuntimeError(f"giving up after {self.tries} attempts: {last_err}")


def sgt_window(date_str):
    """Return (start_utc, end_utc) for the SGT calendar day, as aware datetimes."""
    day = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=LOCAL_TZ)
    start = day.astimezone(dt.timezone.utc)
    return start, start + dt.timedelta(days=1)


def load_network(fetcher, code):
    """Return list of (station_id, name) for a network, or None if it doesn't exist."""
    net = f"{code}__ASOS"
    try:
        r = fetcher.get(NETWORK_URL.format(net=net))
    except RuntimeError as e:
        log(f"[{net}] listing failed: {e}")
        return None
    if r.status_code != 200:
        log(f"[{net}] listing returned HTTP {r.status_code}; skipping network")
        return None
    try:
        feats = r.json().get("features", [])
    except ValueError:
        log(f"[{net}] listing is not valid JSON; skipping network")
        return None
    stations = []
    for f in feats:
        props = f.get("properties") or {}
        sid = props.get("sid") or f.get("id")
        if sid:
            stations.append((str(sid).strip(), props.get("sname", "")))
    if not stations:
        log(f"[{net}] listing has no stations; skipping network")
        return None
    return stations


def parse_csv(text):
    """Parse an IEM onlycomma response; None if it doesn't look like a data CSV."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if not lines or not lines[0].startswith("station,valid"):
        return None
    return pd.read_csv(io.StringIO("\n".join(lines)), dtype=str, keep_default_na=False)


class Requester:
    def __init__(self, fetcher, start, end):
        self.fetcher = fetcher
        self.start = start
        self.end = end
        self.use_components = False  # flips to True if sts/ets is rejected

    def _params(self, stations, components):
        p = {
            "station": list(stations),
            "data": "all",
            "tz": "Etc/UTC",
            "format": "onlycomma",
            "latlon": "yes",
            "elev": "yes",
            "missing": "empty",
            "trace": "T",
            "direct": "no",
            "report_type": ["3", "4"],  # routine + specials
        }
        if components:
            s, e = self.start, self.end
            p.update(year1=s.year, month1=s.month, day1=s.day, hour1=s.hour, minute1=s.minute,
                     year2=e.year, month2=e.month, day2=e.day, hour2=e.hour, minute2=e.minute)
        else:
            p["sts"] = self.start.strftime("%Y-%m-%dT%H:%MZ")
            p["ets"] = self.end.strftime("%Y-%m-%dT%H:%MZ")
        return p

    def fetch(self, stations):
        """Return a DataFrame (possibly empty). Raises RuntimeError if unusable."""
        for components in ([True] if self.use_components else [False, True]):
            r = self.fetcher.get(ASOS_URL, self._params(stations, components))
            df = parse_csv(r.text) if r.status_code == 200 else None
            if df is not None:
                return df
            what = "year1/.. components" if components else "sts/ets"
            log(f"  response to {what} request unusable (HTTP {r.status_code}): {r.text[:120]!r}")
            if not components:
                log("  falling back to year1/month1/... window parameters")
                self.use_components = True
        raise RuntimeError("IEM response was not a valid CSV with either window format")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2026-10-18", help="SGT calendar date, YYYY-MM-DD")
    ap.add_argument("--outdir", default="data/metar")
    ap.add_argument("--networks", default=",".join(COUNTRIES), help="comma-separated country codes")
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--sleep", type=float, default=2.0, help="min seconds between requests")
    ap.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = ap.parse_args()

    try:
        start, end = sgt_window(args.date)
    except ValueError:
        sys.exit(f"--date must be YYYY-MM-DD, got {args.date!r}")

    outdir = Path(args.outdir)
    csv_path = outdir / f"metar_sea_{args.date}.csv"
    summary_path = outdir / f"metar_sea_{args.date}_summary.txt"
    existing = [p for p in (csv_path, summary_path) if p.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(p) for p in existing)
        log(f"Refusing to overwrite existing file(s): {names}. Re-run with overwrite enabled to replace.")
        return 3
    if existing:
        log(f"WARNING: overwriting existing file(s): {', '.join(str(p) for p in existing)}")

    log(f"SGT day {args.date} -> UTC window {start:%Y-%m-%dT%H:%MZ} to {end:%Y-%m-%dT%H:%MZ}")
    fetcher = Fetcher(args.sleep)
    requester = Requester(fetcher, start, end)

    # --- station lists per network ---------------------------------------
    codes = [c.strip().upper() for c in args.networks.split(",") if c.strip()]
    station_net = {}   # station id -> network
    station_name = {}
    skipped_networks = []
    net_stations = {}
    for code in codes:
        listing = load_network(fetcher, code)
        if listing is None:
            skipped_networks.append(f"{code}__ASOS")
            continue
        net_stations[code] = []
        for sid, name in listing:
            if sid not in station_net:
                station_net[sid] = f"{code}__ASOS"
                station_name[sid] = name
                net_stations[code].append(sid)
        log(f"[{code}__ASOS] {len(net_stations[code])} stations")

    # Haze watchlist: request even if absent from the listings (nothing is filtered out).
    watch_not_listed = [s for s in HAZE_WATCHLIST if s not in station_net]
    if watch_not_listed:
        log(f"Watchlist stations missing from network listings (requesting anyway): {watch_not_listed}")
        net_stations.setdefault("ID", [])
        for s in watch_not_listed:
            station_net[s] = "ID__ASOS"
            station_name[s] = HAZE_WATCHLIST[s]
            net_stations["ID"].append(s)
    all_requested = [s for code in net_stations for s in net_stations[code]]
    if not all_requested:
        log("No stations to request.")
        return 1

    # --- fetch data ------------------------------------------------------
    frames = []
    failed = []  # stations whose request failed after retries
    for code, sids in net_stations.items():
        for i in range(0, len(sids), args.batch_size):
            batch = sids[i:i + args.batch_size]
            log(f"[{code}] requesting {len(batch)} stations ({i + 1}-{i + len(batch)} of {len(sids)})")
            try:
                frames.append(requester.fetch(batch))
                continue
            except RuntimeError as e:
                log(f"  batch failed ({e}); retrying stations one at a time")
            for sid in batch:
                try:
                    frames.append(requester.fetch([sid]))
                except RuntimeError as e:
                    log(f"  {sid} failed: {e}")
                    failed.append(sid)

    frames = [f for f in frames if len(f)]
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["station", "valid"])

    # --- post-process ----------------------------------------------------
    df["valid_utc"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    df = df.dropna(subset=["valid_utc"])
    n_before = len(df)
    df = df[(df["valid_utc"] >= start) & (df["valid_utc"] < end)]  # strictly the SGT day
    if len(df) != n_before:
        log(f"Dropped {n_before - len(df)} rows outside the SGT day window")
    key = [c for c in ("station", "valid_utc", "metar") if c in df.columns]
    n_before = len(df)
    df = df.drop_duplicates(subset=key)
    if len(df) != n_before:
        log(f"Dropped {n_before - len(df)} exact duplicate rows")

    df["valid_sgt"] = df["valid_utc"].dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m-%d %H:%M")
    df["valid_utc"] = df["valid_utc"].dt.strftime("%Y-%m-%d %H:%M")
    df["network"] = df["station"].map(station_net)
    df["country"] = [COUNTRIES.get(country_code(st, nw)) for st, nw in zip(df["station"], df["network"])]
    moved = df[df["network"].str[:2].map(COUNTRIES) != df["country"]]
    for st, grp in moved.groupby("station"):
        log(f"Country override: {st} listed in {grp['network'].iloc[0]} -> {grp['country'].iloc[0]}")
    df = df.drop(columns=["valid"])
    lead = ["station", "network", "country", "valid_utc", "valid_sgt"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df = df.sort_values(["country", "station", "valid_utc"], kind="stable").reset_index(drop=True)

    missing_cols = [c for c in EXPECTED_COLS if c not in df.columns] if len(df) else []
    returned = sorted(df["station"].unique())
    zero = [s for s in all_requested if s not in set(returned) and s not in set(failed)]
    unrequested = sorted(set(returned) - set(all_requested))

    # --- write -----------------------------------------------------------
    if not len(df):
        log("No observations returned at all; writing nothing.")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    per_country = df.groupby("country").agg(stations=("station", "nunique"), observations=("station", "size"))
    lines = [
        f"METAR Southeast Asia, SGT day {args.date}",
        f"UTC window: {start:%Y-%m-%dT%H:%MZ} to {end:%Y-%m-%dT%H:%MZ} (end exclusive)",
        f"Generated: {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%MZ}",
        "",
        f"Stations requested: {len(all_requested)}",
        f"Stations returning data: {len(returned)}",
        f"Stations with zero data: {len(zero)}",
        f"Stations whose request failed after retries: {len(failed)}",
        f"Total observations: {len(df)}",
        "",
        "Observations per country:",
    ]
    for country in COUNTRIES.values():
        if country in per_country.index:
            r = per_country.loc[country]
            lines.append(f"  {country}: {r['observations']} obs from {r['stations']} stations")
        else:
            lines.append(f"  {country}: 0 obs")
    lines += ["", "Networks skipped (missing or empty listing): " + (", ".join(skipped_networks) or "none")]
    lines += ["Stations with zero data: " + (", ".join(f"{s} ({station_net[s]}{' - not in listing' if s in watch_not_listed else ''})" for s in zero) or "none")]
    lines += ["Stations whose request failed (data status unknown): " + (", ".join(failed) or "none")]
    lines += ["", "Haze watchlist (Sumatra/Kalimantan):"]
    counts = df["station"].value_counts()
    for s, name in HAZE_WATCHLIST.items():
        note = "; not in any IEM network listing (no data available from IEM)" if s in watch_not_listed else ""
        lines.append(f"  {s} {name}: {int(counts.get(s, 0))} obs{note}")
    if unrequested:
        lines.append("Returned but not requested (unexpected): " + ", ".join(unrequested))
    if missing_cols:
        lines.append("WARNING: expected columns missing from download: " + ", ".join(missing_cols))
    summary = "\n".join(lines) + "\n"
    summary_path.write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
