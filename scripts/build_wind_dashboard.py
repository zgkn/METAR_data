#!/usr/bin/env python3
"""Build a single-file HTML page: hourly METAR wind-barb map + collapsible 48 h time-series overlay.

Pulls the last 48 h of METARs from IEM for every station inside the box, reusing the
fetch/retry code in fetch_metar_sea.py. Times are UTC in the data, shown as SGT.
Needs only requests and pandas; no secrets.

  python scripts/build_wind_dashboard.py            # writes site/index.html
"""
import argparse
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

import fetch_metar_sea as fm  # noqa: E402

LON_RANGE, LAT_RANGE = (95.0, 120.0), (-6.0, 6.0)
LAND_BOX = (88.0, -13.0, 127.0, 13.0)  # land clip, wider than the station box so panning has context
NSTEPS = 48
LAND_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"


def listing(fetcher, code):
    """(sid, name, lon, lat) for a network, [] if missing/empty."""
    r = fetcher.get(fm.NETWORK_URL.format(net=f"{code}__ASOS"))
    if r.status_code != 200:
        return []
    out = []
    for f in r.json().get("features", []):
        p = f.get("properties") or {}
        sid = p.get("sid") or f.get("id")
        coords = (f.get("geometry") or {}).get("coordinates")
        if sid and coords:
            out.append((str(sid).strip(), p.get("sname") or str(sid), float(coords[0]), float(coords[1])))
    return out


def clip_ring(ring, box):
    """Sutherland-Hodgman clip of a polygon ring to an axis-aligned box."""
    x0, y0, x1, y1 = box
    edges = [
        (lambda p: p[0] >= x0, lambda a, b: (x0, a[1] + (b[1] - a[1]) * (x0 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= x1, lambda a, b: (x1, a[1] + (b[1] - a[1]) * (x1 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= y0, lambda a, b: (a[0] + (b[0] - a[0]) * (y0 - a[1]) / (b[1] - a[1]), y0)),
        (lambda p: p[1] <= y1, lambda a, b: (a[0] + (b[0] - a[0]) * (y1 - a[1]) / (b[1] - a[1]), y1)),
    ]
    pts = [tuple(p[:2]) for p in ring]
    for inside, cross in edges:
        if not pts:
            break
        out, prev = [], pts[-1]
        for cur in pts:
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
            prev = cur
        pts = out
    return pts


def land_rings(path):
    gj = json.loads(Path(path).read_text())
    rings = []
    for ft in gj["features"]:
        g = ft["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = poly[0]  # exterior only
            xs, ys = [p[0] for p in ring], [p[1] for p in ring]
            if max(xs) < LAND_BOX[0] or min(xs) > LAND_BOX[2] or max(ys) < LAND_BOX[1] or min(ys) > LAND_BOX[3]:
                continue
            clipped = clip_ring(ring, LAND_BOX)
            flat, last = [], None
            for x, y in clipped:
                q = (round(x, 2), round(y, 2))
                if q != last:
                    flat += [q[0], q[1]]
                    last = q
            if len(flat) >= 8:
                rings.append(flat)
    return rings


def num(s):
    return pd.to_numeric(s, errors="coerce")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", default=HERE / "wind_dashboard_template.html")
    ap.add_argument("--land", default="output/ne_50m_land.geojson", help="Natural Earth land outline (downloaded if missing)")
    ap.add_argument("--output", default="site/index.html")
    ap.add_argument("--sleep", type=float, default=1.5)
    args = ap.parse_args()

    fetcher = fm.Fetcher(args.sleep)

    # last full hour whose +/-30 min obs window has closed
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    h_end = (now - dt.timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    step0 = h_end - dt.timedelta(hours=NSTEPS - 1)
    axis_start = step0 - dt.timedelta(minutes=30)
    axis_end = h_end + dt.timedelta(minutes=30)
    print(f"now {now:%Y-%m-%d %H:%M}Z; steps {step0:%m-%d %H}Z .. {h_end:%m-%d %H}Z; axis {axis_start:%m-%d %H:%M}..{axis_end:%m-%d %H:%M}Z", file=sys.stderr)

    # stations inside the box, from every SEA network listing
    stations = {}
    for code in fm.COUNTRIES:
        for sid, name, lon, lat in listing(fetcher, code):
            if LON_RANGE[0] <= lon <= LON_RANGE[1] and LAT_RANGE[0] <= lat <= LAT_RANGE[1]:
                stations.setdefault(sid, dict(id=sid, name=name.title(), lon=round(lon, 3), lat=round(lat, 3)))
    ids = sorted(stations)
    print(f"{len(ids)} stations in box", file=sys.stderr)

    utc = dt.timezone.utc
    req = fm.Requester(fetcher, (axis_start - dt.timedelta(minutes=1)).replace(tzinfo=utc), (axis_end + dt.timedelta(minutes=31)).replace(tzinfo=utc))
    frames = []
    for i in range(0, len(ids), 15):
        batch = ids[i:i + 15]
        print(f"requesting {batch[0]}..{batch[-1]} ({len(batch)})", file=sys.stderr)
        try:
            frames.append(req.fetch(batch))
        except RuntimeError as e:
            print(f"  batch failed ({e}); trying singly", file=sys.stderr)
            for sid in batch:
                try:
                    frames.append(req.fetch([sid]))
                except RuntimeError as e2:
                    print(f"  {sid} failed: {e2}", file=sys.stderr)
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    df["t"] = pd.to_datetime(df["valid"])
    df = df[(df.t >= axis_start) & (df.t <= axis_end)].drop_duplicates(["station", "t", "metar"]).sort_values(["station", "t"])

    def col(g, name, f=lambda v: v, nd=1):
        out = []
        for v in num(g[name]):
            out.append(None if pd.isna(v) else round(float(f(v)), nd))
        return out

    for sid, g in df.groupby("station"):
        if sid not in stations:
            continue
        s = stations[sid]
        s["t"] = [int((t - axis_start).total_seconds() // 60) for t in g.t]
        s["spd"] = col(g, "sknt", nd=0)
        s["gst"] = col(g, "gust", nd=0)
        s["dir"] = col(g, "drct", nd=0)
        s["vis"] = col(g, "vsby", lambda v: min(v * 1.609344, 10.0))   # miles -> km, capped at 10 (9999 m)
        s["tmp"] = col(g, "tmpf", lambda v: (v - 32) * 5 / 9)
        s["dew"] = col(g, "dwpf", lambda v: (v - 32) * 5 / 9)
        s["metar"] = [str(m) for m in g["metar"]]
    for s in stations.values():
        for k in ("t", "spd", "gst", "dir", "vis", "tmp", "dew", "metar"):
            s.setdefault(k, [])

    land = Path(args.land)
    if not land.exists():
        land.parent.mkdir(parents=True, exist_ok=True)
        land.write_bytes(urllib.request.urlopen(LAND_URL, timeout=120).read())

    epoch = dt.datetime(1970, 1, 1)
    data = dict(
        built=now.strftime("%Y-%m-%d %H:%M"),
        t0=int((axis_start - epoch).total_seconds() * 1000),
        step0=int((step0 - epoch).total_seconds() * 1000),
        nsteps=NSTEPS,
        span=int((axis_end - axis_start).total_seconds() // 60),
        box=dict(lon=LON_RANGE, lat=LAT_RANGE),
        stations=[stations[i] for i in ids],
        land=land_rings(land),
    )
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = Path(args.template).read_text().replace("__DATA__", blob)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(html)
    n_with = sum(1 for s in data["stations"] if s["t"])
    print(f"wrote {args.output} ({len(html)/1024:.0f} KB); {n_with}/{len(ids)} stations with data; "
          f"{sum(len(s['t']) for s in data['stations'])} obs; {len(data['land'])} land rings", file=sys.stderr)
    print("no data:", [s["id"] for s in data["stations"] if not s["t"]], file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
