import requests
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
from math import atan2, sqrt, asin
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time

ANIMATION_UPDATE_INTERVAL = 30
DATA_UPDATE_INTERVAL = 7200
MAX_STEPS = 4000

def FetchData():
    url = "https://celestrak.org/NORAD/elements/stations.txt"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    Lines = resp.text.splitlines()
    for idx, line in enumerate(Lines):
        if "ISS" in line:
            return Lines[idx + 1].strip(), Lines[idx + 2].strip()
    raise RuntimeError("ISS TLE not found")

def ECItoGeodetic(x: float, y: float, z: float, dt: datetime):
    jd = (
        367 * dt.year
        - (7 * (dt.year + (dt.month + 9) // 12)) // 4
        + (275 * dt.month) // 9
        + dt.day
        + 1721013.5
        + (dt.hour + dt.minute / 60 + dt.second / 3600) / 24
    )
    T = (jd - 2451545.0) / 36525
    theta = (
        280.46061837
        + 360.98564736629 * (jd - 2451545)
        + 0.000387933 * T**2
        - T**3 / 38710000.0
    ) % 360
    theta = np.radians(theta)
    x_ecef = np.cos(theta) * x + np.sin(theta) * y
    y_ecef = -np.sin(theta) * x + np.cos(theta) * y
    z_ecef = z
    r = sqrt(x_ecef**2 + y_ecef**2 + z_ecef**2)
    Lat = asin(z_ecef / r)
    Lon = atan2(y_ecef, x_ecef)
    return Lat, Lon

def StepSat(Satellite: Satrec, t: datetime):
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
    err, pos, _ = Satellite.sgp4(jd, fr)
    if err != 0: raise RuntimeError(f"SGP4 error: {err}")
    x, y, z = pos
    Lat, Lon = ECItoGeodetic(x, y, z, t)
    return np.degrees(Lat), np.degrees(Lon)

def PropagateTracks(Satellite: Satrec, start: datetime, direction: int):
    Lats, Lons = [], []
    t = start
    LastLon = None
    for _ in range(MAX_STEPS):
        t += timedelta(seconds=direction * ANIMATION_UPDATE_INTERVAL)
        Lat, Lon = StepSat(Satellite, t)
        LonWrapped = (Lon + 180) % 360 - 180
        Lats.append(Lat)
        Lons.append(LonWrapped)
        if LastLon is not None and abs(LastLon - LonWrapped) > 180 - 1e-6:
            break
        LastLon = LonWrapped
    return Lats, Lons

def ComputeTracks(TLE1: str, TLE2: str):
    Satellite = Satrec.twoline2rv(TLE1, TLE2)
    CurrentTime = datetime.utcnow()
    Lat2, Lon2 = StepSat(Satellite, CurrentTime)
    Lon2 = (Lon2 + 180) % 360 - 180
    Lat1s, Lon1s = PropagateTracks(Satellite, CurrentTime, direction=-1)
    FutureLats, FutureLons = PropagateTracks(Satellite, CurrentTime, direction=+1)
    Lat1s.reverse()
    Lon1s.reverse()
    return Lat1s, Lon1s, Lat2, Lon2, FutureLats, FutureLons, CurrentTime, Satellite

def Update():
    plt.ion()
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(14, 7), facecolor="#1e1e1e")
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    ax = plt.axes(projection=proj, facecolor="#1e1e1e")

    Running = True
    def OnClose(event):
        nonlocal Running
        Running = False
    fig.canvas.mpl_connect('close_event', OnClose)

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#2b2b2b", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#1e2a40", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="white")
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.3, edgecolor="gray")
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, linestyle="--", color="gray")
    gl.xlabel_style = {"size": 8, "color": "white"}
    gl.ylabel_style = {"size": 8, "color": "white"}
    gl.top_labels = gl.right_labels = False

    try:
        TLE1, TLE2 = FetchData()
        Lat1, Lon1, Lat2, Lon2, Lat3, Lon3, LastUpdate, Satellite = ComputeTracks(TLE1, TLE2)
    except Exception as e:
        print("Initial data fetch error:", e)
        return

    LastCompute = time.time()
    LastPositionUpdate = time.time()

    Lines = []
    Scatters = []

    while Running:
        try:
            CurrentTime = time.time()
            CurrentUTC = datetime.utcnow()

            if CurrentTime - LastCompute >= DATA_UPDATE_INTERVAL:
                TLE1, TLE2 = FetchData()
                Lat1, Lon1, Lat3, Lon3, _, _, LastUpdate, Satellite = ComputeTracks(TLE1, TLE2)
                LastCompute = CurrentTime

            if CurrentTime - LastPositionUpdate >= ANIMATION_UPDATE_INTERVAL:
                Lat2, Lon2 = StepSat(Satellite, CurrentUTC)
                Lon2 = (Lon2 + 180) % 360 - 180
                LastPositionUpdate = CurrentTime

            for line in Lines:
                line.remove()
            for scatter in Scatters:
                scatter.remove()
            Lines.clear()
            Scatters.clear()

            l1, = ax.plot(Lon1, Lat1, color="#ff4e50", linewidth=2.5, transform=ccrs.Geodetic())
            l2, = ax.plot(Lon3, Lat3, color="#ff4e50", linewidth=2.5, linestyle="--", transform=ccrs.Geodetic())
            s1 = ax.scatter(Lon2, Lat2, s=100, color="#ffffff", edgecolor="black", zorder=5, transform=ccrs.PlateCarree())

            Lines.extend([l1, l2])
            Scatters.append(s1)

            ax.set_title(
                f"International Space Station\n{CurrentUTC:%Y-%m-%d} | UTC {CurrentUTC:%H:%M:%S}",
                fontsize=16,
                pad=20,
                color="#ffffff"
            )

            plt.draw()
            plt.pause(0.1)

        except Exception as e:
            print("Error in update loop:", e)
            plt.pause(5)

if __name__ == "__main__":
    Update()