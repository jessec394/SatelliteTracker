import requests
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
from math import atan2, sqrt, asin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
from matplotlib.widgets import CheckButtons, TextBox, Slider

ANIMATION_UPDATE_INTERVAL = 10       # Seconds between position updates
TIMESTAMP_REFRESH_INTERVAL = 0.5     # Seconds between timestamp updates
DATA_UPDATE_INTERVAL = 7200          # Seconds between TLE refetches (2h)
MAX_STEPS = 4000                     # Steps to propagate trajectory segments
CHECKBOX_DISPLAY_LIMIT = 15          # Max checkboxes visible at once

TLE_SOURCES = [
    "https://celestrak.org/NORAD/elements/stations.txt",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
]

def SatelliteCatalog(URLs):
    Catalog = {}
    for link in URLs:
        try:
            Response = requests.get(link, timeout=10)
            Response.raise_for_status()
            Lines = Response.text.splitlines()
            for idx in range(0, len(Lines) - 2, 3):
                Name = Lines[idx].strip()
                if Name: Catalog[Name] = link
        except Exception as e:
            print(f"Could not load Catalog from {link}: {e}")
    return Catalog

class SatelliteTracker:
    def __init__(self, Catalog):
        self.Catalog = Catalog
        self.DataCache = {}
        self.SatObjects = {}
        self.LastFetch = {}

    def _download_file(self, URL):
        Response = requests.get(URL, timeout=10)
        Response.raise_for_status()
        return Response.text.splitlines()

    def FetchTLE(self, SatelliteName, Interval=DATA_UPDATE_INTERVAL):
        if SatelliteName not in self.Catalog: return
        NeedNewData = (
            SatelliteName not in self.DataCache
            or (time.time() - self.LastFetch.get(SatelliteName, 0)) > Interval
        )
        if not NeedNewData: return

        URL = self.Catalog[SatelliteName]
        try:
            Lines = self._download_file(URL)
            for idx in range(0, len(Lines) - 2):
                if Lines[idx].strip().upper() == SatelliteName.upper():
                    l1, l2 = Lines[idx + 1].strip(), Lines[idx + 2].strip()
                    self.DataCache[SatelliteName] = (l1, l2)
                    self.SatObjects[SatelliteName] = Satrec.twoline2rv(l1, l2)
                    self.LastFetch[SatelliteName] = time.time()
                    return
            print(f"{SatelliteName} not found inside {URL}")
        except Exception as e:
            print(f"Error fetching {SatelliteName} from {URL}: {e}")

    def RefreshTLE(self, Satellites, Interval=DATA_UPDATE_INTERVAL):
        for name in Satellites: self.FetchTLE(name, Interval)

def ECItoGeodetic(x, y, z, dt):
    def JulianDate(dt):
        return (
            367 * dt.year
            - (7 * (dt.year + (dt.month + 9) // 12)) // 4
            + (275 * dt.month) // 9
            + dt.day
            + 1721013.5
            + (dt.hour + dt.minute / 60 + dt.second / 3600) / 24
        )

    JD = JulianDate(dt)
    T = (JD - 2451545.0) / 36525
    theta = np.radians((280.46061837 + 360.98564736629 * (JD - 2451545) + 0.000387933 * T**2 - T**3 / 38710000.0) % 360)
    xECEF = np.cos(theta) * x + np.sin(theta) * y
    yECEF = -np.sin(theta) * x + np.cos(theta) * y
    r = sqrt(xECEF**2 + yECEF**2 + z**2)
    return asin(z / r), atan2(yECEF, xECEF)

def StepSat(Satellite, t):
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
    err, pos, _ = Satellite.sgp4(jd, fr)
    if err != 0: raise RuntimeError(f"SGP4 error: {err}")
    x, y, z = pos
    Lat, Lon = ECItoGeodetic(x, y, z, t)
    return np.degrees(Lat), np.degrees(Lon)

def PropagateTracks(Satellite, Start, Direction):
    Lats, Lons = [], []
    t = Start
    LastLon = None
    for _ in range(MAX_STEPS):
        t += timedelta(seconds=Direction * ANIMATION_UPDATE_INTERVAL)
        Lat, Lon = StepSat(Satellite, t)
        LonWrapped = (Lon + 180) % 360 - 180
        if LastLon is not None and abs(LastLon - LonWrapped) > 180 - 1e-6: break
        Lats.append(Lat)
        Lons.append(LonWrapped)
        LastLon = LonWrapped
    return Lats, Lons

def UpdatePositions(Tracker, CurrentTime, TrackedSats):
    Positions = {}
    for SatelliteName in TrackedSats:
        SatObject = Tracker.SatObjects.get(SatelliteName)
        if SatObject is None: continue
        try:
            lat, lon = StepSat(SatObject, CurrentTime)
            lon = (lon + 180) % 360 - 180
            lat_past, lon_past = PropagateTracks(SatObject, CurrentTime, Direction=-1)
            lat_future, lon_future = PropagateTracks(SatObject, CurrentTime, Direction=+1)
            lat_past.reverse()
            lon_past.reverse()
            Positions[SatelliteName] = {
                "lat": lat,
                "lon": lon,
                "lat_past": lat_past,
                "lon_past": lon_past,
                "lat_future": lat_future,
                "lon_future": lon_future,
            }
        except Exception as e:
            print(f"Error updating {SatelliteName}: {e}")
    return Positions

def UpdateAnimation():
    Catalog = SatelliteCatalog(TLE_SOURCES)
    Tracker = SatelliteTracker(Catalog)

    plt.ion()
    fig = plt.figure(figsize=(18, 8), facecolor="#121212")
    manager = plt.get_current_fig_manager()
    try: manager.window.showFullScreen()
    except Exception: pass

    axMap = plt.axes([0.3, 0.05, 0.65, 0.9], projection=ccrs.PlateCarree(), facecolor="#121212")
    axMap.set_global()
    axMap.add_feature(cfeature.LAND, facecolor="#262626", zorder=0)
    axMap.add_feature(cfeature.OCEAN, facecolor="#0a224a", zorder=0)
    axMap.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#a0a0a0")
    axMap.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, edgecolor="#888888")
    gl = axMap.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", color="#555555")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {"size": 9, "color": "#bbbbbb", "weight": "bold"}
    gl.ylabel_style = {"size": 9, "color": "#bbbbbb", "weight": "bold"}

    axSearch = plt.axes([0.02, 0.87, 0.23, 0.02], facecolor="#262626")
    Text = TextBox(axSearch, "Search:", initial="", textalignment="left", color="white")
    axCheckbox = plt.axes([0.02, 0.15, 0.23, 0.7], facecolor="#2b2b2b")
    axScrollbar = plt.axes([0.255, 0.15, 0.01, 0.7], facecolor="#1d1d1d")
    ScrollSlider = Slider(
        axScrollbar,
        label="",
        valmin=0,
        valmax=1,
        valinit=0,
        orientation="vertical",
        facecolor="#4d4d4d",
        handle_style={"facecolor": "#bbbbbb", "edgecolor": "#eeeeee"},
    )

    ScrollSlider.ax.tick_params(left=False, labelleft=False)
    for spine in ScrollSlider.ax.spines.values(): spine.set_visible(False)

    SatListFull = sorted(Catalog.keys())
    SatListFiltered = SatListFull.copy()
    ScrollPos = 0
    CheckedSatellites = set()

    ColorMap = plt.get_cmap("tab20")
    Colors = [ColorMap(i) for i in range(ColorMap.N)]

    ButtonCheck = None
    Lines, Scatters = [], []
    TimestampText = None
    CachedPositions = {}

    Running = True

    def on_close(event):
        nonlocal Running
        Running = False

    fig.canvas.mpl_connect("close_event", on_close)

    def RedrawCheckboxes():
        nonlocal ScrollPos, ButtonCheck, SatListFiltered

        MaxScroll = max(0, len(SatListFiltered) - CHECKBOX_DISPLAY_LIMIT)
        ScrollPos = max(0, min(ScrollPos, MaxScroll))

        ScrollSlider.valmax = MaxScroll if MaxScroll else 1
        ScrollSlider.ax.set_ylim(ScrollSlider.valmin, ScrollSlider.valmax)
        OldEventsOn = ScrollSlider.eventson
        ScrollSlider.eventson = False
        ScrollSlider.set_val(ScrollPos)
        ScrollSlider.eventson = OldEventsOn

        plt.sca(axCheckbox)
        axCheckbox.clear()
        axCheckbox.set_xticks([])
        axCheckbox.set_yticks([])
        for sp in axCheckbox.spines.values(): sp.set_visible(False)

        VisibleSats = SatListFiltered[ScrollPos : ScrollPos + CHECKBOX_DISPLAY_LIMIT]
        ActiveSats = [name in CheckedSatellites for name in VisibleSats]

        if not VisibleSats:
            axCheckbox.text(
                0.5,
                0.5,
                "No matches",
                color="white",
                ha="center",
                va="center",
                fontsize=12,
            )
            fig.canvas.draw_idle()
            return

        ButtonCheck = CheckButtons(axCheckbox, VisibleSats, ActiveSats)
        for text in ButtonCheck.labels: text.set_color("white")

        def on_check(label):
            if label in CheckedSatellites: CheckedSatellites.remove(label)
            else: CheckedSatellites.add(label)
            RedrawMap(ForcePositionUpdate=True)

        ButtonCheck.on_clicked(on_check)
        fig.canvas.draw_idle()

    def RedrawMap(ForcePositionUpdate=False):
        nonlocal TimestampText, CachedPositions

        CurrentTime = datetime.utcnow()
        if ForcePositionUpdate:
            Tracker.RefreshTLE(CheckedSatellites)
            CachedPositions = UpdatePositions(Tracker, CurrentTime, CheckedSatellites)

        if TimestampText is not None: TimestampText.remove()
        for artist in (*Lines, *Scatters): artist.remove()
        Lines.clear()
        Scatters.clear()

        for idx, (SatelliteName, pos) in enumerate(CachedPositions.items()):
            color = Colors[idx % len(Colors)]
            l1, = axMap.plot(
                pos["lon_past"],
                pos["lat_past"],
                color=color,
                linewidth=2.5,
                transform=ccrs.Geodetic(),
            )
            l2, = axMap.plot(
                pos["lon_future"],
                pos["lat_future"],
                color=color,
                linewidth=2.5,
                linestyle="--",
                transform=ccrs.Geodetic(),
            )
            s1 = axMap.scatter(
                pos["lon"],
                pos["lat"],
                s=100,
                color=color,
                edgecolor="black",
                zorder=5,
                transform=ccrs.PlateCarree(),
            )
            txt = axMap.text(
                pos["lon"] + 3,
                pos["lat"] + 1.5,
                SatelliteName,
                color=color,
                fontsize=9,
                transform=ccrs.PlateCarree(),
                zorder=6,
                weight="bold",
                ha="left",
                va="center",
            )
            txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground="black"), path_effects.Normal()])
            Lines.extend([l1, l2])
            Scatters.extend([s1, txt])

        axMap.set_title(
            f"Tracking {len(CheckedSatellites)} satellite{'s' if len(CheckedSatellites) != 1 else ''}",
            fontsize=16,
            pad=24,
            fontweight="heavy",
            fontfamily="DejaVu Sans",
            color="#ffffff",
            ha="center",
            va="bottom",
        )
        TimestampText = axMap.text(
            0.5, 1.02,
            f"{CurrentTime:%Y-%m-%d} (UTC {CurrentTime:%H:%M:%S})",
            fontsize=12,
            fontweight="normal",
            fontfamily="DejaVu Sans",
            color="#ffffff",
            ha="center",
            va="bottom",
            transform=axMap.transAxes,
        )
        fig.canvas.draw_idle()

    def SearchEnter(text):
        nonlocal SatListFiltered, ScrollPos
        SatListFiltered = [s for s in SatListFull if text.lower() in s.lower()]
        ScrollPos = 0
        RedrawCheckboxes()

    Text.on_submit(SearchEnter)

    def OnSlider(val):
        nonlocal ScrollPos
        NewPos = int(round(val))
        if NewPos != ScrollPos:
            ScrollPos = NewPos
            RedrawCheckboxes()

    ScrollSlider.on_changed(OnSlider)

    def OnScroll(event):
        nonlocal ScrollPos
        if event.inaxes == axCheckbox:
            delta = -1 if event.button == "up" else 1
            MaxScroll = max(0, len(SatListFiltered) - CHECKBOX_DISPLAY_LIMIT)
            ScrollPos = max(0, min(ScrollPos + delta, MaxScroll))
            OldEventsOn = ScrollSlider.eventson
            ScrollSlider.eventson = False
            ScrollSlider.set_val(ScrollPos)
            ScrollSlider.eventson = OldEventsOn
            RedrawCheckboxes()

    fig.canvas.mpl_connect("scroll_event", OnScroll)

    RedrawCheckboxes()
    RedrawMap(ForcePositionUpdate=True)

    LastPositionUpdate = time.time()
    LastTimestampRefresh = time.time()

    while Running:
        try:
            Now = time.time()
            if Now - LastPositionUpdate >= ANIMATION_UPDATE_INTERVAL:
                RedrawMap(ForcePositionUpdate=True)
                LastPositionUpdate = Now
                LastTimestampRefresh = Now
            elif Now - LastTimestampRefresh >= TIMESTAMP_REFRESH_INTERVAL:
                RedrawMap(ForcePositionUpdate=False)
                LastTimestampRefresh = Now
            plt.pause(0.1)
        except Exception as e:
            print("Error in main loop:", e)
            plt.pause(1)

if __name__ == "__main__":
    UpdateAnimation()