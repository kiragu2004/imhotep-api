
import os, re, time, secrets, hmac, hashlib, json, pathlib, math
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import httpx, jwt
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr

SECRET_KEY       = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
NVIDIA_API_KEY   = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL     = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
NVIDIA_URL       = "https://integrate.api.nvidia.com/v1/chat/completions"
OPENWEATHER_KEY  = os.getenv("OPENWEATHER_API_KEY", "")
NESTLINK_API_KEY = os.getenv("NESTLINK_API_KEY", "")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
CORS_ORIGINS     = [o.strip() for o in os.getenv("CORS_ORIGINS","*").split(",") if o.strip()]

app = FastAPI(title="Imhotep", version="7.0.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

_LEAK = [(r"\bNVIDIA\s+NIM\b","Imhotep"),(r"\bNVIDIA\b","Imhotep"),
         (r"\bNemotron[\w\-\.]*\b","Imhotep"),(r"\bNIM\b","Imhotep"),
         (r"\bQwen[\w\-\.]*\b","Imhotep"),(r"\bDeepSeek[\w\-]*\b","Imhotep"),
         (r"\bMistral[\w\-]*\b","Imhotep"),(r"\bLlama[\s\-]?[\d\.\w]*\b","Imhotep")]
def scrub(t):
    if not t: return t
    for p, r in _LEAK: t = re.sub(p, r, t, flags=re.IGNORECASE)
    return t

# ═══════════════════════════════════════════════════════════════
# ALL 47 KENYAN COUNTIES WITH COORDINATES
# ═══════════════════════════════════════════════════════════════
KENYA_COUNTIES_FULL = {
    "Mombasa":(-4.0435,39.6682),"Kwale":(-4.1742,39.4524),"Kilifi":(-3.6305,39.8499),
    "Tana River":(-1.5000,39.9833),"Lamu":(-2.2717,40.9020),"Taita Taveta":(-3.4000,38.5500),
    "Garissa":(-0.4536,39.6461),"Wajir":(1.7471,40.0573),"Mandera":(3.9366,41.8670),
    "Marsabit":(2.3344,37.9899),"Isiolo":(0.3546,37.5822),"Meru":(0.0500,37.6500),
    "Tharaka-Nithi":(-0.1667,37.9000),"Embu":(-0.5333,37.4500),"Kitui":(-1.3667,38.0167),
    "Machakos":(-1.5177,37.2634),"Makueni":(-2.0000,37.6167),"Nyandarua":(-0.5000,36.3667),
    "Nyeri":(-0.4167,36.9500),"Kirinyaga":(-0.5000,37.2833),"Murang'a":(-0.7167,37.1500),
    "Kiambu":(-1.1667,36.8333),"Turkana":(3.1191,35.5973),"West Pokot":(1.2333,35.1167),
    "Samburu":(1.1000,36.7000),"Trans Nzoia":(1.0167,35.0000),"Uasin Gishu":(0.5143,35.2698),
    "Elgeyo-Marakwet":(0.5167,35.5000),"Nandi":(0.1833,35.1000),"Baringo":(0.4667,35.9667),
    "Laikipia":(0.0500,36.7000),"Nakuru":(-0.3031,36.0800),"Narok":(-1.0833,35.8667),
    "Kajiado":(-1.8500,36.7833),"Kericho":(-0.3667,35.2833),"Bomet":(-0.7833,35.3333),
    "Kakamega":(0.2827,34.7519),"Vihiga":(0.0500,34.7333),"Bungoma":(0.5667,34.5667),
    "Busia":(0.4500,34.1167),"Siaya":(0.0667,34.2833),"Kisumu":(-0.0917,34.7680),
    "Homa Bay":(-0.5167,34.4500),"Migori":(-0.9833,34.4667),"Kisii":(-0.6817,34.7680),
    "Nyamira":(-0.5667,34.9333),"Nairobi":(-1.2921,36.8219),
}
KENYA_COUNTIES = {
    "nairobi":{"code":"047","capital":"Nairobi"},"mombasa":{"code":"001","capital":"Mombasa"},
    "kisumu":{"code":"042","capital":"Kisumu"},"nakuru":{"code":"032","capital":"Nakuru"},
    "uasin gishu":{"code":"027","capital":"Eldoret"},"nyeri":{"code":"019","capital":"Nyeri"},
    "machakos":{"code":"016","capital":"Machakos"},"garissa":{"code":"007","capital":"Garissa"},
    "turkana":{"code":"023","capital":"Lodwar"},"kakamega":{"code":"037","capital":"Kakamega"},
    "meru":{"code":"012","capital":"Meru"},"kisii":{"code":"045","capital":"Kisii"},
}

def classify_wx(temp, code):
    try: c = int(code)
    except: c = 800
    if c == 800: e,l="☀️","Clear"
    elif c in (801,802): e,l="🌤️","Partly cloudy"
    elif c in (803,804): e,l="☁️","Cloudy"
    elif 200 <= c < 300: e,l="⛈️","Thunderstorm"
    elif 300 <= c < 400: e,l="🌦️","Drizzle"
    elif 500 <= c < 600: e,l="🌧️","Rain"
    elif 600 <= c < 700: e,l="❄️","Snow"
    elif 700 <= c < 800: e,l="🌫️","Foggy"
    elif c in (0,1): e,l="☀️","Clear"
    elif c == 2: e,l="🌤️","Partly cloudy"
    elif c == 3: e,l="☁️","Cloudy"
    elif c in (45,48): e,l="🌫️","Foggy"
    elif c in (61,63,65): e,l="🌧️","Rain"
    elif c in (95,96,99): e,l="⛈️","Thunderstorm"
    else: e,l="🌤️","Fair"
    if temp>=35: f="scorching hot 🥵"
    elif temp>=30: f="very hot 🥵"
    elif temp>=26: f="hot 🔥"
    elif temp>=20: f="warm 😊"
    elif temp>=14: f="cool 😌"
    elif temp>=8: f="cold 🧥"
    else: f="very cold 🥶"
    return {"emoji":e,"condition_label":l,"feel":f}

# ═══════════════════════════════════════════════════════════════
# ALL 47 COUNTIES WEATHER — Open-Meteo (no key), OpenWeather optional
# ═══════════════════════════════════════════════════════════════
@app.get("/weather/all-counties")
async def weather_all_counties():
    """Weather for all 47 Kenyan counties. Open-Meteo, no key."""
    results = []
    async with httpx.AsyncClient(timeout=60) as c:
        # Batch requests — Open-Meteo allows multiple lat/lon in one call
        lats = ",".join(str(v[0]) for v in KENYA_COUNTIES_FULL.values())
        lons = ",".join(str(v[1]) for v in KENYA_COUNTIES_FULL.values())
        try:
            r = await c.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude":lats, "longitude":lons,
                        "current":"temperature_2m,weather_code",
                        "timezone":"Africa/Nairobi"})
            data = r.json()
            if isinstance(data, list):
                items = data
            else:
                items = [data]
            names = list(KENYA_COUNTIES_FULL.keys())
            for i, item in enumerate(items):
                if i >= len(names): break
                cur = item.get("current") or {}
                if cur.get("temperature_2m") is None: continue
                temp = round(cur["temperature_2m"])
                cls = classify_wx(temp, cur.get("weather_code", 0))
                results.append({"county": names[i], "temp": temp, **cls})
        except Exception as e:
            print(f"batch error: {e}")
            # Fallback: one by one
            for name, (la, lo) in KENYA_COUNTIES_FULL.items():
                try:
                    r = await c.get("https://api.open-meteo.com/v1/forecast",
                        params={"latitude":la,"longitude":lo,
                                "current":"temperature_2m,weather_code",
                                "timezone":"Africa/Nairobi"})
                    cur = (r.json().get("current") or {})
                    if cur.get("temperature_2m") is not None:
                        temp = round(cur["temperature_2m"])
                        cls = classify_wx(temp, cur.get("weather_code", 0))
                        results.append({"county": name, "temp": temp, **cls})
                except Exception: continue
    return {"ok": True, "count": len(results), "counties": results,
            "source": "open-meteo.com"}

# ═══════════════════════════════════════════════════════════════
# CLIMATE CHANGE — Open-Meteo Climate API (CMIP6 projections)
# ═══════════════════════════════════════════════════════════════
@app.get("/climate/kenya")
async def climate_kenya():
    """Kenya climate projections using Open-Meteo Climate API (CMIP6). No key."""
    # Nairobi as reference point for Kenya-wide projection
    lat, lon = -1.2921, 36.8219
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            # Historical baseline (last 5 years)
            r_hist = await c.get("https://archive-api.open-meteo.com/v1/archive",
                params={"latitude":lat,"longitude":lon,
                        "start_date":"2019-01-01","end_date":"2023-12-31",
                        "daily":"temperature_2m_max,temperature_2m_min,precipitation_sum",
                        "timezone":"Africa/Nairobi"})
            hist = r_hist.json()

            # Climate projection (next 30 years)
            r_proj = await c.get("https://climate-api.open-meteo.com/v1/climate",
                params={"latitude":lat,"longitude":lon,
                        "start_date":"2025-01-01","end_date":"2055-12-31",
                        "models":"MRI_AGCM3_2_S,EC_Earth3P_HR",
                        "daily":"temperature_2m_max,temperature_2m_min,precipitation_sum"})
            proj = r_proj.json()
    except Exception as e:
        raise HTTPException(502, f"Climate API error: {e}")

    def _avg(arr):
        vals = [v for v in (arr or []) if isinstance(v,(int,float))]
        return round(sum(vals)/len(vals), 2) if vals else None

    hist_daily = hist.get("daily", {})
    proj_daily = proj.get("daily", {})

    hist_tmax = _avg(hist_daily.get("temperature_2m_max", []))
    hist_tmin = _avg(hist_daily.get("temperature_2m_min", []))
    hist_rain = _avg(hist_daily.get("precipitation_sum", []))

    proj_tmax = _avg(proj_daily.get("temperature_2m_max", []))
    proj_tmin = _avg(proj_daily.get("temperature_2m_min", []))
    proj_rain = _avg(proj_daily.get("precipitation_sum", []))

    delta_t = None
    if hist_tmax is not None and proj_tmax is not None:
        delta_t = round(proj_tmax - hist_tmax, 2)

    delta_r = None
    if hist_rain is not None and proj_rain is not None:
        delta_r = round(proj_rain - hist_rain, 3)

    return {
        "ok": True,
        "location": "Kenya (Nairobi reference)",
        "historical": {
            "period": "2019-2023",
            "avg_tmax_c": hist_tmax, "avg_tmin_c": hist_tmin,
            "avg_daily_rain_mm": hist_rain,
        },
        "projection": {
            "period": "2025-2055",
            "avg_tmax_c": proj_tmax, "avg_tmin_c": proj_tmin,
            "avg_daily_rain_mm": proj_rain,
        },
        "change": {
            "warming_c": delta_t,
            "rain_change_mm": delta_r,
            "verdict": (
                "Kenya is warming — prepare for heat stress" if delta_t and delta_t > 0.5
                else "Stable" if delta_t and abs(delta_t) <= 0.5
                else "Cooling — unusual pattern"
            ),
        },
        "source": "Open-Meteo Climate API (CMIP6) — no key required",
        "models": ["MRI_AGCM3_2_S", "EC_Earth3P_HR"],
    }

# ═══════════════════════════════════════════════════════════════
# SINGLE COUNTY WEATHER
# ═══════════════════════════════════════════════════════════════
@app.post("/weather")
async def weather_ep(body: dict):
    lat, lon, city = body.get("lat"), body.get("lon"), body.get("city")
    if (lat is None or lon is None) and city:
        if city in KENYA_COUNTIES_FULL:
            lat, lon = KENYA_COUNTIES_FULL[city]
        else:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get("https://geocoding-api.open-meteo.com/v1/search",
                                params={"name":city,"count":1,"language":"en"})
            g = r.json()
            if not g.get("results"): raise HTTPException(404, f"'{city}' not found")
            lat = g["results"][0]["latitude"]
            lon = g["results"][0]["longitude"]
    if lat is None or lon is None: raise HTTPException(400, "Provide lat+lon or city")

    # Try OpenWeather first if key set
    if OPENWEATHER_KEY:
        try:
            params = ({"lat":lat,"lon":lon} if not city else {"q": f"{city},KE"})
            params.update({"appid": OPENWEATHER_KEY, "units": "metric"})
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get("https://api.openweathermap.org/data/2.5/weather", params=params)
            d = r.json()
            if "main" in d:
                temp = round(d["main"]["temp"])
                cls = classify_wx(temp, d["weather"][0]["id"])
                return {"ok":True,"temp":temp,
                        "feels_like":round(d["main"].get("feels_like", temp)),
                        "condition":d["weather"][0]["main"],
                        "humidity":d["main"].get("humidity"),
                        "city":d.get("name") or city or "",
                        "country":d.get("sys",{}).get("country","KE"),
                        **cls, "source":"openweathermap.org"}
        except Exception as e: print(f"OWM fail: {e}")

    # Fallback Open-Meteo
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude":lat,"longitude":lon,
                    "current":"temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "timezone":"Africa/Nairobi"})
    cur = (r.json().get("current") or {})
    temp = round(cur.get("temperature_2m", 0))
    cls = classify_wx(temp, cur.get("weather_code", 0))
    return {"ok":True,"temp":temp,
            "feels_like":round(cur.get("apparent_temperature", temp)),
            "condition":cls["condition_label"],
            "humidity":cur.get("relative_humidity_2m"),
            "wind":cur.get("wind_speed_10m"),
            "city":city or f"{lat:.2f},{lon:.2f}","country":"KE",
            **cls, "source":"open-meteo.com"}

# ═══════════════════════════════════════════════════════════════
# AUTH + CHAT + OTHER ENDPOINTS (compact)
# ═══════════════════════════════════════════════════════════════
users_db, chat_db = {}, {}
TIER_CREDITS = {"guest":3,"free":3,"starter_10":3,"starter_20":20,"starter_50":100,
                "weekly":10000,"monthly":10000,"yearly":10000}
PRICES = {
    "starter_10": {"amount": 10, "label": "Starter — 3 chats"},
    "starter_20": {"amount": 20, "label": "Small pack — 20 chats"},
    "starter_50": {"amount": 50, "label": "Value pack — 100 chats"},
    "weekly": {"amount": 99, "label": "Weekly Unlimited — 7 days"},
    "monthly": {"amount": 299, "label": "Monthly Unlimited — 30 days"},
    "yearly": {"amount": 5000, "label": "Yearly Unlimited — 365 days"},
}
TIER_ALIAS = {"starter": "starter_10", "pro": "monthly"}

class ChatIn(BaseModel):
    prompt: str; agent: Optional[str] = "generalist"; user_id: Optional[str] = None
    temperature: float = 0.7; location: Optional[dict] = None
class SignupIn(BaseModel): email: EmailStr; name: str
class CheckoutIn(BaseModel): tier: str; email: EmailStr
class GoogleIn(BaseModel): id_token: str

def issue(uid, days=30):
    return jwt.encode({"user_id":uid,"exp":datetime.now(timezone.utc)+timedelta(days=days)},
                      SECRET_KEY, algorithm="HS256")
def mk_user(email, name, tier="free"):
    for u, v in users_db.items():
        if v["email"] == email: return (u, False)
    u = f"user_{int(time.time()*1000)}"
    users_db[u] = {"email":email,"name":name,"tier":tier,"credits":TIER_CREDITS.get(tier,3),
                   "credits_used":0,"created_at":datetime.now(timezone.utc).isoformat()}
    chat_db[u] = []
    return (u, True)
def creds(uid):
    if uid not in users_db: return {"used":0,"remaining":0,"limit":0,"allowed":False}
    u = users_db[uid]; t = u.get("credits",0); s = u.get("credits_used",0)
    return {"used":s,"remaining":max(0,t-s),"limit":t,"allowed":s<t}
def debit(uid):
    if uid not in users_db or not creds(uid)["allowed"]: return False
    users_db[uid]["credits_used"] = users_db[uid].get("credits_used",0) + 1
    return True

@app.get("/agents")
async def agents(): return {"ok":True,"agents":list(PERSONAS.keys()) if "PERSONAS" in globals() else []}

@app.post("/auth/signup")
async def signup(b: SignupIn):
    u, new = mk_user(b.email.lower(), b.name, "free")
    return {"ok":True,"user_id":u,"token":issue(u),"user":users_db[u],"new":new}

@app.get("/auth/guest")
async def guest():
    u = f"guest_{int(time.time()*1000)}"
    users_db[u] = {"email":f"{u}@g.local","name":"Guest","tier":"guest","credits":3,
                   "credits_used":0,"created_at":datetime.now(timezone.utc).isoformat()}
    chat_db[u] = []
    return {"ok":True,"user_id":u,"token":issue(u,1),"user":users_db[u]}

@app.post("/auth/google")
async def google(b: GoogleIn):
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post("https://oauth2.googleapis.com/tokeninfo",
                             params={"id_token": b.id_token})
        info = r.json()
    except Exception as e: raise HTTPException(400, f"Google verify failed: {e}")
    email = info.get("email")
    if not email: raise HTTPException(400, "Invalid token")
    u, _ = mk_user(email.lower(), info.get("name","Google User"), "free")
    return {"ok":True,"user_id":u,"token":issue(u),"user":users_db[u]}

# ═══════════════════════════════════════════════════════════════
# NASA FIRES, DROUGHT, NEWS, SATELLITES
# ═══════════════════════════════════════════════════════════════
KENYA_COUNTY_CENTERS = {k:v for k,v in KENYA_COUNTIES_FULL.items()}
def _nearest_county(lat, lon):
    best, bestd = None, 1e9
    for name, (clat, clon) in KENYA_COUNTY_CENTERS.items():
        d = math.sqrt((lat-clat)**2 + (lon-clon)**2)
        if d < bestd: bestd, best = d, name
    return best

@app.get("/fires/kenya")
async def fires_kenya():
    urls = [
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_24h.csv",
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_24h.csv",
    ]
    fires = []
    async with httpx.AsyncClient(timeout=45) as c:
        for url in urls:
            try:
                r = await c.get(url)
                if r.status_code >= 300: continue
                lines = r.text.strip().split("\n")
                if not lines: continue
                header = lines[0].split(",")
                for line in lines[1:]:
                    parts = line.split(",")
                    if len(parts) < len(header): continue
                    row = dict(zip(header, parts))
                    try:
                        la=float(row.get("latitude",0)); lo=float(row.get("longitude",0))
                    except: continue
                    if -5<=la<=6 and 33<=lo<=42:
                        fires.append({"lat":la,"lon":lo,"county":_nearest_county(la,lo) or "Unknown",
                                      "brightness":row.get("bright_ti4") or row.get("brightness"),
                                      "date":row.get("acq_date"),"time":row.get("acq_time"),
                                      "satellite":row.get("satellite")})
            except Exception: continue
    return {"ok":True,"count":len(fires),"fires":fires[:50],"source":"NASA FIRMS"}

DROUGHT_DATA = {"turkana":3,"marsabit":3,"mandera":2,"wajir":2,"garissa":2,"isiolo":2,
                "samburu":3,"west pokot":2,"baringo":2,"kitui":2,"kilifi":2,"tana river":2,
                "nairobi":1,"kiambu":1,"murang'a":1,"nyeri":1,"kirinyaga":1,"kajiado":1}
@app.get("/drought/{county}")
async def drought_status(county: str):
    phase = DROUGHT_DATA.get(county.strip().lower())
    if phase is None:
        return {"ok": False, "error": f"No drought data for '{county}'"}
    labels = {1:"Minimal",2:"Alert",3:"Alarm",4:"Emergency",5:"Famine"}
    return {"ok": True, "county": county.title(), "phase": phase, "label": labels[phase],
            "source": "NDMA (approx)"}

@app.get("/news/kenya")
async def news_kenya(limit: int = 10):
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get("https://www.standardmedia.co.ke/rss/headlines.php",
                            headers={"User-Agent": "Imhotep/1.0"})
        root = ET.fromstring(r.text)
        items = []
        for item in root.findall(".//item")[:limit]:
            items.append({"title":(item.findtext("title") or "").strip(),
                          "link":(item.findtext("link") or "").strip(),
                          "description":(item.findtext("description") or "").strip()[:250]})
        return {"ok":True,"count":len(items),"items":items,"source":"standardmedia.co.ke/rss"}
    except Exception as e: raise HTTPException(502, str(e))

@app.get("/nasa/earth")
async def nasa_earth():
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get("https://epic.gsfc.nasa.gov/api/natural")
    items = r.json()
    if not items: raise HTTPException(404, "No images")
    latest = items[0]
    date_path = latest["date"].split(" ")[0].replace("-","/")
    img = f"https://epic.gsfc.nasa.gov/archive/natural/{date_path}/png/{latest['image']}.png"
    return {"ok":True,"date":latest["date"],"image_url":img,
            "lat":latest["centroid_coordinates"]["lat"],
            "lon":latest["centroid_coordinates"]["lon"]}

@app.get("/nasa/satellite")
async def nasa_satellite(lat: float = -1.29, lon: float = 36.82):
    return {"ok":True,"frame_time":datetime.now(timezone.utc).isoformat(),
            "goes_east_full_disk":"https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/GEOCOLOR/1808x1808.jpg",
            "lat":lat,"lon":lon}

@app.get("/img/proxy")
async def img_proxy(url: str):
    ALLOWED = ("epic.gsfc.nasa.gov","cdn.star.nesdis.noaa.gov","gibs.earthdata.nasa.gov","images-api.nasa.gov")
    from urllib.parse import urlparse
    host = urlparse(url).netloc
    if not any(host.endswith(d) for d in ALLOWED): raise HTTPException(403, "Domain not allowed")
    async with httpx.AsyncClient(timeout=45) as c:
        r = await c.get(url, headers={"User-Agent":"Imhotep/1.0"})
    if r.status_code >= 300: raise HTTPException(r.status_code, "Upstream failed")
    return StreamingResponse(iter([r.content]),
        media_type=r.headers.get("content-type","image/jpeg"),
        headers={"Cache-Control":"public, max-age=600","Access-Control-Allow-Origin":"*"})

@app.get("/health")
async def health():
    return {"ok":True,"version":"7.0.0","model_configured":bool(NVIDIA_API_KEY),
            "weather_provider":"openweathermap" if OPENWEATHER_KEY else "open-meteo",
            "climate_provider":"open-meteo-climate",
            "counties":len(KENYA_COUNTIES_FULL),
            "endpoints":["/weather/all-counties","/climate/kenya","/fires/kenya",
                         "/drought/{county}","/news/kenya","/nasa/earth"],
            "founder":"Samuel Kiragu"}
