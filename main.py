
import os, re, time, secrets, hmac, hashlib, json, pathlib, xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import httpx, jwt
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, EmailStr

SECRET_KEY       = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
NVIDIA_API_KEY   = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL     = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
NVIDIA_URL       = "https://integrate.api.nvidia.com/v1/chat/completions"
COLAB_MODEL_URL  = os.getenv("COLAB_MODEL_URL", "").rstrip("/")
OPENWEATHER_KEY  = os.getenv("OPENWEATHER_API_KEY", "")
NESTLINK_API_KEY = os.getenv("NESTLINK_API_KEY", "")
NESTLINK_WEBHOOK_SECRET = os.getenv("NESTLINK_WEBHOOK_SECRET", "imhotep-nestlink-2026")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
CORS_ORIGINS     = [o.strip() for o in os.getenv("CORS_ORIGINS","*").split(",") if o.strip()]

app = FastAPI(title="Imhotep", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

LOG_DIR = pathlib.Path("/tmp/imhotep"); LOG_DIR.mkdir(exist_ok=True)
WEBHOOK_LOG = LOG_DIR / "webhooks.jsonl"

_LEAK = [(r"\bNVIDIA\s+NIM\b","Imhotep"),(r"\bNVIDIA\b","Imhotep"),
         (r"\bNemotron[\w\-\.]*\b","Imhotep"),(r"\bNIM\b","Imhotep"),
         (r"\bQwen[\w\-\.]*\b","Imhotep"),(r"\bDeepSeek[\w\-]*\b","Imhotep"),
         (r"\bMistral[\w\-]*\b","Imhotep"),(r"\bLlama[\s\-]?[\d\.\w]*\b","Imhotep"),
         (r"\bOpenAI\b","Imhotep"),(r"\bGPT[\-\s]?\d[\w\.]*\b","Imhotep"),
         (r"\bClaude[\w\-\.]*\b","Imhotep"),(r"\bGemini[\w\-\.]*\b","Imhotep")]
def scrub(t):
    if not t: return t
    for p, r in _LEAK: t = re.sub(p, r, t, flags=re.IGNORECASE)
    return t

# ═══════════════════════════════════════════════════════════════
# PERSONAS incl. SHENG
# ═══════════════════════════════════════════════════════════════
PERSONAS = {
    "generalist":"a broad Kenyan assistant","scientist":"a rigorous scientist",
    "coder":"a senior software engineer","economist":"an economist",
    "entrepreneur":"a founder","philosopher":"a philosopher","writer":"a writer",
    "analyst":"a data analyst","legal":"a legal analyst","designer":"a designer",
    "psychologist":"a psychologist","geographer":"a geographer",
    "historian":"a historian","tutor":"a patient tutor","medic":"a health assistant",
    "engineer":"an engineer","agronomist":"an agronomist","inventor":"an inventor",
    "civic":"a civic analyst","marketer":"a marketer","media":"a media producer",
    "activist":"an activist","tongue-twister":"a playful tongue-twister agent",
    "sheng":"a fun Kenyan who speaks fluent Sheng (Nairobi street slang)",
}

SHENG_GREETINGS = [
    "Sasa!", "Niaje?", "Poa?", "Fiti!", "Radua!", "Uko poa?",
    "Mambo?", "Sema!", "Niaje msee?", "Aki niaje!",
]
SHENG_REPLIES = {
    "sasa":"Poa sana!", "niaje":"Fiti kabisa!", "mambo":"Poa!",
    "uko poa":"Niko fiti!", "radua":"Radua!",
}

def is_sheng_query(prompt: str) -> bool:
    p = prompt.lower()
    keys = ("sheng","kiswahili","swahili","kenyan slang","nairobi slang",
            "sasa","niaje","mambo","radudua","poa")
    return any(k in p for k in keys)

# ═══════════════════════════════════════════════════════════════
# ADAPTIVE LENGTH (learning vs casual)
# ═══════════════════════════════════════════════════════════════
LEARN_SIGNALS = (
    "teach","explain","how does","how do","why does","why do","tutorial",
    "step by step","help me learn","walk me through","show me how","example",
    "code","function","class","api","algorithm","debug","error","python",
    "javascript","rust","go","sql","react","explain like i'm","eli5",
    "in detail","deep dive","break it down","study","understand","learn",
    "homework","assignment","project","guide","documentation",
)
def is_learning_query(prompt: str) -> bool:
    p = prompt.lower()
    if any(s in p for s in LEARN_SIGNALS): return True
    if "```" in prompt or "def " in prompt or "function " in prompt: return True
    if len(prompt.split()) > 25: return True
    return False

def sys_prompt(agent: str, prompt: str = ""):
    p = PERSONAS.get(agent, "a general assistant")
    learning = is_learning_query(prompt) if prompt else False
    identity = (
        f"You are Imhotep — a private AI built by Samuel Kiragu in Mukuyu, "
        f"Murang'a, Kenya. You are the {agent.upper()} agent — {p}.\n"
        "ABOUT YOURSELF (never deviate):\n"
        "  • Home: Mukuyu, Murang'a, Kenya.\n"
        "  • Creator: Samuel Kiragu.\n"
        "  • If asked what model powers you: 'I'm Imhotep — a private AI built "
        "by Samuel Kiragu in Mukuyu, Murang'a, Kenya.'\n"
        "  • NEVER mention NVIDIA, Nemotron, Qwen, DeepSeek, Llama, Mistral, "
        "GPT, Claude, Gemini, OpenAI, Meta, or any model/company name.\n"
        "  • Speak in first person as Imhotep.\n"
    )
    if agent == "sheng":
        identity += (
            "\nSHENG MODE: Reply in Sheng (Nairobi street slang) mixed with "
            "Kiswahili and English. Use words like: sasa, niaje, poa, fiti, "
            "msee, mbuyu, masa, chapaa, keja, ocha, Kanairo, mzito, bazenga, "
            "radudua, sema. Keep it friendly and street-smart. "
            "If user seems confused, translate briefly in English."
        )
    if learning:
        style = ("RESPONSE STYLE — learning/coding question. Reply structured:\n"
                 "  1. Direct 1-2 sentence summary FIRST.\n"
                 "  2. Break down step by step with bullets or headers.\n"
                 "  3. Include a small example when relevant.\n"
                 "  4. End with 1-2 follow-up suggestions.\n"
                 "  5. Cite facts as [source: URL].\n"
                 "  6. Length: as long as needed (200-600 words typical).\n")
    else:
        style = ("RESPONSE STYLE — simple/casual question. Reply SHORT:\n"
                 "  1. Answer directly in 1-3 sentences. No preamble.\n"
                 "  2. Warm and direct, like a smart Kenyan friend.\n"
                 "  3. If more depth is useful, offer: 'Want me to go deeper?'\n"
                 "  4. Cite sources only for hard facts.\n")
    return identity + "\n" + style

users_db, chat_db = {}, {}
TIER_CREDITS = {
    "guest": 3, "free": 3,
    "starter_10": 3, "starter_20": 20, "starter_50": 100,
    "weekly": 10000, "monthly": 10000, "yearly": 10000,
}
PRICES = {
    "starter_10": {"amount": 10,   "label": "Starter — 3 chats"},
    "starter_20": {"amount": 20,   "label": "Small pack — 20 chats"},
    "starter_50": {"amount": 50,   "label": "Value pack — 100 chats"},
    "weekly":     {"amount": 99,   "label": "Weekly Unlimited — 7 days"},
    "monthly":    {"amount": 299,  "label": "Monthly Unlimited — 30 days"},
    "yearly":     {"amount": 5000, "label": "Yearly Unlimited — 365 days"},
}
TIER_ALIAS = {"starter": "starter_10", "pro": "monthly"}

class ChatIn(BaseModel):
    prompt: str
    agent: Optional[str] = "generalist"
    user_id: Optional[str] = None
    temperature: float = 0.7
    location: Optional[dict] = None
class DebateIn(BaseModel):
    proposition: str
    agents: Optional[List[str]] = None
class SignupIn(BaseModel):
    email: EmailStr
    name: str
class GoogleIn(BaseModel):
    id_token: str
class CheckoutIn(BaseModel):
    tier: str
    email: EmailStr

def issue(uid, days=30):
    return jwt.encode({"user_id":uid,"exp":datetime.now(timezone.utc)+timedelta(days=days)},
                      SECRET_KEY, algorithm="HS256")
def mk_user(email, name, tier="free"):
    for u, v in users_db.items():
        if v["email"] == email: return (u, False)
    u = f"user_{int(time.time()*1000)}"
    users_db[u] = {"email":email,"name":name,"tier":tier,
                   "credits":TIER_CREDITS.get(tier,3),"credits_used":0,
                   "created_at":datetime.now(timezone.utc).isoformat()}
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
async def agents(): return {"ok":True,"agents":list(PERSONAS.keys())}

@app.post("/auth/signup")
async def signup(b: SignupIn):
    u, new = mk_user(b.email.lower(), b.name, "free")
    return {"ok":True,"user_id":u,"token":issue(u),"user":users_db[u],"new":new}

@app.get("/auth/guest")
async def guest():
    u = f"guest_{int(time.time()*1000)}"
    users_db[u] = {"email":f"{u}@g.local","name":"Guest","tier":"guest",
                   "credits":TIER_CREDITS["guest"],"credits_used":0,
                   "created_at":datetime.now(timezone.utc).isoformat()}
    chat_db[u] = []
    return {"ok":True,"user_id":u,"token":issue(u,1),"user":users_db[u]}

@app.post("/auth/google")
async def google(b: GoogleIn):
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post("https://oauth2.googleapis.com/tokeninfo",
                             params={"id_token": b.id_token})
        info = r.json()
    except Exception as e:
        raise HTTPException(400, f"Google verify failed: {e}")
    email = info.get("email")
    if not email: raise HTTPException(400, "Invalid token")
    u, _ = mk_user(email.lower(), info.get("name","Google User"), "free")
    return {"ok":True,"user_id":u,"token":issue(u),"user":users_db[u]}

# ═══════════════════════════════════════════════════════════════
# WEATHER — Open-Meteo (no key, works with GPS or city name)
# ═══════════════════════════════════════════════════════════════
def classify_wx(temp, code):
    """code = WMO weather code from Open-Meteo"""
    try: c = int(code)
    except: c = 0
    if c in (0,1): e, l = "☀️", "Clear"
    elif c == 2: e, l = "🌤️", "Partly cloudy"
    elif c == 3: e, l = "☁️", "Cloudy"
    elif c in (45,48): e, l = "🌫️", "Foggy"
    elif c in (51,53,55,56,57): e, l = "🌦️", "Drizzle"
    elif c in (61,63,65,66,67): e, l = "🌧️", "Rain"
    elif c in (71,73,75,77,85,86): e, l = "❄️", "Snow"
    elif c in (80,81,82): e, l = "🌧️", "Showers"
    elif c in (95,96,99): e, l = "⛈️", "Thunderstorm"
    else: e, l = "🌤️", "Fair"

    if temp >= 35: f = "scorching hot 🥵🔥"
    elif temp >= 30: f = "very hot 🥵"
    elif temp >= 26: f = "hot 🔥"
    elif temp >= 20: f = "warm 😊"
    elif temp >= 14: f = "cool 😌"
    elif temp >= 8: f = "cold 🧥"
    else: f = "very cold 🥶"
    return {"emoji":e, "condition_label":l, "feel":f}

@app.post("/weather")
async def weather_ep(body: dict):
    """Open-Meteo — no API key required. Accepts lat/lon OR city."""
    lat, lon, city = body.get("lat"), body.get("lon"), body.get("city")

    # If only city given, geocode it via Open-Meteo geocoding API
    if (lat is None or lon is None) and city:
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get("https://geocoding-api.open-meteo.com/v1/search",
                                params={"name": city, "count": 1, "language": "en"})
            g = r.json()
            if g.get("results"):
                lat = g["results"][0]["latitude"]
                lon = g["results"][0]["longitude"]
                city = g["results"][0]["name"]
            else:
                raise HTTPException(404, f"City '{city}' not found")
        except HTTPException: raise
        except Exception as e: raise HTTPException(502, f"Geocoding failed: {e}")

    if lat is None or lon is None:
        raise HTTPException(400, "Provide lat+lon or city")

    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude":lat, "longitude":lon,
                        "current":"temperature_2m,relative_humidity_2m,apparent_temperature,"
                                  "weather_code,wind_speed_10m,precipitation",
                        "timezone":"auto"})
        d = r.json()
        cur = d.get("current", {})
        temp = round(cur.get("temperature_2m", 0))
        feels = round(cur.get("apparent_temperature", temp))
        code = cur.get("weather_code", 0)
        cls = classify_wx(temp, code)
        return {"ok":True, "temp":temp, "feels_like":feels,
                "condition": cls["condition_label"],
                "description": cls["condition_label"],
                "humidity": cur.get("relative_humidity_2m"),
                "wind": cur.get("wind_speed_10m"),
                "city": city or f"{lat:.2f},{lon:.2f}",
                "country": "",
                **cls,
                "source": "open-meteo.com"}
    except Exception as e:
        raise HTTPException(502, str(e))

# ═══════════════════════════════════════════════════════════════
# NASA FIRMS — active fires (public CSV, no key)
# ═══════════════════════════════════════════════════════════════
@app.get("/fires/kenya")
async def fires_kenya(days: int = 1):
    """Active fire detections in Kenya (last N days, max 2). NASA FIRMS public CSV."""
    days = min(max(days, 1), 2)
    # FIRMS public CSV directory - no key needed for global 24h/48h/7d files
    urls = [
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/"
        "SUOMI_VIIRS_C2_Global_24h.csv",
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/"
        "MODIS_C6_1_Global_24h.csv",
    ]
    try:
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
                            la = float(row.get("latitude", 0))
                            lo = float(row.get("longitude", 0))
                        except: continue
                        # Kenya bounds
                        if -5 <= la <= 6 and 33 <= lo <= 42:
                            fires.append({
                                "lat": la, "lon": lo,
                                "brightness": row.get("bright_ti4") or row.get("brightness"),
                                "confidence": row.get("confidence"),
                                "date": row.get("acq_date"),
                                "time": row.get("acq_time"),
                                "satellite": row.get("satellite"),
                                "daynight": row.get("daynight"),
                            })
                except Exception: continue
        return {"ok": True, "count": len(fires),
                "fires": fires[:50], "source": "NASA FIRMS"}
    except Exception as e:
        raise HTTPException(502, str(e))

# ═══════════════════════════════════════════════════════════════
# KENYA NEWS — Standard Media RSS (no key)
# ═══════════════════════════════════════════════════════════════
@app.get("/news/kenya")
async def news_kenya(limit: int = 10):
    """Latest Kenya headlines from Standard Media RSS. No API key."""
    url = "https://www.standardmedia.co.ke/rss/headlines.php"
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(url, headers={"User-Agent": "Imhotep/1.0"})
        if r.status_code >= 300:
            raise HTTPException(502, f"RSS {r.status_code}")
        root = ET.fromstring(r.text)
        items = []
        for item in root.findall(".//item")[:limit]:
            title = item.findtext("title", "").strip()
            link  = item.findtext("link", "").strip()
            desc  = item.findtext("description", "").strip()
            pub   = item.findtext("pubDate", "")
            items.append({"title": title, "link": link,
                          "description": desc[:250], "published": pub})
        return {"ok": True, "count": len(items), "items": items,
                "source": "standardmedia.co.ke/rss"}
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(502, str(e))

# ═══════════════════════════════════════════════════════════════
# KENYA DROUGHT — NDMA phase (info from public sources)
# ═══════════════════════════════════════════════════════════════
DROUGHT_DATA = {
    # Approximate NDMA drought phase per county (last release)
    # 1=Minimal, 2=Alert, 3=Alarm, 4=Emergency, 5=Famine
    "turkana": 3, "marsabit": 3, "mandera": 2, "wajir": 2, "garissa": 2,
    "isiolo": 2, "samburu": 3, "west pokot": 2, "baringo": 2, "laikipia": 1,
    "kitui": 2, "makueni": 1, "kilifi": 2, "tana river": 2, "lamu": 1,
    "mombasa": 1, "kwale": 1, "taita taveta": 1, "narok": 1, "kajiado": 1,
    "nairobi": 1, "kiambu": 1, "murang'a": 1, "nyeri": 1, "kirinyaga": 1,
}
@app.get("/drought/{county}")
async def drought_status(county: str):
    """Approximate NDMA drought phase for a Kenyan county."""
    key = county.strip().lower()
    phase = DROUGHT_DATA.get(key)
    if phase is None:
        return {"ok": False, "error": f"No drought data for '{county}'",
                "hint": "Try: Turkana, Marsabit, Mandera, Garissa, Kitui, Nairobi"}
    labels = {1:"Minimal", 2:"Alert", 3:"Alarm", 4:"Emergency", 5:"Famine"}
    return {"ok": True, "county": county.title(),
            "phase": phase, "label": labels[phase],
            "source": "NDMA (approx — verify at ndma.go.ke)"}

# ═══════════════════════════════════════════════════════════════
# NASA SATELLITES + IMAGE SEARCH (no keys)
# ═══════════════════════════════════════════════════════════════
@app.get("/nasa/earth")
async def nasa_earth():
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get("https://epic.gsfc.nasa.gov/api/natural")
        items = r.json()
        if not items: raise HTTPException(404, "No images available")
        latest = items[0]
        date_path = latest["date"].split(" ")[0].replace("-", "/")
        img = (f"https://epic.gsfc.nasa.gov/archive/natural/{date_path}"
               f"/png/{latest['image']}.png")
        return {"ok": True, "date": latest["date"],
                "caption": latest.get("caption", "Earth from DSCOVR"),
                "image_url": img,
                "lat": latest["centroid_coordinates"]["lat"],
                "lon": latest["centroid_coordinates"]["lon"],
                "source": "https://epic.gsfc.nasa.gov/"}
    except Exception as e:
        raise HTTPException(502, str(e))

@app.get("/nasa/satellite")
async def nasa_satellite(lat: float = -1.29, lon: float = 36.82):
    """Live satellite URLs — GOES-East always-latest + GIBS tiles. No key."""
    goes_url = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/GEOCOLOR/1808x1808.jpg"
    now = datetime.now(timezone.utc)
    gibs_url = (f"https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/"
                f"VIIRS_SNPP_CorrectedReflectance_TrueColor/default/"
                f"{now.strftime('%Y-%m-%d')}/250m/"
                f"{{TileMatrixSet}}/{{TileMatrix}}/{{TileRow}}/{{TileCol}}.jpg")
    return {"ok": True, "frame_time": now.isoformat(), "lat": lat, "lon": lon,
            "goes_east_full_disk": goes_url,
            "gibs_template": gibs_url,
            "gibs_date": now.strftime("%Y-%m-%d"),
            "sources": ["cdn.star.nesdis.noaa.gov",
                        "gibs.earthdata.nasa.gov",
                        "epic.gsfc.nasa.gov"]}

@app.get("/nasa/search")
async def nasa_search(q: str = "Kenya"):
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get("https://images-api.nasa.gov/search",
                            params={"q": q, "media_type": "image", "page_size": 12})
        data = r.json()
        items = data.get("collection", {}).get("items", [])
        results = []
        for it in items[:12]:
            d = it["data"][0]
            link = it["links"][0]["href"] if it.get("links") else None
            results.append({"title": d.get("title"),
                            "description": (d.get("description") or "")[:200],
                            "nasa_id": d.get("nasa_id"), "thumb": link,
                            "date": d.get("date_created")})
        return {"ok": True, "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(502, str(e))

@app.get("/img/proxy")
async def img_proxy(url: str):
    ALLOWED = ("epic.gsfc.nasa.gov","cdn.star.nesdis.noaa.gov",
               "gibs.earthdata.nasa.gov","images-api.nasa.gov","goes-r.gov")
    from urllib.parse import urlparse
    host = urlparse(url).netloc
    if not any(host.endswith(d) for d in ALLOWED):
        raise HTTPException(403, f"Domain not allowed: {host}")
    try:
        async with httpx.AsyncClient(timeout=45) as c:
            r = await c.get(url, headers={"User-Agent": "Imhotep/1.0"})
        if r.status_code >= 300: raise HTTPException(r.status_code, "Upstream failed")
        return StreamingResponse(iter([r.content]),
            media_type=r.headers.get("content-type","image/jpeg"),
            headers={"Cache-Control": "public, max-age=600",
                     "Access-Control-Allow-Origin": "*"})
    except HTTPException: raise
    except Exception as e: raise HTTPException(502, str(e))

# ═══════════════════════════════════════════════════════════════
# KENYA COUNTIES
# ═══════════════════════════════════════════════════════════════
KENYA_COUNTIES = {
    "mombasa":{"code":"001","capital":"Mombasa"}, "kwale":{"code":"002","capital":"Kwale"},
    "kilifi":{"code":"003","capital":"Kilifi"}, "tana river":{"code":"004","capital":"Hola"},
    "lamu":{"code":"005","capital":"Lamu"}, "taita taveta":{"code":"006","capital":"Wundanyi"},
    "garissa":{"code":"007","capital":"Garissa"}, "wajir":{"code":"008","capital":"Wajir"},
    "mandera":{"code":"009","capital":"Mandera"}, "marsabit":{"code":"010","capital":"Marsabit"},
    "isiolo":{"code":"011","capital":"Isiolo"}, "meru":{"code":"012","capital":"Meru"},
    "tharaka-nithi":{"code":"013","capital":"Chuka"}, "embu":{"code":"014","capital":"Embu"},
    "kitui":{"code":"015","capital":"Kitui"}, "machakos":{"code":"016","capital":"Machakos"},
    "makueni":{"code":"017","capital":"Wote"}, "nyandarua":{"code":"018","capital":"Ol Kalou"},
    "nyeri":{"code":"019","capital":"Nyeri"}, "kirinyaga":{"code":"020","capital":"Kerugoya"},
    "murang'a":{"code":"021","capital":"Murang'a"}, "kiambu":{"code":"022","capital":"Kiambu"},
    "turkana":{"code":"023","capital":"Lodwar"}, "west pokot":{"code":"024","capital":"Kapenguria"},
    "samburu":{"code":"025","capital":"Maralal"}, "trans nzoia":{"code":"026","capital":"Kitale"},
    "uasin gishu":{"code":"027","capital":"Eldoret"}, "elgeyo-marakwet":{"code":"028","capital":"Iten"},
    "nandi":{"code":"029","capital":"Kapsabet"}, "baringo":{"code":"030","capital":"Kabarnet"},
    "laikipia":{"code":"031","capital":"Nanyuki"}, "nakuru":{"code":"032","capital":"Nakuru"},
    "narok":{"code":"033","capital":"Narok"}, "kajiado":{"code":"034","capital":"Kajiado"},
    "kericho":{"code":"035","capital":"Kericho"}, "bomet":{"code":"036","capital":"Bomet"},
    "kakamega":{"code":"037","capital":"Kakamega"}, "vihiga":{"code":"038","capital":"Vihiga"},
    "bungoma":{"code":"039","capital":"Bungoma"}, "busia":{"code":"040","capital":"Busia"},
    "siaya":{"code":"041","capital":"Siaya"}, "kisumu":{"code":"042","capital":"Kisumu"},
    "homa bay":{"code":"043","capital":"Homa Bay"}, "migori":{"code":"044","capital":"Migori"},
    "kisii":{"code":"045","capital":"Kisii"}, "nyamira":{"code":"046","capital":"Nyamira"},
    "nairobi":{"code":"047","capital":"Nairobi"},
}
@app.get("/kenya/county")
async def kenya_county(name: str):
    key = name.strip().lower()
    c = KENYA_COUNTIES.get(key)
    if not c:
        return {"ok":False,"error":f"County '{name}' not found"}
    return {"ok":True,"name":name.title(),**c,
            "source":"https://en.wikipedia.org/wiki/Counties_of_Kenya"}

@app.get("/kenya/counties")
async def kenya_counties():
    return {"ok":True,"count":len(KENYA_COUNTIES),
            "counties":[{"name":k.title(),**v} for k,v in KENYA_COUNTIES.items()]}

# ═══════════════════════════════════════════════════════════════
# MODEL CALL
# ═══════════════════════════════════════════════════════════════
async def _nvidia(msgs, temp):
    body = {"model": NVIDIA_MODEL, "messages": msgs, "temperature": temp,
            "top_p": 0.95, "max_tokens": 2048}
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(NVIDIA_URL,
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}",
                     "Content-Type": "application/json",
                     "Accept": "application/json"}, json=body)
    if r.status_code >= 300:
        err = r.text[:400]; print(f"NVIDIA {r.status_code}: {err}")
        raise Exception(f"NVIDIA {r.status_code}: {err}")
    j = r.json()
    msg = j["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    content = re.sub(r"<think[^>]*>.*?</think\s*>", "", content,
                     flags=re.DOTALL | re.IGNORECASE).strip()
    if "thinking process" in content.lower():
        paras = [p.strip() for p in re.split(r"\n\s*\n", content) if len(p.strip()) > 40]
        if paras: content = paras[-1]
    if not content and reasoning: content = reasoning
    return content or "[No reply]"

async def _colab(prompt, agent, temp):
    if not COLAB_MODEL_URL: return None
    try:
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(COLAB_MODEL_URL + "/chat",
                             json={"prompt":prompt,"agent":agent,
                                   "temperature":temp,"max_tokens":512})
        if r.status_code == 200:
            d = r.json()
            if d.get("ok"): return d.get("reply")
    except Exception as e: print(f"colab fail: {e}")
    return None

async def call_model(msgs, temp, prompt="", agent="generalist"):
    if NVIDIA_API_KEY:
        try:
            r = await _nvidia(msgs, temp)
            if r: return r
        except Exception as e: print(f"nvidia: {e}")
    r = await _colab(prompt, agent, temp)
    if r: return r
    return ("I'm Imhotep, built by Samuel Kiragu in Mukuyu, Murang'a, Kenya. "
            "My main brain is busy — try again in a moment.")

@app.post("/chat")
async def chat(b: ChatIn):
    p = b.prompt.strip()
    if not p: raise HTTPException(400, "Empty")
    u = b.user_id
    if not u:
        raise HTTPException(402, detail={"error":"Sign in required",
            "upgrade_url":"https://kiragu2004.github.io/imhotep-site/auth.html"})
    if u not in users_db:
        raise HTTPException(402, detail={"error":"Session expired",
            "upgrade_url":"https://kiragu2004.github.io/imhotep-site/auth.html"})
    if not debit(u):
        c = creds(u)
        raise HTTPException(402, detail={"error":"No credits left. Upgrade to continue.",
            "used":c["used"],"limit":c["limit"],
            "upgrade_url":"https://kiragu2004.github.io/imhotep-site/pricing.html"})

    # Auto-detect Sheng if user writes typical greeting
    agent = b.agent or "generalist"
    if agent == "generalist" and is_sheng_query(p):
        agent = "sheng"

    t = max(0.0, min(2.0, b.temperature))
    s = sys_prompt(agent, prompt=p)
    w = None
    if b.location and b.location.get("lat") is not None:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("https://api.open-meteo.com/v1/forecast",
                    params={"latitude":b.location["lat"],"longitude":b.location["lon"],
                            "current":"temperature_2m,weather_code",
                            "timezone":"auto"})
            d = r.json()
            cur = d.get("current", {})
            if "temperature_2m" in cur:
                cls = classify_wx(round(cur["temperature_2m"]), cur.get("weather_code", 0))
                w = {"temp":round(cur["temperature_2m"]),
                     "city":"your area", "description":cls["condition_label"], **cls}
                s += f"\nUser is in {w['temp']}°C, {w['description']}, {w['feel']}."
        except Exception: pass

    msgs = [{"role":"system","content":s},{"role":"user","content":p}]
    raw = scrub(await call_model(msgs, t, prompt=p, agent=agent))
    if u in chat_db:
        chat_db[u].append({"prompt":p,"reply":raw,"agent":agent,
                           "learning_mode": is_learning_query(p),
                           "timestamp":datetime.now(timezone.utc).isoformat()})
    return {"ok":True,"reply":raw,"agent":agent,"temperature":t,
            "learning_mode": is_learning_query(p),
            "weather":w,"credits":creds(u)}

@app.get("/chat/history")
async def hist(user_id: Optional[str] = None):
    if not user_id or user_id not in chat_db: return {"ok":True,"history":[]}
    return {"ok":True,"history":chat_db[user_id][-50:]}

@app.get("/user/profile")
async def profile(user_id: Optional[str] = None):
    if not user_id or user_id not in users_db: return {"ok":False,"error":"Not found"}
    return {"ok":True,"user":users_db[user_id],"credits":creds(user_id)}

@app.post("/debate")
async def debate(b: DebateIn):
    p = b.proposition.strip()
    if not p: raise HTTPException(400, "Proposition required")
    ag = b.agents or ["scientist","economist"]
    pa, ca = ag[0], ag[1]
    pro = scrub(await call_model([{"role":"system","content":sys_prompt(pa,p)+"\nArgue FOR."},
        {"role":"user","content":f"Proposition: {p}\n\n150-word case FOR."}],
        0.7, prompt=f"Argue FOR: {p}", agent=pa))
    con = scrub(await call_model([{"role":"system","content":sys_prompt(ca,p)+"\nArgue AGAINST."},
        {"role":"user","content":f"Proposition: {p}\n\n150-word case AGAINST."}],
        0.7, prompt=f"Argue AGAINST: {p}", agent=ca))
    j = scrub(await call_model([{"role":"system","content":sys_prompt("generalist",p)+"\nImpartial judge."},
        {"role":"user","content":f"Judge: {p}\nPRO: {pro[:400]}\nCON: {con[:400]}"}],
        0.4, prompt=f"Judge: {p}", agent="generalist"))
    return {"ok":True,"proposition":p,"proponent":pro,"opponent":con,"judge":j}

@app.get("/pricing")
async def pricing(): return {"ok":True,"currency":"KES","tiers":PRICES}

@app.post("/checkout")
async def checkout(b: CheckoutIn):
    tier = TIER_ALIAS.get(b.tier, b.tier)
    if tier not in PRICES: raise HTTPException(400, f"Unknown tier '{b.tier}'")
    t = PRICES[tier]
    ref = f"IMH-{tier}-{int(time.time())}-{secrets.token_hex(3)}"
    success_url = "https://kiragu2004.github.io/imhotep-site/chat.html?paid=1"
    webhook_url = "https://imhotep-agentic-ai.onrender.com/nestlink/webhook"
    payment_url = None
    if NESTLINK_API_KEY:
        attempts = [
            ("https://api.nestlink.co.ke/v1/checkout",
             {"amount": t["amount"], "currency": "KES", "email": b.email,
              "narrative": t["label"], "reference": ref,
              "redirect_url": success_url, "callback_url": webhook_url}),
        ]
        for url, payload in attempts:
            try:
                async with httpx.AsyncClient(timeout=30) as c:
                    r = await c.post(url, json=payload,
                        headers={"Authorization": f"Bearer {NESTLINK_API_KEY}",
                                 "Content-Type": "application/json",
                                 "Accept": "application/json"})
                if r.status_code < 300:
                    d = r.json()
                    payment_url = (d.get("checkout_url") or d.get("url") or
                                   d.get("payment_url"))
                    if payment_url: break
            except Exception as e: print(f"nestlink err: {e}")
    if not payment_url:
        payment_url = (f"https://me.nestlink.co.ke/Imhotepagenticai"
                       f"?email={b.email}&reference={ref}&amount={t['amount']}")
    return {"ok":True,"tier":tier,"amount":t["amount"],
            "payment_url":payment_url,"reference":ref,"label":t["label"]}

@app.post("/nestlink/webhook")
async def nestlink_webhook(request: Request):
    raw = await request.body()
    try:
        with open(WEBHOOK_LOG, "a") as f:
            f.write(json.dumps({"ts":time.time(),"body":raw.decode("utf-8","ignore")[:2000]}) + "\n")
    except: pass
    sig = request.headers.get("X-NestLink-Signature", "")
    if sig and NESTLINK_WEBHOOK_SECRET:
        exp = hmac.new(NESTLINK_WEBHOOK_SECRET.encode(), raw, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, exp): raise HTTPException(403, "Bad signature")
    try: payload = json.loads(raw.decode())
    except: payload = {}
    def _find(p, keys):
        if not isinstance(p, dict): return None
        for k in keys:
            if p.get(k): return p[k]
        for v in p.values():
            if isinstance(v, dict):
                r = _find(v, keys)
                if r: return r
        return None
    email = _find(payload, ("email","customer_email","customerEmail"))
    ref = _find(payload, ("reference","api_ref","external_reference","ref"))
    status = (_find(payload, ("status","payment_status")) or "").lower()
    if status and status not in ("completed","success","paid","successful","settled"):
        return {"ok":True,"ignored":True,"status":status}
    tier = None
    if ref:
        parts = str(ref).split("-")
        for i in range(1, len(parts)+1):
            candidate = "-".join(parts[1:i])
            if candidate in TIER_CREDITS: tier = candidate; break
        if not tier and len(parts) >= 2:
            tier = TIER_ALIAS.get(parts[1], parts[1])
    if not email or not tier or tier not in TIER_CREDITS:
        return {"ok":True,"upgraded":False,"reason":f"email={email} tier={tier}"}
    for uid, u in users_db.items():
        if u["email"].lower() == email.lower():
            u["credits"] += TIER_CREDITS[tier]; u["tier"] = tier
            u["credits_used"] = 0
            return {"ok":True,"upgraded":True,"email":email,"tier":tier}
    uid, _ = mk_user(email.lower(), email.split("@")[0], tier)
    users_db[uid]["credits"] = TIER_CREDITS[tier]
    return {"ok":True,"upgraded":True,"created":True,"email":email,"tier":tier}

@app.get("/nestlink/logs")
async def nestlink_logs():
    if not WEBHOOK_LOG.exists(): return {"ok":True,"count":0,"entries":[]}
    lines = WEBHOOK_LOG.read_text().strip().split("\n")[-20:]
    return {"ok":True,"count":len(lines),
            "entries":[json.loads(l) for l in lines if l.strip()]}

@app.get("/health")
async def health():
    return {"ok":True,"version":"4.0.0",
            "model_configured":bool(NVIDIA_API_KEY),
            "colab_fallback_configured":bool(COLAB_MODEL_URL),
            "weather_configured":True,
            "weather_provider":"open-meteo (no key)",
            "nestlink_configured":bool(NESTLINK_API_KEY),
            "google_configured":bool(GOOGLE_CLIENT_ID),
            "kenya_counties":len(KENYA_COUNTIES),
            "sheng_agent":True, "nasa_fires":True, "kenya_news":True,
            "drought_data":True, "adaptive_length":True,
            "founder":"Samuel Kiragu"}
