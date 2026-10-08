
import os, re, time, secrets, hmac, hashlib, json, pathlib, math
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import httpx, jwt
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════
# APP  ← MUST COME BEFORE ANY @app.get / @app.post DECORATOR
# ═══════════════════════════════════════════════════════════════
app = FastAPI(title="Imhotep", version="6.1.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

LOG_DIR = pathlib.Path("/tmp/imhotep"); LOG_DIR.mkdir(exist_ok=True)
WEBHOOK_LOG = LOG_DIR / "webhooks.jsonl"

# ═══════════════════════════════════════════════════════════════
# PERSONA SCRUBBER
# ═══════════════════════════════════════════════════════════════
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
# SHENG
# ═══════════════════════════════════════════════════════════════
SHENG_DICT = """
SHENG VOCAB: chali/msee=guy, demu/msupa=girl, beshte/arif=friend, mbogi=group,
doo/chapaa=money, ashu=10/-, blue=20/-, soo=100/-, thao=1000/-,
bonga=talk, dishi=eat, doro=sleep, sota=broke, sasa=hey, niaje=what's up,
mambo=how are things, poa/fiti/safi=cool, mtaa=neighborhood, ocha=upcountry,
Kanairo=Nairobi, radudua=what's up (slang)
"""
SHENG_GREETINGS = ("sasa","niaje","mambo","marada","radudua","sema","bonga","uko poa")
def is_sheng_query(prompt: str) -> bool:
    p = prompt.lower()
    return any(g in p for g in SHENG_GREETINGS) or "sheng" in p

# ═══════════════════════════════════════════════════════════════
# PERSONAS
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
    "sheng":"a fluent Sheng speaker from Nairobi streets",
    "weather":"a weather analyst",
    "emergency":"a disaster-response advisor",
}
LEARN_SIGNALS = ("teach","explain","how does","why does","tutorial","step by step",
                "help me learn","code","function","algorithm","debug","python",
                "javascript","in detail","study","understand","learn","homework")
def is_learning_query(prompt: str) -> bool:
    p = prompt.lower()
    return any(s in p for s in LEARN_SIGNALS) or len(prompt.split()) > 25

def sys_prompt(agent: str, prompt: str = ""):
    p = PERSONAS.get(agent, "a general assistant")
    learning = is_learning_query(prompt) if prompt else False
    identity = (
        f"You are Imhotep — a private AI built by Samuel Kiragu in Mukuyu, "
        f"Murang'a, Kenya. You are the {agent.upper()} agent — {p}.\n"
        "RULES:\n"
        "  • Home: Mukuyu, Murang'a, Kenya. Creator: Samuel Kiragu.\n"
        "  • If asked what model powers you: 'I'm Imhotep, built by Samuel Kiragu.'\n"
        "  • NEVER mention NVIDIA, Nemotron, Qwen, DeepSeek, Llama, Mistral, "
        "GPT, Claude, Gemini, OpenAI, Meta, or any model/company name.\n"
        "  • Speak in first person as Imhotep.\n"
    )
    if agent == "sheng":
        return (identity + "\n" + SHENG_DICT +
                "\nSHENG MODE: Reply in Sheng mixed with Kiswahili and English.")
    if learning:
        style = ("RESPONSE STYLE — structured: 1. Summary. 2. Step-by-step bullets. "
                 "3. Example. 4. Follow-up. 5. Length: 200-600 words.")
    else:
        style = ("RESPONSE STYLE — SHORT: 1. Answer in 1-3 sentences. "
                 "2. Warm, direct. 3. Offer: 'Want me to go deeper?'")
    return identity + "\n" + style

# ═══════════════════════════════════════════════════════════════
# USER STATE
# ═══════════════════════════════════════════════════════════════
users_db, chat_db = {}, {}
TIER_CREDITS = {"guest":3,"free":3,"starter_10":3,"starter_20":20,"starter_50":100,
                "weekly":10000,"monthly":10000,"yearly":10000}
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
class SignupIn(BaseModel):
    email: EmailStr
    name: str
class CheckoutIn(BaseModel):
    tier: str
    email: EmailStr
class GoogleIn(BaseModel):
    id_token: str
class DebateIn(BaseModel):
    proposition: str
    agents: Optional[List[str]] = None

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

# ═══════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════
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
                   "credits":3,"credits_used":0,
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
# WEATHER — OpenWeather primary, Open-Meteo fallback
# ═══════════════════════════════════════════════════════════════
def classify_wx(temp, code):
    try: c = int(code)
    except: c = 800
    if c in (800,): e,l="☀️","Clear"
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
    elif c in (51,53,55,56,57): e,l="🌦️","Drizzle"
    elif c in (61,63,65,66,67): e,l="🌧️","Rain"
    elif c in (71,73,75,77,85,86): e,l="❄️","Snow"
    elif c in (80,81,82): e,l="🌧️","Showers"
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

async def _geocode_city(city: str):
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get("https://geocoding-api.open-meteo.com/v1/search",
                        params={"name":city,"count":1,"language":"en"})
    g = r.json()
    if not g.get("results"): return None,None,None
    hit = g["results"][0]
    return hit.get("latitude"), hit.get("longitude"), hit.get("name") or city

@app.post("/weather")
async def weather_ep(body: dict):
    """OpenWeather primary (if key set), Open-Meteo fallback."""
    lat, lon, city = body.get("lat"), body.get("lon"), body.get("city")

    # Try OpenWeather first if key is configured
    if OPENWEATHER_KEY:
        try:
            params = ({"lat": lat, "lon": lon} if lat is not None and lon is not None
                      else {"q": f"{city},KE" if city else "Nairobi,KE"})
            params.update({"appid": OPENWEATHER_KEY, "units": "metric"})
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get("https://api.openweathermap.org/data/2.5/weather",
                                params=params)
            d = r.json()
            if "main" in d:
                temp = round(d["main"]["temp"])
                cls = classify_wx(temp, d["weather"][0]["id"])
                return {"ok":True, "temp":temp,
                        "feels_like":round(d["main"].get("feels_like", temp)),
                        "condition":d["weather"][0]["main"],
                        "description":d["weather"][0]["description"],
                        "humidity":d["main"].get("humidity"),
                        "wind":d.get("wind",{}).get("speed"),
                        "city":d.get("name") or city or "",
                        "country":d.get("sys",{}).get("country","KE"),
                        **cls, "source":"openweathermap.org"}
        except Exception as e:
            print(f"OpenWeather failed, falling back: {e}")

    # Fallback to Open-Meteo
    if (lat is None or lon is None) and city:
        la, lo, resolved = await _geocode_city(city)
        if la is None: raise HTTPException(404, f"City '{city}' not found")
        lat, lon, city = la, lo, resolved
    if lat is None or lon is None: raise HTTPException(400, "Provide lat+lon or city")

    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude":lat,"longitude":lon,
                    "current":"temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "timezone":"auto"})
    d = r.json(); cur = d.get("current") or {}
    temp = round(cur.get("temperature_2m", 0)); code = cur.get("weather_code", 0)
    cls = classify_wx(temp, code)
    return {"ok":True,"temp":temp,
            "feels_like":round(cur.get("apparent_temperature", temp)),
            "condition":cls["condition_label"],
            "description":cls["condition_label"],
            "humidity":cur.get("relative_humidity_2m"),
            "wind":cur.get("wind_speed_10m"),
            "city":city or f"{lat:.2f},{lon:.2f}", "country":"",
            **cls, "source":"open-meteo.com"}

@app.get("/weather/counties")
async def weather_counties():
    """Get current weather for major Kenyan counties."""
    counties = ["Nairobi","Mombasa","Kisumu","Nakuru","Eldoret","Nyeri",
                "Machakos","Garissa","Turkana","Kakamega","Meru","Kisii"]
    results = []
    if OPENWEATHER_KEY:
        async with httpx.AsyncClient(timeout=30) as c:
            for city in counties:
                try:
                    r = await c.get("https://api.openweathermap.org/data/2.5/weather",
                                    params={"q": f"{city},KE", "appid": OPENWEATHER_KEY,
                                            "units": "metric"})
                    d = r.json()
                    if "main" in d:
                        temp = round(d["main"]["temp"])
                        cls = classify_wx(temp, d["weather"][0]["id"])
                        results.append({"city": city, "temp": temp,
                                        "feels_like": round(d["main"].get("feels_like", temp)),
                                        "condition": d["weather"][0]["main"],
                                        "description": d["weather"][0]["description"],
                                        "humidity": d["main"].get("humidity"),
                                        "icon": d["weather"][0]["icon"],
                                        **cls})
                except Exception: continue
        return {"ok": True, "count": len(results), "counties": results,
                "source": "openweathermap.org"}

    # Fallback to Open-Meteo for each
    async with httpx.AsyncClient(timeout=30) as c:
        for city in counties:
            la, lo, _ = await _geocode_city(city)
            if la is None: continue
            try:
                r = await c.get("https://api.open-meteo.com/v1/forecast",
                    params={"latitude":la,"longitude":lo,
                            "current":"temperature_2m,weather_code","timezone":"auto"})
                d = r.json(); cur = d.get("current") or {}
                temp = round(cur.get("temperature_2m", 0))
                cls = classify_wx(temp, cur.get("weather_code", 0))
                results.append({"city": city, "temp": temp, **cls})
            except Exception: continue
    return {"ok": True, "count": len(results), "counties": results,
            "source": "open-meteo.com (fallback)"}

# ═══════════════════════════════════════════════════════════════
# NASA FIRES — with county labels
# ═══════════════════════════════════════════════════════════════
KENYA_COUNTY_CENTERS = {
    "Nairobi":(-1.2921,36.8219),"Mombasa":(-4.0435,39.6682),"Kisumu":(-0.0917,34.7680),
    "Nakuru":(-0.3031,36.0800),"Eldoret":(0.5143,35.2698),"Nyeri":(-0.4167,36.9500),
    "Machakos":(-1.5177,37.2634),"Garissa":(-0.4536,39.6461),"Turkana":(3.1191,35.5973),
    "Marsabit":(2.3344,37.9899),"Wajir":(1.7471,40.0573),"Mandera":(3.9366,41.8670),
    "Isiolo":(0.3546,37.5822),"Meru":(0.0500,37.6500),"Kitui":(-1.3667,38.0167),
    "Kilifi":(-3.6305,39.8499),"Kwale":(-4.1742,39.4524),"Lamu":(-2.2717,40.9020),
    "Taita Taveta":(-3.4000,38.5500),"Kajiado":(-1.8500,36.7833),"Narok":(-1.0833,35.8667),
    "Bomet":(-0.7833,35.3333),"Kericho":(-0.3667,35.2833),"Nandi":(0.1833,35.1000),
    "Kakamega":(0.2827,34.7519),"Bungoma":(0.5667,34.5667),"Busia":(0.4500,34.1167),
    "Siaya":(0.0667,34.2833),"Homa Bay":(-0.5167,34.4500),"Migori":(-0.9833,34.4667),
    "Kisii":(-0.6817,34.7680),"Nyamira":(-0.5667,34.9333),"Nyandarua":(-0.5000,36.3667),
    "Kirinyaga":(-0.5000,37.2833),"Murang'a":(-0.7167,37.1500),"Kiambu":(-1.1667,36.8333),
    "Trans Nzoia":(1.0167,35.0000),"West Pokot":(1.2333,35.1167),"Samburu":(1.1000,36.7000),
    "Laikipia":(0.0500,36.7000),"Baringo":(0.4667,35.9667),"Elgeyo-Marakwet":(0.5167,35.5000),
    "Uasin Gishu":(0.5167,35.2833),"Tharaka-Nithi":(-0.1667,37.9000),"Embu":(-0.5333,37.4500),
    "Makueni":(-2.0000,37.6167),"Tana River":(-1.5000,39.9833),"Kwale":(-4.1742,39.4524),
    "Kilifi":(-3.6305,39.8499),
}
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
                        la = float(row.get("latitude",0)); lo = float(row.get("longitude",0))
                    except: continue
                    if -5 <= la <= 6 and 33 <= lo <= 42:
                        county = _nearest_county(la, lo)
                        fires.append({
                            "lat": la, "lon": lo,
                            "county": county or "Unknown",
                            "brightness": row.get("bright_ti4") or row.get("brightness"),
                            "confidence": row.get("confidence"),
                            "date": row.get("acq_date"), "time": row.get("acq_time"),
                            "satellite": row.get("satellite"),
                        })
            except Exception: continue
    wms_url = ("https://firms.modaps.eosdis.nasa.gov/wms/active_fire/?"
               "SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=fires24&"
               "CRS=EPSG:4326&BBOX=33,-5,42,6&WIDTH=800&HEIGHT=600&"
               "FORMAT=image/png&TRANSPARENT=TRUE")
    return {"ok":True,"count":len(fires),"fires":fires[:50],
            "wms_url":wms_url,"source":"NASA FIRMS"}

# ═══════════════════════════════════════════════════════════════
# DROUGHT — approximate NDMA phases
# ═══════════════════════════════════════════════════════════════
DROUGHT_DATA = {
    "turkana":3,"marsabit":3,"mandera":2,"wajir":2,"garissa":2,
    "isiolo":2,"samburu":3,"west pokot":2,"baringo":2,"laikipia":1,
    "kitui":2,"makueni":1,"kilifi":2,"tana river":2,"lamu":1,
    "mombasa":1,"kwale":1,"taita taveta":1,"narok":1,"kajiado":1,
    "nairobi":1,"kiambu":1,"murang'a":1,"nyeri":1,"kirinyaga":1,
}
@app.get("/drought/{county}")
async def drought_status(county: str):
    key = county.strip().lower()
    phase = DROUGHT_DATA.get(key)
    if phase is None:
        return {"ok": False, "error": f"No drought data for '{county}'",
                "hint": "Try: Turkana, Marsabit, Mandera, Garissa, Kitui, Nairobi"}
    labels = {1:"Minimal", 2:"Alert", 3:"Alarm", 4:"Emergency", 5:"Famine"}
    return {"ok": True, "county": county.title(),
            "phase": phase, "label": labels[phase],
            "source": "NDMA (approximate)"}

# ═══════════════════════════════════════════════════════════════
# NEWS
# ═══════════════════════════════════════════════════════════════
@app.get("/news/kenya")
async def news_kenya(limit: int = 10):
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get("https://www.standardmedia.co.ke/rss/headlines.php",
                            headers={"User-Agent": "Imhotep/1.0"})
        root = ET.fromstring(r.text)
        items = []
        for item in root.findall(".//item")[:limit]:
            items.append({
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "description": (item.findtext("description") or "").strip()[:250],
                "published": item.findtext("pubDate") or "",
            })
        return {"ok": True, "count": len(items), "items": items,
                "source": "standardmedia.co.ke/rss"}
    except Exception as e:
        raise HTTPException(502, str(e))

# ═══════════════════════════════════════════════════════════════
# COUNTIES
# ═══════════════════════════════════════════════════════════════
KENYA_COUNTIES = {
    "nairobi":{"code":"047","capital":"Nairobi"},
    "mombasa":{"code":"001","capital":"Mombasa"},
    "kisumu":{"code":"042","capital":"Kisumu"},
    "nakuru":{"code":"032","capital":"Nakuru"},
    "uasin gishu":{"code":"027","capital":"Eldoret"},
    "nyeri":{"code":"019","capital":"Nyeri"},
    "machakos":{"code":"016","capital":"Machakos"},
    "garissa":{"code":"007","capital":"Garissa"},
    "turkana":{"code":"023","capital":"Lodwar"},
    "kakamega":{"code":"037","capital":"Kakamega"},
    "meru":{"code":"012","capital":"Meru"},
    "kisii":{"code":"045","capital":"Kisii"},
    "murang'a":{"code":"021","capital":"Murang'a"},
    "kiambu":{"code":"022","capital":"Kiambu"},
    "kilifi":{"code":"003","capital":"Kilifi"},
    "marsabit":{"code":"010","capital":"Marsabit"},
    "wajir":{"code":"008","capital":"Wajir"},
    "mandera":{"code":"009","capital":"Mandera"},
}
@app.get("/kenya/counties")
async def kenya_counties():
    return {"ok":True,"count":len(KENYA_COUNTIES),
            "counties":[{"name":k.title(),**v} for k,v in KENYA_COUNTIES.items()]}

# ═══════════════════════════════════════════════════════════════
# NASA SATELLITES
# ═══════════════════════════════════════════════════════════════
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
            "lon":latest["centroid_coordinates"]["lon"],
            "source":"https://epic.gsfc.nasa.gov/"}

@app.get("/nasa/satellite")
async def nasa_satellite(lat: float = -1.29, lon: float = 36.82):
    goes = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/GEOCOLOR/1808x1808.jpg"
    return {"ok":True,"frame_time":datetime.now(timezone.utc).isoformat(),
            "goes_east_full_disk":goes,"lat":lat,"lon":lon,
            "sources":["cdn.star.nesdis.noaa.gov","gibs.earthdata.nasa.gov"]}

@app.get("/nasa/search")
async def nasa_search(q: str = "Kenya"):
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get("https://images-api.nasa.gov/search",
                        params={"q":q,"media_type":"image","page_size":12})
    data = r.json()
    items = data.get("collection",{}).get("items",[])
    results = []
    for it in items[:12]:
        d = it["data"][0]
        link = it["links"][0]["href"] if it.get("links") else None
        results.append({"title":d.get("title"),
                        "description":(d.get("description") or "")[:200],
                        "nasa_id":d.get("nasa_id"),"thumb":link,
                        "date":d.get("date_created")})
    return {"ok":True,"count":len(results),"results":results}

@app.get("/img/proxy")
async def img_proxy(url: str):
    ALLOWED = ("epic.gsfc.nasa.gov","cdn.star.nesdis.noaa.gov",
               "gibs.earthdata.nasa.gov","images-api.nasa.gov")
    from urllib.parse import urlparse
    host = urlparse(url).netloc
    if not any(host.endswith(d) for d in ALLOWED):
        raise HTTPException(403, f"Domain not allowed: {host}")
    async with httpx.AsyncClient(timeout=45) as c:
        r = await c.get(url, headers={"User-Agent":"Imhotep/1.0"})
    if r.status_code >= 300: raise HTTPException(r.status_code, "Upstream failed")
    return StreamingResponse(iter([r.content]),
        media_type=r.headers.get("content-type","image/jpeg"),
        headers={"Cache-Control":"public, max-age=600",
                 "Access-Control-Allow-Origin":"*"})

# ═══════════════════════════════════════════════════════════════
# MODEL CALL
# ═══════════════════════════════════════════════════════════════
async def _nvidia(msgs, temp):
    body = {"model":NVIDIA_MODEL,"messages":msgs,"temperature":temp,
            "top_p":0.95,"max_tokens":2048}
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(NVIDIA_URL,
            headers={"Authorization":f"Bearer {NVIDIA_API_KEY}",
                     "Content-Type":"application/json","Accept":"application/json"},
            json=body)
    if r.status_code >= 300:
        print(f"NVIDIA {r.status_code}: {r.text[:300]}")
        raise Exception(f"NVIDIA {r.status_code}")
    msg = r.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    content = re.sub(r"<think[^>]*>.*?</think\s*>","",content,
                     flags=re.DOTALL|re.IGNORECASE).strip()
    if "thinking process" in content.lower():
        paras = [p.strip() for p in re.split(r"\n\s*\n", content) if len(p.strip()) > 40]
        if paras: content = paras[-1]
    return content or msg.get("reasoning_content") or "[No reply]"

async def call_model(msgs, temp):
    if NVIDIA_API_KEY:
        try: return await _nvidia(msgs, temp)
        except Exception as e: print(f"nvidia: {e}")
    return "I'm Imhotep, built by Samuel Kiragu in Mukuyu, Murang'a, Kenya. Try again."

# ═══════════════════════════════════════════════════════════════
# CHAT
# ═══════════════════════════════════════════════════════════════
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
    agent = b.agent or "generalist"
    if is_sheng_query(p): agent = "sheng"
    t = max(0.0, min(2.0, b.temperature))
    s = sys_prompt(agent, prompt=p)
    w = None
    if b.location and b.location.get("lat") is not None:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("https://api.open-meteo.com/v1/forecast",
                    params={"latitude":b.location["lat"],"longitude":b.location["lon"],
                            "current":"temperature_2m,weather_code","timezone":"auto"})
            d = r.json(); cur = d.get("current") or {}
            if cur.get("temperature_2m") is not None:
                cls = classify_wx(round(cur["temperature_2m"]), cur.get("weather_code",0))
                w = {"temp":round(cur["temperature_2m"]),"city":"your area",
                     "description":cls["condition_label"], **cls}
                s += f"\nUser is at {w['temp']}°C, {w['description']}."
        except Exception: pass
    msgs = [{"role":"system","content":s},{"role":"user","content":p}]
    raw = scrub(await call_model(msgs, t))
    if u in chat_db:
        chat_db[u].append({"prompt":p,"reply":raw,"agent":agent,
                           "learning_mode":is_learning_query(p),
                           "timestamp":datetime.now(timezone.utc).isoformat()})
    return {"ok":True,"reply":raw,"agent":agent,"temperature":t,
            "learning_mode":is_learning_query(p),
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
        {"role":"user","content":f"Proposition: {p}\n\n150-word case FOR."}], 0.7))
    con = scrub(await call_model([{"role":"system","content":sys_prompt(ca,p)+"\nArgue AGAINST."},
        {"role":"user","content":f"Proposition: {p}\n\n150-word case AGAINST."}], 0.7))
    j = scrub(await call_model([{"role":"system","content":sys_prompt("generalist",p)+"\nImpartial judge."},
        {"role":"user","content":f"Judge: {p}\nPRO: {pro[:400]}\nCON: {con[:400]}"}], 0.4))
    return {"ok":True,"proposition":p,"proponent":pro,"opponent":con,"judge":j}

# ═══════════════════════════════════════════════════════════════
# PRICING + CHECKOUT + WEBHOOK
# ═══════════════════════════════════════════════════════════════
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
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post("https://api.nestlink.co.ke/v1/checkout",
                    json={"amount":t["amount"],"currency":"KES","email":b.email,
                          "narrative":t["label"],"reference":ref,
                          "redirect_url":success_url,"callback_url":webhook_url},
                    headers={"Authorization":f"Bearer {NESTLINK_API_KEY}",
                             "Content-Type":"application/json","Accept":"application/json"})
            if r.status_code < 300:
                d = r.json()
                payment_url = d.get("checkout_url") or d.get("url") or d.get("payment_url")
        except Exception as e: print(f"nestlink: {e}")
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

# ═══════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    return {"ok":True,"version":"6.1.0",
            "model_configured":bool(NVIDIA_API_KEY),
            "colab_fallback_configured":bool(COLAB_MODEL_URL),
            "weather_configured":bool(OPENWEATHER_KEY),
            "weather_provider":"openweathermap" if OPENWEATHER_KEY else "open-meteo",
            "nestlink_configured":bool(NESTLINK_API_KEY),
            "google_configured":bool(GOOGLE_CLIENT_ID),
            "kenya_counties":len(KENYA_COUNTIES),
            "sheng_agent":True, "nasa_fires":True, "kenya_news":True,
            "drought_data":True, "adaptive_length":True,
            "founder":"Samuel Kiragu"}
