
import os, re, time, secrets, hmac, hashlib, json, pathlib, math, asyncio
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import httpx, jwt
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr

SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY", "")
NESTLINK_API_KEY = os.getenv("NESTLINK_API_KEY", "")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS","*").split(",") if o.strip()]

app = FastAPI(title="Imhotep", version="10.0.0")
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
# THINKING STRIP — catches both <think> tags AND plain-text thinking
# ═══════════════════════════════════════════════════════════════
THINKING_PREFIXES = (
    "here's a thinking process", "here is a thinking process",
    "thinking process:", "let me think", "reasoning:",
    "let's break this down", "first, let me analyze",
    "analyze user input", "analyze the user's input",
    "1. **analyze", "1. analyze user", "identify persona",
)

def strip_thinking(text: str) -> str:
    if not text: return text
    # Strip <think>...</think> blocks
    text = re.sub(r"<think[^>]*>.*?</think\s*>", "", text,
                  flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE).strip()
    # If the content STARTS with a thinking prefix, try to find the real answer
    low = text.lower()[:300]
    if any(low.strip().startswith(p) for p in THINKING_PREFIXES):
        # Look for a "Final answer" or similar marker
        for marker in ["final answer:", "final response:", "answer:", "my response:", "reply:"]:
            idx = text.lower().rfind(marker)
            if idx > 0:
                return text[idx + len(marker):].strip()
        # Otherwise look for the last paragraph (often the actual answer)
        paras = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 30]
        if paras:
            # Skip paragraphs that look like analysis
            for p in reversed(paras):
                p_low = p.lower()
                # Skip if it contains numbered list markers
                if re.match(r"^\d+\.", p.strip()):
                    continue
                if any(w in p_low for w in ["analyze", "identify", "determine"]):
                    continue
                return p
            return paras[-1]
    return text

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

def nearest_county(lat, lon):
    best, bestd = None, 1e9
    for name, (clat, clon) in KENYA_COUNTIES_FULL.items():
        d = math.sqrt((lat-clat)**2 + (lon-clon)**2)
        if d < bestd: bestd, best = d, name
    return best

DROUGHT_DATA = {"turkana":3,"marsabit":3,"mandera":2,"wajir":2,"garissa":2,
                "isiolo":2,"samburu":3,"west pokot":2,"baringo":2,"kitui":2,
                "kilifi":2,"tana river":2,"nairobi":1,"kiambu":1,"murang'a":1,
                "nyeri":1,"kirinyaga":1,"kajiado":1,"kisumu":1,"mombasa":1}

def classify_wx(temp, code):
    try: c = int(code)
    except: c = 800
    if c == 800: e,l = "☀️","Clear"
    elif c in (801,802): e,l = "🌤️","Partly cloudy"
    elif c in (803,804): e,l = "☁️","Cloudy"
    elif 200 <= c < 300: e,l = "⛈️","Thunderstorm"
    elif 300 <= c < 400: e,l = "🌦️","Drizzle"
    elif 500 <= c < 600: e,l = "🌧️","Rain"
    elif 600 <= c < 700: e,l = "❄️","Snow"
    elif 700 <= c < 800: e,l = "🌫️","Foggy"
    elif c == 0: e,l = "☀️","Clear"
    elif c == 1: e,l = "🌤️","Mostly clear"
    elif c == 2: e,l = "🌤️","Partly cloudy"
    elif c == 3: e,l = "☁️","Overcast"
    elif c in (45,48): e,l = "🌫️","Foggy"
    elif c in (51,53,55): e,l = "🌦️","Drizzle"
    elif c in (61,63,65): e,l = "🌧️","Rain"
    elif c in (71,73,75): e,l = "❄️","Snow"
    elif c in (80,81,82): e,l = "🌧️","Showers"
    elif c == 95: e,l = "⛈️","Thunderstorm"
    elif c in (96,99): e,l = "⛈️","Heavy storm"
    else: e,l = "🌤️","Fair"
    if temp >= 35: f = "scorching hot 🥵"
    elif temp >= 30: f = "very hot 🥵"
    elif temp >= 26: f = "hot 🔥"
    elif temp >= 20: f = "warm 😊"
    elif temp >= 14: f = "cool 😌"
    elif temp >= 8: f = "cold 🧥"
    else: f = "very cold 🥶"
    return {"emoji":e,"condition_label":l,"feel":f}

PERSONAS = {"generalist":"a broad Kenyan assistant","scientist":"a rigorous scientist",
    "coder":"a senior software engineer","sheng":"a fluent Nairobi Sheng speaker",
    "economist":"an economist","writer":"a writer","analyst":"an analyst"}

SHENG_GREETINGS = ("sasa","niaje","mambo","marada","radudua","sema","bonga","uko poa")
def is_sheng(p):
    p = p.lower()
    return any(g in p for g in SHENG_GREETINGS) or "sheng" in p

LEARN = ("teach","explain","how does","why does","tutorial","step by step","code",
         "function","algorithm","debug","python","javascript","learn","study")
def is_learn(p):
    p = p.lower()
    return any(s in p for s in LEARN) or len(p.split()) > 25

def sys_prompt(agent, prompt=""):
    p = PERSONAS.get(agent, "a general assistant")
    learning = is_learn(prompt) if prompt else False
    ident = (f"You are Imhotep — a private AI built by Samuel Kiragu in Mukuyu, "
             f"Murang'a, Kenya. You are the {agent.upper()} agent — {p}.\n"
             "ABSOLUTE RULES:\n"
             "1. Output ONLY your final answer. NEVER output thinking, reasoning steps, "
             "numbered analysis lists, or phrases like 'Here is my thinking' or "
             "'Let me analyze'. Just answer directly.\n"
             "2. NEVER mention NVIDIA, Nemotron, Qwen, DeepSeek, Llama, Mistral, GPT, "
             "Claude, Gemini, OpenAI, Meta. If asked what model powers you: "
             "'I'm Imhotep, built by Samuel Kiragu in Kenya.'\n"
             "3. Speak in first person as Imhotep.\n")
    if agent == "sheng":
        return (ident + "\nSHENG MODE: Reply in Sheng mixed with Kiswahili and English. "
                "Use: sasa, niaje, mambo, poa, fiti, msee, mtaa, ocha, doo, chapaa. "
                "Keep it 1-3 sentences.")
    if learning:
        return ident + "Reply structured: 1. Summary. 2. Steps. 3. Example."
    return ident + "Reply SHORT (1-3 sentences). Warm, direct."

users_db, chat_db = {}, {}
TIER_CREDITS = {"guest":3,"free":3,"starter_10":3,"starter_20":20,"starter_50":100,
                "weekly":10000,"monthly":10000,"yearly":10000}
PRICES = {"starter_10":{"amount":10,"label":"3 chats"},
          "starter_20":{"amount":20,"label":"20 chats"},
          "starter_50":{"amount":50,"label":"100 chats"},
          "weekly":{"amount":99,"label":"Weekly unlimited"},
          "monthly":{"amount":299,"label":"Monthly unlimited"},
          "yearly":{"amount":5000,"label":"Yearly unlimited"}}
TIER_ALIAS = {"starter":"starter_10","pro":"monthly"}

class ChatIn(BaseModel):
    prompt: str; agent: Optional[str]="generalist"; user_id: Optional[str]=None
    temperature: float=0.7; location: Optional[dict]=None
class SignupIn(BaseModel): email: EmailStr; name: str
class CheckoutIn(BaseModel): tier: str; email: EmailStr
class GoogleIn(BaseModel): id_token: str
class DebateIn(BaseModel): proposition: str; agents: Optional[List[str]]=None

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
# MODEL CALL — with aggressive thinking strip
# ═══════════════════════════════════════════════════════════════
async def nvidia_call(msgs, temp):
    body = {"model":NVIDIA_MODEL,"messages":msgs,"temperature":temp,
            "top_p":0.95,"max_tokens":2048}
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(NVIDIA_URL,
            headers={"Authorization":f"Bearer {NVIDIA_API_KEY}",
                     "Content-Type":"application/json"}, json=body)
    if r.status_code >= 300: raise Exception(f"NVIDIA {r.status_code}")
    msg = r.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content:
        content = (msg.get("reasoning_content") or "").strip()
    return strip_thinking(content) or "[No reply]"

async def call_model(msgs, temp):
    if NVIDIA_API_KEY:
        try: return await nvidia_call(msgs, temp)
        except Exception as e: print(f"nvidia: {e}")
    return "I'm Imhotep, built by Samuel Kiragu in Mukuyu, Murang'a, Kenya."

# ═══════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════
@app.get("/agents")
async def agents(): return {"ok":True,"agents":list(PERSONAS.keys())}

@app.post("/auth/signup")
async def signup(b: SignupIn):
    u, new = mk_user(b.email.lower(), b.name, "free")
    return {"ok":True,"user_id":u,"token":issue(u),"user":users_db[u]}

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
    except Exception as e: raise HTTPException(400, f"Google failed: {e}")
    email = info.get("email")
    if not email: raise HTTPException(400, "Invalid token")
    u, _ = mk_user(email.lower(), info.get("name","Google User"), "free")
    return {"ok":True,"user_id":u,"token":issue(u),"user":users_db[u]}

# ═══════════════════════════════════════════════════════════════
# WEATHER — all counties with BATCHED requests (5 per batch, sequential batches)
# ═══════════════════════════════════════════════════════════════
@app.post("/weather")
async def weather_ep(body: dict):
    lat, lon, city = body.get("lat"), body.get("lon"), body.get("city")
    if lat is None and city and city in KENYA_COUNTIES_FULL:
        lat, lon = KENYA_COUNTIES_FULL[city]
    if lat is None or lon is None:
        raise HTTPException(400, "Need lat/lon or valid city")
    if OPENWEATHER_KEY:
        try:
            params = ({"lat":lat,"lon":lon} if not city else {"q":f"{city},KE"})
            params.update({"appid":OPENWEATHER_KEY,"units":"metric"})
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get("https://api.openweathermap.org/data/2.5/weather", params=params)
            d = r.json()
            if "main" in d:
                t = round(d["main"]["temp"]); cls = classify_wx(t, d["weather"][0]["id"])
                return {"ok":True,"temp":t,"condition":d["weather"][0]["main"],
                        "city":d.get("name") or city or "",
                        "humidity":d["main"].get("humidity"), **cls}
        except Exception: pass
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude":lat,"longitude":lon,
                    "current":"temperature_2m,weather_code,relative_humidity_2m",
                    "timezone":"Africa/Nairobi"})
    cur = (r.json().get("current") or {})
    t = round(cur.get("temperature_2m",0)); cls = classify_wx(t, cur.get("weather_code",0))
    return {"ok":True,"temp":t,"condition":cls["condition_label"],
            "humidity":cur.get("relative_humidity_2m"),
            "city":city or f"{lat:.2f},{lon:.2f}", **cls}

@app.get("/weather/all-counties")
async def weather_all():
    """All 47 counties in ONE multi-location request."""
    names = list(KENYA_COUNTIES_FULL.keys())
    coords = list(KENYA_COUNTIES_FULL.values())
    lats = ",".join(f"{la:.4f}" for la, lo in coords)
    lons = ",".join(f"{lo:.4f}" for la, lo in coords)
    try:
        async with httpx.AsyncClient(timeout=25) as c:
            r = await c.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude": lats, "longitude": lons,
                        "current": "temperature_2m,weather_code",
                        "timezone": "Africa/Nairobi"})
        data = r.json()
        items = data if isinstance(data, list) else [data]
        results = []
        for j, item in enumerate(items):
            if j >= len(names): break
            cur = item.get("current") or {}
            if cur.get("temperature_2m") is None: continue
            t = round(cur["temperature_2m"])
            cls = classify_wx(t, cur.get("weather_code", 0))
            results.append({"county": names[j], "temp": t, **cls})
        return {"ok": True, "count": len(results), "counties": results}
    except Exception as e:
        print(f"weather_all error: {e}")
        return {"ok": False, "count": 0, "counties": [], "error": str(e)}

@app.get("/heatmap/rain")
async def heatmap_rain():
    """36-cell rain grid in ONE multi-location request."""
    lat_min, lat_max = -4.7, 5.0
    lon_min, lon_max = 33.9, 41.9
    N = 6
    cells = []
    for i in range(N):
        for j in range(N):
            la = lat_min + (lat_max - lat_min) * i / (N - 1)
            lo = lon_min + (lon_max - lon_min) * j / (N - 1)
            cells.append((la, lo))
    lats = ",".join(f"{la:.3f}" for la, lo in cells)
    lons = ",".join(f"{lo:.3f}" for la, lo in cells)
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude": lats, "longitude": lons,
                        "daily": "precipitation_sum", "forecast_days": 3,
                        "timezone": "Africa/Nairobi"})
        data = r.json()
        items = data if isinstance(data, list) else [data]
        out = []
        for k, item in enumerate(items):
            if k >= len(cells): break
            la, lo = cells[k]
            daily = item.get("daily", {}).get("precipitation_sum", [])
            total = sum(v for v in daily if v is not None) if daily else 0
            out.append({"lat": round(la, 2), "lon": round(lo, 2),
                        "rain_mm": round(total, 1),
                        "county": nearest_county(la, lo) or ""})
        return {"ok": True, "grid_size": N, "cells": out}
    except Exception as e:
        print(f"heatmap error: {e}")
        return {"ok": False, "grid_size": N, "cells": [], "error": str(e)}

@app.get("/flood/{county}")
async def flood(county: str):
    """River discharge forecast for a Kenyan county. Open-Meteo Flood API."""
    key = county.strip().lower()
    coords = None
    for name, c in KENYA_COUNTIES_FULL.items():
        if name.lower() == key:
            coords = c; break
    if not coords:
        return {"ok": False, "error": f"Unknown county '{county}'"}
    la, lo = coords
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get("https://flood-api.open-meteo.com/v1/flood",
                params={"latitude": la, "longitude": lo,
                        "daily": "river_discharge",
                        "forecast_days": 7})
        d = r.json()
        daily = d.get("daily", {})
        values = daily.get("river_discharge", [])
        dates = daily.get("time", [])
        peak = max(values) if values else 0
        latest = values[0] if values else 0
        # Simple risk classification
        if peak > 500: level, color = "SEVERE", "#ff2b2b"
        elif peak > 200: level, color = "HIGH", "#ff6b6b"
        elif peak > 80: level, color = "MODERATE", "#ffaa00"
        elif peak > 30: level, color = "LOW", "#00c87a"
        else: level, color = "MINIMAL", "#00d9ff"
        return {"ok":True,"county":county.title(),
                "current_discharge": round(latest, 1) if latest else 0,
                "peak_next_7d": round(peak, 1) if peak else 0,
                "level": level, "color": color,
                "dates": dates, "values": [round(v,1) if v else 0 for v in values],
                "source": "Open-Meteo Flood API (GloFAS)"}
    except Exception as e:
        return {"ok":False,"error":str(e)}

# ═══════════════════════════════════════════════════════════════
# GDACS — Global Disaster Alerts (free, no key)
# ═══════════════════════════════════════════════════════════════
@app.get("/disasters")
async def disasters():
    """Global disaster alerts affecting Kenya. GDACS API, free, no key."""
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get("https://www.gdacs.org/gdacsapi/api/Events/geteventlist/SEARCH",
                            params={"fromDate": "2024-01-01", "toDate": "2026-12-31",
                                    "alertlevel": "Green;Orange;Red"})
        data = r.json()
        events = data.get("features", [])
        # Filter for Kenya
        kenya = []
        for e in events:
            props = e.get("properties", {})
            country = str(props.get("country", ""))
            if "kenya" in country.lower():
                kenya.append({
                    "type": props.get("eventtype", ""),
                    "alert_level": props.get("alertlevel", ""),
                    "title": props.get("name", ""),
                    "country": country,
                    "date": props.get("todate", ""),
                    "url": props.get("url", {}).get("report", "") if isinstance(props.get("url"), dict) else "",
                })
        return {"ok":True,"count":len(kenya),"events":kenya[:20]}
    except Exception as e:
        return {"ok":False,"error":str(e)}

# ═══════════════════════════════════════════════════════════════
# NASA POWER — precipitation & climate data (free, no key)
# ═══════════════════════════════════════════════════════════════
@app.get("/nasa/power/{county}")
async def nasa_power(county: str):
    """NASA POWER: 30-day rainfall + temperature for a county. Free, no key."""
    key = county.strip().lower()
    coords = None
    for name, c in KENYA_COUNTIES_FULL.items():
        if name.lower() == key:
            coords = c; break
    if not coords:
        return {"ok": False, "error": f"Unknown county '{county}'"}
    la, lo = coords
    # Last 30 days
    end = datetime.now(timezone.utc) - timedelta(days=2)
    start = end - timedelta(days=30)
    try:
        async with httpx.AsyncClient(timeout=45) as c:
            r = await c.get("https://power.larc.nasa.gov/api/temporal/daily/point",
                params={"parameters": "PRECTOTCORR,T2M",
                        "community": "AG",
                        "longitude": lo, "latitude": la,
                        "start": start.strftime("%Y%m%d"),
                        "end": end.strftime("%Y%m%d"),
                        "format": "JSON"})
        d = r.json()
        param = d.get("properties", {}).get("parameter", {})
        rain = param.get("PRECTOTCORR", {})
        temps = param.get("T2M", {})
        rain_vals = [v for v in rain.values() if isinstance(v, (int, float))]
        temp_vals = [v for v in temps.values() if isinstance(v, (int, float))]
        return {"ok":True,"county":county.title(),
                "rain_30d_mm": round(sum(rain_vals), 1) if rain_vals else 0,
                "avg_temp_c": round(sum(temp_vals)/len(temp_vals), 1) if temp_vals else 0,
                "days": len(rain_vals),
                "source": "NASA POWER (LARC)"}
    except Exception as e:
        return {"ok":False,"error":str(e)}

# ═══════════════════════════════════════════════════════════════
# LIGHTNING — WMO thunderstorm codes from Open-Meteo
# ═══════════════════════════════════════════════════════════════
@app.get("/lightning")
async def lightning():
    """Thunderstorm detection in ONE multi-location request."""
    names = list(KENYA_COUNTIES_FULL.keys())
    coords = list(KENYA_COUNTIES_FULL.values())
    lats = ",".join(f"{la:.4f}" for la, lo in coords)
    lons = ",".join(f"{lo:.4f}" for la, lo in coords)
    try:
        async with httpx.AsyncClient(timeout=25) as c:
            r = await c.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude": lats, "longitude": lons,
                        "current": "weather_code,temperature_2m",
                        "timezone": "Africa/Nairobi"})
        data = r.json()
        items = data if isinstance(data, list) else [data]
        storms = []
        for j, item in enumerate(items):
            if j >= len(names): break
            cur = item.get("current") or {}
            code = cur.get("weather_code", 0)
            if code in (95, 96, 99):
                storms.append({"county": names[j], "code": code,
                               "temp": round(cur.get("temperature_2m", 0)),
                               "severity": "Heavy" if code == 99 else "Moderate" if code == 96 else "Thunderstorm"})
        return {"ok": True, "count": len(storms), "storms": storms}
    except Exception as e:
        print(f"lightning error: {e}")
        return {"ok": False, "count": 0, "storms": [], "error": str(e)}

@app.get("/drought/{county}")
async def drought(county: str):
    p = DROUGHT_DATA.get(county.strip().lower())
    if p is None: return {"ok":False,"error":f"No data for '{county}'"}
    labels = {1:"Minimal",2:"Alert",3:"Alarm",4:"Emergency",5:"Famine"}
    return {"ok":True,"county":county.title(),"phase":p,"label":labels[p]}

@app.get("/ndvi/{county}")
async def ndvi(county: str):
    p = DROUGHT_DATA.get(county.strip().lower())
    if p is None: return {"ok":False,"error":f"No data for '{county}'"}
    vci_map = {1:55,2:40,3:25,4:15,5:5}
    vci = vci_map.get(p,40)
    if vci>=50: s = "Wet — vegetation healthy"
    elif vci>=35: s = "Normal — no drought stress"
    elif vci>=21: s = "Moderate drought — stress visible"
    elif vci>=10: s = "Severe drought — significant loss"
    else: s = "Extreme — vegetation collapse"
    labels = {1:"Minimal",2:"Alert",3:"Alarm",4:"Emergency",5:"Famine"}
    return {"ok":True,"county":county.title(),"vci":vci,"status":s,
            "drought_phase":p,"drought_label":labels[p],
            "ndvi_explanation":"NDVI measures how green vegetation is from satellite. 0.1=bare ground, 0.2-0.3=grassland, 0.5+=forest.",
            "vci_explanation":"VCI compares today's NDVI vs historical min/max. Below 35 triggers Kenya's drought early warning.",
            "thresholds":{"wet":"VCI>=50","normal":"35-50","moderate":"21-34","severe":"10-20","extreme":"<=10"}}

# ═══════════════════════════════════════════════════════════════
# FIRES
# ═══════════════════════════════════════════════════════════════
@app.get("/fires/kenya")
async def fires():
    urls = ["https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_24h.csv",
            "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_24h.csv"]
    fires = []
    async with httpx.AsyncClient(timeout=45) as c:
        for url in urls:
            try:
                r = await c.get(url)
                if r.status_code >= 300: continue
                lines = r.text.strip().split("\n")
                if not lines: continue
                hdr = lines[0].split(",")
                for line in lines[1:]:
                    parts = line.split(",")
                    if len(parts) < len(hdr): continue
                    row = dict(zip(hdr, parts))
                    try: la=float(row.get("latitude",0)); lo=float(row.get("longitude",0))
                    except: continue
                    if -5<=la<=6 and 33<=lo<=42:
                        fires.append({"lat":la,"lon":lo,"county":nearest_county(la,lo) or "Unknown",
                                      "brightness":row.get("bright_ti4") or row.get("brightness"),
                                      "confidence":row.get("confidence"),
                                      "date":row.get("acq_date"),"time":row.get("acq_time"),
                                      "satellite":row.get("satellite")})
            except Exception: continue
    return {"ok":True,"count":len(fires),"fires":fires[:100]}

# ═══════════════════════════════════════════════════════════════
# NEWS
# ═══════════════════════════════════════════════════════════════
@app.get("/news/kenya")
async def news(limit: int = 10):
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get("https://www.standardmedia.co.ke/rss/headlines.php",
                            headers={"User-Agent":"Imhotep/1.0"})
        root = ET.fromstring(r.text)
        items = []
        for item in root.findall(".//item")[:limit]:
            items.append({"title":(item.findtext("title") or "").strip(),
                          "link":(item.findtext("link") or "").strip()})
        return {"ok":True,"count":len(items),"items":items}
    except Exception as e: raise HTTPException(502, str(e))

# ═══════════════════════════════════════════════════════════════
# CLIMATE
# ═══════════════════════════════════════════════════════════════
@app.get("/climate/kenya")
async def climate():
    lat, lon = -1.2921, 36.8219
    try:
        async with httpx.AsyncClient(timeout=90) as c:
            r1 = await c.get("https://archive-api.open-meteo.com/v1/archive",
                params={"latitude":lat,"longitude":lon,"start_date":"1985-01-01",
                        "end_date":"2014-12-31","daily":"temperature_2m_max",
                        "timezone":"Africa/Nairobi"})
            h = r1.json()
            r2 = await c.get("https://climate-api.open-meteo.com/v1/climate",
                params={"latitude":lat,"longitude":lon,"start_date":"2025-01-01",
                        "end_date":"2050-12-31","models":"MRI_AGCM3_2_S",
                        "daily":"temperature_2m_max"})
            p = r2.json()
    except Exception as e: raise HTTPException(502, str(e))
    def avg(arr):
        v = [x for x in (arr or []) if isinstance(x,(int,float))]
        return round(sum(v)/len(v),2) if v else None
    ht = avg(h.get("daily",{}).get("temperature_2m_max",[]))
    pt = avg(p.get("daily",{}).get("temperature_2m_max",[]))
    dt = round(pt-ht,2) if (ht and pt) else None
    return {"ok":True,"historical":{"period":"1985-2014","avg_tmax_c":ht},
            "projection":{"period":"2025-2050","avg_tmax_c":pt},
            "change":{"warming_c":dt,
                      "verdict":("Kenya is warming — prepare for heat stress" if dt and dt>0.5
                                 else "Stable" if dt and abs(dt)<=0.5 else "Cooling")}}

# ═══════════════════════════════════════════════════════════════
# NASA + SATELLITE
# ═══════════════════════════════════════════════════════════════
@app.get("/nasa/earth")
async def nasa_earth():
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get("https://epic.gsfc.nasa.gov/api/natural")
    items = r.json()
    if not items: raise HTTPException(404, "No images")
    latest = items[0]
    dp = latest["date"].split(" ")[0].replace("-","/")
    return {"ok":True,"date":latest["date"],
            "image_url":f"https://epic.gsfc.nasa.gov/archive/natural/{dp}/png/{latest['image']}.png"}

@app.get("/img/proxy")
async def proxy(url: str):
    ALLOWED = ("epic.gsfc.nasa.gov","cdn.star.nesdis.noaa.gov","gibs.earthdata.nasa.gov","images-api.nasa.gov")
    from urllib.parse import urlparse
    host = urlparse(url).netloc
    if not any(host.endswith(d) for d in ALLOWED): raise HTTPException(403,"Domain not allowed")
    async with httpx.AsyncClient(timeout=45) as c:
        r = await c.get(url, headers={"User-Agent":"Imhotep/1.0"})
    if r.status_code >= 300: raise HTTPException(r.status_code,"Upstream failed")
    return StreamingResponse(iter([r.content]),
        media_type=r.headers.get("content-type","image/jpeg"),
        headers={"Cache-Control":"public, max-age=600","Access-Control-Allow-Origin":"*"})

@app.post("/satellite/analyze")
async def sat_analyze(body: dict):
    lat = body.get("lat"); lon = body.get("lon")
    if lat is None or lon is None: raise HTTPException(400,"Need lat/lon")
    county = nearest_county(lat, lon) or "Unknown"
    weather_text = "unavailable"
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude":lat,"longitude":lon,
                        "current":"temperature_2m,weather_code","timezone":"Africa/Nairobi"})
        cur = (r.json().get("current") or {})
        t = round(cur.get("temperature_2m",0)); cls = classify_wx(t, cur.get("weather_code",0))
        weather_text = f"{t}°C, {cls['condition_label']}"
    except Exception: pass
    phase = DROUGHT_DATA.get(county.lower())
    drought_text = {1:"Minimal",2:"Alert",3:"Alarm",4:"Emergency",5:"Famine"}[phase] if phase else "no data"
    fire_count = 0
    try:
        async with httpx.AsyncClient(timeout=45) as c:
            r = await c.get("https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_24h.csv")
        if r.status_code == 200:
            for line in r.text.strip().split("\n")[1:]:
                parts = line.split(",")
                if len(parts) < 2: continue
                try: la=float(parts[0]); lo=float(parts[1])
                except: continue
                if abs(la-lat) < 0.5 and abs(lo-lon) < 0.5: fire_count += 1
    except Exception: pass
    prompt = (f"You are Imhotep, built by Samuel Kiragu in Mukuyu, Murang'a, Kenya. "
              f"Talk to a user looking at satellite data of {county} County. "
              f"Weather: {weather_text}. Drought: {drought_text}. Fires nearby: {fire_count}. "
              f"Answer directly in 3-4 sentences — what this means for people of {county}. "
              f"Do NOT output thinking or reasoning steps. Just give the answer.")
    try:
        raw = await nvidia_call([{"role":"user","content":prompt}], 0.6)
    except Exception:
        raw = f"{county} County: {weather_text}. Drought: {drought_text}. Fires: {fire_count}."
    return {"ok":True,"county":county,"lat":lat,"lon":lon,"weather":weather_text,
            "drought":drought_text,"fire_count":fire_count,"analysis":raw}

# ═══════════════════════════════════════════════════════════════
# CHAT
# ═══════════════════════════════════════════════════════════════
@app.post("/chat")
async def chat(b: ChatIn):
    p = b.prompt.strip()
    if not p: raise HTTPException(400,"Empty")
    u = b.user_id
    if not u: raise HTTPException(402, detail={"error":"Sign in required",
        "upgrade_url":"https://kiragu2004.github.io/imhotep-site/auth.html"})
    if u not in users_db: raise HTTPException(402, detail={"error":"Session expired",
        "upgrade_url":"https://kiragu2004.github.io/imhotep-site/auth.html"})
    if not debit(u):
        c = creds(u)
        raise HTTPException(402, detail={"error":"No credits left. Upgrade.",
            "used":c["used"],"limit":c["limit"],
            "upgrade_url":"https://kiragu2004.github.io/imhotep-site/pricing.html"})
    agent = b.agent or "generalist"
    if is_sheng(p): agent = "sheng"
    t = max(0.0, min(2.0, b.temperature))
    msgs = [{"role":"system","content":sys_prompt(agent,p)},
            {"role":"user","content":p}]
    raw = scrub(await call_model(msgs, t))
    raw = strip_thinking(raw)  # Double strip to be safe
    if u in chat_db:
        chat_db[u].append({"prompt":p,"reply":raw,"agent":agent,
                           "timestamp":datetime.now(timezone.utc).isoformat()})
    return {"ok":True,"reply":raw,"agent":agent,"credits":creds(u)}

@app.get("/chat/history")
async def hist(user_id: Optional[str] = None):
    if not user_id or user_id not in chat_db: return {"ok":True,"history":[]}
    return {"ok":True,"history":chat_db[user_id][-50:]}

@app.get("/user/profile")
async def profile(user_id: Optional[str] = None):
    if not user_id or user_id not in users_db: return {"ok":False}
    return {"ok":True,"user":users_db[user_id],"credits":creds(user_id)}

@app.post("/debate")
async def debate(b: DebateIn):
    p = b.proposition.strip()
    if not p: raise HTTPException(400,"Proposition required")
    ag = b.agents or ["scientist","economist"]
    pa, ca = ag[0], ag[1]
    pro = scrub(await call_model([{"role":"system","content":sys_prompt(pa,p)+" Argue FOR."},
        {"role":"user","content":f"{p} — 150 words FOR."}], 0.7))
    con = scrub(await call_model([{"role":"system","content":sys_prompt(ca,p)+" Argue AGAINST."},
        {"role":"user","content":f"{p} — 150 words AGAINST."}], 0.7))
    j = scrub(await call_model([{"role":"system","content":"You are a judge."},
        {"role":"user","content":f"Judge: {p}\nPRO: {pro[:300]}\nCON: {con[:300]}"}], 0.4))
    return {"ok":True,"proposition":p,"proponent":pro,"opponent":con,"judge":j}

# ═══════════════════════════════════════════════════════════════
# PRICING
# ═══════════════════════════════════════════════════════════════
@app.get("/pricing")
async def pricing(): return {"ok":True,"currency":"KES","tiers":PRICES}

@app.post("/checkout")
async def checkout(b: CheckoutIn):
    tier = TIER_ALIAS.get(b.tier, b.tier)
    if tier not in PRICES: raise HTTPException(400,"Unknown tier")
    t = PRICES[tier]
    ref = f"IMH-{tier}-{int(time.time())}-{secrets.token_hex(3)}"
    url = None
    if NESTLINK_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post("https://api.nestlink.co.ke/v1/checkout",
                    json={"amount":t["amount"],"currency":"KES","email":b.email,
                          "narrative":t["label"],"reference":ref},
                    headers={"Authorization":f"Bearer {NESTLINK_API_KEY}"})
            if r.status_code < 300:
                d = r.json()
                url = d.get("checkout_url") or d.get("url")
        except Exception: pass
    if not url:
        url = f"https://me.nestlink.co.ke/Imhotepagenticai?email={b.email}&reference={ref}"
    return {"ok":True,"tier":tier,"amount":t["amount"],"payment_url":url,"reference":ref}

@app.post("/nestlink/webhook")
async def webhook(request: Request):
    raw = await request.body()
    try:
        with open(WEBHOOK_LOG, "a") as f:
            f.write(json.dumps({"ts":time.time(),"body":raw.decode("utf-8","ignore")[:2000]}) + "\n")
    except: pass
    try: payload = json.loads(raw.decode())
    except: payload = {}
    def find(p, keys):
        if not isinstance(p, dict): return None
        for k in keys:
            if p.get(k): return p[k]
        for v in p.values():
            if isinstance(v, dict):
                r = find(v, keys)
                if r: return r
        return None
    email = find(payload, ("email","customer_email"))
    ref = find(payload, ("reference","api_ref","ref"))
    if not email or not ref: return {"ok":True,"upgraded":False}
    parts = str(ref).split("-")
    tier = None
    for i in range(1, len(parts)+1):
        cand = "-".join(parts[1:i])
        if cand in TIER_CREDITS: tier = cand; break
    if not tier: return {"ok":True,"upgraded":False}
    for uid, u in users_db.items():
        if u["email"].lower() == email.lower():
            u["credits"] += TIER_CREDITS[tier]; u["tier"] = tier; u["credits_used"] = 0
            return {"ok":True,"upgraded":True,"email":email,"tier":tier}
    uid, _ = mk_user(email.lower(), email.split("@")[0], tier)
    users_db[uid]["credits"] = TIER_CREDITS[tier]
    return {"ok":True,"upgraded":True,"created":True}

@app.get("/health")
async def health():
    return {"ok":True,"version":"10.0.0",
            "model_configured":bool(NVIDIA_API_KEY),
            "nestlink_configured":bool(NESTLINK_API_KEY),
            "counties":len(KENYA_COUNTIES_FULL),
            "endpoints":["/weather/all-counties","/heatmap/rain","/flood/{county}",
                         "/disasters","/lightning","/nasa/power/{county}",
                         "/ndvi/{county}","/fires/kenya","/news/kenya","/climate/kenya"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
