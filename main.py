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
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS","*").split(",") if o.strip()]

app = FastAPI(title="Imhotep", version="13.0.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

_C = {}
def cget(k, ttl):
    e = _C.get(k)
    if e and (time.time() - e["t"]) < ttl: return e["v"]
    return None
def cset(k, v): _C[k] = {"t": time.time(), "v": v}

_LEAK = [(r"\bNVIDIA\s+NIM\b","Imhotep"),(r"\bNVIDIA\b","Imhotep"),
         (r"\bNemotron[\w\-\.]*\b","Imhotep"),(r"\bNIM\b","Imhotep"),
         (r"\bQwen[\w\-\.]*\b","Imhotep"),(r"\bDeepSeek[\w\-]*\b","Imhotep"),
         (r"\bMistral[\w\-]*\b","Imhotep"),(r"\bLlama[\s\-]?[\d\.\w]*\b","Imhotep")]
def scrub(t):
    if not t: return t
    for p, r in _LEAK: t = re.sub(p, r, t, flags=re.IGNORECASE)
    return t

THINK = ("here's a thinking","thinking process","let me think","analyze user input",
         "1. **analyze","1. analyze user","identify persona","analyze the user")
def strip_thinking(text):
    if not text: return text
    text = re.sub(r"<think[^>]*>.*?</think\s*>","",text,flags=re.DOTALL|re.IGNORECASE).strip()
    text = re.sub(r"</?think\b[^>]*>","",text,flags=re.IGNORECASE).strip()
    low = text.lower()[:250]
    if any(low.strip().startswith(p) for p in THINK):
        for m in ["final answer:","answer:","reply:"]:
            i = text.lower().rfind(m)
            if i > 0: return text[i+len(m):].strip()
        paras = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip())>30]
        if paras:
            for p in reversed(paras):
                if re.match(r"^\d+\.", p.strip()): continue
                if any(w in p.lower() for w in ["analyze","identify"]): continue
                return p
            return paras[-1]
    return text

KENYA = {
    "Mombasa":(-4.0435,39.6682),"Kwale":(-4.1742,39.4524),"Kilifi":(-3.6305,39.8499),
    "Tana River":(-1.5,39.9833),"Lamu":(-2.2717,40.902),"Taita Taveta":(-3.4,38.55),
    "Garissa":(-0.4536,39.6461),"Wajir":(1.7471,40.0573),"Mandera":(3.9366,41.867),
    "Marsabit":(2.3344,37.9899),"Isiolo":(0.3546,37.5822),"Meru":(0.05,37.65),
    "Tharaka-Nithi":(-0.1667,37.9),"Embu":(-0.5333,37.45),"Kitui":(-1.3667,38.0167),
    "Machakos":(-1.5177,37.2634),"Makueni":(-2.0,37.6167),"Nyandarua":(-0.5,36.3667),
    "Nyeri":(-0.4167,36.95),"Kirinyaga":(-0.5,37.2833),"Murang'a":(-0.7167,37.15),
    "Kiambu":(-1.1667,36.8333),"Turkana":(3.1191,35.5973),"West Pokot":(1.2333,35.1167),
    "Samburu":(1.1,36.7),"Trans Nzoia":(1.0167,35.0),"Uasin Gishu":(0.5143,35.2698),
    "Elgeyo-Marakwet":(0.5167,35.5),"Nandi":(0.1833,35.1),"Baringo":(0.4667,35.9667),
    "Laikipia":(0.05,36.7),"Nakuru":(-0.3031,36.08),"Narok":(-1.0833,35.8667),
    "Kajiado":(-1.85,36.7833),"Kericho":(-0.3667,35.2833),"Bomet":(-0.7833,35.3333),
    "Kakamega":(0.2827,34.7519),"Vihiga":(0.05,34.7333),"Bungoma":(0.5667,34.5667),
    "Busia":(0.45,34.1167),"Siaya":(0.0667,34.2833),"Kisumu":(-0.0917,34.768),
    "Homa Bay":(-0.5167,34.45),"Migori":(-0.9833,34.4667),"Kisii":(-0.6817,34.768),
    "Nyamira":(-0.5667,34.9333),"Nairobi":(-1.2921,36.8219),
}

def nearest_county(lat, lon):
    b, bd = None, 1e9
    for n, (cl, cn) in KENYA.items():
        d = math.sqrt((lat-cl)**2 + (lon-cn)**2)
        if d < bd: bd, b = d, n
    return b

DROUGHT = {"turkana":3,"marsabit":3,"mandera":2,"wajir":2,"garissa":2,"isiolo":2,
           "samburu":3,"west pokot":2,"baringo":2,"kitui":2,"kilifi":2,"tana river":2,
           "nairobi":1,"kiambu":1,"murang'a":1,"nyeri":1,"kirinyaga":1,"kajiado":1}

def classify_wx(t, c):
    try: c = int(c)
    except: c = 0
    if c == 0: e,l = "☀️","Clear"
    elif c == 1: e,l = "🌤️","Mostly clear"
    elif c == 2: e,l = "🌤️","Partly cloudy"
    elif c == 3: e,l = "☁️","Overcast"
    elif c in (45,48): e,l = "🌫️","Foggy"
    elif c in (51,53,55): e,l = "🌦️","Drizzle"
    elif c in (61,63,65): e,l = "🌧️","Rain"
    elif c in (80,81,82): e,l = "🌧️","Showers"
    elif c == 95: e,l = "⛈️","Thunderstorm"
    elif c in (96,99): e,l = "⛈️","Heavy storm"
    else: e,l = "🌤️","Fair"
    if t >= 35: f = "scorching 🥵"
    elif t >= 30: f = "very hot 🥵"
    elif t >= 26: f = "hot 🔥"
    elif t >= 20: f = "warm 😊"
    elif t >= 14: f = "cool 😌"
    elif t >= 8: f = "cold 🧥"
    else: f = "very cold 🥶"
    return {"emoji":e,"condition_label":l,"feel":f}

PERSONAS = {"generalist":"a broad Kenyan assistant","scientist":"a rigorous scientist",
    "coder":"a senior software engineer","sheng":"a fluent Nairobi Sheng speaker",
    "economist":"an economist","writer":"a writer","analyst":"an analyst"}
SHENG_G = ("sasa","niaje","mambo","marada","radudua","sema","bonga","uko poa")
def is_sheng(p):
    p = p.lower()
    return any(g in p for g in SHENG_G) or "sheng" in p
LEARN = ("teach","explain","how does","why does","tutorial","step by step","code",
         "function","algorithm","debug","python","learn","study")
def is_learn(p):
    p = p.lower()
    return any(s in p for s in LEARN) or len(p.split()) > 25

def sys_prompt(a, p=""):
    pp = PERSONAS.get(a, "an assistant")
    l = is_learn(p) if p else False
    ident = (f"You are Imhotep — a private AI built by Samuel Kiragu in Mukuyu, "
             f"Murang'a, Kenya. You are the {a.upper()} agent — {pp}.\n"
             "RULES: 1. Output ONLY final answer. No thinking. "
             "2. NEVER mention NVIDIA, Nemotron, Qwen, DeepSeek, Llama, Mistral, GPT, "
             "Claude, Gemini, OpenAI. If asked: 'I'm Imhotep, built by Samuel Kiragu.'\n")
    if a == "sheng":
        return (ident + "Reply in Sheng mixed with Kiswahili and English. "
                "Use: sasa, niaje, mambo, poa, fiti, msee, mtaa.")
    if l: return ident + "Reply structured: summary, steps, example."
    return ident + "Reply SHORT (1-3 sentences)."

users_db, chat_db = {}, {}
TIER_C = {"guest":3,"free":3,"starter_10":3,"starter_20":20,"starter_50":100,
          "weekly":10000,"monthly":10000,"yearly":10000}
PRICES = {"starter_10":{"amount":10,"label":"3 chats"},
          "starter_20":{"amount":20,"label":"20 chats"},
          "starter_50":{"amount":50,"label":"100 chats"},
          "weekly":{"amount":99,"label":"Weekly unlimited"},
          "monthly":{"amount":299,"label":"Monthly unlimited"},
          "yearly":{"amount":5000,"label":"Yearly unlimited"}}
TALIAS = {"starter":"starter_10","pro":"monthly"}

class ChatIn(BaseModel):
    prompt: str; agent: Optional[str]="generalist"; user_id: Optional[str]=None
    temperature: float=0.7; location: Optional[dict]=None
class SignupIn(BaseModel): email: EmailStr; name: str
class CheckoutIn(BaseModel): tier: str; email: EmailStr
class GoogleIn(BaseModel): id_token: str
class DebateIn(BaseModel): proposition: str; agents: Optional[List[str]]=None

def issue(u, days=30):
    return jwt.encode({"user_id":u,"exp":datetime.now(timezone.utc)+timedelta(days=days)},
                      SECRET_KEY, algorithm="HS256")
def mk_user(e, n, t="free"):
    for u, v in users_db.items():
        if v["email"] == e: return (u, False)
    u = f"user_{int(time.time()*1000)}"
    users_db[u] = {"email":e,"name":n,"tier":t,"credits":TIER_C.get(t,3),
                   "credits_used":0,"created_at":datetime.now(timezone.utc).isoformat()}
    chat_db[u] = []
    return (u, True)
def creds(u):
    if u not in users_db: return {"used":0,"remaining":0,"limit":0,"allowed":False}
    x = users_db[u]; t = x.get("credits",0); s = x.get("credits_used",0)
    return {"used":s,"remaining":max(0,t-s),"limit":t,"allowed":s<t}
def debit(u):
    if u not in users_db or not creds(u)["allowed"]: return False
    users_db[u]["credits_used"] = users_db[u].get("credits_used",0) + 1
    return True

# ═══════════════════════════════════════════════════════════════
# WEATHER — SINGLE county per request (no batches)
# ═══════════════════════════════════════════════════════════════
@app.get("/counties")
async def counties():
    return {"ok": True, "counties": list(KENYA.keys())}

@app.get("/weather/county/{name}")
async def weather_county(name: str):
    """Weather for ONE county. Cached 5 min."""
    ck = f"wx_{name}"
    c = cget(ck, 300)
    if c: return c
    key = next((k for k in KENYA if k.lower() == name.lower()), None)
    if not key: raise HTTPException(404, f"Unknown county '{name}'")
    la, lo = KENYA[key]
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude": f"{la:.4f}", "longitude": f"{lo:.4f}",
                        "current": "temperature_2m,weather_code,relative_humidity_2m",
                        "timezone": "Africa/Nairobi"})
        if r.status_code != 200: raise HTTPException(502, f"HTTP {r.status_code}")
        cur = (r.json().get("current") or {})
        t = round(cur.get("temperature_2m", 0))
        cls = classify_wx(t, cur.get("weather_code", 0))
        result = {"ok": True, "county": key, "temp": t,
                  "humidity": cur.get("relative_humidity_2m"), **cls}
        cset(ck, result)
        return result
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(502, str(e))

@app.get("/weather/forecast/{name}")
async def weather_forecast(name: str):
    """7-day forecast for one county."""
    ck = f"fc_{name}"
    c = cget(ck, 1800)
    if c: return c
    key = next((k for k in KENYA if k.lower() == name.lower()), None)
    if not key: raise HTTPException(404, f"Unknown county")
    la, lo = KENYA[key]
    try:
        async with httpx.AsyncClient(timeout=12) as c:
            r = await c.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude": f"{la:.4f}", "longitude": f"{lo:.4f}",
                        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                        "forecast_days": 7,
                        "timezone": "Africa/Nairobi"})
        d = r.json().get("daily", {})
        dates = d.get("time", [])
        tmax = d.get("temperature_2m_max", [])
        tmin = d.get("temperature_2m_min", [])
        rain = d.get("precipitation_sum", [])
        codes = d.get("weather_code", [])
        days = []
        for i in range(min(7, len(dates))):
            cls = classify_wx(round(tmax[i] or 0), codes[i] if i < len(codes) else 0)
            days.append({"date": dates[i],
                         "tmax": round(tmax[i]) if tmax[i] is not None else None,
                         "tmin": round(tmin[i]) if tmin[i] is not None else None,
                         "rain_mm": round(rain[i], 1) if i < len(rain) and rain[i] is not None else 0,
                         "emoji": cls["emoji"],
                         "condition": cls["condition_label"]})
        result = {"ok": True, "county": key, "days": days}
        cset(ck, result)
        return result
    except Exception as e:
        raise HTTPException(502, str(e))

# ── Legacy: single-weather POST (kept for chat.html) ──
@app.post("/weather")
async def weather_ep(b: dict):
    lat, lon, city = b.get("lat"), b.get("lon"), b.get("city")
    if lat is None and city and city in KENYA:
        lat, lon = KENYA[city]
    if lat is None or lon is None: raise HTTPException(400, "Need lat/lon or city")
    if city:
        return await weather_county(city)
    # lat/lon path
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude":lat,"longitude":lon,
                        "current":"temperature_2m,weather_code,relative_humidity_2m",
                        "timezone":"Africa/Nairobi"})
        cur = (r.json().get("current") or {})
        t = round(cur.get("temperature_2m",0))
        cls = classify_wx(t, cur.get("weather_code",0))
        return {"ok":True,"temp":t,"county":nearest_county(lat,lon) or "Unknown",
                "humidity":cur.get("relative_humidity_2m"), **cls}
    except Exception as e:
        raise HTTPException(502, str(e))

# ── Fires ──
@app.get("/fires/kenya")
async def fires():
    ck = "fires"
    c = cget(ck, 300)
    if c: return c
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
                    p = line.split(",")
                    if len(p) < len(hdr): continue
                    row = dict(zip(hdr, p))
                    try: la=float(row.get("latitude",0)); lo=float(row.get("longitude",0))
                    except: continue
                    if -5<=la<=6 and 33<=lo<=42:
                        fires.append({"lat":la,"lon":lo,
                                      "county":nearest_county(la,lo) or "Unknown",
                                      "brightness":row.get("bright_ti4") or row.get("brightness"),
                                      "confidence":row.get("confidence"),
                                      "date":row.get("acq_date"),"time":row.get("acq_time"),
                                      "satellite":row.get("satellite")})
            except Exception: continue
    result = {"ok":True,"count":len(fires),"fires":fires[:100]}
    cset(ck, result)
    return result

# ── Drought / NDVI ──
@app.get("/drought/{county}")
async def drought(county: str):
    p = DROUGHT.get(county.strip().lower())
    if p is None: return {"ok":False,"error":f"No data for '{county}'"}
    labels = {1:"Minimal",2:"Alert",3:"Alarm",4:"Emergency",5:"Famine"}
    return {"ok":True,"county":county.title(),"phase":p,"label":labels[p]}

@app.get("/ndvi/{county}")
async def ndvi(county: str):
    p = DROUGHT.get(county.strip().lower())
    if p is None: return {"ok":False,"error":f"No data for '{county}'"}
    vci = {1:55,2:40,3:25,4:15,5:5}.get(p,40)
    s = "Wet" if vci>=50 else "Normal" if vci>=35 else "Moderate drought" if vci>=21 else "Severe drought" if vci>=10 else "Extreme"
    return {"ok":True,"county":county.title(),"vci":vci,"status":s,
            "ndvi_explanation":"NDVI = vegetation greenness from satellite.",
            "vci_explanation":"VCI = today vs historical. Below 35 = early warning."}

# ── Flood ──
@app.get("/flood/{county}")
async def flood(county: str):
    k = county.strip().lower()
    coords = next((c for n, c in KENYA.items() if n.lower()==k), None)
    if not coords: return {"ok":False,"error":"Unknown county"}
    la, lo = coords
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get("https://flood-api.open-meteo.com/v1/flood",
                params={"latitude":la,"longitude":lo,
                        "daily":"river_discharge","forecast_days":7})
        vals = r.json().get("daily",{}).get("river_discharge",[]) or []
        peak = max(vals) if vals else 0
        if peak>500: level,color = "SEVERE","#ff2b2b"
        elif peak>200: level,color = "HIGH","#ff6b6b"
        elif peak>80: level,color = "MODERATE","#ffaa00"
        elif peak>30: level,color = "LOW","#00c87a"
        else: level,color = "MINIMAL","#00d9ff"
        return {"ok":True,"county":county.title(),
                "current_discharge": round(vals[0],1) if vals else 0,
                "peak_next_7d": round(peak,1),"level":level,"color":color}
    except Exception as e:
        return {"ok":False,"error":str(e)}

# ── Disasters ──
@app.get("/disasters")
async def disasters():
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get("https://www.gdacs.org/gdacsapi/api/Events/geteventlist/SEARCH",
                            params={"fromDate":"2024-01-01","toDate":"2026-12-31",
                                    "alertlevel":"Green;Orange;Red"})
        evs = r.json().get("features",[])
        kn = [{"type":e.get("properties",{}).get("eventtype",""),
               "alert_level":e.get("properties",{}).get("alertlevel",""),
               "title":e.get("properties",{}).get("name",""),
               "country":e.get("properties",{}).get("country",""),
               "date":e.get("properties",{}).get("todate","")}
              for e in evs if "kenya" in str(e.get("properties",{}).get("country","")).lower()]
        return {"ok":True,"count":len(kn),"events":kn[:20]}
    except Exception as e:
        return {"ok":False,"error":str(e)}

# ── NASA POWER ──
@app.get("/nasa/power/{county}")
async def nasa_power(county: str):
    k = county.strip().lower()
    coords = next((c for n, c in KENYA.items() if n.lower()==k), None)
    if not coords: return {"ok":False,"error":"Unknown county"}
    la, lo = coords
    end = datetime.now(timezone.utc) - timedelta(days=2)
    start = end - timedelta(days=30)
    try:
        async with httpx.AsyncClient(timeout=45) as c:
            r = await c.get("https://power.larc.nasa.gov/api/temporal/daily/point",
                params={"parameters":"PRECTOTCORR,T2M","community":"AG",
                        "longitude":lo,"latitude":la,
                        "start":start.strftime("%Y%m%d"),
                        "end":end.strftime("%Y%m%d"),"format":"JSON"})
        pr = r.json().get("properties",{}).get("parameter",{})
        rv = [v for v in pr.get("PRECTOTCORR",{}).values() if isinstance(v,(int,float))]
        tv = [v for v in pr.get("T2M",{}).values() if isinstance(v,(int,float))]
        return {"ok":True,"county":county.title(),
                "rain_30d_mm": round(sum(rv),1) if rv else 0,
                "avg_temp_c": round(sum(tv)/len(tv),1) if tv else 0,
                "days": len(rv)}
    except Exception as e:
        return {"ok":False,"error":str(e)}

# ── News ──
@app.get("/news/kenya")
async def news(limit: int = 10):
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get("https://www.standardmedia.co.ke/rss/headlines.php",
                            headers={"User-Agent":"Imhotep/1.0"})
        root = ET.fromstring(r.text)
        items = [{"title":(i.findtext("title") or "").strip(),
                  "link":(i.findtext("link") or "").strip()}
                 for i in root.findall(".//item")[:limit]]
        return {"ok":True,"count":len(items),"items":items}
    except Exception as e: raise HTTPException(502, str(e))

# ── Climate ──
@app.get("/climate/kenya")
async def climate():
    c = cget("climate", 3600)
    if c: return c
    lat, lon = -1.2921, 36.8219
    try:
        async with httpx.AsyncClient(timeout=90) as cc:
            r1 = await cc.get("https://archive-api.open-meteo.com/v1/archive",
                params={"latitude":lat,"longitude":lon,"start_date":"1985-01-01",
                        "end_date":"2014-12-31","daily":"temperature_2m_max",
                        "timezone":"Africa/Nairobi"})
            h = r1.json()
            r2 = await cc.get("https://climate-api.open-meteo.com/v1/climate",
                params={"latitude":lat,"longitude":lon,"start_date":"2025-01-01",
                        "end_date":"2050-12-31","models":"MRI_AGCM3_2_S",
                        "daily":"temperature_2m_max"})
            p = r2.json()
    except Exception as e: raise HTTPException(502, str(e))
    def avg(a):
        v = [x for x in (a or []) if isinstance(x,(int,float))]
        return round(sum(v)/len(v),2) if v else None
    ht = avg(h.get("daily",{}).get("temperature_2m_max",[]))
    pt = avg(p.get("daily",{}).get("temperature_2m_max",[]))
    dt = round(pt-ht,2) if (ht and pt) else None
    result = {"ok":True,"historical":{"period":"1985-2014","avg_tmax_c":ht},
              "projection":{"period":"2025-2050","avg_tmax_c":pt},
              "change":{"warming_c":dt,
                        "verdict":"Kenya is warming — prepare for heat stress" if dt and dt>0.5
                                  else "Stable" if dt and abs(dt)<=0.5 else "Cooling"}}
    cset("climate", result)
    return result

# ── NASA Earth ──
@app.get("/nasa/earth")
async def earth():
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get("https://epic.gsfc.nasa.gov/api/natural")
    items = r.json()
    if not items: raise HTTPException(404, "No images")
    l = items[0]
    dp = l["date"].split(" ")[0].replace("-","/")
    return {"ok":True,"date":l["date"],
            "image_url":f"https://epic.gsfc.nasa.gov/archive/natural/{dp}/png/{l['image']}.png"}

@app.get("/img/proxy")
async def proxy(url: str):
    ALLOWED = ("epic.gsfc.nasa.gov","cdn.star.nesdis.noaa.gov","gibs.earthdata.nasa.gov","images-api.nasa.gov")
    from urllib.parse import urlparse
    h = urlparse(url).netloc
    if not any(h.endswith(d) for d in ALLOWED): raise HTTPException(403,"Domain not allowed")
    async with httpx.AsyncClient(timeout=45) as c:
        r = await c.get(url, headers={"User-Agent":"Imhotep/1.0"})
    if r.status_code >= 300: raise HTTPException(r.status_code,"Upstream")
    return StreamingResponse(iter([r.content]),
        media_type=r.headers.get("content-type","image/jpeg"),
        headers={"Cache-Control":"public, max-age=600","Access-Control-Allow-Origin":"*"})

# ── Model ──
async def nvidia_call(msgs, temp):
    body = {"model":NVIDIA_MODEL,"messages":msgs,"temperature":temp,
            "top_p":0.95,"max_tokens":2048}
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(NVIDIA_URL,
            headers={"Authorization":f"Bearer {NVIDIA_API_KEY}",
                     "Content-Type":"application/json"}, json=body)
    if r.status_code >= 300: raise Exception(f"NVIDIA {r.status_code}")
    m = r.json()["choices"][0]["message"]
    content = (m.get("content") or "").strip()
    if not content: content = (m.get("reasoning_content") or "").strip()
    return strip_thinking(content) or "[No reply]"

async def call_model(msgs, temp):
    if NVIDIA_API_KEY:
        try: return await nvidia_call(msgs, temp)
        except Exception as e: print(f"nvidia: {e}")
    return "I'm Imhotep, built by Samuel Kiragu in Mukuyu, Murang'a, Kenya."

# ── Chat ──
@app.get("/agents")
async def agents(): return {"ok":True,"agents":list(PERSONAS.keys())}

@app.post("/auth/signup")
async def signup(b: SignupIn):
    u, _ = mk_user(b.email.lower(), b.name, "free")
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
    except Exception as e: raise HTTPException(400, f"Google: {e}")
    e = info.get("email")
    if not e: raise HTTPException(400, "Invalid token")
    u, _ = mk_user(e.lower(), info.get("name","Google User"), "free")
    return {"ok":True,"user_id":u,"token":issue(u),"user":users_db[u]}

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
        raise HTTPException(402, detail={"error":"No credits left.",
            "used":c["used"],"limit":c["limit"],
            "upgrade_url":"https://kiragu2004.github.io/imhotep-site/pricing.html"})
    a = b.agent or "generalist"
    if is_sheng(p): a = "sheng"
    t = max(0.0, min(2.0, b.temperature))
    raw = scrub(await call_model([{"role":"system","content":sys_prompt(a,p)},
                                  {"role":"user","content":p}], t))
    raw = strip_thinking(raw)
    if u in chat_db:
        chat_db[u].append({"prompt":p,"reply":raw,"agent":a,
                           "timestamp":datetime.now(timezone.utc).isoformat()})
    return {"ok":True,"reply":raw,"agent":a,"credits":creds(u)}

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
    pro = scrub(await call_model([{"role":"system","content":sys_prompt(ag[0],p)+" Argue FOR."},
        {"role":"user","content":f"{p} — 150 words FOR."}], 0.7))
    con = scrub(await call_model([{"role":"system","content":sys_prompt(ag[1],p)+" Argue AGAINST."},
        {"role":"user","content":f"{p} — 150 words AGAINST."}], 0.7))
    j = scrub(await call_model([{"role":"system","content":"You are a judge."},
        {"role":"user","content":f"Judge: {p}\nPRO: {pro[:300]}\nCON: {con[:300]}"}], 0.4))
    return {"ok":True,"proposition":p,"proponent":pro,"opponent":con,"judge":j}

@app.get("/pricing")
async def pricing(): return {"ok":True,"currency":"KES","tiers":PRICES}

@app.post("/checkout")
async def checkout(b: CheckoutIn):
    t = TALIAS.get(b.tier, b.tier)
    if t not in PRICES: raise HTTPException(400,"Unknown tier")
    info = PRICES[t]
    ref = f"IMH-{t}-{int(time.time())}-{secrets.token_hex(3)}"
    url = None
    if NESTLINK_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post("https://api.nestlink.co.ke/v1/checkout",
                    json={"amount":info["amount"],"currency":"KES","email":b.email,
                          "narrative":info["label"],"reference":ref},
                    headers={"Authorization":f"Bearer {NESTLINK_API_KEY}"})
            if r.status_code < 300:
                d = r.json()
                url = d.get("checkout_url") or d.get("url")
        except Exception: pass
    if not url:
        url = f"https://me.nestlink.co.ke/Imhotepagenticai?email={b.email}&reference={ref}"
    return {"ok":True,"tier":t,"amount":info["amount"],"payment_url":url,"reference":ref}

@app.get("/health")
async def health():
    return {"ok":True,"version":"13.0.0",
            "model_configured":bool(NVIDIA_API_KEY),
            "nestlink_configured":bool(NESTLINK_API_KEY),
            "counties":len(KENYA),
            "cache_keys":len(_C)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
