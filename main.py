@app.get("/drought/{county}")
async def drought_status(county: str):
    """Fetch live NDMA drought phase for a Kenyan county."""
    # This is the public API endpoint for the wapimaji-mcp server
    api_url = f"https://api.wapimaji.dev/drought/{county.lower()}"
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(api_url)
        if r.status_code != 200:
            return {"ok": False, "error": f"No drought data for '{county}'"}
        data = r.json()
        # The exact response format may vary; assume a JSON like:
        # {"phase": 3, "label": "Alarm", "rainfall_deficit": 30}
        return {"ok": True, "county": county.title(), **data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


import os, re, time, secrets, hmac, hashlib, json, pathlib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import httpx, jwt
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr

# ── CONFIG ──
SECRET_KEY       = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
NVIDIA_API_KEY   = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL     = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
NVIDIA_URL       = "https://integrate.api.nvidia.com/v1/chat/completions"
COLAB_MODEL_URL  = os.getenv("COLAB_MODEL_URL", "").rstrip("/")
NESTLINK_API_KEY = os.getenv("NESTLINK_API_KEY", "")
NESTLINK_WEBHOOK_SECRET = os.getenv("NESTLINK_WEBHOOK_SECRET", "imhotep-nestlink-2026")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
CORS_ORIGINS     = [o.strip() for o in os.getenv("CORS_ORIGINS","*").split(",") if o.strip()]

app = FastAPI(title="Imhotep", version="6.0.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ── STATE ──
users_db, chat_db = {}, {}
LOG_DIR = pathlib.Path("/tmp/imhotep"); LOG_DIR.mkdir(exist_ok=True)
WEBHOOK_LOG = LOG_DIR / "webhooks.jsonl"

# ── PERSONA SCRUBBER ──
_LEAK = [(r"\bNVIDIA\s+NIM\b","Imhotep"),(r"\bNVIDIA\b","Imhotep"),
         (r"\bNemotron[\w\-\.]*\b","Imhotep"),(r"\bNIM\b","Imhotep"),
         (r"\bQwen[\w\-\.]*\b","Imhotep"),(r"\bDeepSeek[\w\-]*\b","Imhotep"),
         (r"\bMistral[\w\-]*\b","Imhotep"),(r"\bLlama[\s\-]?[\d\.\w]*\b","Imhotep")]
def scrub(t):
    if not t: return t
    for p, r in _LEAK: t = re.sub(p, r, t, flags=re.IGNORECASE)
    return t

# ── SHENG DICTIONARY ──
SHENG_DICT = """
SHENG VOCABULARY:
PEOPLE: chali/msee=guy, demu/msupa=girl, beshte/arif=friend, mbogi=group
MONEY: doo/chapaa=money, ashu=10/-, blue=20/-, soo=100/-, thao=1000/-
ACTIONS: bonga=talk, dishi=eat, doro=sleep, sota=broke, rada=watch
GREETINGS: sasa=hey, niaje=what's up, mambo=how are things, poa=cool,
           fiti=good, safi=fine, radudua=what's up (slang)
PLACES: mtaa=neighborhood, ocha=upcountry, mat/nganya=matatu, Kanairo=Nairobi
"""
SHENG_GREETINGS = ("sasa","niaje","mambo","marada","radudua","sema","bonga","uko poa")
def is_sheng_query(prompt: str) -> bool:
    p = prompt.lower()
    return any(g in p for g in SHENG_GREETINGS) or "sheng" in p

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
    )
    if agent == "sheng":
        return identity + "\n" + SHENG_DICT + "\nSHENG MODE: Reply in Sheng mixed with Kiswahili and English. Never break character."
    if learning:
        style = "RESPONSE STYLE — structured: 1. Summary. 2. Step-by-step bullets. 3. Example. 4. Follow-up. 5. Length: 200-600 words."
    else:
        style = "RESPONSE STYLE — SHORT: 1. Answer in 1-3 sentences. 2. Warm, direct. 3. Offer: 'Want me to go deeper?'"
    return identity + "\n" + style

# ── AUTH ──
TIER_CREDITS = {"guest":3,"free":3,"starter_10":3,"starter_20":20,"starter_50":100,"weekly":10000,"monthly":10000,"yearly":10000}
class ChatIn(BaseModel): prompt: str; agent: Optional[str] = "generalist"; user_id: Optional[str] = None; temperature: float = 0.7; location: Optional[dict] = None
class SignupIn(BaseModel): email: EmailStr; name: str
class CheckoutIn(BaseModel): tier: str; email: EmailStr
class GoogleIn(BaseModel): id_token: str

def issue(uid, days=30):
    return jwt.encode({"user_id":uid,"exp":datetime.now(timezone.utc)+timedelta(days=days)}, SECRET_KEY, algorithm="HS256")
def mk_user(email, name, tier="free"):
    for u, v in users_db.items():
        if v["email"] == email: return (u, False)
    u = f"user_{int(time.time()*1000)}"
    users_db[u] = {"email":email,"name":name,"tier":tier,"credits":TIER_CREDITS.get(tier,3),"credits_used":0,"created_at":datetime.now(timezone.utc).isoformat()}
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

@app.post("/auth/signup")
async def signup(b: SignupIn):
    u, new = mk_user(b.email.lower(), b.name, "free")
    return {"ok":True,"user_id":u,"token":issue(u),"user":users_db[u],"new":new}
@app.get("/auth/guest")
async def guest():
    u = f"guest_{int(time.time()*1000)}"
    users_db[u] = {"email":f"{u}@g.local","name":"Guest","tier":"guest","credits":3,"credits_used":0,"created_at":datetime.now(timezone.utc).isoformat()}
    chat_db[u] = []
    return {"ok":True,"user_id":u,"token":issue(u,1),"user":users_db[u]}
@app.post("/auth/google")
async def google(b: GoogleIn):
    r = httpx.post("https://oauth2.googleapis.com/tokeninfo", params={"id_token": b.id_token}); info = r.json()
    email = info.get("email")
    if not email: raise HTTPException(400, "Invalid token")
    u, _ = mk_user(email.lower(), info.get("name","Google User"), "free")
    return {"ok":True,"user_id":u,"token":issue(u),"user":users_db[u]}

# ── WEATHER (Open-Meteo, no key) ──
def classify_wx(temp, code):
    try: c = int(code)
    except: c = 0
    if c in (0,1): e,l="☀️","Clear"
    elif c == 2: e,l="🌤️","Partly cloudy"
    elif c == 3: e,l="☁️","Cloudy"
    elif c in (45,48): e,l="🌫️","Foggy"
    elif c in (61,63,65): e,l="🌧️","Rain"
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
        r = await c.get("https://geocoding-api.open-meteo.com/v1/search", params={"name":city,"count":1,"language":"en"})
    g = r.json()
    if not g.get("results"): return None,None,None
    hit = g["results"][0]
    return hit.get("latitude"), hit.get("longitude"), hit.get("name") or city

@app.post("/weather")
async def weather_ep(body: dict):
    lat, lon, city = body.get("lat"), body.get("lon"), body.get("city")
    if (lat is None or lon is None) and city:
        la, lo, resolved = await _geocode_city(city)
        if la is None: raise HTTPException(404, f"City '{city}' not found")
        lat, lon, city = la, lo, resolved
    if lat is None or lon is None: raise HTTPException(400, "Provide lat+lon or city")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get("https://api.open-meteo.com/v1/forecast",
                        params={"latitude":lat,"longitude":lon,"current":"temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m","timezone":"auto"})
    d = r.json(); cur = d.get("current") or {}
    temp = round(cur.get("temperature_2m",0)); code = cur.get("weather_code",0)
    cls = classify_wx(temp, code)
    return {"ok":True,"temp":temp,"feels_like":round(cur.get("apparent_temperature",temp)),"condition":cls["condition_label"],"humidity":cur.get("relative_humidity_2m"),"wind":cur.get("wind_speed_10m"),"city":city or f"{lat:.2f},{lon:.2f}","country":"",**cls,"source":"open-meteo.com"}

# ── KENYA FIRES (NASA FIRMS, no key) ──
@app.get("/fires/kenya")
async def fires_kenya():
    urls = [
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_24h.csv",
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_24h.csv"
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
                        fires.append({"lat":la,"lon":lo,"brightness":row.get("bright_ti4") or row.get("brightness"),"date":row.get("acq_date"),"time":row.get("acq_time"),"satellite":row.get("satellite")})
            except: continue
    return {"ok":True,"count":len(fires),"fires":fires[:50],"source":"NASA FIRMS"}

# ── KENYA NEWS (Standard RSS, no key) ──
@app.get("/news/kenya")
async def news_kenya(limit: int = 10):
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get("https://www.standardmedia.co.ke/rss/headlines.php", headers={"User-Agent":"Imhotep/1.0"})
        root = ET.fromstring(r.text)
        items = []
        for item in root.findall(".//item")[:limit]:
            items.append({"title":(item.findtext("title") or "").strip(),"link":(item.findtext("link") or "").strip(),"description":(item.findtext("description") or "").strip()[:250]})
        return {"ok":True,"count":len(items),"items":items,"source":"standardmedia.co.ke/rss"}
    except Exception as e: raise HTTPException(502, str(e))

# ── KENYA COUNTIES ──
KENYA_COUNTIES = {
    "mombasa":{"code":"001","capital":"Mombasa"},"kwale":{"code":"002","capital":"Kwale"},
    "kilifi":{"code":"003","capital":"Kilifi"},"lamu":{"code":"005","capital":"Lamu"},
    "garissa":{"code":"007","capital":"Garissa"},"wajir":{"code":"008","capital":"Wajir"},
    "mandera":{"code":"009","capital":"Mandera"},"marsabit":{"code":"010","capital":"Marsabit"},
    "isiolo":{"code":"011","capital":"Isiolo"},"meru":{"code":"012","capital":"Meru"},
    "embu":{"code":"014","capital":"Embu"},"kitui":{"code":"015","capital":"Kitui"},
    "machakos":{"code":"016","capital":"Machakos"},"makueni":{"code":"017","capital":"Wote"},
    "nyeri":{"code":"019","capital":"Nyeri"},"kirinyaga":{"code":"020","capital":"Kerugoya"},
    "murang'a":{"code":"021","capital":"Murang'a"},"kiambu":{"code":"022","capital":"Kiambu"},
    "turkana":{"code":"023","capital":"Lodwar"},"west pokot":{"code":"024","capital":"Kapenguria"},
    "samburu":{"code":"025","capital":"Maralal"},"trans nzoia":{"code":"026","capital":"Kitale"},
    "uasin gishu":{"code":"027","capital":"Eldoret"},"nakuru":{"code":"032","capital":"Nakuru"},
    "narok":{"code":"033","capital":"Narok"},"kajiado":{"code":"034","capital":"Kajiado"},
    "kericho":{"code":"035","capital":"Kericho"},"kakamega":{"code":"037","capital":"Kakamega"},
    "bungoma":{"code":"039","capital":"Bungoma"},"busia":{"code":"040","capital":"Busia"},
    "siaya":{"code":"041","capital":"Siaya"},"kisumu":{"code":"042","capital":"Kisumu"},
    "homa bay":{"code":"043","capital":"Homa Bay"},"migori":{"code":"044","capital":"Migori"},
    "kisii":{"code":"045","capital":"Kisii"},"nairobi":{"code":"047","capital":"Nairobi"},
}

@app.get("/kenya/counties")
async def kenya_counties():
    return {"ok":True,"count":len(KENYA_COUNTIES),"counties":[{"name":k.title(),**v} for k,v in KENYA_COUNTIES.items()]}

# ── SATELLITES ──
@app.get("/nasa/earth")
async def nasa_earth():
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get("https://epic.gsfc.nasa.gov/api/natural")
    items = r.json()
    if not items: raise HTTPException(404, "No images")
    latest = items[0]
    date_path = latest["date"].split(" ")[0].replace("-","/")
    img = f"https://epic.gsfc.nasa.gov/archive/natural/{date_path}/png/{latest['image']}.png"
    return {"ok":True,"date":latest["date"],"image_url":img,"lat":latest["centroid_coordinates"]["lat"],"lon":latest["centroid_coordinates"]["lon"]}

@app.get("/img/proxy")
async def img_proxy(url: str):
    ALLOWED = ("epic.gsfc.nasa.gov","cdn.star.nesdis.noaa.gov","gibs.earthdata.nasa.gov","images-api.nasa.gov")
    from urllib.parse import urlparse
    host = urlparse(url).netloc
    if not any(host.endswith(d) for d in ALLOWED): raise HTTPException(403,"Domain not allowed")
    async with httpx.AsyncClient(timeout=45) as c:
        r = await c.get(url, headers={"User-Agent":"Imhotep/1.0"})
    if r.status_code >= 300: raise HTTPException(r.status_code,"Upstream failed")
    return StreamingResponse(iter([r.content]), media_type=r.headers.get("content-type","image/jpeg"), headers={"Cache-Control":"public, max-age=600","Access-Control-Allow-Origin":"*"})

# ── MODEL CALL ──
async def _nvidia(msgs, temp):
    body = {"model":NVIDIA_MODEL,"messages":msgs,"temperature":temp,"top_p":0.95,"max_tokens":2048}
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(NVIDIA_URL, headers={"Authorization":f"Bearer {NVIDIA_API_KEY}","Content-Type":"application/json","Accept":"application/json"}, json=body)
    if r.status_code >= 300: raise Exception(f"NVIDIA {r.status_code}")
    msg = r.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    content = re.sub(r"<think[^>]*>.*?</think\s*>","",content,flags=re.DOTALL|re.IGNORECASE).strip()
    return content or msg.get("reasoning_content") or "[No reply]"

async def call_model(msgs, temp, prompt="", agent="generalist"):
    if NVIDIA_API_KEY:
        try: return await _nvidia(msgs, temp)
        except Exception as e: print(f"nvidia: {e}")
    return "I'm Imhotep, built by Samuel Kiragu in Mukuyu, Murang'a, Kenya. Try again."

@app.post("/chat")
async def chat(b: ChatIn):
    p = b.prompt.strip()
    if not p: raise HTTPException(400,"Empty")
    u = b.user_id
    if not u: raise HTTPException(402, detail={"error":"Sign in required","upgrade_url":"https://kiragu2004.github.io/imhotep-site/auth.html"})
    if u not in users_db: raise HTTPException(402, detail={"error":"Session expired","upgrade_url":"https://kiragu2004.github.io/imhotep-site/auth.html"})
    if not debit(u):
        c = creds(u); raise HTTPException(402, detail={"error":"No credits left.","used":c["used"],"limit":c["limit"],"upgrade_url":"https://kiragu2004.github.io/imhotep-site/pricing.html"})
    agent = b.agent or "generalist"
    if is_sheng_query(p): agent = "sheng"
    t = max(0.0, min(2.0, b.temperature))
    s = sys_prompt(agent, prompt=p)
    msgs = [{"role":"system","content":s},{"role":"user","content":p}]
    raw = scrub(await call_model(msgs, t, prompt=p, agent=agent))
    if u in chat_db: chat_db[u].append({"prompt":p,"reply":raw,"agent":agent,"timestamp":datetime.now(timezone.utc).isoformat()})
    return {"ok":True,"reply":raw,"agent":agent,"credits":creds(u)}

@app.get("/health")
async def health():
    return {"ok":True,"version":"6.0.0","model_configured":bool(NVIDIA_API_KEY),"founder":"Samuel Kiragu"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
