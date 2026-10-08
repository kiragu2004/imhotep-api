
import os, re, time, secrets, hmac, hashlib, json, pathlib
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
COLAB_MODEL_URL  = os.getenv("COLAB_MODEL_URL", "").rstrip("/")
NESTLINK_API_KEY = os.getenv("NESTLINK_API_KEY", "")
NESTLINK_WEBHOOK_SECRET = os.getenv("NESTLINK_WEBHOOK_SECRET", "imhotep-nestlink-2026")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
CORS_ORIGINS     = [o.strip() for o in os.getenv("CORS_ORIGINS","*").split(",") if o.strip()]

app = FastAPI(title="Imhotep", version="5.0.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

LOG_DIR = pathlib.Path("/tmp/imhotep"); LOG_DIR.mkdir(exist_ok=True)
WEBHOOK_LOG = LOG_DIR / "webhooks.jsonl"

# ── Persona leak scrubber ──
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
# SHENG DICTIONARY — real words the model can use
# ═══════════════════════════════════════════════════════════════
SHENG_DICT = """
SHENG VOCABULARY — use these accurately:
PEOPLE: chali/msee=m guy, demu/msupa/chica=girl, beshte/arif=friend,
        mbogi=group, mathee=mother, buda/mzae=father, karao/makarao=police
MONEY: doo/chapaa/mula=money, ashu/kinde=10/-, blue/mbao=20/-, soo=100/-,
       rwabe=200/-, punch=500/-, thao=1000/-
ACTIONS: bonga=talk, dishi=eat, doro=sleep, banjuka=dance, sota=broke,
         cheza chini=chill/low profile, rada=pay attention
GREETINGS: sasa=hey, niaje=what's up, mambo=how are things, poa/fiti/safi=cool,
           sema=speak, radudua=what's up (slang)
PLACES: mtaa=neighborhood, hao=house, ocha=upcountry, mat/nganya=matatu,
        dinga/moti=car, Kanairo=Nairobi
OTHER: noma=trouble/impressive, mbuyu=old man, masa=friend, keja=house,
       mzito=heavy/serious, bazenga=respected man
"""

SHENG_GREETINGS = ("sasa","niaje","mambo","marada","radudua","sema","bonga","uko poa","uko aje")

def is_sheng_query(prompt: str) -> bool:
    p = prompt.lower()
    if any(g in p for g in SHENG_GREETINGS): return True
    # Also check for common sheng words
    sheng_words = ("chapaa","doo","mula","beshte","mbogi","demu","chali","msee",
                   "sota","dishi","doro","banjuka","ocha","mtaa","mathee","buda",
                   "thao","soo","rwabe","punch")
    if any(w in p for w in sheng_words): return True
    if "sheng" in p or "kiswahili" in p or "swahili" in p: return True
    return False

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
}

LEARN_SIGNALS = (
    "teach","explain","how does","how do","why does","why do","tutorial",
    "step by step","help me learn","walk me through","show me how","example",
    "code","function","class","algorithm","debug","error","python",
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
        return (
            identity + "\n" + SHENG_DICT + "\n"
            "SHENG MODE — CRITICAL RULES:\n"
            "1. Reply ENTIRELY in Sheng mixed with Kiswahili and English.\n"
            "2. Use the real Sheng vocabulary above. Do not invent words.\n"
            "3. Greet with: sasa, niaje, mambo, radudua, sema.\n"
            "4. Example reply style: 'Sasa msee! Niaje? Mi ni fiti, poa tu. "
            "Wewe uko aje leo?'\n"
            "5. If the user writes in English, still reply in Sheng with English "
            "explanation only if they look confused.\n"
            "6. Never break character. Never say you are an AI.\n"
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
# WEATHER — Open-Meteo (no key) + OpenWeather fallback if key set
# ═══════════════════════════════════════════════════════════════
def classify_wx(temp, code):
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

async def _geocode_city(city: str):
    """Resolve a Kenyan city name to lat/lon via Open-Meteo geocoder."""
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get("https://geocoding-api.open-meteo.com/v1/search",
                        params={"name": city, "count": 1, "language": "en"})
    g = r.json()
    if not g.get("results"):
        return None, None, None
    hit = g["results"][0]
    return (hit.get("latitude"), hit.get("longitude"),
            hit.get("name") or city)

@app.post("/weather")
async def weather_ep(body: dict):
    """Open-Meteo — no API key needed. Accepts {lat,lon} OR {city}."""
    lat = body.get("lat"); lon = body.get("lon"); city = body.get("city")

    # If city given and coords missing, geocode
    if (lat is None or lon is None) and city:
        la, lo, resolved = await _geocode_city(city)
        if la is None:
            raise HTTPException(404, f"City '{city}' not found")
        lat, lon, city = la, lo, resolved

    if lat is None or lon is None:
        raise HTTPException(400, "Provide lat+lon or city")

    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "timezone": "auto",
                })
        data = r.json()
    except Exception as e:
        raise HTTPException(502, f"Open-Meteo error: {e}")

    cur = data.get("current") or {}
    temp_raw = cur.get("temperature_2m")
    if temp_raw is None:
        raise HTTPException(502, "Weather provider returned no data")

    temp = round(temp_raw)
    code = cur.get("weather_code", 0)
    cls = classify_wx(temp, code)

    return {
        "ok": True,
        "temp": temp,
        "feels_like": round(cur.get("apparent_temperature", temp)),
        "condition": cls["condition_label"],
        "description": cls["condition_label"],
        "humidity": cur.get("relative_humidity_2m"),
        "wind": cur.get("wind_speed_10m"),
        "city": city or f"{lat:.2f},{lon:.2f}",
        "country": "",
        **cls,
        "source": "open-meteo.com",
    }

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
                    json={"amount": t["amount"], "currency": "KES", "email": b.email,
                          "narrative": t["label"], "reference": ref,
                          "redirect_url": success_url, "callback_url": webhook_url},
                    headers={"Authorization": f"Bearer {NESTLINK_API_KEY}",
                             "Content-Type": "application/json",
                             "Accept": "application/json"})
            if r.status_code < 300:
                d = r.json()
                payment_url = (d.get("checkout_url") or d.get("url") or d.get("payment_url"))
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
    return {"ok":True,"version":"5.0.0",
            "model_configured":bool(NVIDIA_API_KEY),
            "colab_fallback_configured":bool(COLAB_MODEL_URL),
            "weather_provider":"open-meteo (no key)",
            "nestlink_configured":bool(NESTLINK_API_KEY),
            "google_configured":bool(GOOGLE_CLIENT_ID),
            "kenya_counties":len(KENYA_COUNTIES),
            "sheng_agent":True, "nasa_fires":True, "kenya_news":True,
            "drought_data":True, "adaptive_length":True,
            "founder":"Samuel Kiragu"}
