
import os, re, time, secrets, hmac, hashlib, json, pathlib
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import httpx, jwt
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(title="Imhotep", version="3.2.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ── Log file for webhook debugging (on Render /tmp) ──
LOG_DIR = pathlib.Path("/tmp/imhotep")
LOG_DIR.mkdir(exist_ok=True)
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
}

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

# ── NEW PRICING — 6 tiers ──
# credits granted per tier when paid
TIER_CREDITS = {
    "guest":       3,
    "free":        3,
    "starter_10":  3,
    "starter_20":  20,
    "starter_50":  100,
    "weekly":      10000,
    "monthly":     10000,
    "yearly":      10000,
}

# price displayed + amount charged (KES)
PRICES = {
    "starter_10": {"amount": 10,   "label": "Starter — 3 chats"},
    "starter_20": {"amount": 20,   "label": "Small pack — 20 chats"},
    "starter_50": {"amount": 50,   "label": "Value pack — 100 chats"},
    "weekly":     {"amount": 99,   "label": "Weekly Unlimited — 7 days"},
    "monthly":    {"amount": 299,  "label": "Monthly Unlimited — 30 days"},
    "yearly":     {"amount": 5000, "label": "Yearly Unlimited — 365 days"},
}

# legacy names → new names (so old references don't break)
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
    if GOOGLE_CLIENT_ID and info.get("aud") and info.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(400, "Token not for this app")
    u, _ = mk_user(email.lower(), info.get("name","Google User"), "free")
    return {"ok":True,"user_id":u,"token":issue(u),"user":users_db[u]}

def classify_wx(temp, cond):
    cond = (cond or "").lower()
    if "rain" in cond: e, l = "🌧️", "Rainy"
    elif "thunder" in cond: e, l = "⛈️", "Stormy"
    elif "snow" in cond: e, l = "❄️", "Snowy"
    elif "cloud" in cond: e, l = "☁️", "Cloudy"
    elif "clear" in cond or "sun" in cond: e, l = "☀️", "Sunny"
    elif "mist" in cond or "fog" in cond: e, l = "🌫️", "Foggy"
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
    if not OPENWEATHER_KEY: raise HTTPException(503, "Weather not configured")
    lat, lon, city = body.get("lat"), body.get("lon"), body.get("city")
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            params = {"lat":lat,"lon":lon} if lat is not None and lon is not None else {"q":city}
            params.update({"appid":OPENWEATHER_KEY,"units":"metric"})
            r = await c.get("https://api.openweathermap.org/data/2.5/weather", params=params)
        d = r.json()
        if "main" not in d: raise HTTPException(404, "Not found")
        cls = classify_wx(d["main"]["temp"], d["weather"][0]["main"])
        return {"ok":True,"temp":round(d["main"]["temp"]),
                "feels_like":round(d["main"].get("feels_like", d["main"]["temp"])),
                "condition":d["weather"][0]["main"],
                "description":d["weather"][0]["description"],
                "humidity":d["main"].get("humidity"),
                "city":d.get("name",""), "country":d.get("sys",{}).get("country",""),
                **cls}
    except HTTPException: raise
    except Exception as e: raise HTTPException(502, str(e))

# ── Kenyan counties ──
KENYA_COUNTIES = {
    "mombasa":{"code":"001","capital":"Mombasa","region":"Coast"},
    "kwale":{"code":"002","capital":"Kwale","region":"Coast"},
    "kilifi":{"code":"003","capital":"Kilifi","region":"Coast"},
    "tana river":{"code":"004","capital":"Hola","region":"Coast"},
    "lamu":{"code":"005","capital":"Lamu","region":"Coast"},
    "taita taveta":{"code":"006","capital":"Wundanyi","region":"Coast"},
    "garissa":{"code":"007","capital":"Garissa","region":"North Eastern"},
    "wajir":{"code":"008","capital":"Wajir","region":"North Eastern"},
    "mandera":{"code":"009","capital":"Mandera","region":"North Eastern"},
    "marsabit":{"code":"010","capital":"Marsabit","region":"Eastern"},
    "isiolo":{"code":"011","capital":"Isiolo","region":"Eastern"},
    "meru":{"code":"012","capital":"Meru","region":"Eastern"},
    "tharaka-nithi":{"code":"013","capital":"Chuka","region":"Eastern"},
    "embu":{"code":"014","capital":"Embu","region":"Eastern"},
    "kitui":{"code":"015","capital":"Kitui","region":"Eastern"},
    "machakos":{"code":"016","capital":"Machakos","region":"Eastern"},
    "makueni":{"code":"017","capital":"Wote","region":"Eastern"},
    "nyandarua":{"code":"018","capital":"Ol Kalou","region":"Central"},
    "nyeri":{"code":"019","capital":"Nyeri","region":"Central"},
    "kirinyaga":{"code":"020","capital":"Kerugoya","region":"Central"},
    "murang'a":{"code":"021","capital":"Murang'a","region":"Central"},
    "kiambu":{"code":"022","capital":"Kiambu","region":"Central"},
    "turkana":{"code":"023","capital":"Lodwar","region":"Rift Valley"},
    "west pokot":{"code":"024","capital":"Kapenguria","region":"Rift Valley"},
    "samburu":{"code":"025","capital":"Maralal","region":"Rift Valley"},
    "trans nzoia":{"code":"026","capital":"Kitale","region":"Rift Valley"},
    "uasin gishu":{"code":"027","capital":"Eldoret","region":"Rift Valley"},
    "elgeyo-marakwet":{"code":"028","capital":"Iten","region":"Rift Valley"},
    "nandi":{"code":"029","capital":"Kapsabet","region":"Rift Valley"},
    "baringo":{"code":"030","capital":"Kabarnet","region":"Rift Valley"},
    "laikipia":{"code":"031","capital":"Nanyuki","region":"Rift Valley"},
    "nakuru":{"code":"032","capital":"Nakuru","region":"Rift Valley"},
    "narok":{"code":"033","capital":"Narok","region":"Rift Valley"},
    "kajiado":{"code":"034","capital":"Kajiado","region":"Rift Valley"},
    "kericho":{"code":"035","capital":"Kericho","region":"Rift Valley"},
    "bomet":{"code":"036","capital":"Bomet","region":"Rift Valley"},
    "kakamega":{"code":"037","capital":"Kakamega","region":"Western"},
    "vihiga":{"code":"038","capital":"Vihiga","region":"Western"},
    "bungoma":{"code":"039","capital":"Bungoma","region":"Western"},
    "busia":{"code":"040","capital":"Busia","region":"Western"},
    "siaya":{"code":"041","capital":"Siaya","region":"Nyanza"},
    "kisumu":{"code":"042","capital":"Kisumu","region":"Nyanza"},
    "homa bay":{"code":"043","capital":"Homa Bay","region":"Nyanza"},
    "migori":{"code":"044","capital":"Migori","region":"Nyanza"},
    "kisii":{"code":"045","capital":"Kisii","region":"Nyanza"},
    "nyamira":{"code":"046","capital":"Nyamira","region":"Nyanza"},
    "nairobi":{"code":"047","capital":"Nairobi","region":"Nairobi"},
}

@app.get("/kenya/county")
async def kenya_county(name: str):
    key = name.strip().lower()
    c = KENYA_COUNTIES.get(key)
    if not c:
        return {"ok":False,"error":f"County '{name}' not found",
                "hint":"Try one of the 47 counties, e.g. Nairobi, Mombasa, Murang'a"}
    return {"ok":True,"name":name.title(),**c,
            "source":"https://en.wikipedia.org/wiki/Counties_of_Kenya"}

@app.get("/kenya/counties")
async def kenya_counties():
    return {"ok":True,"count":len(KENYA_COUNTIES),
            "counties":[{"name":k.title(),**v} for k,v in KENYA_COUNTIES.items()]}

async def _nvidia(msgs, temp):
    body = {"model": NVIDIA_MODEL, "messages": msgs, "temperature": temp,
            "top_p": 0.95, "max_tokens": 2048}
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(NVIDIA_URL,
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}",
                     "Content-Type": "application/json",
                     "Accept": "application/json"}, json=body)
    if r.status_code >= 300:
        err = r.text[:400]
        print(f"NVIDIA {r.status_code}: {err}")
        raise Exception(f"NVIDIA {r.status_code}: {err}")
    j = r.json()
    msg = j["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    content = re.sub(r"<think[^>]*>.*?</think\s*>", "", content,
                     flags=re.DOTALL | re.IGNORECASE).strip()
    if "thinking process" in content.lower() or "analyze user" in content.lower():
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
    t = max(0.0, min(2.0, b.temperature))
    agent = b.agent or "generalist"
    s = sys_prompt(agent, prompt=p)
    w = None
    if b.location and b.location.get("lat") is not None and OPENWEATHER_KEY:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("https://api.openweathermap.org/data/2.5/weather",
                    params={"lat":b.location["lat"],"lon":b.location["lon"],
                            "appid":OPENWEATHER_KEY,"units":"metric"})
            d = r.json()
            if "main" in d:
                cls = classify_wx(d["main"]["temp"], d["weather"][0]["main"])
                w = {"temp":round(d["main"]["temp"]),"city":d.get("name",""),
                     "description":d["weather"][0]["description"], **cls}
                s += f"\nUser is in {w['city']} ({w['temp']}°C, {w['description']}, {w['feel']})."
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
    # Accept legacy tier names
    tier = TIER_ALIAS.get(b.tier, b.tier)
    if tier not in PRICES: raise HTTPException(400, f"Unknown tier '{b.tier}'")
    t = PRICES[tier]
    ref = f"IMH-{tier}-{int(time.time())}-{secrets.token_hex(3)}"
    success_url = "https://kiragu2004.github.io/imhotep-site/chat.html?paid=1"
    webhook_url = "https://imhotep-agentic-ai.onrender.com/nestlink/webhook"

    payment_url = None
    if NESTLINK_API_KEY:
        # Try multiple possible NestLink endpoint shapes
        attempts = [
            ("https://api.nestlink.co.ke/v1/checkout",
             {"amount": t["amount"], "currency": "KES", "email": b.email,
              "narrative": t["label"], "reference": ref,
              "redirect_url": success_url, "callback_url": webhook_url}),
            ("https://api.nestlink.co.ke/v1/payments/checkout",
             {"amount": t["amount"], "currency": "KES", "email": b.email,
              "description": t["label"], "reference": ref,
              "redirect_url": success_url, "callback_url": webhook_url}),
            ("https://api.nestlink.co.ke/api/v1/checkout",
             {"amount": t["amount"], "currency": "KES", "email": b.email,
              "narrative": t["label"], "reference": ref}),
        ]
        for url, payload in attempts:
            try:
                async with httpx.AsyncClient(timeout=30) as c:
                    r = await c.post(url, json=payload,
                        headers={"Authorization": f"Bearer {NESTLINK_API_KEY}",
                                 "Content-Type": "application/json",
                                 "Accept": "application/json"})
                print(f"nestlink {url} → {r.status_code}: {r.text[:200]}")
                if r.status_code < 300:
                    d = r.json()
                    payment_url = (d.get("checkout_url") or d.get("url") or
                                   d.get("payment_url") or d.get("link"))
                    if payment_url:
                        break
            except Exception as e:
                print(f"nestlink {url} err: {e}")

    if not payment_url:
        # Fallback: manual payment page with reference embedded
        payment_url = ("https://me.nestlink.co.ke/Imhotepagenticai"
                       f"?email={b.email}&reference={ref}&amount={t['amount']}")

    return {"ok":True,"tier":tier,"amount":t["amount"],
            "payment_url":payment_url,"reference":ref,
            "label": t["label"]}

@app.post("/nestlink/webhook")
async def nestlink_webhook(request: Request):
    """NestLink calls this on payment. Upgrades user immediately."""
    raw = await request.body()
    headers = dict(request.headers)
    body_str = raw.decode("utf-8", errors="ignore")

    # Log everything for debugging
    entry = {"ts": time.time(), "headers": headers, "body": body_str[:2000]}
    try:
        with open(WEBHOOK_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"log write failed: {e}")

    print(f"WEBHOOK HIT: {body_str[:500]}")

    # Signature check (if header present)
    sig = request.headers.get("X-NestLink-Signature", "")
    if sig and NESTLINK_WEBHOOK_SECRET:
        exp = hmac.new(NESTLINK_WEBHOOK_SECRET.encode(), raw, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, exp):
            print("⚠️ bad signature — continuing anyway for debug")

    # Parse JSON body
    try:
        payload = json.loads(body_str) if body_str else {}
    except Exception:
        payload = {}
    if not payload and request.query_params:
        payload = dict(request.query_params)

    # Find email — try every possible location
    def _find_email(p):
        if not isinstance(p, dict): return None
        for k in ("email","customer_email","customerEmail","user_email"):
            if p.get(k): return p[k]
        for k in ("customer","user","payer"):
            if isinstance(p.get(k), dict):
                e = _find_email(p[k])
                if e: return e
        for k in ("data","transaction","payment","meta"):
            if isinstance(p.get(k), dict):
                e = _find_email(p[k])
                if e: return e
        return None

    # Find reference — try every location
    def _find_ref(p):
        if not isinstance(p, dict): return None
        for k in ("reference","api_ref","external_reference","ref",
                  "transaction_reference","account_reference"):
            if p.get(k): return p[k]
        for k in ("data","transaction","payment","meta"):
            if isinstance(p.get(k), dict):
                r = _find_ref(p[k])
                if r: return r
        return None

    # Find status
    def _find_status(p):
        if not isinstance(p, dict): return ""
        for k in ("status","payment_status","transaction_status","state"):
            v = p.get(k)
            if v: return str(v).lower()
        for k in ("data","transaction","payment"):
            if isinstance(p.get(k), dict):
                s = _find_status(p[k])
                if s: return s
        return ""

    email = _find_email(payload)
    ref   = _find_ref(payload)
    status = _find_status(payload)

    print(f"PARSED — email={email}, ref={ref}, status={status}")

    if status and status not in ("completed","success","paid","successful",
                                 "settled","confirmed"):
        return {"ok":True,"ignored":True,"status":status}

    # Extract tier from reference like IMH-starter_20-1234567-abc
    tier = None
    if ref:
        parts = str(ref).split("-")
        for i, p in enumerate(parts):
            candidate = "-".join(parts[1:i+1]) if i >= 1 else p
            if candidate in TIER_CREDITS:
                tier = candidate; break
        if not tier and len(parts) >= 2:
            tier = TIER_ALIAS.get(parts[1], parts[1])

    if not email or not tier or tier not in TIER_CREDITS:
        return {"ok":True,"upgraded":False,
                "reason":f"email={email} tier={tier} ref={ref}"}

    # Unlock
    for uid, u in users_db.items():
        if u["email"].lower() == email.lower():
            u["credits"] += TIER_CREDITS[tier]
            u["tier"] = tier
            u["credits_used"] = 0
            u["paid_at"] = datetime.now(timezone.utc).isoformat()
            print(f"✅ UPGRADED {email} → {tier} (+{TIER_CREDITS[tier]} credits)")
            return {"ok":True,"upgraded":True,"email":email,"tier":tier}

    # Pre-register
    uid, _ = mk_user(email.lower(), email.split("@")[0], tier)
    users_db[uid]["credits"] = TIER_CREDITS[tier]
    print(f"✅ PRE-REGISTERED {email} → {tier}")
    return {"ok":True,"upgraded":True,"created":True,"email":email,"tier":tier}

@app.get("/nestlink/logs")
async def nestlink_logs():
    """See recent webhook calls."""
    if not WEBHOOK_LOG.exists(): return {"ok":True,"count":0,"entries":[]}
    lines = WEBHOOK_LOG.read_text().strip().split("\n")[-20:]
    return {"ok":True,"count":len(lines),
            "entries":[json.loads(l) for l in lines if l.strip()]}

@app.get("/cfc/status")
async def cfc_status_proxy():
    if not COLAB_MODEL_URL: raise HTTPException(503, "CFC offline")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(COLAB_MODEL_URL + "/cfc/status")
    return r.json()

@app.get("/cfc/active")
async def cfc_active():
    if not COLAB_MODEL_URL: return {"ok":True,"count":0,"agents":[]}
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(COLAB_MODEL_URL + "/cfc/status")
        d = r.json(); pr = d.get("predictors", {})
        return {"ok":True,"count":len(pr),"agents":list(pr.keys())}
    except Exception:
        return {"ok":True,"count":0,"agents":[]}

@app.post("/agent/run")
async def agent_proxy(body: dict):
    if not COLAB_MODEL_URL: raise HTTPException(503, "Local brain offline")
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(COLAB_MODEL_URL + "/agent/run", json=body)
    return r.json()

@app.get("/health")
async def health():
    return {"ok":True,"version":"3.2.0",
            "model_configured":bool(NVIDIA_API_KEY),
            "model":NVIDIA_MODEL if NVIDIA_API_KEY else "not-set",
            "colab_fallback_configured":bool(COLAB_MODEL_URL),
            "weather_configured":bool(OPENWEATHER_KEY),
            "nestlink_configured":bool(NESTLINK_API_KEY),
            "google_configured":bool(GOOGLE_CLIENT_ID),
            "kenya_counties":len(KENYA_COUNTIES),
            "pricing_tiers":len(PRICES),
            "adaptive_length":True,
            "founder":"Samuel Kiragu"}
