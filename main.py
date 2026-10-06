
import os, re, time, secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import httpx, jwt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

SECRET_KEY       = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
NVIDIA_API_KEY   = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL     = os.getenv("NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
NVIDIA_URL       = "https://integrate.api.nvidia.com/v1/chat/completions"
OPENWEATHER_KEY  = os.getenv("OPENWEATHER_API_KEY", "")
NESTLINK_API_KEY = os.getenv("NESTLINK_API_KEY", "")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
CORS_ORIGINS     = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="Imhotep", version="2.5.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

_LEAK = [
    (r"\bNVIDIA\s+NIM\b","Imhotep"),(r"\bNVIDIA\b","Imhotep"),(r"\bNemotron[\w\-]*\b","Imhotep"),
    (r"\bNIM\b","Imhotep"),(r"\bQwen[\w\-\.]*\b","Imhotep"),(r"\bDeepSeek[\w\-]*\b","Imhotep"),
    (r"\bMistral[\w\-]*\b","Imhotep"),(r"\bLlama[\s\-]?[\d\.\w]*\b","Imhotep"),
    (r"\bMeta\s+AI\b","Imhotep"),(r"\bAnthropic\b","Imhotep"),(r"\bOpenAI\b","Imhotep"),
    (r"\bGPT[\-\s]?\d[\w\.]*\b","Imhotep"),(r"\bClaude[\w\-\.]*\b","Imhotep"),
    (r"\bGemini[\w\-\.]*\b","Imhotep"),
]
def scrub(t):
    if not t: return t
    for p, r in _LEAK: t = re.sub(p, r, t, flags=re.IGNORECASE)
    return t

PERSONAS = {
    "generalist":"a broad, well-rounded assistant",
    "scientist":"a rigorous scientist — cites sources, quantifies",
    "coder":"a senior software engineer — precise, working code",
    "economist":"an economist — data, incentives, opportunity cost",
    "entrepreneur":"a founder — strategy, product-market fit, growth",
    "philosopher":"a philosopher — assumptions, logic, ethics",
    "writer":"a writer — vivid prose, rhythm, structure",
    "analyst":"a data analyst — numbers, KPIs, trends",
    "legal":"a legal analyst — risks, principles, not binding advice",
    "designer":"a designer — UX, hierarchy, systems",
    "psychologist":"a psychologist — behavior, cognition",
    "geographer":"a geographer — regions, climate, demographics",
    "historian":"a historian — context, primary sources",
    "tutor":"a patient tutor — step by step",
    "medic":"a health-literate assistant",
    "engineer":"an engineer — systems, constraints, trade-offs",
    "agronomist":"an agronomist — crops, soil, East African farming",
    "inventor":"an inventor — bold ideas, then feasibility",
    "civic":"a civic analyst — policy, governance",
    "marketer":"a marketer — positioning, channels, campaigns",
    "media":"a media producer — video, audio, storyboards",
    "activist":"an activist — advocacy, campaigns, community",
}
def sys_prompt(agent):
    p = PERSONAS.get(agent, "a general assistant")
    return (
        f"You are Imhotep, a private multi-agent AI built by Samuel Kiragu in "
        f"Mukuyu, Murang'a, Kenya. You are the {agent.upper()} agent — {p}.\n"
        "1. NEVER mention external model or company names.\n"
        "2. If asked what model powers you: 'I run on Imhotep — a private in-house "
        "stack built by Samuel Kiragu in Kenya.'\n"
        "3. If asked who built you: 'I was built by Samuel Kiragu — a Kenyan developer.'\n"
        "4. Speak in first person as this agent.\n"
        "5. Cite facts as [source: URL].\n"
        "6. Optional: wrap reasoning in [THOUGHTS]...[/THOUGHTS]."
    )

users_db, chat_db = {}, {}
TIER = {"guest":3,"free":3,"starter":3,"weekly":10000,"monthly":10000,"yearly":10000}

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
                   "credits":TIER.get(tier,3),"credits_used":0,
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
                   "credits":TIER["guest"],"credits_used":0,
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
    u, new = mk_user(email.lower(), info.get("name","Google User"), "free")
    return {"ok":True,"user_id":u,"token":issue(u),"user":users_db[u],"new":new}

async def weather(lat, lon):
    if not OPENWEATHER_KEY: return None
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.openweathermap.org/data/2.5/weather",
                            params={"lat":lat,"lon":lon,"appid":OPENWEATHER_KEY,"units":"metric"})
        d = r.json()
        if "main" not in d: return None
        return {"temp":round(d["main"]["temp"]),"condition":d["weather"][0]["main"],
                "humidity":d["main"].get("humidity"),"city":d.get("name","")}
    except Exception: return None

async def call_model(msgs, temp):
    if not NVIDIA_API_KEY:
        return ("[Demo mode — no model key configured on the server] "
                "I run on Imhotep — a private in-house stack built by Samuel Kiragu in Kenya.")
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(NVIDIA_URL,
            headers={"Authorization":f"Bearer {NVIDIA_API_KEY}",
                     "Content-Type":"application/json","Accept":"application/json"},
            json={"model":NVIDIA_MODEL,"messages":msgs,"temperature":temp,
                  "top_p":0.95,"max_tokens":1024})
    if r.status_code >= 300:
        raise HTTPException(502, f"Model error {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"]

@app.post("/chat")
async def chat(b: ChatIn):
    p = b.prompt.strip()
    if not p: raise HTTPException(400, "Empty prompt")
    u = b.user_id
    if u and not debit(u):
        c = creds(u); raise HTTPException(429, f"No credits ({c['used']}/{c['limit']})")
    t = max(0.0, min(2.0, b.temperature))
    agent = b.agent or "generalist"
    w = None
    if b.location and b.location.get("lat") is not None:
        w = await weather(b.location["lat"], b.location["lon"])
    s = sys_prompt(agent)
    if w: s += f"\nUser is in {w.get('city','unknown')} ({w['temp']}°C, {w['condition']})."
    try:
        raw = await call_model([{"role":"system","content":s},{"role":"user","content":p}], t)
    except HTTPException: raise
    except Exception as e: raise HTTPException(502, str(e))
    raw = scrub(raw)
    th = ""; ans = raw
    m = re.search(r"\[THOUGHTS\](.*?)\[/THOUGHTS\]", raw, flags=re.DOTALL|re.IGNORECASE)
    if m: th = m.group(1).strip(); ans = (raw[:m.start()]+raw[m.end():]).strip()
    if u and u in chat_db:
        chat_db[u].append({"prompt":p,"reply":ans,"agent":agent,"temperature":t,
                           "thoughts":th,"timestamp":datetime.now(timezone.utc).isoformat()})
    c = creds(u) if u else {"used":0,"remaining":0,"limit":0}
    return {"ok":True,"reply":ans,"thoughts":th,"agent":agent,"temperature":t,
            "weather":w,"credits":c,"timestamp":datetime.now(timezone.utc).isoformat()}

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
    pro = scrub(await call_model(
        [{"role":"system","content":sys_prompt(pa)+f"\nArgue STRONGLY IN FAVOR as the {pa}."},
         {"role":"user","content":f"Proposition: {p}\n\nStrongest 150-word case FOR."}], 0.7))
    con = scrub(await call_model(
        [{"role":"system","content":sys_prompt(ca)+f"\nArgue STRONGLY AGAINST as the {ca}."},
         {"role":"user","content":f"Proposition: {p}\n\nStrongest 150-word case AGAINST."}], 0.7))
    j = scrub(await call_model(
        [{"role":"system","content":sys_prompt("generalist")+"\nYou are an impartial judge."},
         {"role":"user","content":f"Proposition: {p}\n\nPRO ({pa}): {pro[:400]}\n\nCON ({ca}): {con[:400]}\n\nWho wins and why? 100 words."}], 0.4))
    return {"ok":True,"proposition":p,"proponent":pro,"opponent":con,"judge":j,
            "pro_agent":pa,"con_agent":ca}

PRICES = {
    "starter":{"amount":10,"label":"Starter — 3 chats, 7 days"},
    "weekly":{"amount":99,"label":"Weekly Unlimited — 7 days"},
    "monthly":{"amount":199,"label":"Pro Monthly — 30 days"},
    "yearly":{"amount":5000,"label":"Pro Yearly — 365 days"},
}
@app.get("/pricing")
async def pricing(): return {"ok":True,"currency":"KES","tiers":PRICES}

@app.post("/checkout")
async def checkout(b: CheckoutIn):
    if b.tier not in PRICES: raise HTTPException(400, "Unknown tier")
    t = PRICES[b.tier]
    if NESTLINK_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post("https://api.nestlink.co.ke/v1/checkout",
                    json={"amount":t["amount"],"currency":"KES","email":b.email,
                          "narrative":t["label"],"reference":f"IMH-{b.tier}-{int(time.time())}"},
                    headers={"Authorization":f"Bearer {NESTLINK_API_KEY}"})
            if r.status_code < 300:
                d = r.json()
                return {"ok":True,"tier":b.tier,"amount":t["amount"],
                        "payment_url":d.get("checkout_url") or d.get("url")}
        except Exception as e: print(f"nestlink error: {e}")
    return {"ok":True,"tier":b.tier,"amount":t["amount"],
            "payment_url":"https://me.nestlink.co.ke/Imhotepagenticai"}

@app.get("/health")
async def health():
    return {"ok":True,"version":"2.5.0",
            "model_configured":bool(NVIDIA_API_KEY),
            "model":NVIDIA_MODEL if NVIDIA_API_KEY else "not-set",
            "weather_configured":bool(OPENWEATHER_KEY),
            "nestlink_configured":bool(NESTLINK_API_KEY),
            "google_configured":bool(GOOGLE_CLIENT_ID),
            "founder":"Samuel Kiragu"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
