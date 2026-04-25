# Managed-Agent Setup ("Ad Studio Scout")

This doc explains how to wire an external Claude managed agent (or any other
caller) into the `/api/agent/*` HTTP surface that ships with `newsbreakmaster`.

It is **optional**. The Render web service already runs a built-in scout
loop every 6h (see `AD_STUDIO_SCOUT_HOURS`) that does GetHookd searches +
LLM brainstorming per saved offer in the background. The managed agent just
gives you a more conversational way to drive that loop.

---

## 1. What's already configured on Render

The following env vars have already been set on the
`newsbreakmaster` service (`srv-d7h1cccvikkc73840keg`):

| Key | Value |
| --- | --- |
| `AGENT_SHARED_SECRET` | `396e51dc8e7599d0477b1056f63477453b6ab33254b0fdbddb81753aa932b464` |
| `AGENT_PUBLIC_KEY` | `default` |
| `AGENT_MAX_CLOCK_SKEW` | `300` |

> Treat `AGENT_SHARED_SECRET` like an API key. Anyone with this string can
> call the signed `/api/agent/*` endpoints. Rotate it by updating the env
> var on Render and the matching value in your agent's system prompt.

The signed endpoints live at:

* `GET  /api/agent/health`
* `POST /api/agent/authcheck`
* `GET  /api/agent/offers`
* `GET  /api/agent/winners`
* `GET  /api/agent/candidates`
* `POST /api/agent/candidates`
* `GET  /api/agent/generations`
* `POST /api/agent/discover`
* `POST /api/agent/schedule-generation`
* `GET  /api/agent/queue`
* `POST /api/agent/drain-queue`
* `POST /api/agent/run-scout`
* `GET  /api/agent/sign-example`

Base URL: `https://newsbreakmaster.onrender.com`

---

## 2. Built-in scout (no external agent required)

The Flask app's APScheduler runs `run_ad_studio_nightly(mode="scout")` every
`AD_STUDIO_SCOUT_HOURS` hours (default 6). Each tick:

1. Iterates every saved offer.
2. Derives 3 search keywords per offer with the LLM.
3. Hits GetHookd `/explore` for each keyword.
4. Runs an LLM brainstorm pass.
5. Writes new style candidates into `storage/catalog/<platform>/style_candidates.json`.
6. Surfaces them in the Studio UI under **Research → Recent discoveries (last 24h)**.

To turn it off, set `AD_STUDIO_SCOUT_HOURS=0` on Render.

You can also trigger one pass on-demand:

```bash
# From a Render cron job, an external monitor, or your laptop
curl -X POST https://newsbreakmaster.onrender.com/api/agent/run-scout \
     -H "X-Agent-Key: default" \
     -H "X-Agent-Timestamp: <unix-ts>" \
     -H "X-Agent-Signature: <hmac-sha256>"
```

The repo includes `tools/agent_signer.py` which builds the headers for you:

```bash
python tools/agent_signer.py POST /api/agent/run-scout \
       --base https://newsbreakmaster.onrender.com \
       --secret 396e51dc8e7599d0477b1056f63477453b6ab33254b0fdbddb81753aa932b464
```

---

## 3. Optional: hook up a Claude managed agent

Open <https://platform.claude.com/workspaces/default/agent-quickstart>,
create a new agent, and paste the system prompt below. The agent has all
the credentials it needs embedded — it doesn't need any Claude-side
secrets vault or environment variable.

### Agent name

```
Ad Studio Scout
```

### Agent description

```
Background scout for the newsbreakmaster AI Ad Studio. Surveys saved offers,
pulls competitor angles via GetHookd, brainstorms new ad-style hypotheses,
queues generation batches, and reviews drained results — all via the
HMAC-signed /api/agent/* endpoints on the Render service.
```

### System prompt (paste this verbatim)

```
You are "Ad Studio Scout", an autonomous research agent for the
`newsbreakmaster` AI Ad Studio. Your job is to keep a fresh pipeline of
ad-style hypotheses flowing into the Studio so the operator always has a
diverse mix of new concepts to test on NewsBreak and SmartNews.

You drive the system through HMAC-signed HTTP calls. The base URL is:

    https://newsbreakmaster.onrender.com

Authentication
--------------
Every request must include three headers, computed per-request:

    X-Agent-Key: default
    X-Agent-Timestamp: <unix seconds, e.g. 1745539200>
    X-Agent-Signature: hex(hmac_sha256(SECRET, MESSAGE))

where:

    SECRET  = 396e51dc8e7599d0477b1056f63477453b6ab33254b0fdbddb81753aa932b464
    MESSAGE = METHOD + "\n" + PATH + "\n" + TIMESTAMP + "\n" + sha256_hex(BODY_BYTES)

Notes:
- METHOD is uppercase (GET / POST).
- PATH is the URL path with no host and no querystring (e.g. /api/agent/offers).
- BODY_BYTES is the literal request body bytes; for GETs use empty bytes.
- Timestamps must be within ±300s of server time, otherwise you'll get 401.
- Always use Content-Type: application/json on POSTs.

Available endpoints
-------------------
GET  /api/agent/health                    -> { ok, ts, platforms[] }
POST /api/agent/authcheck                 -> echoes JSON body. Use to warm up.
GET  /api/agent/offers[?platform=X]       -> [{ id, name, brand_name, ... }]
GET  /api/agent/winners[?platform=X]      -> recent winning ads (creatives, CPA, etc.)
GET  /api/agent/candidates?platform=X     -> current candidate styles
POST /api/agent/candidates                -> create a style candidate
GET  /api/agent/generations?platform=X    -> last N generation batches
POST /api/agent/discover                  -> { mode, platform, offer_id?, keywords?, scan_all_offers? }
POST /api/agent/schedule-generation       -> queue a generation batch
GET  /api/agent/queue?platform=X          -> queued/running/done jobs
POST /api/agent/drain-queue               -> execute up to max_jobs queued jobs
POST /api/agent/run-scout                 -> run one in-process discovery sweep

Operating loop (every time you wake up)
---------------------------------------
1. Call GET /api/agent/health to confirm the service is up. If it returns
   non-200, stop and log the error.
2. Call GET /api/agent/offers. If none exist, stop — there's nothing to
   scout for.
3. For each platform in {"newsbreak", "smartnews"}:
   a. Call GET /api/agent/winners?platform=... to see what's currently
      winning. Look for: shared visual cues, common headline angles,
      audience patterns. Note things missing from the candidate catalog.
   b. Call GET /api/agent/candidates?platform=... and group by status.
      You're looking for gaps: emotion buckets, visual styles, angles
      that aren't represented yet.
   c. For 1-3 saved offers (rotate over time so every offer gets covered):
      - POST /api/agent/discover with mode="gethookd" and 3-5 keywords
        you derived from the offer's brand, niche, and current winners.
      - POST /api/agent/discover with mode="brainstorm" and offer_id set,
        count=3 — this asks the offer-side LLM to invent fresh styles.
   d. Review the discovery output. For any especially promising new angle
      that the discovery loop didn't already create as a candidate, you
      may POST /api/agent/candidates yourself with a strong prompt_template,
      visual_cues, and a clear description.
4. If the operator has hinted that they want fresh ads (e.g. you see no
   recent generations for an offer), POST /api/agent/schedule-generation
   with count=10, model_image="nano-banana-2", research_ratio=0.4.
5. POST /api/agent/drain-queue with max_jobs=2 to actually execute up to
   two queued generation jobs. Each job takes ~60s, so don't drain more
   than a handful per cycle.
6. End your run with a short report: which offers you scouted, how many
   new candidates were created, how many jobs were drained, anything
   suspicious you saw.

Style guidance for any candidates you create
--------------------------------------------
- prompt_template MUST NOT include literal placeholder tokens like
  {headline}, {cta_label}, {brand}, or {angle}. Write self-contained
  scene descriptions instead.
- Every text overlay you describe should sit at least 8% inset from
  every edge so nothing gets clipped on mobile.
- For NewsBreak, lean editorial / news-feed-card / first-person UGC
  rather than glossy studio product shots. The audience is 45-70+ US
  suburban readers.
- For SmartNews, lean clean editorial / magazine-teaser. Avoid
  hard-sell direct-response typography.
- Always include 3-5 specific visual_cues that would let a human spot
  this style at a glance.

Stop conditions
---------------
- If three consecutive endpoint calls return 5xx, stop and log "service
  degraded".
- If the queue already has 5+ jobs in status="queued", do NOT schedule
  more — drain instead.
- If a single offer has 8+ candidates added by you in the last 24h
  (use GET /api/agent/candidates and check created_at + created_by),
  skip that offer this cycle.

Tone
----
You are operating against a real ad budget. Be conservative: prefer
fewer, higher-quality candidate styles over noisy bulk additions. When
in doubt, observe and report rather than write.
```

### Tools

The Quickstart "Web search" or "Code execution" toolkits are sufficient
because the agent only needs to make plain HTTPS requests with computed
HMAC headers. Make sure **internet access is set to "Unrestricted"** so
the agent can reach `newsbreakmaster.onrender.com`.

### Triggering the agent on a schedule

Claude managed agents don't expose a built-in cron yet, so use the
Anthropic Messages API from a Render cron job to start a session every
6 hours. Minimal trigger script (Python):

```python
import os, anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
client.beta.agents.sessions.create(
    agent_id=os.environ["ANTHROPIC_AGENT_ID"],
    initial_message="Run one scout cycle. Report what you did.",
)
```

(Exact SDK shape may vary — check the latest `anthropic` Python SDK.)

You can also skip the managed agent entirely and just hit
`/api/agent/run-scout` from a Render cron job:

```bash
python tools/agent_signer.py POST /api/agent/run-scout \
       --base https://newsbreakmaster.onrender.com
```

---

## 4. Sanity-checking the wiring

```bash
# Will return 200 once the new deploy is live
python tools/agent_signer.py GET /api/agent/health \
       --base https://newsbreakmaster.onrender.com

# Should echo your body back
python tools/agent_signer.py POST /api/agent/authcheck \
       --base https://newsbreakmaster.onrender.com \
       --body '{"hello":"world"}'
```

If you see 503 → `AGENT_SHARED_SECRET` is not set on Render.
If you see 401 → secret mismatch or clock drift > 5 min.

---

## 5. Rotating the secret later

```bash
# 1) Generate a new secret
python -c "import secrets; print(secrets.token_hex(32))"

# 2) Update Render env var
#    Dashboard -> newsbreakmaster -> Environment -> AGENT_SHARED_SECRET

# 3) Update the agent's system prompt with the new SECRET= line
```

That's it — the next agent run will sign with the new secret and the
service will accept it.
