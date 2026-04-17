# NewsBreak Ads Launcher (`newsbreakmaster`)

Standalone Flask app to **bulk-launch** NewsBreak ads and run **rules-based scaling/cuts** using the [NewsBreak Advertising API](https://advertising-api.newsbreak.com/hc/en-us/categories/37825505060237-API-Reference).

## Features

- Paste **Access-Token** + **Organization ID(s)** (no OAuth).
- **Dashboard**: list ad accounts (`getGroupsByOrgIds`).
- **Bulk launch**: upload multiple creatives, optional new campaign, grouping (all-in-one / isolate / groups of N).
- **Scaling**: integrated report table with pause/enable ads and ±20% ad set budget (when report includes ids/budgets).
- **Rules**: templates (kill zero conversions, high CPA, scale winners, low ROAS), dry-run, scheduler every 15 minutes.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env
# Set FLASK_SECRET_KEY and optional NEWSBREAK_DEFAULT_ORG_IDS
python app.py
```

Open http://127.0.0.1:5055

## Configuration

| Variable | Description |
|----------|-------------|
| `FLASK_SECRET_KEY` | Flask session secret (required in production) |
| `NEWSBREAK_DEFAULT_ORG_IDS` | Comma-separated org IDs prefilled on login |
| `BASIC_AUTH_USER` / `BASIC_AUTH_PASSWORD` | If both set, the entire app requires HTTP Basic Auth first (recommended on public hosts) |
| `PORT` | Listen port (default 5055) |

## API notes

- Base URL used in code: `https://business.newsbreak.com/business-api/v1`
- Auth header: `Access-Token`
- Campaign / ad set / ad **create** payloads may require extra fields (targeting, etc.) per your account — adjust `bulk_launcher` / `app.py` as needed.
- **Budget** values in NewsBreak APIs are often in **cents**; the UI uses dollars and converts.
- Integrated report **field names** vary; `normalize_report_rows` maps common aliases. Tune `rules_engine.py` if your responses differ.

## Deploy (Render)

- Connect repo, use `render.yaml` or set build `pip install -r requirements.txt` and start `gunicorn app:app`.
- Ephemeral disk: rules/audit reset on redeploy unless you add a persistent disk or database.

## Scheduler

`APScheduler` runs `run_scheduled_rules()` every **15 minutes** for all saved tokens with enabled rules. For multiple Gunicorn workers, consider disabling in-app scheduler and using Render **Cron** hitting `/api/scheduler/run` with a secret header (add in a future iteration).
