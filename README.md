# Metathon

Marathon finish time predictor powered by your Strava data.

## Deploy in 5 minutes

### 1. Create a Strava API Application

1. Go to [strava.com/settings/api](https://www.strava.com/settings/api)
2. Fill in any app name (e.g. "Metathon")
3. Set **Category** to Training
4. Set **Website** to your Vercel URL (e.g. `https://metathon.vercel.app`) — you can update this after deploy
5. Set **Authorization Callback Domain** to your Vercel domain (e.g. `metathon.vercel.app`)
6. Note down your **Client ID** and **Client Secret**

### 2. Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

Or via CLI:

```bash
npm i -g vercel
vercel
```

Set these environment variables in Vercel's dashboard (Settings → Environment Variables):

| Variable | Value |
|----------|-------|
| `STRAVA_CLIENT_ID` | Your Strava Client ID |
| `STRAVA_CLIENT_SECRET` | Your Strava Client Secret |
| `VITE_STRAVA_CLIENT_ID` | Same as `STRAVA_CLIENT_ID` |

### 3. Register your callback URL with Strava

Back in your Strava API app settings, add your deployed URL as the callback domain:
- `metathon.vercel.app` (or whatever your Vercel URL is)

That's it. Visit your app and click "Connect with Strava."

---

## Local development

```bash
npm install
cp .env.example .env.local   # fill in your credentials
npm run dev
```

For local OAuth to work, also add `localhost` to your Strava app's Authorization Callback Domain.

The dev server proxies `/api/*` requests to Vercel Functions locally via `vite.config.js`.

## How the prediction works

The predicted finish time uses a **Riegel formula** across all your run activities from the last 8 weeks:

```
T_marathon = T_run × (42195 / distance_metres) ^ 1.06
```

Each run is weighted by:
- **Recency** — exponential decay with a 6-week half-life (recent runs count more)
- **Distance** — longer runs are stronger predictors than short ones
- **Correction factor** — Riegel overestimates from short distances, so shorter runs are adjusted down slightly

The progress chart reruns the prediction algorithm for each of the past 14 days, showing how your forecast has evolved.

## Stack

- Vite + React
- Recharts for the progress chart
- Vercel Serverless Functions for the OAuth token exchange
- No database — tokens stored in localStorage, activities fetched fresh on each sync
