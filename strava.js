// api/strava.js
// Vercel serverless function — keeps STRAVA_CLIENT_SECRET off the client.
// Called by the frontend as:
//   POST /api/strava  { action: "exchange", code: "..." }
//   POST /api/strava  { action: "refresh",  refresh_token: "..." }

const STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token";

export default async function handler(req, res) {
  // CORS headers (needed if you ever serve from a different origin)
  res.setHeader("Access-Control-Allow-Origin", req.headers.origin || "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const CLIENT_ID = process.env.STRAVA_CLIENT_ID;
  const CLIENT_SECRET = process.env.STRAVA_CLIENT_SECRET;

  if (!CLIENT_ID || !CLIENT_SECRET) {
    return res
      .status(500)
      .json({ error: "STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET must be set in environment variables." });
  }

  const { action, code, refresh_token } = req.body;

  try {
    if (action === "exchange" && code) {
      const body = new URLSearchParams({
        client_id: CLIENT_ID,
        client_secret: CLIENT_SECRET,
        code,
        grant_type: "authorization_code",
      });

      const response = await fetch(STRAVA_TOKEN_URL, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: body.toString(),
      });

      const data = await response.json();

      if (!data.access_token) {
        return res.status(400).json({ error: data.message || "Authorization failed" });
      }

      // Return token data (but NOT the client secret — it never leaves this function)
      return res.json({
        access_token: data.access_token,
        refresh_token: data.refresh_token,
        expires_at: data.expires_at,
        athlete: data.athlete,
      });
    }

    if (action === "refresh" && refresh_token) {
      const body = new URLSearchParams({
        client_id: CLIENT_ID,
        client_secret: CLIENT_SECRET,
        refresh_token,
        grant_type: "refresh_token",
      });

      const response = await fetch(STRAVA_TOKEN_URL, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: body.toString(),
      });

      const data = await response.json();

      if (!data.access_token) {
        return res.status(400).json({ error: data.message || "Token refresh failed" });
      }

      return res.json({
        access_token: data.access_token,
        refresh_token: data.refresh_token,
        expires_at: data.expires_at,
      });
    }

    return res.status(400).json({ error: `Unknown action: ${action}` });
  } catch (err) {
    console.error("Strava API error:", err);
    return res.status(500).json({ error: err.message || "Internal server error" });
  }
}
