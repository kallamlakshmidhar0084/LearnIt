const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function generatePoster({ prompt, sessionId, useMemory }) {
  const res = await fetch(`${API_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      session_id: sessionId ?? null,
      use_memory: Boolean(useMemory),
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Generate failed (${res.status}): ${text}`);
  }

  const data = await res.json();
  return {
    sessionId: data.session_id,
    html: data.html,
    css: data.css,
    usedMemory: data.used_memory,
    historyLength: data.history_length,
  };
}
