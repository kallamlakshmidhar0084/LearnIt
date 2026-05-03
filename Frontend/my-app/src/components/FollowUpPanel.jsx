import { useState } from "react";

export default function FollowUpPanel({
  history,
  onFollowUp,
  onGenerateNew,
  loading,
  error,
  sessionId,
}) {
  const [prompt, setPrompt] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = prompt.trim();
    if (!trimmed) return;
    onFollowUp(trimmed);
    setPrompt("");
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <div>
          <span className="pill pill--ghost">Session</span>
          <h2>Iterate on this poster</h2>
          <p className="panel-sub">
            Follow-ups keep the agent's memory of prior turns. Generate New
            starts a fresh session.
          </p>
        </div>
        <button
          type="button"
          className="btn btn--ghost"
          onClick={onGenerateNew}
          disabled={loading}
        >
          ＋ Generate New
        </button>
      </div>

      <div className="history">
        {history.map((msg, i) => (
          <div key={i} className={`bubble bubble--${msg.kind}`}>
            <span className="bubble-label">
              {msg.kind === "initial" ? "Initial brief" : "Follow-up"}
            </span>
            <p>{msg.content}</p>
          </div>
        ))}
        {loading && (
          <div className="bubble bubble--loading">
            <span className="spinner spinner--sm" />
            <span>Working on it…</span>
          </div>
        )}
      </div>

      {error && <div className="form-error">{error}</div>}

      <form onSubmit={handleSubmit} className="followup-form">
        <textarea
          rows={3}
          placeholder="Make the headline larger, switch to a darker palette…"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={loading}
        />
        <div className="followup-actions">
          <span className="session-id">
            {sessionId ? `session: ${sessionId.slice(0, 8)}…` : ""}
          </span>
          <button
            type="submit"
            className="btn btn--primary"
            disabled={loading || !prompt.trim()}
          >
            {loading ? "Sending…" : "Send Follow-up"}
          </button>
        </div>
      </form>
    </div>
  );
}
