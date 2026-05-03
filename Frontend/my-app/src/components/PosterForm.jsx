import { useState } from "react";

export default function PosterForm({ onSubmit, loading, error }) {
  const [title, setTitle] = useState("");
  const [audience, setAudience] = useState("");
  const [style, setStyle] = useState("");
  const [details, setDetails] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!title.trim()) return;

    const parts = [
      `Title: ${title.trim()}`,
      audience.trim() && `Audience: ${audience.trim()}`,
      style.trim() && `Style: ${style.trim()}`,
      details.trim() && `Details: ${details.trim()}`,
    ].filter(Boolean);

    onSubmit(parts.join(" | "));
  };

  return (
    <div className="form-shell">
      <div className="form-card">
        <div className="form-header">
          <span className="pill">AI Poster Studio</span>
          <h1>Describe the poster you want</h1>
          <p>
            Give a few details and the agent will generate a poster you can
            iterate on.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="form-grid">
          <label className="field field--full">
            <span>Poster title *</span>
            <input
              type="text"
              placeholder="e.g. Summer Music Festival 2026"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              required
            />
          </label>

          <label className="field">
            <span>Audience</span>
            <input
              type="text"
              placeholder="e.g. College students"
              value={audience}
              onChange={(e) => setAudience(e.target.value)}
            />
          </label>

          <label className="field">
            <span>Style</span>
            <input
              type="text"
              placeholder="e.g. Bold, retro, minimal"
              value={style}
              onChange={(e) => setStyle(e.target.value)}
            />
          </label>

          <label className="field field--full">
            <span>Other details</span>
            <textarea
              rows={4}
              placeholder="Date, venue, palette, vibe, anything else…"
              value={details}
              onChange={(e) => setDetails(e.target.value)}
            />
          </label>

          {error && <div className="form-error">{error}</div>}

          <button
            type="submit"
            className="btn btn--primary btn--lg"
            disabled={loading || !title.trim()}
          >
            {loading ? "Generating…" : "Generate Poster"}
          </button>
        </form>
      </div>
    </div>
  );
}
