export default function RefusalPanel({ onGenerateNew, lastPrompt }) {
  return (
    <div className="panel panel--refusal">
      <div className="panel-header">
        <div>
          <span className="pill pill--ghost">Out of scope</span>
          <h2>That wasn't a poster brief</h2>
          <p className="panel-sub">
            I only generate posters. Try describing a topic, event, message,
            or product you want visualized — for example,
            <em> "Yoga workshop next Saturday, calm minimal style."</em>
          </p>
        </div>
      </div>

      {lastPrompt && (
        <div className="bubble bubble--initial">
          <span className="bubble-label">You sent</span>
          <p>{lastPrompt}</p>
        </div>
      )}

      <button
        type="button"
        className="btn btn--primary btn--lg"
        onClick={onGenerateNew}
      >
        Try a different brief
      </button>
    </div>
  );
}
