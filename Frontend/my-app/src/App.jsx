import { useEffect, useState } from "react";
import PosterForm from "./components/PosterForm";
import PosterPreview from "./components/PosterPreview";
import FollowUpPanel from "./components/FollowUpPanel";
import { usePosterAgent } from "./hooks/usePosterAgent";
import { checkHealth } from "./api/posterApi";
import "./styles/app.css";

export default function App() {
  const { state, generate, followUp, reset } = usePosterAgent();
  const [backendOk, setBackendOk] = useState(null);

  useEffect(() => {
    let alive = true;
    checkHealth()
      .then(() => alive && setBackendOk(true))
      .catch(() => alive && setBackendOk(false));
    return () => {
      alive = false;
    };
  }, []);

  const loading = state.status === "loading";
  const showResult = state.poster !== null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">◆</span>
          <span>Poster Agent</span>
        </div>
        <div
          className={`status status--${
            backendOk === null ? "idle" : backendOk ? "ok" : "down"
          }`}
        >
          <span className="status-dot" />
          {backendOk === null
            ? "checking backend…"
            : backendOk
              ? "backend connected"
              : "backend offline"}
        </div>
      </header>

      <main className={showResult ? "stage stage--split" : "stage"}>
        {!showResult && (
          <PosterForm
            onSubmit={generate}
            loading={loading}
            error={state.error}
          />
        )}

        {showResult && (
          <>
            <section className="stage-left">
              <PosterPreview
                html={state.poster.html}
                css={state.poster.css}
                loading={loading}
              />
            </section>
            <section className="stage-right">
              <FollowUpPanel
                history={state.history}
                onFollowUp={followUp}
                onGenerateNew={reset}
                loading={loading}
                error={state.error}
                sessionId={state.sessionId}
              />
            </section>
          </>
        )}
      </main>
    </div>
  );
}
