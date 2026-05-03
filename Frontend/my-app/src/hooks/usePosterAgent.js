import { useCallback, useState } from "react";
import { generatePoster } from "../api/posterApi";

const initialState = {
  sessionId: null,
  poster: null,
  history: [],
  status: "idle",
  error: null,
};

export function usePosterAgent() {
  const [state, setState] = useState(initialState);

  const generate = useCallback(async (prompt) => {
    setState((s) => ({ ...s, status: "loading", error: null }));
    try {
      const result = await generatePoster({
        prompt,
        sessionId: null,
        useMemory: false,
      });
      setState({
        sessionId: result.sessionId,
        poster: { html: result.html, css: result.css },
        history: [{ role: "user", content: prompt, kind: "initial" }],
        status: "ready",
        error: null,
      });
    } catch (err) {
      setState((s) => ({ ...s, status: "error", error: err.message }));
    }
  }, []);

  const followUp = useCallback(
    async (prompt) => {
      setState((s) => ({ ...s, status: "loading", error: null }));
      try {
        const result = await generatePoster({
          prompt,
          sessionId: state.sessionId,
          useMemory: true,
        });
        setState((s) => ({
          ...s,
          sessionId: result.sessionId,
          poster: { html: result.html, css: result.css },
          history: [
            ...s.history,
            { role: "user", content: prompt, kind: "followup" },
          ],
          status: "ready",
          error: null,
        }));
      } catch (err) {
        setState((s) => ({ ...s, status: "error", error: err.message }));
      }
    },
    [state.sessionId]
  );

  const reset = useCallback(() => {
    setState(initialState);
  }, []);

  return { state, generate, followUp, reset };
}
