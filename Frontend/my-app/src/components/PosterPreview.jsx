import { useEffect, useMemo, useRef } from "react";

export default function PosterPreview({ html, css, loading }) {
  const iframeRef = useRef(null);

  const document = useMemo(() => {
    if (!html && !css) return "";
    return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>${css ?? ""}</style>
  </head>
  <body>${html ?? ""}</body>
</html>`;
  }, [html, css]);

  useEffect(() => {
    const frame = iframeRef.current;
    if (!frame) return;
    frame.srcdoc = document;
  }, [document]);

  return (
    <div className="preview-shell">
      <div className="preview-toolbar">
        <span className="dot dot--r" />
        <span className="dot dot--y" />
        <span className="dot dot--g" />
        <span className="preview-title">poster preview</span>
      </div>
      <div className="preview-stage">
        {loading && (
          <div className="preview-loading">
            <div className="spinner" />
            <span>Generating poster…</span>
          </div>
        )}
        <iframe
          ref={iframeRef}
          title="poster"
          sandbox=""
          className={loading ? "preview-frame is-loading" : "preview-frame"}
        />
      </div>
    </div>
  );
}
