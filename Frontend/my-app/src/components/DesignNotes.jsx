export default function DesignNotes({ rationale, palette }) {
  if (!rationale && (!palette || palette.length === 0)) return null;

  return (
    <div className="design-notes">
      <span className="bubble-label">Design notes</span>
      {rationale && <p className="design-notes__rationale">{rationale}</p>}
      {palette && palette.length > 0 && (
        <div className="design-notes__swatches">
          {palette.map((hex) => (
            <span
              key={hex}
              className="swatch"
              style={{ background: hex }}
              title={hex}
            >
              <span className="swatch__label">{hex}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
