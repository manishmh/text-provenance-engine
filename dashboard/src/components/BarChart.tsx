export function BarChart({ data, title }: { data: { label: string; value: number; color?: string }[]; title: string }) {
  const max = Math.max(...data.map((d) => d.value), 1);

  return (
    <div className="card" style={{ padding: "16px 20px" }}>
      <h3 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "12px", color: "#374151" }}>{title}</h3>
      {data.length === 0 ? (
        <div className="empty-state" style={{ padding: "16px" }}>
          <div style={{ fontSize: "13px" }}>No data available</div>
        </div>
      ) : (
        data.map((item, i) => (
          <div key={item.label + i} style={{ marginBottom: "8px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", marginBottom: "3px" }}>
              <span style={{ color: "#374151" }}>{item.label}</span>
              <span className="mono" style={{ color: "#6b7280", fontWeight: 600 }}>{item.value.toLocaleString()}</span>
            </div>
            <div style={{ height: "8px", background: "#f3f4f6", borderRadius: "4px", overflow: "hidden" }}>
              <div
                data-testid="bar-fill"
                style={{
                  height: "100%",
                  width: `${(item.value / max) * 100}%`,
                  background: item.color || "#e94560",
                  borderRadius: "4px",
                  transition: "width 0.3s ease",
                }}
              />
            </div>
          </div>
        ))
      )}
    </div>
  );
}
