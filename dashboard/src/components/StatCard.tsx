export function StatCard({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div data-testid="stat-card" className="stat-card">
      <div className="stat-card__label">{label}</div>
      <div className="stat-card__value" style={color ? { color } : undefined}>
        {value}
      </div>
    </div>
  );
}
