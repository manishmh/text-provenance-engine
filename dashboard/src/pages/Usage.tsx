import { useConfig } from "../hooks/useConfig";
import { StatCard } from "../components/StatCard";
import { BarChart } from "../components/BarChart";

export function UsagePage() {
  const { usage, refreshUsage } = useConfig();

  if (!usage) {
    return (
      <div>
        <h2 className="page-header">Usage</h2>
        <div className="empty-state">
          <div className="empty-state__icon">📈</div>
          <div className="empty-state__text">No usage data available</div>
        </div>
      </div>
    );
  }

  const endpointData = Object.entries(usage.by_endpoint).map(([label, value]) => ({ label, value }));
  const detectorData = Object.entries(usage.by_detector).map(([label, value]) => ({ label, value }));
  const successRate = usage.total_requests > 0
    ? ((usage.successful_requests / usage.total_requests) * 100).toFixed(1)
    : "—";

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
        <h2 className="page-header" style={{ marginBottom: 0 }}>Usage</h2>
        <button onClick={refreshUsage} data-testid="btn-refresh-usage" className="btn">Refresh</button>
      </div>

      <div className="grid-stats">
        <StatCard label="Total Requests" value={usage.total_requests} />
        <StatCard label="Successful" value={usage.successful_requests} color="#166534" />
        <StatCard label="Failed" value={usage.failed_requests} color="#991b1b" />
        <StatCard label="Characters Analyzed" value={usage.total_characters.toLocaleString()} />
        <StatCard label="Success Rate" value={`${successRate}%`} />
      </div>

      <div className="grid-charts">
        <BarChart data={endpointData} title="Requests by Endpoint" />
        <BarChart data={detectorData} title="Detector Usage" />
      </div>

      <div className="card">
        <h3 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "12px" }}>Usage Details</h3>
        <table className="table">
          <tbody>
            <tr><td style={{ color: "#6b7280" }}>Total Requests</td><td style={{ fontWeight: 600 }}>{usage.total_requests.toLocaleString()}</td></tr>
            <tr><td style={{ color: "#6b7280" }}>Successful</td><td style={{ fontWeight: 600, color: "#166534" }}>{usage.successful_requests.toLocaleString()}</td></tr>
            <tr><td style={{ color: "#6b7280" }}>Failed</td><td style={{ fontWeight: 600, color: "#991b1b" }}>{usage.failed_requests.toLocaleString()}</td></tr>
            <tr><td style={{ color: "#6b7280" }}>Total Characters</td><td style={{ fontWeight: 600 }}>{usage.total_characters.toLocaleString()}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
