import type { DashboardConfig } from "../types/api";
import { ConnectionForm } from "./ConnectionForm";

export function SetupScreen({ onSave }: { onSave: (c: DashboardConfig) => void }) {
  return (
    <div className="welcome">
      <div className="welcome__inner">
        <div className="welcome__brand">
          <div className="welcome__logo">Provenance Engine</div>
          <p className="welcome__tagline">
            Detect text-level provenance signals, evaluate watermark robustness,
            and review every result with statistical rigor.
          </p>
          <ul className="welcome__points">
            <li>Analyze text across unicode, KGW, and SynthID detectors</li>
            <li>Run robustness benchmarks with Wilson confidence intervals</li>
            <li>Track history, jobs, usage, and detector capabilities</li>
          </ul>
        </div>
        <div className="card welcome__card">
          <h1 className="welcome__title">Connect to your API server</h1>
          <p className="welcome__subtitle">
            Runs locally by default. Your key is sent only to this server
            and kept in this browser session.
          </p>
          <ConnectionForm onSave={onSave} />
        </div>
      </div>
    </div>
  );
}
