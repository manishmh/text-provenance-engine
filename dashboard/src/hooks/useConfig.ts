import { createContext, useContext } from "react";
import type { DashboardConfig, UsageResponse } from "../types/api";
import { ProvenanceApiClient } from "../api/client";

export interface ConfigContextType {
  config: DashboardConfig | null;
  client: ProvenanceApiClient | null;
  setConfig: (config: DashboardConfig) => void;
  clearConfig: () => void;
  usage: UsageResponse | null;
  refreshUsage: () => Promise<void>;
}

export const ConfigContext = createContext<ConfigContextType>({
  config: null,
  client: null,
  setConfig: () => {},
  clearConfig: () => {},
  usage: null,
  refreshUsage: async () => {},
});

export function useConfig() {
  return useContext(ConfigContext);
}
