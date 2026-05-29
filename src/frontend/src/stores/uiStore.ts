import { create } from 'zustand';
import type { RunDetailTab } from '../types';

interface UiState {
  activeRunTab: RunDetailTab;
  compareRunIds: string[];
  sidebarCollapsed: boolean;
  compactTables: boolean;
  setActiveRunTab: (tab: RunDetailTab) => void;
  toggleCompareRun: (runId: string) => void;
  clearCompare: () => void;
  setSidebarCollapsed: (value: boolean) => void;
  setCompactTables: (value: boolean) => void;
}

export const useUiStore = create<UiState>((set) => ({
  activeRunTab: 'overview',
  compareRunIds: [],
  sidebarCollapsed: false,
  compactTables: false,
  setActiveRunTab: (activeRunTab) => set({ activeRunTab }),
  toggleCompareRun: (runId) =>
    set((state) => {
      const exists = state.compareRunIds.includes(runId);
      return {
        compareRunIds: exists
          ? state.compareRunIds.filter((id) => id !== runId)
          : state.compareRunIds.length < 4
            ? [...state.compareRunIds, runId]
            : state.compareRunIds,
      };
    }),
  clearCompare: () => set({ compareRunIds: [] }),
  setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),
  setCompactTables: (compactTables) => set({ compactTables }),
}));