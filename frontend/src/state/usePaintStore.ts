/**
 * Frontend State & Precedence Resolver (`Region Override > Part Assignment > Color-Group Assignment`)
 * Specification: SDD Section 4.2
 */

export type SelectionMode = 'COLOR_GROUP' | 'PART' | 'REGION';
export type TopcoatFinishMode = 'NONE_INHERIT' | 'GLOSS' | 'SEMI_GLOSS' | 'MATTE';
export type LightingEnvironment = 'STANDARD' | 'DARK_ROOM' | 'DARK_UV_365NM';

export interface LayerItemConfig {
  paintId: string;
  coats: number;
  hidingPowerOverride?: number;
  intrinsicSheenPreset?: 'DEFAULT' | 'GLOSS' | 'SEMI_GLOSS' | 'MATTE';
  intrinsicRoughnessOverride?: number; // [0.03, 0.90] -> active when topcoatFinish == 'NONE_INHERIT'
  enabled: boolean;
}

export interface LayerStackConfig {
  primer?: LayerItemConfig;        // e.g., #500, #1000, #1200, #1500 or Gloss Black Surfacer
  basePaints: LayerItemConfig[];   // Solid, Metallic, Mirror Chrome, Chameleon, or Fluorescent Solid
  clearCoat?: LayerItemConfig;     // Clear Candy or Fluorescent Clear
  topcoatFinish: TopcoatFinishMode;// Defaults to 'NONE_INHERIT' (uses topmost layer's intrinsic gloss!)
  topcoatRoughnessOverride?: number;
  wash?: { colorHex: string; intensity: number; isFluorescent: boolean; enabled: boolean };
}

export interface PaintSimulatorState {
  activeSelectionMode: SelectionMode;       // Defaults to 'COLOR_GROUP' (Palette-First)
  lightingEnvironment: LightingEnvironment; // 'STANDARD' | 'DARK_ROOM' | 'DARK_UV_365NM'
  ambientDimFactor: number;                 // Default 0.12 in DARK_ROOM, 0.05 in DARK_UV_365NM
  uvIntensity: number;                      // Default 1.5 in DARK_UV_365NM
  colorGroupStacks: Record<string, LayerStackConfig>;
  partStacks: Record<string, LayerStackConfig>;
  regionOverrides: Record<number, LayerStackConfig>;
}

export function resolveEffectiveStack(
  state: PaintSimulatorState,
  regionId: number,
  partId: string,
  colorGroupId: string,
  isProtected: boolean
): LayerStackConfig | null {
  if (isProtected) {
    return null;
  }
  if (state.regionOverrides[regionId]) {
    return state.regionOverrides[regionId];
  }
  if (state.partStacks[partId]) {
    return state.partStacks[partId];
  }
  if (state.colorGroupStacks[colorGroupId]) {
    return state.colorGroupStacks[colorGroupId];
  }
  return null;
}
