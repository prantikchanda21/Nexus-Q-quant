import { useEffect, useMemo, useRef } from "react";
import ForceGraph3D from "3d-force-graph";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import * as THREE from "three";
import type { ContagionShockResponse } from "../types/nexusq";

/** Node in either network (interbank ids are numeric, market ids are tickers). */
export interface TopologyNode {
  id: number | string;
  label: string;
  capital: number;
  degree: number;
  /** Market assets carry a sector for coloring; interbank nodes don't. */
  sector?: string;
}

/** Edge in either network. */
export interface TopologyEdge {
  source: number | string;
  target: number | string;
  exposure: number;
}

/** Minimal renderer contract needed to attach the bloom composer. */
interface FGInstanceWithRenderer extends FGInstance {
  renderer(): { domElement: HTMLCanvasElement }; 
  scene(): THREE.Scene;
  camera(): THREE.PerspectiveCamera;
}

/** Structural superset of both graph payloads. */
export interface NetworkTopology {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
}

/** Visual state of one asset/bank in the network. */
export interface BankVisualNode extends TopologyNode {
  /** Resting color: solvent green (interbank) or sector color (market). */
  baseColor: string;
  /** Marginal failure probability [0,1] driving color + size. */
  failure: number;
  /** Quantum walker probability [0,1] driving glow intensity. */
  walker: number;
  /** True while the shock wavefront is passing through this node. */
  pulsing: boolean;
}

/** Visual state of one exposure/correlation corridor. */
export interface BankVisualLink {
  source: number | string;
  target: number | string;
  exposure: number;
}

/** Structural view of the 3d-force-graph instance surface we use. */
interface FGInstance {
  width(px: number): FGInstance;
  height(px: number): FGInstance;
  backgroundColor(color: string): FGInstance;
  showNavInfo(show: boolean): FGInstance;
  graphData(data: { nodes: BankVisualNode[]; links: BankVisualLink[] }): FGInstance;
  graphData(): { nodes: BankVisualNode[]; links: BankVisualLink[] };
  refresh(): FGInstance;
  nodeLabel(accessor: (node: BankVisualNode) => string): FGInstance;
  nodeColor(accessor: (node: BankVisualNode) => string): FGInstance;
  nodeVal(accessor: (node: BankVisualNode) => number): FGInstance;
  nodeOpacity(opacity: number): FGInstance;
  linkColor(accessor: (link: BankVisualLink) => string): FGInstance;
  linkWidth(accessor: (link: BankVisualLink) => number): FGInstance;
  linkOpacity(opacity: number): FGInstance;
  linkDirectionalParticles(count: number): FGInstance;
  onNodeClick(handler: (node: BankVisualNode) => void): FGInstance;
  onRenderFramePre?(hook: () => void): FGInstance;
  d3Force(name: "charge" | "link" | "center"): { strength(value: number): unknown } | undefined;
  _destructor(): void;
}

interface Network3DCanvasProps {
  summary: NetworkTopology | null;
  shock: ContagionShockResponse | null;
  /** True while a quantum job is in flight (drives the scan overlay). */
  loading: boolean;
  /** "market" switches the legend + tooltip wording. */
  mode: "interbank" | "market";
  onNodeClick: (nodeId: number | string) => void;
}

const SOLVENT_COLOR = "#22c55e"; // green — solvent / unshocked
const LINK_BASE_OPACITY = 0.25;

/** Sector palette for the live-market view. */
const SECTOR_COLORS: Record<string, string> = {
  index: "#f59e0b",
  etf: "#a78bfa",
  bank: "#38bdf8",
  it: "#34d399",
  metal: "#f97316",
  energy: "#eab308",
  consumer: "#f472b6",
  stock: "#94a3b8",
};

/** Interpolate base color → crimson by failure probability. */
function failureColor(baseHex: string, failure: number): string {
  const clamped = Math.min(1, Math.max(0, failure));
  const parse = (hex: string): [number, number, number] => {
    const value = hex.replace("#", "");
    return [
      parseInt(value.slice(0, 2), 16),
      parseInt(value.slice(2, 4), 16),
      parseInt(value.slice(4, 6), 16),
    ];
  };
  const [r0, g0, b0] = parse(baseHex.startsWith("#") ? baseHex : "#22c55e");
  const [r1, g1, b1] = parse("#e11d48");
  const mix = (a: number, b: number) => Math.round(a + (b - a) * clamped);
  return `rgb(${mix(r0, r1)},${mix(g0, g1)},${mix(b0, b1)})`;
}

/** Construct the WebGL graph bound to our typed accessors. */
function createInstance(container: HTMLElement): FGInstanceWithRenderer {
  const ctor = ForceGraph3D as unknown as new (
    element: HTMLElement,
    config: Record<string, unknown>,
  ) => FGInstanceWithRenderer;
  // Transparent canvas so the quantum-streak artwork shows through the graph.
  return new ctor(container, { rendererConfig: { alpha: true, antialias: true } });
}

/** Attach an UnrealBloom composer so glowing nodes read as light, not dots. */
function attachBloom(instance: FGInstanceWithRenderer, container: HTMLElement) {
  try {
    const renderer = instance.renderer() as unknown as {
      domElement: HTMLCanvasElement;
      setRenderTarget: (target: unknown) => void;
      getPixelRatio: () => number;
      setSize: (w: number, h: number, updateStyle?: boolean) => void;
      render: (scene: THREE.Scene, camera: THREE.Camera) => void;
    };
    const scene = instance.scene();
    const camera = instance.camera();

    const composer = new EffectComposer(renderer as unknown as THREE.WebGLRenderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(
      new THREE.Vector2(container.clientWidth, container.clientHeight),
      0.9,   // strength
      0.55,  // radius
      0.12,  // threshold — only bright nodes bloom
    );
    composer.addPass(bloom);

    const resize = () => {
      composer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", resize);

    // Drive the render loop through the composer instead of the plain renderer.
    instance.onRenderFramePre?.(() => {
      composer.render();
    });
    return () => window.removeEventListener("resize", resize);
  } catch {
    // Bloom is cosmetic — never let it break the graph.
    return () => undefined;
  }
}

/**
 * WebGL force-directed network for both terminal modes.
 *
 * Interbank: green = solvent, crimson = defaulted; edge width ∝ exposure.
 * Market: nodes rest in sector colors, edges ∝ correlation exposure; the
 * CTQW walker probability renders as a glow so the distress wave is visible
 * flowing from the shocked ticker into correlated stocks and funds.
 */
export default function Network3DCanvas({
  summary,
  shock,
  loading,
  mode,
  onNodeClick,
}: Network3DCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<FGInstance | null>(null);
  const onNodeClickRef = useRef(onNodeClick);
  onNodeClickRef.current = onNodeClick;

  const shockLookup = useMemo(() => {
    const map = new Map<string, { failure: number; walker: number }>();
    if (!shock) return map;
    for (const row of shock.node_probabilities) {
      map.set(String(row.node_id), { failure: row.failure_marginal, walker: row.walker_probability });
    }
    return map;
  }, [shock]);

  // --- Instance lifecycle -------------------------------------------------
  useEffect(() => {
    if (!containerRef.current || graphRef.current) return;
    const container = containerRef.current;

    const instance = createInstance(container)
      .backgroundColor("rgba(0,0,0,0)") // transparent — app background shows through
      .showNavInfo(false)
      .nodeLabel((node) => {
        const kind = mode === "market" ? "vol buffer" : "capital";
        const sectorTag = node.sector ? `  [${node.sector}]` : "";
        return `${node.label}${sectorTag}\n${kind} ${node.capital.toFixed(1)}  ·  failure ${(node.failure * 100).toFixed(1)}%`;
      })
      .nodeColor((node) =>
        node.pulsing || node.failure > 0
          ? failureColor(node.baseColor, node.failure)
          : node.baseColor,
      )
      .nodeVal((node) => {
        const glow = 1 + node.walker * 26; // quantum walker glow
        return 1.4 + node.degree * 0.32 + node.capital * 0.05 + glow;
      })
      .nodeOpacity(0.92)
      .linkColor(() => "rgba(56,189,248,0.35)")
      .linkWidth((corridor) => 0.35 + Math.min(4.2, corridor.exposure * 1.6))
      .linkOpacity(LINK_BASE_OPACITY)
      .linkDirectionalParticles(0)
      .onNodeClick((node) => onNodeClickRef.current(node.id));

    instance.d3Force("charge")?.strength(-140);
    graphRef.current = instance;

    const detachBloom = attachBloom(instance as unknown as FGInstanceWithRenderer, container);

    const resize = () => {
      if (containerRef.current) {
        instance.width(containerRef.current.clientWidth).height(containerRef.current.clientHeight);
      }
    };
    resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      detachBloom();
      instance._destructor();
      graphRef.current = null;
    };
  }, [mode]);

  // --- Topology load ------------------------------------------------------
  useEffect(() => {
    const instance = graphRef.current;
    if (!instance || !summary) return;

    const nodes: BankVisualNode[] = summary.nodes.map((n) => ({
      ...n,
      baseColor:
        n.sector !== undefined
          ? SECTOR_COLORS[n.sector] ?? SECTOR_COLORS.stock
          : SOLVENT_COLOR,
      failure: 0,
      walker: 0,
      pulsing: false,
    }));
    const links: BankVisualLink[] = summary.edges.map((e) => ({
      source: e.source,
      target: e.target,
      exposure: e.exposure,
    }));

    instance.graphData({ nodes, links });
  }, [summary]);

  // --- Shock wave animation ------------------------------------------------
  useEffect(() => {
    const instance = graphRef.current;
    if (!instance || !summary) return;

    const data = instance.graphData();
    if (!data.nodes.length) return;

    if (!shock) {
      for (const node of data.nodes) {
        node.failure = 0;
        node.walker = 0;
        node.pulsing = false;
      }
      instance.refresh();
      return;
    }

    // Wavefront: nodes light up in order of walker probability mass — an
    // approximation of CTQW propagation order at measurement time t.
    const ranked = [...shock.node_probabilities].sort(
      (a, b) => b.walker_probability - a.walker_probability,
    );
    const waveIndex = new Map<string, number>();
    ranked.forEach((row, order) => waveIndex.set(String(row.node_id), order));

    const total = ranked.length;
    const timers: number[] = [];
    const applyWaveStep = (step: number) => {
      for (const node of data.nodes) {
        const stats = shockLookup.get(String(node.id));
        if (!stats) continue;
        const reached = (waveIndex.get(String(node.id)) ?? total) <= step;
        node.failure = reached ? stats.failure : 0;
        node.walker = reached ? stats.walker : 0;
        node.pulsing = reached;
      }
      instance.refresh();
    };

    // Stagger the wavefront across ~1.2 s for visible propagation.
    const stepDelay = Math.max(90, 1200 / total);
    ranked.forEach((_, step) => {
      timers.push(window.setTimeout(() => applyWaveStep(step), step * stepDelay));
    });
    timers.push(
      window.setTimeout(() => applyWaveStep(total - 1), stepDelay * total + 200),
    );

    return () => {
      for (const timer of timers) window.clearTimeout(timer);
    };
  }, [shock, shockLookup, summary]);

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full" />

      {/* Cinematic vignette over the graph volume */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 75% 65% at 50% 46%, transparent 55%, rgba(2,6,16,0.5) 100%)",
        }}
      />

      {loading && (
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="scan-sweep h-16 w-full bg-gradient-to-b from-transparent via-sky-400/20 to-transparent" />
        </div>
      )}

      <div className="absolute bottom-3 left-3 flex flex-col gap-1 rounded border border-sky-400/15 bg-[rgba(2,8,20,0.55)] px-3 py-2 text-[11px] text-slate-300 backdrop-blur-md">
        {mode === "market" ? (
          <>
            <span className="text-slate-400">live correlation network · yfinance</span>
            <span className="flex flex-wrap gap-x-3 gap-y-1">
              {Object.entries(SECTOR_COLORS).map(([sector, color]) => (
                <span key={sector} className="flex items-center gap-1.5">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
                  {sector}
                </span>
              ))}
            </span>
            <span className="text-slate-500">edge width ∝ ρ² correlation exposure</span>
          </>
        ) : (
          <>
            <span className="flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full bg-[#22c55e]" /> solvent
            </span>
            <span className="flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full bg-[#e11d48]" /> defaulted / shocked
            </span>
            <span className="text-slate-500">edge width ∝ bilateral exposure</span>
          </>
        )}
        <span className="text-slate-500">glow ∝ CTQW walker probability</span>
      </div>
    </div>
  );
}
