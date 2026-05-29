// ── Catalog items (from GET /api/options) ────────────────────────────────────

export interface DatasetOption {
  id: string
  label: string
  category: string
}

export interface ModelOption {
  id: string
  label: string
}

// ── Existing results (from GET /api/results) ──────────────────────────────────

export interface ExistingResult {
  agent: string
  agent_label: string
  dataset: string
  dataset_label: string
  category: string
  model: string
  base_model: string
  has_summary: boolean
}

// ── Task state ────────────────────────────────────────────────────────────────

export type TaskStatus = 'idle' | 'running' | 'completed' | 'failed'
export type Phase = 'idle' | 'data' | 'generation' | 'evaluation' | 'aggregation' | 'done'

export interface ProgressSnapshot {
  status: TaskStatus
  phase: Phase
  total: number
  completed: number
  logs: string[]
  error: string | null
}

// ── Task creation request (mirrors backend TaskCreateRequest) ─────────────────

export interface TaskParams {
  // Paths
  raw_data_dir: string
  processed_data_dir: string
  output_dir: string
  // Generation model
  model_name: string
  api_key: string
  base_url: string
  temperature: number
  max_tokens: number
  // Execution
  parallel: number
  do_generation: boolean
  do_evaluation: boolean
  // Judge model
  judge_model_type: string
  judge_model_name: string
  // Aggregation
  agg_by_metric: boolean
  agg_turn_stat: string
  agg_dialog_stat: string
  agg_dataset_level: string
}

export const DEFAULT_PARAMS: TaskParams = {
  raw_data_dir: './raw_data',
  processed_data_dir: './data',
  output_dir: './output',
  model_name: '',
  api_key: '',
  base_url: '',
  temperature: 0.7,
  max_tokens: 1024,
  parallel: 4,
  do_generation: true,
  do_evaluation: true,
  judge_model_type: 'openai',
  judge_model_name: 'gpt-4.1-2025-04-14',
  agg_by_metric: false,
  agg_turn_stat: 'mean',
  agg_dialog_stat: 'min',
  agg_dataset_level: 'dialog',
}

// ── Dialog detail (from GET /api/browse/.../dialogs/{id}) ─────────────────────

export interface DialogTurnItem {
  turn_id: number
  role: 'user' | 'assistant'
  content: string
  reference?: string | null
  eval_config: {
    do_eval: boolean
    metrics: Array<{ class_name: string; args: Record<string, unknown> }>
  }
  turn_labels: Record<string, string>
}

export interface TurnEval {
  dialog_id: number
  turn_id: number
  metric_name: string
  score: number
  details: {
    score: number
    rationale?: string
    raw_output?: string
  }
  dialog_labels: Record<string, string>
  turn_labels: Record<string, string>
}

export interface AgentLog {
  metadata: {
    dataset: string
    dialog_id: number
    turn_index: number
    query: string
    timestamp: number
    latency_seconds: number
  }
  memory_update: {
    new_raw_inputs: string[]
    chunked_documents: Array<{ content: string; meta_info: Record<string, unknown> }>
    extracted_knowledge: Record<string, unknown>
  }
  retrieval: {
    search_queries: string[]
    retrieved_contexts: Array<{ source: string; content: string; score: number | null }>
  }
  generation: { generated_response: string }
  system_logs: unknown[]
}

export interface DialogDetail {
  dialog_id: number
  dialog_labels: Record<string, string>
  dialog_turns: DialogTurnItem[]
  eval_details: TurnEval[]
  agent_logs: AgentLog[]
}

// ── Parsed summary ────────────────────────────────────────────────────────────

export interface LabelEntry {
  value: string
  count: number
  mean: number
  min: number
  max: number
}

export interface LabelDimension {
  dim: string
  entries: LabelEntry[]
}

export interface ParsedSummary {
  global: { count: number; mean: number; min: number; max: number }
  config: { turn_stat: string; dialog_stat: string; dataset_level: string }
  dialogDims: LabelDimension[]
  turnDims: LabelDimension[]
}

// ── Summary parsing ───────────────────────────────────────────────────────────

function parseFlatLabels(flat: Record<string, number>): LabelDimension[] {
  const dimMap = new Map<string, Map<string, Partial<LabelEntry> & { value: string }>>()

  for (const [key, val] of Object.entries(flat)) {
    // Format: "dim_name:('category_value', 'stat_name')"
    const match = key.match(/^(.+?):\('(.+?)',\s*'(.+?)'\)$/)
    if (!match) continue
    const [, dim, value, stat] = match

    if (!dimMap.has(dim)) dimMap.set(dim, new Map())
    const vMap = dimMap.get(dim)!
    if (!vMap.has(value)) vMap.set(value, { value })
    const entry = vMap.get(value)!
    ;(entry as Record<string, unknown>)[stat] = val
  }

  return Array.from(dimMap.entries()).map(([dim, vMap]) => ({
    dim,
    entries: Array.from(vMap.values())
      .map((e) => ({
        value: e.value,
        count: (e.count as number) ?? 0,
        mean: (e.mean as number) ?? 0,
        min: (e.min as number) ?? 0,
        max: (e.max as number) ?? 0,
      }))
      .sort((a, b) => b.mean - a.mean),
  }))
}

export function parseSummary(raw: Record<string, unknown>): ParsedSummary {
  const summary = (raw.summary ?? raw) as Record<string, unknown>
  const global = (summary.global as ParsedSummary['global']) ?? {
    count: 0, mean: 0, min: 0, max: 0,
  }
  const config = (summary.config as ParsedSummary['config']) ?? {
    turn_stat: 'mean', dialog_stat: 'min', dataset_level: 'dialog',
  }
  const byLabel = (summary.by_label as Record<string, Record<string, number>>) ?? {}

  return {
    global,
    config,
    dialogDims: parseFlatLabels(byLabel.dialog ?? {}),
    turnDims: parseFlatLabels(byLabel.turn ?? {}),
  }
}

// ── Score helpers ─────────────────────────────────────────────────────────────

export function scoreColor(score: number): string {
  if (score >= 0.8) return '#10b981'
  if (score >= 0.6) return '#6366f1'
  if (score >= 0.4) return '#f59e0b'
  return '#f43f5e'
}

export function scoreBg(score: number): string {
  if (score >= 0.8) return 'bg-emerald-50 border-emerald-200 text-emerald-800'
  if (score >= 0.6) return 'bg-indigo-50 border-indigo-200 text-indigo-800'
  if (score >= 0.4) return 'bg-amber-50 border-amber-200 text-amber-800'
  return 'bg-rose-50 border-rose-200 text-rose-800'
}
