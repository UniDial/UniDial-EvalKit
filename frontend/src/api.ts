/**
 * API layer — all backend calls go through here.
 * WebSocket URL uses ws:// directly (bypasses Vite HTTP proxy).
 */
import type {
  DatasetOption,
  DialogDetail,
  ExistingResult,
  ModelOption,
  ProgressSnapshot,
  TaskParams,
} from './types'

const BASE = '/api'
const WS_URL = 'ws://localhost:8000/api/tasks/ws'

// ── Catalog ───────────────────────────────────────────────────────────────────

export async function fetchOptions(): Promise<{
  datasets: DatasetOption[]
  models: ModelOption[]
}> {
  const res = await fetch(`${BASE}/options`)
  if (!res.ok) throw new Error('Failed to fetch options')
  return res.json()
}

// ── Browse existing results ───────────────────────────────────────────────────

export async function fetchExistingResults(): Promise<ExistingResult[]> {
  const res = await fetch(`${BASE}/results`)
  if (!res.ok) return []
  return res.json()
}

export async function fetchBrowseSummary(
  agent: string,
  dataset: string,
  model: string,
): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE}/browse/${agent}/${dataset}/${model}/summary`)
  if (!res.ok) throw new Error('Summary not available')
  return res.json()
}

export async function fetchBrowseDialogs(
  agent: string,
  dataset: string,
  model: string,
): Promise<string[]> {
  const res = await fetch(`${BASE}/browse/${agent}/${dataset}/${model}/dialogs`)
  if (!res.ok) return []
  return res.json()
}

export async function fetchBrowseDialog(
  agent: string,
  dataset: string,
  model: string,
  dialogId: string,
): Promise<DialogDetail> {
  const res = await fetch(`${BASE}/browse/${agent}/${dataset}/${model}/dialogs/${dialogId}`)
  if (!res.ok) throw new Error('Dialog not available')
  return res.json()
}

// ── Task lifecycle ────────────────────────────────────────────────────────────

export async function startTask(
  dataset: string,
  modelType: string,
  params: TaskParams,
): Promise<void> {
  const res = await fetch(`${BASE}/tasks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset, model_type: modelType, ...params }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Failed to start task')
  }
}

export async function fetchResult(): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE}/tasks/result`)
  if (!res.ok) throw new Error('Result not available')
  return res.json()
}

export async function fetchTaskPreview(): Promise<Record<string, unknown> | null> {
  const res = await fetch(`${BASE}/tasks/preview`)
  if (!res.ok) return null
  return res.json()
}

export async function fetchTaskDialogList(): Promise<
  Array<{ dialog_id: number; dialog_labels: Record<string, string> }>
> {
  const res = await fetch(`${BASE}/tasks/dialog_list`)
  if (!res.ok) return []
  return res.json()
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

/**
 * Opens a WebSocket to the progress endpoint and calls the provided callbacks.
 * Returns a cleanup function to close the socket.
 */
export function openProgressSocket(
  onSnapshot: (snap: ProgressSnapshot) => void,
  onClose: () => void,
): () => void {
  const ws = new WebSocket(WS_URL)

  ws.onmessage = (e) => {
    try {
      onSnapshot(JSON.parse(e.data) as ProgressSnapshot)
    } catch {
      // ignore malformed frames
    }
  }

  ws.onclose = onClose
  ws.onerror = () => ws.close()

  return () => ws.close()
}
