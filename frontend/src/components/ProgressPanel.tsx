/**
 * Progress panel — shown while a task is running.
 *
 * During generation/evaluation phases, shows a split view:
 *   Left  — compact list of ALL dialogs processed so far (latest auto-selected)
 *   Right — full DialogViewer for the selected dialog
 *
 * During data/aggregation phases, shows only phase strip + progress bar.
 */
import type { Phase, ProgressSnapshot } from '../types'
import type { DialogDetail } from '../types'
import { DialogViewer } from './DialogViewer'

// ── Phase strip ───────────────────────────────────────────────────────────────

const PHASES: { id: Phase; label: string }[] = [
  { id: 'data', label: 'Data' },
  { id: 'generation', label: 'Generation' },
  { id: 'evaluation', label: 'Evaluation' },
  { id: 'aggregation', label: 'Aggregation' },
]
const PHASE_ORDER: Phase[] = ['idle', 'data', 'generation', 'evaluation', 'aggregation', 'done']

function PhaseStrip({ current }: { current: Phase }) {
  const currentIdx = PHASE_ORDER.indexOf(current)
  return (
    <div className="flex items-center gap-1.5">
      {PHASES.map((p, i) => {
        const phaseIdx = PHASE_ORDER.indexOf(p.id)
        const done = phaseIdx < currentIdx
        const active = p.id === current
        return (
          <div key={p.id} className="flex items-center gap-1.5">
            {i > 0 && <div className={`h-px w-5 ${done ? 'bg-indigo-400' : 'bg-gray-200'}`} />}
            <span
              className={[
                'rounded-full px-3 py-0.5 text-xs font-medium',
                active ? 'bg-indigo-600 text-white'
                  : done ? 'bg-indigo-100 text-indigo-700'
                  : 'bg-gray-100 text-gray-400',
              ].join(' ')}
            >
              {done ? '✓ ' : ''}{p.label}
            </span>
          </div>
        )
      })}
    </div>
  )
}

// ── Progress bar ──────────────────────────────────────────────────────────────

function ProgressBar({ completed, total, phase }: { completed: number; total: number; phase: Phase }) {
  const pct = total > 0 ? Math.min(100, Math.round((completed / total) * 100)) : 0
  const isIndeterminate = phase === 'data' || phase === 'aggregation'
  return (
    <div className="flex items-center gap-3">
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-gray-100">
        {isIndeterminate ? (
          <div className="h-full w-1/3 animate-pulse rounded-full bg-indigo-400" />
        ) : (
          <div
            className="h-full rounded-full bg-indigo-500 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
      <span className="min-w-[6rem] text-right text-xs text-gray-500">
        {isIndeterminate
          ? phase === 'data' ? 'Loading data…' : 'Aggregating…'
          : `${completed} / ${total > 0 ? total : '?'} (${pct}%)`}
      </span>
    </div>
  )
}

// ── Compact dialog list item ──────────────────────────────────────────────────

interface DialogSummary {
  dialog_id: number
  dialog_labels: Record<string, string>
}

function DialogListItem({
  item,
  isSelected,
  isLatest,
  onClick,
}: {
  item: DialogSummary
  isSelected: boolean
  isLatest: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={[
        'w-full rounded-lg border px-3 py-2 text-left transition-colors',
        isSelected
          ? 'border-indigo-400 bg-indigo-50'
          : 'border-transparent bg-white hover:border-gray-200 hover:bg-gray-50',
      ].join(' ')}
    >
      <div className="flex items-center gap-1.5">
        <span className="font-mono text-xs font-semibold text-gray-600">
          #{item.dialog_id}
        </span>
        {isLatest && (
          <span className="rounded-full bg-indigo-600 px-1.5 py-0.5 text-[9px] font-semibold text-white">
            latest
          </span>
        )}
      </div>
      {Object.keys(item.dialog_labels).length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {Object.values(item.dialog_labels).map((v, i) => (
            <span key={i} className="text-[10px] text-gray-400">
              {v}
            </span>
          ))}
        </div>
      )}
    </button>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

interface Props {
  snapshot: ProgressSnapshot
  /** Compact list of all dialogs generated so far */
  dialogList: DialogSummary[]
  /** Full detail of the selected dialog (null while loading) */
  selectedDialog: DialogDetail | null
  selectedDialogId: number | null
  onSelectDialog: (id: number) => void
}

export function ProgressPanel({
  snapshot,
  dialogList,
  selectedDialog,
  selectedDialogId,
  onSelectDialog,
}: Props) {
  const showDialogView =
    snapshot.phase === 'generation' || snapshot.phase === 'evaluation'
  const latestId =
    dialogList.length > 0 ? dialogList[dialogList.length - 1].dialog_id : null

  return (
    <div className="flex flex-1 flex-col gap-0 min-h-0">
      {/* Phase strip + progress bar */}
      <div className="flex-shrink-0 border-b border-gray-100 bg-white px-5 py-4">
        <PhaseStrip current={snapshot.phase} />
        <div className="mt-3">
          <ProgressBar
            completed={snapshot.completed}
            total={snapshot.total}
            phase={snapshot.phase}
          />
        </div>
        {snapshot.error && (
          <div className="mt-3 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
            <strong>Error:</strong> {snapshot.error}
          </div>
        )}
      </div>

      {/* Split-view dialog browser (generation / evaluation phases) */}
      {showDialogView && (
        <div className="flex flex-1 min-h-0">
          {/* Left: dialog list */}
          <div className="flex w-44 flex-shrink-0 flex-col border-r border-gray-100 bg-gray-50">
            <div className="border-b border-gray-100 px-3 py-2">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">
                {dialogList.length > 0
                  ? `${dialogList.length} dialog${dialogList.length > 1 ? 's' : ''} done`
                  : 'Waiting…'}
              </span>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
              {dialogList.length === 0 ? (
                <div className="flex h-20 items-center justify-center">
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-indigo-300 border-t-transparent" />
                </div>
              ) : (
                // Show all dialogs, newest at bottom
                [...dialogList].reverse().map((item) => (
                  <DialogListItem
                    key={item.dialog_id}
                    item={item}
                    isSelected={selectedDialogId === item.dialog_id}
                    isLatest={item.dialog_id === latestId}
                    onClick={() => onSelectDialog(item.dialog_id)}
                  />
                ))
              )}
            </div>
          </div>

          {/* Right: selected dialog viewer */}
          <div className="flex-1 overflow-y-auto p-4">
            {selectedDialog ? (
              <DialogViewer dialog={selectedDialog} immediateStart />
            ) : dialogList.length > 0 ? (
              <div className="flex h-32 items-center justify-center text-sm text-gray-400">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-indigo-300 border-t-transparent mr-2" />
                Loading dialog…
              </div>
            ) : (
              <div className="flex h-32 items-center justify-center text-sm text-gray-400">
                Dialogs will appear here as they complete…
              </div>
            )}
          </div>
        </div>
      )}

      {/* Data / Aggregation phase: simple waiting message */}
      {!showDialogView && (
        <div className="flex flex-1 items-center justify-center text-sm text-gray-400 p-8">
          {snapshot.phase === 'data' && 'Preparing dataset…'}
          {snapshot.phase === 'aggregation' && 'Aggregating results…'}
          {snapshot.phase === 'idle' && 'Starting…'}
        </div>
      )}
    </div>
  )
}
