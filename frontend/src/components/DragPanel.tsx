/**
 * Drag-source panel — flat list of dataset cards and model cards.
 *
 * Datasets appear with a green dot if they have existing results (browse mode indicator).
 * Models appear with their type label.
 */
import { useDraggable } from '@dnd-kit/core'
import { CSS } from '@dnd-kit/utilities'
import type { DatasetOption, ExistingResult, ModelOption } from '../types'

// ── Single draggable card ─────────────────────────────────────────────────────

function DraggableCard({
  id,
  label,
  hasResult,
  sub,
}: {
  id: string
  label: string
  hasResult?: boolean
  sub?: string
}) {
  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({ id })

  return (
    <div
      ref={setNodeRef}
      {...listeners}
      {...attributes}
      style={{ transform: CSS.Translate.toString(transform) }}
      className={[
        'cursor-grab select-none rounded-lg border px-3 py-2 text-sm',
        'bg-white shadow-sm transition-shadow',
        isDragging
          ? 'opacity-50 shadow-lg ring-2 ring-indigo-400'
          : 'border-gray-200 hover:border-indigo-300 hover:shadow',
      ].join(' ')}
    >
      <div className="flex items-center gap-1.5">
        {hasResult !== undefined && (
          <span
            title={hasResult ? 'Has existing results' : 'No results yet'}
            className={[
              'h-1.5 w-1.5 flex-shrink-0 rounded-full',
              hasResult ? 'bg-emerald-400' : 'bg-gray-200',
            ].join(' ')}
          />
        )}
        <span className="font-medium text-gray-800">{label}</span>
      </div>
      {sub && <div className="mt-0.5 text-xs text-gray-400">{sub}</div>}
    </div>
  )
}

// ── Section group ─────────────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-4">
      <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-gray-400">
        {title}
      </h3>
      <div className="flex flex-col gap-1.5">{children}</div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

interface Props {
  datasets: DatasetOption[]
  models: ModelOption[]
  existingResults: ExistingResult[]
  /** In browse mode, highlight only models that have results for this dataset */
  filterDatasetId?: string | null
}

export function DragPanel({ datasets, models, existingResults, filterDatasetId }: Props) {
  // Count how many model results exist per dataset
  const resultCountByDataset = existingResults.reduce<Record<string, number>>((acc, r) => {
    acc[r.dataset] = (acc[r.dataset] ?? 0) + 1
    return acc
  }, {})

  // In browse mode with a dataset selected: which models have results for it?
  const modelsWithResultsForDataset: Set<string> | null = filterDatasetId
    ? new Set(
        existingResults
          .filter((r) => r.dataset === filterDatasetId)
          .map((r) => {
            // r.model is like "deepseek-v3.2" or "hipporag-deepseek-v3.2"
            // Match against model IDs: "openai", "hipporag", "amem", "memoryos"
            if (r.model.startsWith('hipporag')) return 'hipporag'
            if (r.model.startsWith('memoryos')) return 'memoryos'
            if (r.model.startsWith('amem')) return 'amem'
            return 'openai'
          }),
      )
    : null

  return (
    <aside className="flex h-full w-48 flex-shrink-0 flex-col overflow-y-auto border-r border-gray-100 bg-gray-50 p-3">
      {datasets.length === 0 && models.length === 0 && (
        <div className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700">
          Backend offline — start the FastAPI server.
        </div>
      )}

      {datasets.length > 0 && (
        <Section title="Datasets">
          {datasets.map((d) => {
            const count = resultCountByDataset[d.id] ?? 0
            return (
              <DraggableCard
                key={d.id}
                id={`dataset::${d.id}`}
                label={d.label}
                sub={count > 0 ? `${count} result${count > 1 ? 's' : ''}` : undefined}
                hasResult={count > 0}
              />
            )
          })}
        </Section>
      )}

      {models.length > 0 && (
        <Section title="Agents / Models">
          {models.map((m) => {
            const available = modelsWithResultsForDataset
              ? modelsWithResultsForDataset.has(m.id)
              : undefined
            return (
              <DraggableCard
                key={m.id}
                id={`model::${m.id}`}
                label={m.label}
                hasResult={available}
                sub={
                  modelsWithResultsForDataset && !available
                    ? 'no results for this dataset'
                    : undefined
                }
              />
            )
          })}
        </Section>
      )}
    </aside>
  )
}
