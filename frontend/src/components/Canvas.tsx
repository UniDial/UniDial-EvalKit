import type { DatasetOption, ExistingResult, ModelOption } from '../types'

interface Props {
  datasets: DatasetOption[]
  models: ModelOption[]
  existingResults: ExistingResult[]
  selectedDataset: DatasetOption | null
  selectedModel: ModelOption | null
  onSelectDataset: (d: DatasetOption | null) => void
  onSelectModel: (m: ModelOption | null) => void
  isBrowseMode: boolean
  isRunning: boolean
  onRun: () => void
  onBrowse: () => void
}

export function Canvas({
  datasets,
  models,
  existingResults,
  selectedDataset,
  selectedModel,
  onSelectDataset,
  onSelectModel,
  isBrowseMode,
  isRunning,
  onRun,
  onBrowse,
}: Props) {
  const hasSelection = !!selectedDataset && !!selectedModel

  // In browse mode, filter agents to those with results for selected dataset
  const filteredModels =
    isBrowseMode && selectedDataset
      ? models.filter((m) => {
          const agentKey = m.id === 'openai' ? '_base' : m.id
          return existingResults.some(
            (r) => r.agent === agentKey && r.dataset === selectedDataset.id,
          )
        })
      : models

  return (
    <div className="flex flex-shrink-0 items-end gap-3 border-b border-gray-100 bg-white px-5 py-4">
      {/* Dataset dropdown */}
      <div className="flex-1">
        <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wider text-gray-400">
          Dataset
        </label>
        <select
          value={selectedDataset?.id ?? ''}
          onChange={(e) => {
            const d = datasets.find((x) => x.id === e.target.value) ?? null
            onSelectDataset(d)
            onSelectModel(null)
          }}
          className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-700 shadow-sm transition focus:border-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-300"
        >
          <option value="">Select dataset…</option>
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>
              {d.label}
            </option>
          ))}
        </select>
      </div>

      {/* Agent / Model dropdown */}
      <div className="flex-1">
        <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wider text-gray-400">
          Agent / Model
        </label>
        <select
          value={selectedModel?.id ?? ''}
          onChange={(e) => {
            const m = models.find((x) => x.id === e.target.value) ?? null
            onSelectModel(m)
          }}
          className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-700 shadow-sm transition focus:border-violet-400 focus:outline-none focus:ring-1 focus:ring-violet-300"
        >
          <option value="">Select agent…</option>
          {filteredModels.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      {/* Action button */}
      <div className="flex-shrink-0 pb-0.5">
        {isBrowseMode ? (
          <button
            onClick={onBrowse}
            disabled={!hasSelection}
            className={[
              'rounded-lg px-5 py-2 text-sm font-semibold transition-colors',
              hasSelection
                ? 'bg-amber-500 text-white hover:bg-amber-600 active:bg-amber-700'
                : 'cursor-not-allowed bg-gray-200 text-gray-400',
            ].join(' ')}
          >
            View Results
          </button>
        ) : (
          <button
            onClick={onRun}
            disabled={!hasSelection || isRunning}
            className={[
              'rounded-lg px-5 py-2 text-sm font-semibold text-white transition-colors',
              hasSelection && !isRunning
                ? 'bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800'
                : 'cursor-not-allowed bg-gray-300',
            ].join(' ')}
          >
            {isRunning ? (
              <span className="flex items-center gap-2">
                <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                Running…
              </span>
            ) : (
              'Start Evaluation'
            )}
          </button>
        )}
      </div>
    </div>
  )
}
