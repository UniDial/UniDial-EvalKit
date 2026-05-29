import { useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { DialogDetail, LabelDimension } from '../types'
import { parseSummary, scoreColor } from '../types'
import { DialogViewer } from './DialogViewer'

// ── Chart height constant — unified across all sections ───────────────────────

const CHART_H = 220

// ── View type ─────────────────────────────────────────────────────────────────

type ViewType = 'bar' | 'radar' | 'table'

// ── Hero Score Card ───────────────────────────────────────────────────────────

function HeroCard({
  mean,
  min,
  max,
  count,
}: {
  mean: number
  min: number
  max: number
  count: number
}) {
  const pct = Math.round(mean * 100)
  const color = scoreColor(mean)

  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-gray-100 bg-gradient-to-br from-indigo-50 via-white to-violet-50 px-6 py-6 shadow-sm">
      {/* Circular progress ring */}
      <div className="relative mb-3 h-24 w-24">
        <svg className="h-full w-full -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="42" fill="none" stroke="#e5e7eb" strokeWidth="10" />
          <circle
            cx="50"
            cy="50"
            r="42"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={`${2 * Math.PI * 42}`}
            strokeDashoffset={`${2 * Math.PI * 42 * (1 - mean)}`}
            className="transition-all duration-700"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold" style={{ color }}>
            {pct}%
          </span>
        </div>
      </div>

      <span className="text-sm font-semibold text-gray-700">Overall Score</span>
      <span className="mt-1 text-xs text-gray-400">{count.toLocaleString()} dialogs</span>

      {/* Best / Avg / Worst */}
      <div className="mt-4 grid w-full grid-cols-3 gap-2">
        {[
          { label: 'Best', value: max },
          { label: 'Avg', value: mean },
          { label: 'Worst', value: min },
        ].map((stat) => (
          <div key={stat.label} className="rounded-lg bg-white/70 py-2 text-center">
            <div className="text-sm font-bold" style={{ color: scoreColor(stat.value) }}>
              {(stat.value * 100).toFixed(1)}%
            </div>
            <div className="text-[10px] text-gray-400">{stat.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Dimension charts ──────────────────────────────────────────────────────────

const DIM_LABELS: Record<string, string> = {
  task_type: 'Task Type',
  task_subtype: 'Task Subtype',
  task_hierarchy: 'Task (Type › Subtype)',
  language: 'Language',
  cur_task_type: 'Turn Task Type',
}

function DimBarChart({ dim }: { dim: LabelDimension }) {
  const data = dim.entries.map((e) => ({
    name: e.value.length > 18 ? e.value.slice(0, 16) + '…' : e.value,
    fullName: e.value,
    score: Math.round(e.mean * 1000) / 1000,
    count: e.count,
  }))

  return (
    <ResponsiveContainer width="100%" height={CHART_H}>
      <BarChart data={data} layout="vertical" margin={{ left: 2, right: 14, top: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.5} />
        <XAxis
          type="number"
          domain={[0, 1]}
          tickFormatter={(v) => `${Math.round(v * 100)}%`}
          tick={{ fontSize: 10, fill: '#9ca3af' }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={85}
          tick={{ fontSize: 10, fill: '#6b7280' }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          formatter={(v: number) => [`${(Number(v) * 100).toFixed(1)}%`]}
          labelFormatter={() => ''}
          contentStyle={{
            border: '1px solid #e5e7eb',
            borderRadius: '8px',
            fontSize: '12px',
            padding: '6px 10px',
          }}
        />
        <Bar dataKey="score" radius={[0, 5, 5, 0]} maxBarSize={22}>
          {data.map((entry, i) => (
            <Cell key={i} fill={scoreColor(entry.score)} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function DimRadarChart({ dim }: { dim: LabelDimension }) {
  const data = dim.entries.map((e) => ({
    subject: e.value.length > 20 ? e.value.slice(0, 18) + '…' : e.value,
    score: Math.round(e.mean * 100),
    fullName: e.value,
  }))

  return (
    <ResponsiveContainer width="100%" height={CHART_H}>
      <RadarChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
        <PolarGrid stroke="#e5e7eb" />
        <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: '#6b7280' }} />
        <Radar
          dataKey="score"
          stroke="#6366f1"
          fill="#6366f1"
          fillOpacity={0.2}
          strokeWidth={2}
        />
        <Tooltip
          formatter={(v: number) => [`${v}%`]}
          contentStyle={{
            border: '1px solid #e5e7eb',
            borderRadius: '8px',
            fontSize: '12px',
          }}
        />
      </RadarChart>
    </ResponsiveContainer>
  )
}

function DimTableView({ dim }: { dim: LabelDimension }) {
  return (
    <div className="overflow-y-auto" style={{ maxHeight: `${CHART_H}px` }}>
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-white">
          <tr className="border-b text-left text-[10px] font-semibold uppercase tracking-wider text-gray-400">
            <th className="pb-1.5 pr-2">Category</th>
            <th className="pb-1.5 pr-2 text-right">Score</th>
            <th className="pb-1.5 pr-2 text-right">Best</th>
            <th className="pb-1.5 text-right">n</th>
          </tr>
        </thead>
        <tbody>
          {dim.entries.map((e) => (
            <tr key={e.value} className="border-b last:border-0 hover:bg-gray-50">
              <td className="py-1.5 pr-2 font-medium text-gray-700">{e.value}</td>
              <td className="py-1.5 pr-2 text-right font-bold" style={{ color: scoreColor(e.mean) }}>
                {(e.mean * 100).toFixed(1)}%
              </td>
              <td className="py-1.5 pr-2 text-right text-gray-500">
                {(e.max * 100).toFixed(1)}%
              </td>
              <td className="py-1.5 text-right text-gray-500">{e.count.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function DimSection({ dim, viewType }: { dim: LabelDimension; viewType: ViewType }) {
  const label = DIM_LABELS[dim.dim] ?? dim.dim.replace(/_/g, ' ')
  const canRadar = dim.entries.length >= 3

  return (
    <div className="rounded-xl border border-gray-100 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">By {label}</h3>
        <span className="text-xs text-gray-400">{dim.entries.length} categories</span>
      </div>

      {viewType === 'bar' && <DimBarChart dim={dim} />}
      {viewType === 'radar' &&
        (canRadar ? <DimRadarChart dim={dim} /> : <DimBarChart dim={dim} />)}
      {viewType === 'table' && <DimTableView dim={dim} />}
    </div>
  )
}

// ── View type selector ────────────────────────────────────────────────────────

const VIEW_OPTIONS: { type: ViewType; label: string }[] = [
  { type: 'bar', label: 'Bar Chart' },
  { type: 'radar', label: 'Radar Chart' },
  { type: 'table', label: 'Table' },
]

function ViewSelector({
  value,
  onChange,
}: {
  value: ViewType
  onChange: (v: ViewType) => void
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400">View:</span>
      <div className="flex rounded-lg border border-gray-200 bg-gray-50 p-0.5 gap-0.5">
        {VIEW_OPTIONS.map(({ type, label }) => (
          <button
            key={type}
            onClick={() => onChange(type)}
            className={[
              'rounded-md px-3 py-1 text-xs font-medium transition-colors',
              value === type
                ? 'bg-white text-indigo-700 shadow-sm border border-gray-200'
                : 'text-gray-500 hover:text-gray-700',
            ].join(' ')}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Browse mode: result selector grid ─────────────────────────────────────────

export interface BrowseEntry {
  agent: string
  agent_label: string
  dataset: string
  dataset_label: string
  category: string
  model: string
  base_model: string
  has_summary: boolean
}

export function BrowseGrid({
  results,
  onSelect,
}: {
  results: BrowseEntry[]
  onSelect: (agent: string, dataset: string, model: string) => void
}) {
  const byDataset = results.reduce<Record<string, { label: string; items: BrowseEntry[] }>>(
    (acc, r) => {
      if (!acc[r.dataset]) acc[r.dataset] = { label: r.dataset_label, items: [] }
      acc[r.dataset].items.push(r)
      return acc
    },
    {},
  )

  if (results.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-12 text-center">
        <div className="text-4xl text-gray-200">📊</div>
        <p className="text-sm text-gray-500">No evaluation results found.</p>
        <p className="text-xs text-gray-400">
          Run an evaluation, or check that your{' '}
          <code className="rounded bg-gray-100 px-1">output/</code> directory has results.
        </p>
      </div>
    )
  }

  return (
    <div className="p-6">
      <h2 className="mb-4 text-base font-semibold text-gray-800">Available Results</h2>
      <div className="space-y-4">
        {Object.entries(byDataset).map(([_dataset, { label, items }]) => (
          <div key={_dataset} className="rounded-xl border border-gray-100 bg-white p-4 shadow-sm">
            <h3 className="mb-3 text-sm font-semibold text-gray-800">{label}</h3>
            <div className="flex flex-wrap gap-2">
              {items.map((r) => (
                <button
                  key={`${r.agent}/${r.model}`}
                  onClick={() => onSelect(r.agent, r.dataset, r.model)}
                  title={r.has_summary ? undefined : 'Generation only — no evaluation summary yet'}
                  className={[
                    'flex items-center gap-1.5 rounded-lg border px-3 py-2 text-sm transition-all hover:shadow-sm',
                    'border-indigo-200 bg-indigo-50 hover:border-indigo-400',
                    !r.has_summary ? 'opacity-60' : '',
                  ].join(' ')}
                >
                  <span className="rounded bg-indigo-600 px-1.5 py-0.5 text-[10px] font-bold text-white">
                    {r.agent_label}
                  </span>
                  <span className="text-gray-700">{r.base_model}</span>
                  {!r.has_summary && (
                    <span className="rounded bg-gray-100 px-1 py-0.5 text-[10px] text-gray-400">
                      dialogs only
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Dialog browser with pagination ────────────────────────────────────────────

const PAGE_SIZE = 20

function DialogBrowser({
  dialogIds,
  selectedId,
  onSelect,
}: {
  dialogIds: string[]
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  const [page, setPage] = useState(0)

  if (dialogIds.length === 0) return null

  const totalPages = Math.ceil(dialogIds.length / PAGE_SIZE)
  const pageIds = dialogIds.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  return (
    <div className="rounded-xl border border-gray-100 bg-white p-5 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">Dialog Browser</h3>
        <span className="text-xs text-gray-400">{dialogIds.length.toLocaleString()} dialogs</span>
      </div>

      <div
        className="grid gap-1.5"
        style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(52px, 1fr))' }}
      >
        {pageIds.map((id) => (
          <button
            key={id}
            onClick={() => onSelect(id)}
            className={[
              'rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors',
              selectedId === id
                ? 'border-indigo-400 bg-indigo-600 text-white'
                : 'border-gray-200 bg-white text-gray-600 hover:border-indigo-300 hover:text-indigo-700',
            ].join(' ')}
          >
            {id}
          </button>
        ))}
      </div>

      {totalPages > 1 && (
        <div className="mt-3 flex items-center justify-between">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="rounded px-3 py-1 text-xs font-medium text-gray-500 hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-40"
          >
            ← Prev
          </button>
          <span className="text-xs text-gray-400">
            {page + 1} / {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page === totalPages - 1}
            className="rounded px-3 py-1 text-xs font-medium text-gray-500 hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

interface Props {
  result: Record<string, unknown>
  dialogIds?: string[]
  selectedDialogId?: string | null
  dialogDetail?: DialogDetail | null
  onSelectDialog?: (id: string) => void
}

export function ResultsPanel({
  result,
  dialogIds = [],
  selectedDialogId = null,
  dialogDetail = null,
  onSelectDialog,
}: Props) {
  const summary = parseSummary(result)
  const [showHierarchy, setShowHierarchy] = useState(false)
  const [viewType, setViewType] = useState<ViewType>('bar')

  const primaryDims = summary.dialogDims.filter((d) => d.dim !== 'task_hierarchy')
  const hierarchyDims = summary.dialogDims.filter((d) => d.dim === 'task_hierarchy')
  const hasDims = primaryDims.length > 0 || summary.turnDims.length > 0 || hierarchyDims.length > 0

  return (
    <div className="flex flex-col gap-5 p-5">
      {/* ── Top row: hero card + dialog-level breakdown ── */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:self-start">
          <HeroCard
            mean={summary.global.mean}
            min={summary.global.min}
            max={summary.global.max}
            count={summary.global.count}
          />
        </div>

        <div className="flex flex-col gap-3 lg:col-span-2">
          {/* View selector */}
          {hasDims && (
            <ViewSelector value={viewType} onChange={setViewType} />
          )}

          {primaryDims.length > 0 ? (
            <>
              {primaryDims.map((dim) => (
                <DimSection key={dim.dim} dim={dim} viewType={viewType} />
              ))}
            </>
          ) : (
            <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-gray-200 text-xs text-gray-400">
              No category breakdown available
            </div>
          )}
        </div>
      </div>

      {/* ── Hierarchy (collapsible) ── */}
      {hierarchyDims.length > 0 && (
        <div>
          <button
            onClick={() => setShowHierarchy((v) => !v)}
            className="mb-3 flex items-center gap-2 text-sm font-semibold text-gray-600 hover:text-gray-900"
          >
            <span>{showHierarchy ? '▾' : '▸'}</span>
            Type × Subtype Breakdown
          </button>
          {showHierarchy &&
            hierarchyDims.map((dim) => (
              <DimSection key={dim.dim} dim={dim} viewType={viewType} />
            ))}
        </div>
      )}

      {/* ── Turn-level breakdowns ── */}
      {summary.turnDims.length > 0 && (
        <div>
          <h2 className="mb-3 text-sm font-semibold text-gray-700">Turn-Level Breakdown</h2>
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            {summary.turnDims.map((dim) => (
              <DimSection key={dim.dim} dim={dim} viewType={viewType} />
            ))}
          </div>
        </div>
      )}

      {/* ── Dialog browser ── */}
      {onSelectDialog && (
        <DialogBrowser
          dialogIds={dialogIds}
          selectedId={selectedDialogId}
          onSelect={onSelectDialog}
        />
      )}

      {/* ── Selected dialog viewer ── */}
      {dialogDetail && (
        <div>
          <h2 className="mb-3 text-sm font-semibold text-gray-700">
            Dialog {dialogDetail.dialog_id}
          </h2>
          <DialogViewer dialog={dialogDetail} />
        </div>
      )}
    </div>
  )
}
