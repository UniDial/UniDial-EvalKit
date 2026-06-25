/**
 * Collapsible settings sidebar — leftmost column of the app.
 *
 * Collapsed (default): narrow icon strip (~40px)
 * Expanded: full settings form (~260px)
 *
 * Contains: Paths · API Keys · Model · Execution · Judge
 */
import { useState } from 'react'
import type { TaskParams } from '../types'

// ── Tiny helpers ──────────────────────────────────────────────────────────────

const inputCls =
  'w-full rounded border border-gray-200 bg-white px-2.5 py-1.5 text-xs focus:border-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-200'

function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">
        {label}
        {hint && <span className="ml-1 font-normal normal-case text-gray-300">{hint}</span>}
      </span>
      {children}
    </label>
  )
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-xs text-gray-600">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="h-3.5 w-3.5 rounded border-gray-300 accent-indigo-600"
      />
      {label}
    </label>
  )
}

function SectionHead({ title }: { title: string }) {
  return (
    <h4 className="mb-2 mt-4 text-[10px] font-semibold uppercase tracking-wider text-gray-400 first:mt-0">
      {title}
    </h4>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

interface Props {
  open: boolean
  onToggle: () => void
  params: TaskParams
  onParamsChange: (p: TaskParams) => void
}

export function SettingsPanel({ open, onToggle, params, onParamsChange }: Props) {
  const [showJudge, setShowJudge] = useState(false)

  const set = <K extends keyof TaskParams>(key: K, value: TaskParams[K]) =>
    onParamsChange({ ...params, [key]: value })

  const hasApiKey = !!params.api_key.trim()

  return (
    <div
      className={[
        'flex flex-shrink-0 flex-col border-r border-gray-100 bg-gray-50 transition-all duration-200',
        open ? 'w-64' : 'w-10',
      ].join(' ')}
    >
      {/* Toggle button */}
      <button
        onClick={onToggle}
        title={open ? 'Collapse settings' : 'Expand settings'}
        className="flex h-10 w-full flex-shrink-0 items-center justify-center border-b border-gray-100 text-gray-400 hover:bg-gray-100 hover:text-gray-700"
      >
        {open ? (
          <span className="text-sm">‹</span>
        ) : (
          <span className="text-sm">⚙</span>
        )}
      </button>

      {/* API key indicator when collapsed */}
      {!open && (
        <div className="flex flex-col items-center gap-2 pt-2">
          <div
            title={hasApiKey ? 'API key set — Run Mode' : 'No API key — Browse Mode'}
            className={[
              'h-2 w-2 rounded-full',
              hasApiKey ? 'bg-emerald-400' : 'bg-amber-400',
            ].join(' ')}
          />
        </div>
      )}

      {/* Settings form (only when open) */}
      {open && (
        <div className="flex-1 overflow-y-auto p-3">
          {/* Mode indicator */}
          <div
            className={[
              'mb-4 rounded-lg border px-3 py-2 text-xs',
              hasApiKey
                ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                : 'border-amber-200 bg-amber-50 text-amber-700',
            ].join(' ')}
          >
            {hasApiKey ? '● Run Mode — evaluation enabled' : '○ Browse Mode — set API key to run'}
          </div>

          {/* ── Paths ── */}
          <SectionHead title="Paths" />
          <div className="flex flex-col gap-2">
            <Field label="Output Dir">
              <input
                type="text"
                value={params.output_dir}
                onChange={(e) => set('output_dir', e.target.value)}
                placeholder="./output"
                className={inputCls}
              />
            </Field>
            <Field label="Raw Data Dir">
              <input
                type="text"
                value={params.raw_data_dir}
                onChange={(e) => set('raw_data_dir', e.target.value)}
                placeholder="./raw_data"
                className={inputCls}
              />
            </Field>
            <Field label="Processed Data Dir">
              <input
                type="text"
                value={params.processed_data_dir}
                onChange={(e) => set('processed_data_dir', e.target.value)}
                placeholder="./data"
                className={inputCls}
              />
            </Field>
          </div>

          {/* ── API Key ── */}
          <SectionHead title="API Key" />
          <div className="flex flex-col gap-2">
            <Field label="API Key" hint="leave blank to browse">
              <input
                type="password"
                value={params.api_key}
                onChange={(e) => set('api_key', e.target.value)}
                placeholder="sk-..."
                className={inputCls}
              />
            </Field>
            <Field label="Base URL" hint="optional">
              <input
                type="text"
                value={params.base_url}
                onChange={(e) => set('base_url', e.target.value)}
                placeholder="http://localhost:8080/v1"
                className={inputCls}
              />
            </Field>
            <Field label="Model Name">
              <input
                type="text"
                value={params.model_name}
                onChange={(e) => set('model_name', e.target.value)}
                placeholder="gpt-4o"
                className={inputCls}
              />
            </Field>
          </div>

          {/* ── Generation ── */}
          <SectionHead title="Generation" />
          <div className="flex flex-col gap-2">
            <div className="grid grid-cols-2 gap-2">
              <Field label="Temperature">
                <input
                  type="number"
                  value={params.temperature}
                  onChange={(e) => set('temperature', parseFloat(e.target.value) || 0)}
                  step="0.1"
                  min="0"
                  max="2"
                  className={inputCls}
                />
              </Field>
              <Field label="Max Tokens">
                <input
                  type="number"
                  value={params.max_tokens}
                  onChange={(e) => set('max_tokens', parseInt(e.target.value) || 1024)}
                  className={inputCls}
                />
              </Field>
            </div>
            <Field label="Workers">
              <input
                type="number"
                value={params.parallel}
                onChange={(e) => set('parallel', parseInt(e.target.value) || 4)}
                min="1"
                className={inputCls}
              />
            </Field>
            <div className="flex flex-col gap-1.5 pt-1">
              <Toggle
                checked={params.do_generation}
                onChange={(v) => set('do_generation', v)}
                label="Run Generation"
              />
              <Toggle
                checked={params.do_evaluation}
                onChange={(v) => set('do_evaluation', v)}
                label="Run Evaluation"
              />
            </div>
          </div>

          {/* ── Judge (collapsible) ── */}
          <button
            onClick={() => setShowJudge((v) => !v)}
            className="mt-4 flex w-full items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-gray-400 hover:text-gray-600"
          >
            <span>{showJudge ? '▾' : '▸'}</span> Judge Model
          </button>
          {showJudge && (
            <div className="mt-2 flex flex-col gap-2">
              <Field label="Judge Model Name">
                <input
                  type="text"
                  value={params.judge_model_name}
                  onChange={(e) => set('judge_model_name', e.target.value)}
                  className={inputCls}
                />
              </Field>
              <Field label="Judge Model Type">
                <input
                  type="text"
                  value={params.judge_model_type}
                  onChange={(e) => set('judge_model_type', e.target.value)}
                  className={inputCls}
                />
              </Field>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
