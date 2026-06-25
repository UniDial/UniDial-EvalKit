import { useEffect, useState } from 'react'

import {
  fetchBrowseDialog,
  fetchBrowseDialogs,
  fetchBrowseSummary,
  fetchExistingResults,
  fetchOptions,
  fetchResult,
  fetchTaskDialogList,
  startTask,
} from './api'
import { useTaskProgress } from './hooks/useTaskProgress'

import { Canvas } from './components/Canvas'
import { ProgressPanel } from './components/ProgressPanel'
import { ResultsPanel, BrowseGrid } from './components/ResultsPanel'
import { SettingsPanel } from './components/SettingsPanel'

import { DEFAULT_PARAMS } from './types'
import type {
  DatasetOption,
  DialogDetail,
  ExistingResult,
  ModelOption,
  ProgressSnapshot,
  TaskParams,
} from './types'

export default function App() {
  // ── Settings ─────────────────────────────────────────────────────────────────
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [params, setParams] = useState<TaskParams>(DEFAULT_PARAMS)

  // ── Mode ──────────────────────────────────────────────────────────────────────
  const [isBrowseMode, setIsBrowseMode] = useState(true)

  function toggleMode() {
    setIsBrowseMode((v) => !v)
    setSelectedDataset(null)
    setSelectedModel(null)
    setBrowseContext(null)
    setSummaryData(null)
    setRunResult(null)
    setDialogIds([])
    setSelectedDialogId(null)
    setDialogDetail(null)
  }

  // ── Catalog ───────────────────────────────────────────────────────────────────
  const [datasets, setDatasets] = useState<DatasetOption[]>([])
  const [models, setModels] = useState<ModelOption[]>([])
  const [existingResults, setExistingResults] = useState<ExistingResult[]>([])

  useEffect(() => {
    fetchOptions()
      .then(({ datasets, models }) => {
        setDatasets(datasets)
        setModels(models)
      })
      .catch(console.error)

    fetchExistingResults()
      .then(setExistingResults)
      .catch(console.error)
  }, [])

  // ── Selection ─────────────────────────────────────────────────────────────────
  const [selectedDataset, setSelectedDataset] = useState<DatasetOption | null>(null)
  const [selectedModel, setSelectedModel] = useState<ModelOption | null>(null)

  // ── Browse state ──────────────────────────────────────────────────────────────
  const [browseContext, setBrowseContext] = useState<{ agent: string; dataset: string; model: string } | null>(null)
  const [summaryData, setSummaryData] = useState<Record<string, unknown> | null>(null)
  const [dialogIds, setDialogIds] = useState<string[]>([])
  const [selectedDialogId, setSelectedDialogId] = useState<string | null>(null)
  const [dialogDetail, setDialogDetail] = useState<DialogDetail | null>(null)
  const [loadingDialog, setLoadingDialog] = useState(false)

  async function loadBrowseResult(agent: string, dataset: string, model: string) {
    try {
      const [summary, ids] = await Promise.all([
        fetchBrowseSummary(agent, dataset, model),
        fetchBrowseDialogs(agent, dataset, model),
      ])
      setBrowseContext({ agent, dataset, model })
      setSummaryData(summary)
      setDialogIds(ids)
      setSelectedDialogId(null)
      setDialogDetail(null)
      // Auto-load first dialog
      if (ids.length > 0) {
        setSelectedDialogId(ids[0])
        setLoadingDialog(true)
        try {
          const detail = await fetchBrowseDialog(agent, dataset, model, ids[0])
          setDialogDetail(detail)
        } catch {
          // ignore auto-load failure
        } finally {
          setLoadingDialog(false)
        }
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to load results')
    }
  }

  async function loadDialog(dialogId: string) {
    const ctx = browseContext ?? runContext
    if (!ctx) return
    setLoadingDialog(true)
    setSelectedDialogId(dialogId)
    try {
      const detail = await fetchBrowseDialog(ctx.agent, ctx.dataset, ctx.model, dialogId)
      setDialogDetail(detail)
    } catch (err) {
      console.error(err)
    } finally {
      setLoadingDialog(false)
    }
  }

  // ── Run state ─────────────────────────────────────────────────────────────────
  const [isRunning, setIsRunning] = useState(false)
  const [snapshot, setSnapshot] = useState<ProgressSnapshot | null>(null)
  const [runResult, setRunResult] = useState<Record<string, unknown> | null>(null)
  const [runContext, setRunContext] = useState<{ agent: string; dataset: string; model: string } | null>(null)

  type DialogSummary = { dialog_id: number; dialog_labels: Record<string, string> }
  const [progressDialogList, setProgressDialogList] = useState<DialogSummary[]>([])
  const [progressSelectedId, setProgressSelectedId] = useState<number | null>(null)
  const [progressSelectedDialog, setProgressSelectedDialog] = useState<DialogDetail | null>(null)

  const onSnapshot = (snap: ProgressSnapshot) => setSnapshot(snap)
  const onDone = () => {
    setIsRunning(false)
    fetchResult().then(setRunResult).catch(console.error)
    if (runContext) {
      const ctx = runContext
      fetchBrowseDialogs(ctx.agent, ctx.dataset, ctx.model)
        .then((ids) => {
          setDialogIds(ids)
          if (ids.length > 0) {
            setSelectedDialogId(ids[0])
            fetchBrowseDialog(ctx.agent, ctx.dataset, ctx.model, ids[0])
              .then(setDialogDetail)
              .catch(console.error)
          }
        })
        .catch(console.error)
    }
  }

  useTaskProgress(isRunning, onSnapshot, onDone)

  useEffect(() => {
    if (!isRunning) return
    if (snapshot?.phase !== 'generation' && snapshot?.phase !== 'evaluation') return

    const poll = () => {
      fetchTaskDialogList().then((list) => {
        setProgressDialogList(list)
        if (list.length > 0) {
          const latest = list[list.length - 1]
          if (runContext && progressSelectedId !== latest.dialog_id) {
            setProgressSelectedId(latest.dialog_id)
            fetchBrowseDialog(runContext.agent, runContext.dataset, runContext.model, String(latest.dialog_id))
              .then(setProgressSelectedDialog)
              .catch(console.error)
          }
        }
      }).catch(console.error)
    }

    poll()
    const id = setInterval(poll, 2500)
    return () => clearInterval(id)
  }, [isRunning, snapshot?.phase]) // eslint-disable-line react-hooks/exhaustive-deps

  async function handleProgressSelectDialog(dialogId: number) {
    if (!runContext) return
    setProgressSelectedId(dialogId)
    fetchBrowseDialog(runContext.agent, runContext.dataset, runContext.model, String(dialogId))
      .then(setProgressSelectedDialog)
      .catch(console.error)
  }

  async function handleRun() {
    if (!selectedDataset || !selectedModel) return
    setRunResult(null)
    setSnapshot(null)
    setDialogIds([])
    setSelectedDialogId(null)
    setDialogDetail(null)
    setProgressDialogList([])
    setProgressSelectedId(null)
    setProgressSelectedDialog(null)

    const isAgent = selectedModel.id !== 'openai'
    const modelDir = (isAgent ? `${selectedModel.id}-${params.model_name}` : params.model_name).replace(/\s+/g, '-')
    const agentKey = isAgent ? selectedModel.id : '_base'
    setRunContext({ agent: agentKey, dataset: selectedDataset.id, model: modelDir })

    try {
      await startTask(selectedDataset.id, selectedModel.id, params)
      setIsRunning(true)
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to start task')
    }
  }

  async function handleBrowse() {
    if (!selectedDataset || !selectedModel) return
    const agentKey = selectedModel.id === 'openai' ? '_base' : selectedModel.id
    const match = existingResults.find(
      (r) => r.agent === agentKey && r.dataset === selectedDataset.id,
    )
    if (!match) {
      alert('No results found for this agent + dataset combination.')
      return
    }
    await loadBrowseResult(match.agent, match.dataset, match.model)
  }

  // ── View state ────────────────────────────────────────────────────────────────
  const showCanvas = !isBrowseMode || !browseContext
  const showBrowseGrid = isBrowseMode && !browseContext
  const showProgress = isRunning && !!snapshot
  const showResults = (!isRunning && !!runResult) || !!summaryData
  const currentResult = summaryData ?? runResult

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-white font-sans text-gray-900">
      {/* ── Header ── */}
      <header className="flex h-11 flex-shrink-0 items-center border-b border-gray-100 bg-white px-4 gap-3">
        <span className="text-sm font-semibold tracking-tight text-gray-800">
          UniDial <span className="text-indigo-600">EvalKit</span>
        </span>

        {/* Segmented mode toggle */}
        <div className="flex rounded-full border border-gray-200 bg-gray-50 p-0.5 text-[11px] font-medium">
          <button
            onClick={() => { if (!isBrowseMode) toggleMode() }}
            className={[
              'rounded-full px-3 py-1 transition-all',
              isBrowseMode ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-400 hover:text-gray-600',
            ].join(' ')}
          >
            Browse
          </button>
          <button
            onClick={() => { if (isBrowseMode) toggleMode() }}
            className={[
              'rounded-full px-3 py-1 transition-all',
              !isBrowseMode ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-400 hover:text-gray-600',
            ].join(' ')}
          >
            Run
          </button>
        </div>

        {/* Active browse context */}
        {browseContext && (
          <div className="flex items-center gap-1.5 text-xs text-gray-500">
            <span className="text-gray-300">›</span>
            <span className="font-medium text-gray-700">
              {existingResults.find(
                (r) => r.agent === browseContext.agent && r.dataset === browseContext.dataset && r.model === browseContext.model,
              )?.dataset_label ?? browseContext.dataset}
            </span>
            <span className="text-gray-300">/</span>
            {(() => {
              const r = existingResults.find(
                (r) => r.agent === browseContext.agent && r.dataset === browseContext.dataset && r.model === browseContext.model,
              )
              return (
                <>
                  <span className="rounded bg-indigo-600 px-1.5 py-0.5 text-[10px] font-bold text-white">
                    {r?.agent_label ?? browseContext.agent}
                  </span>
                  <span className="text-indigo-600">{r?.base_model ?? browseContext.model}</span>
                </>
              )
            })()}
            <button
              onClick={() => {
                setBrowseContext(null)
                setSummaryData(null)
                setDialogIds([])
                setSelectedDialogId(null)
                setDialogDetail(null)
              }}
              className="ml-1 text-gray-400 hover:text-gray-700"
              title="Back to results list"
            >
              ✕
            </button>
          </div>
        )}
      </header>

      {/* ── Body ── */}
      <div className="flex flex-1 min-h-0">
        {/* Settings sidebar — only in Run mode */}
        {!isBrowseMode && (
          <SettingsPanel
            open={settingsOpen}
            onToggle={() => setSettingsOpen((v) => !v)}
            params={params}
            onParamsChange={setParams}
          />
        )}

        {/* Main content area */}
        <main className="flex flex-1 flex-col min-h-0 overflow-y-auto">
          {/* Selector bar (dataset + model dropdowns + action button) */}
          {showCanvas && (
            <Canvas
              datasets={datasets}
              models={models}
              existingResults={existingResults}
              selectedDataset={selectedDataset}
              selectedModel={selectedModel}
              onSelectDataset={setSelectedDataset}
              onSelectModel={setSelectedModel}
              isBrowseMode={isBrowseMode}
              isRunning={isRunning}
              onRun={handleRun}
              onBrowse={handleBrowse}
            />
          )}

          {/* Browse mode: result grid */}
          {showBrowseGrid && (
            <BrowseGrid
              results={existingResults}
              onSelect={(agent, dataset, model) => loadBrowseResult(agent, dataset, model)}
            />
          )}

          {/* Running: progress panel */}
          {showProgress && snapshot && (
            <ProgressPanel
              snapshot={snapshot}
              dialogList={progressDialogList}
              selectedDialog={progressSelectedDialog}
              selectedDialogId={progressSelectedId}
              onSelectDialog={handleProgressSelectDialog}
            />
          )}

          {/* Results */}
          {showResults && currentResult && (
            <ResultsPanel
              result={currentResult}
              dialogIds={dialogIds}
              selectedDialogId={selectedDialogId}
              dialogDetail={dialogDetail}
              onSelectDialog={browseContext || runContext ? loadDialog : undefined}
            />
          )}

          {/* Loading indicator for dialog */}
          {loadingDialog && (
            <div className="flex items-center justify-center py-8 text-sm text-gray-400">
              <span className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-indigo-400 border-t-transparent" />
              Loading dialog…
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
