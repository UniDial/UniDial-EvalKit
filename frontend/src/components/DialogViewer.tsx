/**
 * DialogViewer — renders a full dialog with:
 *  - User message bubbles (markdown + math rendered)
 *  - Collapsible agent process accordion
 *  - Assistant message bubbles (markdown + math rendered)
 *  - Judge evaluation cards (amber, with score + reasoning)
 *  - Turn-by-turn CSS stagger animation (all turns in DOM immediately, visually appear sequentially)
 */
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { Components } from 'react-markdown'
import { useEffect, useRef, useState } from 'react'
import type { AgentLog, DialogDetail, TurnEval } from '../types'
import { scoreBg, scoreColor } from '../types'

// ── Math preprocessing — wrap bare LaTeX in $...$ for remark-math ─────────────

// Only trigger for these known math-specific commands (avoids false positives)
const MATH_CMD_RE =
  /\\(?:boxed|frac|sqrt|binom|sum|int|prod|lim|times|div|cdot|pm|mp|partial|nabla|infty|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|xi|pi|sigma|phi|psi|omega|Gamma|Delta|Theta|Lambda|Pi|Sigma|Phi|Psi|Omega|vec|hat|bar|overline|underline|overbrace|underbrace|text|mathbb|mathbf|mathrm|mathit|mathcal|approx|equiv|leq|geq|neq|sim|propto|forall|exists|in|notin|subset|supset|cup|cap|rightarrow|leftarrow|Rightarrow|Leftarrow)/

// Normalize LLM markdown quirks: `** text **` → `**text**`, `* text *` → `*text*`
function normalizeMarkdown(text: string): string {
  return text
    .replace(/\*\*\s+([\s\S]*?)\s*\*\*/g, '**$1**')
    .replace(/\*\s+([^\s*][\s\S]*?[^\s*])\s*\*/g, '*$1*')
}

// Convert \(...\) → $...$ and \[...\] → $$...$$ for remark-math
function normalizeLatexDelimiters(text: string): string {
  return text
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, inner) => `$$${inner}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, inner) => `$${inner}$`)
}

function preprocessMath(text: string): string {
  if (!MATH_CMD_RE.test(text)) return text

  // Protect already-delimited math and inline code
  const saved: string[] = []
  const TOK = '\x00'
  let out = text.replace(/(\$\$[\s\S]*?\$\$|\$[^\n$]*?\$|`[^`]*`)/g, (m) => {
    saved.push(m)
    return `${TOK}${saved.length - 1}${TOK}`
  })

  // Match: optional leading digits/operators, one or more \cmd{args} groups,
  // optional trailing digits/operators — but only if a known math cmd is present
  out = out.replace(
    /((?:[0-9+\-*.=, ]*)?(?:\\[a-zA-Z]+(?:\{(?:[^{}]|\{[^{}]*\})*\}|\[[^\]]*\])*\s*)+(?:[=+\-*/^_, .()\s0-9]*(?:\\[a-zA-Z]+(?:\{(?:[^{}]|\{[^{}]*\})*\}|\[[^\]]*\])*\s*)*)*)/g,
    (match) => {
      if (!match.trim() || !MATH_CMD_RE.test(match)) return match
      const start = match.search(/\S/)
      const end = match.search(/\s*$/)
      return match.slice(0, start) + '$' + match.slice(start, end) + '$' + match.slice(end)
    },
  )

  return out.replace(new RegExp(`${TOK}(\\d+)${TOK}`, 'g'), (_, i) => saved[+i])
}

// ── Markdown component sets ───────────────────────────────────────────────────

function makeMdComponents(light: boolean): Components {
  const codeBg = light ? 'bg-white/20' : 'bg-gray-100'
  const codeText = light ? '' : 'text-gray-700'
  const borderColor = light ? 'border-white/40' : 'border-gray-300'
  const quoteText = light ? 'opacity-80' : 'text-gray-600'

  return {
    p: ({ children }) => <p className="mb-1.5 last:mb-0">{children}</p>,
    ul: ({ children }) => <ul className="mb-1 list-disc pl-4">{children}</ul>,
    ol: ({ children }) => <ol className="mb-1 list-decimal pl-4">{children}</ol>,
    li: ({ children }) => <li className="mb-0.5">{children}</li>,
    pre: ({ children }) => (
      <pre className={`mb-1.5 overflow-x-auto rounded p-2 text-[11px] font-mono ${codeBg}`}>
        {children}
      </pre>
    ),
    code: (({ className, children }: { className?: string; children?: React.ReactNode }) => {
      const isBlock =
        !!className?.startsWith('language-') ||
        (typeof children === 'string' && (children as string).endsWith('\n'))
      if (isBlock)
        return (
          <code className={`font-mono text-[11px] ${codeText} ${className ?? ''}`}>
            {children}
          </code>
        )
      return (
        <code className={`rounded px-1 py-0.5 font-mono text-[11px] ${codeBg} ${codeText}`}>
          {children}
        </code>
      )
    }) as Components['code'],
    strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
    em: ({ children }) => <em className="italic">{children}</em>,
    h1: ({ children }) => <h1 className="mb-1 text-sm font-bold">{children}</h1>,
    h2: ({ children }) => <h2 className="mb-1 text-sm font-semibold">{children}</h2>,
    h3: ({ children }) => <h3 className="mb-1 text-xs font-semibold">{children}</h3>,
    blockquote: ({ children }) => (
      <blockquote className={`my-1 border-l-2 pl-2 italic ${borderColor} ${quoteText}`}>
        {children}
      </blockquote>
    ),
    a: (({ href, children }: { href?: string; children?: React.ReactNode }) => (
      <a
        href={href}
        className={light ? 'underline opacity-90' : 'text-indigo-600 underline hover:text-indigo-800'}
        target="_blank"
        rel="noopener noreferrer"
      >
        {children}
      </a>
    )) as Components['a'],
    table: ({ children }) => (
      <table className="mb-1 w-full border-collapse text-xs">{children}</table>
    ),
    thead: ({ children }) => (
      <thead className={light ? 'bg-white/20' : 'bg-gray-50'}>{children}</thead>
    ),
    th: (({ children }: { children?: React.ReactNode }) => (
      <th
        className={`border px-2 py-0.5 text-left font-semibold ${
          light ? 'border-white/30' : 'border-gray-200'
        }`}
      >
        {children}
      </th>
    )) as Components['th'],
    td: (({ children }: { children?: React.ReactNode }) => (
      <td className={`border px-2 py-0.5 ${light ? 'border-white/30' : 'border-gray-200'}`}>
        {children}
      </td>
    )) as Components['td'],
  }
}

// Defined at module level so component identities are stable across renders
const MD_LIGHT = makeMdComponents(true)
const MD_DARK = makeMdComponents(false)

const REMARK_PLUGINS = [remarkGfm, remarkMath]
const REHYPE_PLUGINS = [rehypeKatex]

function Md({ content, variant }: { content: string; variant: 'light' | 'dark' }) {
  return (
    <ReactMarkdown
      remarkPlugins={REMARK_PLUGINS}
      rehypePlugins={REHYPE_PLUGINS}
      components={variant === 'light' ? MD_LIGHT : MD_DARK}
    >
      {preprocessMath(normalizeMarkdown(normalizeLatexDelimiters(content)))}
    </ReactMarkdown>
  )
}

// ── Score badge ───────────────────────────────────────────────────────────────

function ScoreBadge({ score }: { score: number }) {
  return (
    <span
      className={`inline-block rounded-full border px-2 py-0.5 text-xs font-bold ${scoreBg(score)}`}
    >
      {(score * 100).toFixed(0)}%
    </span>
  )
}

// ── Judge card ────────────────────────────────────────────────────────────────

function JudgeCard({ evals }: { evals: TurnEval[] }) {
  const [open, setOpen] = useState(false)
  if (evals.length === 0) return null

  const avgScore = evals.reduce((s, e) => s + e.score, 0) / evals.length

  return (
    <div className="ml-10 mr-2 rounded-lg border border-amber-200 bg-amber-50">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-amber-700"
      >
        <span className="font-semibold">⚖ Judge Evaluation</span>
        <ScoreBadge score={avgScore} />
        {evals.length > 1 && (
          <span className="text-amber-500">({evals.length} metrics)</span>
        )}
        <span className="ml-auto text-amber-400">{open ? '▾' : '▸'}</span>
      </button>

      {open && (
        <div className="border-t border-amber-200 px-3 py-2">
          {evals.map((e, i) => (
            <div key={i} className={i > 0 ? 'mt-3 border-t border-amber-100 pt-3' : ''}>
              <div className="mb-1 flex items-center gap-2">
                <span className="font-mono text-xs text-amber-600">{e.metric_name}</span>
                <span className="text-sm font-bold" style={{ color: scoreColor(e.score) }}>
                  {(e.score * 100).toFixed(0)}%
                </span>
              </div>
              {e.details.rationale && (
                <div className="text-xs leading-relaxed text-amber-800">
                  <Md content={e.details.rationale} variant="dark" />
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Process accordion ─────────────────────────────────────────────────────────

function ProcessAccordion({ log }: { log: AgentLog }) {
  const [open, setOpen] = useState(false)

  const hasMemory = log.memory_update.new_raw_inputs.length > 0
  const hasRetrieval = log.retrieval.retrieved_contexts.length > 0
  const hasKnowledge = Object.keys(log.memory_update.extracted_knowledge).length > 0

  if (!hasMemory && !hasRetrieval && !hasKnowledge) return null

  function renderKnowledge(knowledge: Record<string, unknown>) {
    const triplesKey = Object.keys(knowledge).find((k) =>
      k.toLowerCase().includes('triple'),
    )
    if (!triplesKey) {
      const otherKey = Object.keys(knowledge)[0]
      const items = knowledge[otherKey] as unknown[]
      return (
        <ul className="space-y-1">
          {(items ?? []).map((item, i) => (
            <li key={i} className="text-xs text-gray-600">
              {typeof item === 'string' ? item : JSON.stringify(item)}
            </li>
          ))}
        </ul>
      )
    }

    const tripleData = knowledge[triplesKey] as Array<{
      passage?: string
      entities?: string[]
      triples?: string[][]
    }>

    return (
      <div className="space-y-2">
        {tripleData.map((td, i) => (
          <div key={i}>
            {td.triples && td.triples.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {td.triples.map((t, j) => (
                  <span
                    key={j}
                    className="inline-block rounded bg-indigo-50 px-1.5 py-0.5 font-mono text-[10px] text-indigo-700"
                  >
                    {Array.isArray(t) ? t.join(' → ') : String(t)}
                  </span>
                ))}
              </div>
            )}
            {td.entities && td.entities.length > 0 && (
              <div className="mt-1 flex flex-wrap gap-1">
                {td.entities.map((e, j) => (
                  <span
                    key={j}
                    className="inline-block rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[10px] text-gray-600"
                  >
                    {e}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="ml-2 mr-10 rounded-lg border border-gray-200 bg-gray-50">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs text-gray-500"
      >
        <span className="text-gray-400">⚙</span>
        <span>Agent Process</span>
        {log.metadata.latency_seconds && (
          <span className="rounded bg-gray-200 px-1.5 py-0.5 text-[10px] text-gray-500">
            {log.metadata.latency_seconds.toFixed(1)}s
          </span>
        )}
        <span className="ml-auto text-gray-400">{open ? '▾' : '▸'}</span>
      </button>

      {open && (
        <div className="space-y-3 border-t border-gray-200 px-3 py-2">
          {hasMemory && (
            <div>
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-gray-400">
                New Memory Indexed
              </div>
              <div className="space-y-1">
                {log.memory_update.new_raw_inputs.map((text, i) => (
                  <p
                    key={i}
                    className="rounded bg-white px-2 py-1 text-xs leading-relaxed text-gray-600"
                  >
                    {text}
                  </p>
                ))}
              </div>
            </div>
          )}

          {hasKnowledge && (
            <div>
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-gray-400">
                Knowledge Extracted
              </div>
              {renderKnowledge(log.memory_update.extracted_knowledge)}
            </div>
          )}

          {hasRetrieval && (
            <div>
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-gray-400">
                Retrieved Context
              </div>
              <div className="space-y-1">
                {log.retrieval.retrieved_contexts.map((ctx, i) => (
                  <div
                    key={i}
                    className="rounded bg-white px-2 py-1 text-xs leading-relaxed text-gray-600"
                  >
                    <span className="mr-1 rounded bg-violet-100 px-1 py-0.5 text-[10px] font-medium text-violet-700">
                      {ctx.source}
                    </span>
                    {ctx.content}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Turn pair ─────────────────────────────────────────────────────────────────

interface TurnPair {
  userTurn: DialogDetail['dialog_turns'][number]
  assistantTurn: DialogDetail['dialog_turns'][number] | null
  agentLog: AgentLog | undefined
  evals: TurnEval[]
  pairLabels?: Record<string, string>
}

function TurnPairView({ pair, index }: { pair: TurnPair; index: number }) {
  return (
    <div className="space-y-2">
      {/* Turn number badge */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-300">
          Turn {index + 1}
        </span>
        {pair.pairLabels && Object.entries(pair.pairLabels).length > 0 && (
          <div className="flex gap-1">
            {Object.entries(pair.pairLabels).map(([k, v]) => (
              <span
                key={k}
                className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-500"
              >
                {v}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* User bubble — markdown rendered (light variant for dark bg) */}
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-2.5 text-sm leading-relaxed text-white">
          <Md content={pair.userTurn.content} variant="light" />
        </div>
      </div>

      {/* Agent process accordion */}
      {pair.agentLog && <ProcessAccordion log={pair.agentLog} />}

      {/* Assistant bubble — markdown rendered */}
      {pair.assistantTurn && (
        <div className="flex justify-start">
          <div className="max-w-[85%] rounded-2xl rounded-tl-sm border border-gray-200 bg-white px-4 py-2.5 text-sm leading-relaxed text-gray-800 shadow-sm">
            <Md content={pair.assistantTurn.content} variant="dark" />
            {pair.assistantTurn.reference && (
              <div className="mt-2 border-t border-gray-100 pt-2 text-xs text-gray-400">
                <span className="font-medium">Reference: </span>
                <Md content={pair.assistantTurn.reference} variant="dark" />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Judge evaluation */}
      {pair.evals.length > 0 && <JudgeCard evals={pair.evals} />}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

interface Props {
  dialog: DialogDetail
  title?: string
  /** Skip IntersectionObserver and start revealing turns immediately (for inline panels) */
  immediateStart?: boolean
}

export function DialogViewer({ dialog, title, immediateStart = false }: Props) {
  // Build (user, assistant) pairs
  const pairs: TurnPair[] = []
  const turns = dialog.dialog_turns

  for (let i = 0; i < turns.length; i++) {
    const cur = turns[i]
    if (cur.role !== 'user') continue

    const next = turns[i + 1]
    const assistantTurn = next?.role === 'assistant' ? next : null
    if (assistantTurn) i++

    const agentLog = dialog.agent_logs.find(
      (l) => l.metadata.turn_index === cur.turn_id,
    )
    const evals = assistantTurn
      ? dialog.eval_details.filter((e) => e.turn_id === assistantTurn.turn_id)
      : []

    pairs.push({ userTurn: cur, assistantTurn, agentLog, evals, pairLabels: cur.turn_labels })
  }

  // ── Turn-by-turn reveal (starts when dialog enters viewport) ─────────────────
  const [revealCount, setRevealCount] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setRevealCount(0)
    const el = containerRef.current
    if (!el || pairs.length === 0) return

    // Each effect invocation gets its own cancelled flag — safe under React StrictMode
    const local = { cancelled: false, timerId: null as ReturnType<typeof setTimeout> | null }
    const interval = immediateStart ? 200 : 500

    const startReveal = () => {
      if (local.cancelled) return
      let count = 0
      const total = pairs.length
      const tick = () => {
        if (local.cancelled) return
        count++
        setRevealCount(count)
        if (count < total) local.timerId = setTimeout(tick, interval)
      }
      local.timerId = setTimeout(tick, 0) // first turn appears immediately
    }

    if (immediateStart) {
      startReveal()
      return () => {
        local.cancelled = true
        if (local.timerId !== null) clearTimeout(local.timerId)
      }
    }

    // Start only when the dialog section is actually visible in the viewport
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) startReveal() },
      { threshold: 0.08 },
    )
    obs.observe(el)

    return () => {
      local.cancelled = true
      if (local.timerId !== null) clearTimeout(local.timerId)
      obs.disconnect()
    }
  }, [dialog.dialog_id, immediateStart]) // eslint-disable-line react-hooks/exhaustive-deps

  // ─────────────────────────────────────────────────────────────────────────────

  const avgScore =
    dialog.eval_details.length > 0
      ? dialog.eval_details.reduce((s, e) => s + e.score, 0) / dialog.eval_details.length
      : null

  return (
    <div ref={containerRef} className="flex flex-col rounded-xl border border-gray-100 bg-white shadow-sm">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-gray-100 px-5 py-3">
        <span className="font-medium text-gray-700">
          {title ?? `Dialog ${dialog.dialog_id}`}
        </span>

        {Object.entries(dialog.dialog_labels ?? {}).map(([k, v]) => (
          <span
            key={k}
            className="rounded-full bg-indigo-50 px-2.5 py-0.5 text-xs font-medium text-indigo-700"
          >
            {v}
          </span>
        ))}

        {avgScore !== null && (
          <div className="ml-auto flex items-center gap-1.5">
            <span className="text-xs text-gray-400">avg score</span>
            <ScoreBadge score={avgScore} />
          </div>
        )}
      </div>

      {/* Turns — revealed one by one when visible in viewport */}
      <div className="space-y-6 p-5">
        {pairs.slice(0, revealCount).map((pair, i) => (
          <div key={i} className="turn-in">
            <TurnPairView pair={pair} index={i} />
          </div>
        ))}
        {/* Placeholder dots while loading remaining turns */}
        {revealCount < pairs.length && revealCount > 0 && (
          <div className="flex gap-1 pl-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-300" style={{ animationDelay: '0ms' }} />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-300" style={{ animationDelay: '150ms' }} />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-300" style={{ animationDelay: '300ms' }} />
          </div>
        )}
      </div>
    </div>
  )
}
