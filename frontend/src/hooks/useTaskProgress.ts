/**
 * Manages WebSocket lifecycle for task progress.
 *
 * Opens the socket when `active` becomes true, closes it when the task
 * reaches a terminal state (completed / failed) or when `active` turns false.
 */
import { useEffect, useRef } from 'react'
import { openProgressSocket } from '../api'
import type { ProgressSnapshot } from '../types'

export function useTaskProgress(
  active: boolean,
  onSnapshot: (snap: ProgressSnapshot) => void,
  onDone: () => void,
) {
  const cleanupRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    if (!active) return

    const cleanup = openProgressSocket(
      (snap) => {
        onSnapshot(snap)
        if (snap.status === 'completed' || snap.status === 'failed') {
          cleanup()
          cleanupRef.current = null
          onDone()
        }
      },
      onDone,
    )

    cleanupRef.current = cleanup
    return () => {
      cleanup()
      cleanupRef.current = null
    }
  }, [active]) // eslint-disable-line react-hooks/exhaustive-deps
}
