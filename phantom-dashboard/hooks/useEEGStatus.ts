"use client"

import { useState, useEffect, useCallback } from 'react'

export type EEGStatus = 'none' | 'normal' | 'touch'

export interface EEGStatusData {
  status: EEGStatus
  confidence: number
  timestamp: number
  raw_prediction?: number
  message?: string
}

interface UseEEGStatusReturn {
  status: EEGStatus
  confidence: number
  isConnected: boolean
  lastUpdate: number | null
  error: string | null
}

const WEBSOCKET_URL = 'ws://localhost:8765'
const RECONNECT_DELAY = 3000 // 3 seconds
const MAX_RECONNECT_ATTEMPTS = 10

export function useEEGStatus(): UseEEGStatusReturn {
  const [status, setStatus] = useState<EEGStatus>('none')
  const [confidence, setConfidence] = useState<number>(0)
  const [isConnected, setIsConnected] = useState<boolean>(false)
  const [lastUpdate, setLastUpdate] = useState<number | null>(null)
  const [hasReceivedData, setHasReceivedData] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [reconnectAttempts, setReconnectAttempts] = useState<number>(0)

  const connectWebSocket = useCallback(() => {
    let ws: WebSocket | null = null

    try {
      ws = new WebSocket(WEBSOCKET_URL)

      ws.onopen = () => {
        console.log('Connected to EEG inference server')
        setIsConnected(true)
        setError(null)
        setReconnectAttempts(0)
      }

      ws.onmessage = (event) => {
        try {
          const data: EEGStatusData = JSON.parse(event.data)

          // Update status
          setStatus(data.status)
          setConfidence(data.confidence)
          setLastUpdate(data.timestamp)
          setHasReceivedData(true)

          // Log significant changes
          if (data.status === 'touch' && data.confidence > 0.8) {
            console.log(`High confidence touch detected: ${(data.confidence * 100).toFixed(1)}%`)
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err)
        }
      }

      ws.onerror = (event) => {
        console.error('WebSocket error:', event)
        setError('Connection error')
      }

      ws.onclose = () => {
        console.log('Disconnected from EEG inference server')
        setIsConnected(false)
        setStatus('none')
        setHasReceivedData(false)

        // Attempt to reconnect
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
          setTimeout(() => {
            setReconnectAttempts(prev => prev + 1)
            connectWebSocket()
          }, RECONNECT_DELAY)
        } else {
          setError('Failed to connect after multiple attempts')
        }
      }

      return ws
    } catch (err) {
      console.error('Failed to create WebSocket connection:', err)
      setError('Failed to connect')
      return null
    }
  }, [reconnectAttempts])

  useEffect(() => {
    const ws = connectWebSocket()

    // Cleanup on unmount
    return () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close()
      }
    }
  }, []) // Only connect once on mount

  // Reset to none if no updates for 5 seconds
  useEffect(() => {
    if (!isConnected || !lastUpdate || !hasReceivedData) return

    const timeout = setTimeout(() => {
      const timeSinceUpdate = Date.now() / 1000 - lastUpdate
      if (timeSinceUpdate > 5) {
        setStatus('none')
        setConfidence(0)
        setHasReceivedData(false)
      }
    }, 5000)

    return () => clearTimeout(timeout)
  }, [lastUpdate, isConnected, hasReceivedData])

  return {
    status,
    confidence,
    isConnected,
    lastUpdate,
    error
  }
}