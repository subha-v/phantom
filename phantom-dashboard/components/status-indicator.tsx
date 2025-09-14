"use client"

import { useState, useEffect } from "react"
import { Badge } from "@/components/ui/badge"
import { useEEGStatus } from "@/hooks/useEEGStatus"

type StatusType = "none" | "normal" | "touch"

interface StatusConfig {
  color: string
  bgColor: string
  label: string
  description: string
  badgeVariant: "default" | "secondary" | "destructive" | "outline"
}

const statusConfigs: Record<StatusType, StatusConfig> = {
  none: {
    color: "bg-gray-500",
    bgColor: "bg-gray-400",
    label: "NONE",
    description: "No data stream detected",
    badgeVariant: "outline",
  },
  normal: {
    color: "bg-green-500",
    bgColor: "bg-green-400",
    label: "NORMAL",
    description: "No touch detected, monitoring EEG signals",
    badgeVariant: "secondary",
  },
  touch: {
    color: "bg-red-500",
    bgColor: "bg-red-400",
    label: "TOUCH",
    description: "Touch detected",
    badgeVariant: "destructive",
  },
}

export function StatusIndicator() {
  const { status, confidence, isConnected, error } = useEEGStatus()
  const [isAnimating, setIsAnimating] = useState(false)
  const [previousStatus, setPreviousStatus] = useState<StatusType>("none")

  // Animate on status change
  useEffect(() => {
    if (status !== previousStatus) {
      setIsAnimating(true)
      setTimeout(() => {
        setIsAnimating(false)
      }, 500)
      setPreviousStatus(status)
    }
  }, [status, previousStatus])

  const config = statusConfigs[status]

  // Show connection status if not connected
  if (!isConnected) {
    return (
      <div className="flex flex-col items-center space-y-4">
        <div className="w-24 h-24 rounded-full bg-gray-300 flex items-center justify-center animate-pulse">
          <div className="w-16 h-16 rounded-full bg-gray-200 flex items-center justify-center">
            <span className="text-gray-600 font-bold text-sm">OFFLINE</span>
          </div>
        </div>
        <p className="text-center text-muted-foreground text-sm">
          {error || "Connecting to EEG sensor..."}
        </p>
        <Badge variant="outline">Disconnected</Badge>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center space-y-4">
      <div
        className={`w-24 h-24 rounded-full ${config.color} flex items-center justify-center transition-all duration-500 ${
          isAnimating ? "scale-110 animate-pulse" : ""
        }`}
      >
        <div className={`w-16 h-16 rounded-full ${config.bgColor} flex items-center justify-center`}>
          <span className="text-white font-bold text-sm">{config.label}</span>
        </div>
      </div>

      <p className="text-center text-muted-foreground text-sm">{config.description}</p>

      {/* Confidence indicator */}
      <div className="flex flex-col items-center space-y-1">
        <div className="text-xs text-muted-foreground">
          Confidence: {(confidence * 100).toFixed(1)}%
        </div>
        <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${
              status === "touch" ? "bg-red-500" : status === "normal" ? "bg-green-500" : "bg-gray-500"
            }`}
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>

      <Badge variant={config.badgeVariant}>
        {status === "none" ? "No Data" : status === "normal" ? "Monitoring" : "Touch Detected"}
      </Badge>
    </div>
  )
}