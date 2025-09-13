"use client"

import { useState, useEffect } from "react"
import { Badge } from "@/components/ui/badge"

type StatusType = "normal" | "fear" | "pain"

interface StatusConfig {
  color: string
  bgColor: string
  label: string
  description: string
  badgeVariant: "default" | "secondary" | "destructive" | "outline"
}

const statusConfigs: Record<StatusType, StatusConfig> = {
  normal: {
    color: "bg-green-500",
    bgColor: "bg-green-400",
    label: "NORMAL",
    description: "No pain detected, all systems normal",
    badgeVariant: "secondary",
  },
  fear: {
    color: "bg-yellow-500",
    bgColor: "bg-yellow-400",
    label: "FEAR",
    description: "Fear spike detected, monitoring closely",
    badgeVariant: "outline",
  },
  pain: {
    color: "bg-red-500",
    bgColor: "bg-red-400",
    label: "PAIN",
    description: "Pain detected, haptic triggered",
    badgeVariant: "destructive",
  },
}

export function StatusIndicator() {
  const [currentStatus, setCurrentStatus] = useState<StatusType>("pain")
  const [isAnimating, setIsAnimating] = useState(false)

  // Simulate status changes for demo
  useEffect(() => {
    const interval = setInterval(() => {
      setIsAnimating(true)
      setTimeout(() => {
        const statuses: StatusType[] = ["normal", "fear", "pain"]
        const randomStatus = statuses[Math.floor(Math.random() * statuses.length)]
        setCurrentStatus(randomStatus)
        setIsAnimating(false)
      }, 500)
    }, 8000)

    return () => clearInterval(interval)
  }, [])

  const config = statusConfigs[currentStatus]

  return (
    <div className="flex flex-col items-center space-y-4">
      <div
        className={`w-24 h-24 rounded-full ${config.color} flex items-center justify-center transition-all duration-500 ${isAnimating ? "scale-110 animate-pulse" : ""}`}
      >
        <div className={`w-16 h-16 rounded-full ${config.bgColor} flex items-center justify-center`}>
          <span className="text-white font-bold text-sm">{config.label}</span>
        </div>
      </div>
      <p className="text-center text-muted-foreground text-sm">{config.description}</p>
      <Badge variant={config.badgeVariant}>
        {currentStatus === "normal" ? "All Clear" : currentStatus === "fear" ? "Monitoring" : "Active Episode"}
      </Badge>
    </div>
  )
}
