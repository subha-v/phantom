"use client"

import { useState, useEffect } from "react"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"

interface LogEntry {
  id: string
  time: string
  event: string
  rule: string
  severity: "low" | "medium" | "high"
}

const ruleDescriptions = {
  "Rule 1": "Pain threshold exceeded",
  "Rule 2": "Fear response detected",
  "Rule 3": "Escalation protocol",
  "Rule 4": "Recovery monitoring",
}

export function HapticLog() {
  const [logEntries, setLogEntries] = useState<LogEntry[]>([
    {
      id: "1",
      time: "12:31 PM",
      event: "Haptic Triggered",
      rule: "Rule 1",
      severity: "medium",
    },
    {
      id: "2",
      time: "12:34 PM",
      event: "Escalated to caregiver",
      rule: "Rule 3",
      severity: "high",
    },
    {
      id: "3",
      time: "12:28 PM",
      event: "Pain threshold reached",
      rule: "Rule 2",
      severity: "high",
    },
  ])

  // Simulate new log entries
  useEffect(() => {
    const interval = setInterval(() => {
      if (Math.random() > 0.7) {
        // 30% chance every 5 seconds
        const events = ["Haptic Triggered", "Signal normalized", "Fear response detected", "Recovery initiated"]
        const rules = ["Rule 1", "Rule 2", "Rule 3", "Rule 4"]
        const severities: ("low" | "medium" | "high")[] = ["low", "medium", "high"]

        const newEntry: LogEntry = {
          id: Date.now().toString(),
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          event: events[Math.floor(Math.random() * events.length)],
          rule: rules[Math.floor(Math.random() * rules.length)],
          severity: severities[Math.floor(Math.random() * severities.length)],
        }

        setLogEntries((prev) => [newEntry, ...prev.slice(0, 9)]) // Keep last 10 entries
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "low":
        return "text-green-600"
      case "medium":
        return "text-yellow-600"
      case "high":
        return "text-red-600"
      default:
        return "text-muted-foreground"
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-4 text-sm font-medium text-muted-foreground border-b pb-2">
        <span>Time</span>
        <span>Event</span>
        <span>Rule Applied</span>
      </div>

      <ScrollArea className="h-48">
        <div className="space-y-2">
          {logEntries.map((entry) => (
            <div key={entry.id} className="grid grid-cols-3 gap-4 text-sm py-2 border-b border-border/50">
              <span className="font-mono">{entry.time}</span>
              <span>{entry.event}</span>
              <div className="flex items-center space-x-2">
                <span className={getSeverityColor(entry.severity)}>{entry.rule}</span>
                <Badge variant="outline" className="text-xs">
                  {ruleDescriptions[entry.rule as keyof typeof ruleDescriptions]}
                </Badge>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>

      <div className="text-xs text-muted-foreground">
        <p>Rules are automatically applied based on signal patterns and thresholds.</p>
      </div>
    </div>
  )
}
