"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts"

interface SignalData {
  time: string
  eeg: number
  eda: number
  raw_eeg?: number
  raw_eda?: number
}

export function LiveSignalChart() {
  const [showRaw, setShowRaw] = useState(false)
  const [signalData, setSignalData] = useState<SignalData[]>([])

  // Generate mock real-time data
  useEffect(() => {
    const generateDataPoint = (index: number): SignalData => {
      const time = new Date(Date.now() - (29 - index) * 1000).toLocaleTimeString()
      const baseEeg = Math.sin(index * 0.3) * 30 + 50
      const baseEda = Math.cos(index * 0.2) * 20 + 40

      return {
        time,
        eeg: baseEeg + (Math.random() - 0.5) * 10,
        eda: baseEda + (Math.random() - 0.5) * 8,
        raw_eeg: baseEeg + (Math.random() - 0.5) * 25,
        raw_eda: baseEda + (Math.random() - 0.5) * 20,
      }
    }

    // Initialize with 30 data points
    const initialData = Array.from({ length: 30 }, (_, i) => generateDataPoint(i))
    setSignalData(initialData)

    // Update data every second
    const interval = setInterval(() => {
      setSignalData((prev) => {
        const newData = [...prev.slice(1), generateDataPoint(prev.length)]
        return newData
      })
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="space-y-4">
      <div className="flex space-x-2">
        <Button variant={!showRaw ? "default" : "outline"} size="sm" onClick={() => setShowRaw(false)}>
          Show Cleaned
        </Button>
        <Button variant={showRaw ? "default" : "outline"} size="sm" onClick={() => setShowRaw(true)}>
          Show Raw
        </Button>
      </div>

      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={signalData}>
            <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip
              labelStyle={{ color: "hsl(var(--foreground))" }}
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "6px",
              }}
            />
            <Line
              type="monotone"
              dataKey={showRaw ? "raw_eeg" : "eeg"}
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="EEG"
            />
            <Line
              type="monotone"
              dataKey={showRaw ? "raw_eda" : "eda"}
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              name="EDA"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="flex justify-between text-xs text-muted-foreground">
        <span>EEG (blue): Brain activity</span>
        <span>EDA (green): Skin conductance</span>
      </div>
    </div>
  )
}
