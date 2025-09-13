"use client"

import { useState, useEffect } from "react"
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine } from "recharts"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface PainDataPoint {
  time: string
  painLevel: number
  hapticDelivered: boolean
  timestamp: number
}

interface HapticMarkerProps {
  cx: number
  cy: number
  payload: PainDataPoint
}

const HapticMarker = ({ cx, cy, payload }: HapticMarkerProps) => {
  if (!payload.hapticDelivered) return null

  return (
    <g>
      <circle cx={cx} cy={cy} r={4} fill="#3b82f6" stroke="#ffffff" strokeWidth={2} />
      <circle cx={cx} cy={cy} r={8} fill="none" stroke="#3b82f6" strokeWidth={1} opacity={0.5} />
    </g>
  )
}

export function InsightsChart() {
  const [timeRange, setTimeRange] = useState<"today" | "week" | "month">("today")
  const [painData, setPainData] = useState<PainDataPoint[]>([])

  // Generate mock data based on time range
  useEffect(() => {
    const generateData = () => {
      const now = new Date()
      const dataPoints: PainDataPoint[] = []

      if (timeRange === "today") {
        // Generate hourly data for today
        for (let i = 0; i < 24; i++) {
          const time = new Date(now.getFullYear(), now.getMonth(), now.getDate(), i)
          const baseLevel = Math.sin(i * 0.3) * 3 + 4 // Oscillating between 1-7
          const painLevel = Math.max(0, Math.min(10, baseLevel + (Math.random() - 0.5) * 2))

          dataPoints.push({
            time: time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
            painLevel: Math.round(painLevel * 10) / 10,
            hapticDelivered: painLevel > 5 && Math.random() > 0.6,
            timestamp: time.getTime(),
          })
        }
      } else if (timeRange === "week") {
        // Generate daily data for the week
        for (let i = 6; i >= 0; i--) {
          const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000)
          const baseLevel = Math.sin(i * 0.5) * 2 + 5
          const painLevel = Math.max(0, Math.min(10, baseLevel + (Math.random() - 0.5) * 3))

          dataPoints.push({
            time: date.toLocaleDateString([], { weekday: "short" }),
            painLevel: Math.round(painLevel * 10) / 10,
            hapticDelivered: painLevel > 6 && Math.random() > 0.4,
            timestamp: date.getTime(),
          })
        }
      } else {
        // Generate weekly data for the month
        for (let i = 3; i >= 0; i--) {
          const date = new Date(now.getTime() - i * 7 * 24 * 60 * 60 * 1000)
          const baseLevel = Math.sin(i * 0.8) * 2.5 + 4.5
          const painLevel = Math.max(0, Math.min(10, baseLevel + (Math.random() - 0.5) * 2))

          dataPoints.push({
            time: `Week ${4 - i}`,
            painLevel: Math.round(painLevel * 10) / 10,
            hapticDelivered: painLevel > 5.5 && Math.random() > 0.5,
            timestamp: date.getTime(),
          })
        }
      }

      return dataPoints
    }

    setPainData(generateData())
  }, [timeRange])

  const averagePain =
    painData.length > 0
      ? Math.round((painData.reduce((sum, point) => sum + point.painLevel, 0) / painData.length) * 10) / 10
      : 0

  const hapticCount = painData.filter((point) => point.hapticDelivered).length

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="font-medium">{label}</p>
          <p className="text-sm">
            <span className="text-red-500">Pain Level: {data.painLevel}/10</span>
          </p>
          {data.hapticDelivered && <p className="text-sm text-blue-500">🔵 Haptic delivered</p>}
        </div>
      )
    }
    return null
  }

  return (
    <div className="space-y-4">
      {/* Time Range Selector */}
      <div className="flex items-center justify-between">
        <div className="flex space-x-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-red-500">{averagePain}</p>
            <p className="text-xs text-muted-foreground">Avg Pain</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-500">{hapticCount}</p>
            <p className="text-xs text-muted-foreground">Haptics</p>
          </div>
        </div>

        <Select value={timeRange} onValueChange={(value: "today" | "week" | "month") => setTimeRange(value)}>
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="today">Today</SelectItem>
            <SelectItem value="week">Week</SelectItem>
            <SelectItem value="month">Month</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Chart */}
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={painData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
            <YAxis
              domain={[0, 10]}
              tick={{ fontSize: 10 }}
              label={{ value: "Pain Level", angle: -90, position: "insideLeft" }}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Pain threshold line */}
            <ReferenceLine y={5} stroke="#fbbf24" strokeDasharray="5 5" />

            {/* Pain level line */}
            <Line
              type="monotone"
              dataKey="painLevel"
              stroke="#ef4444"
              strokeWidth={3}
              dot={false}
              activeDot={{ r: 6, fill: "#ef4444" }}
            />

            {/* Haptic delivery markers */}
            <Line
              type="monotone"
              dataKey="painLevel"
              stroke="transparent"
              dot={<HapticMarker cx={0} cy={0} payload={{} as PainDataPoint} />}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-xs">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-0.5 bg-red-500"></div>
          <span>Pain Level</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-0.5 bg-yellow-400 border-dashed border-t"></div>
          <span>Pain Threshold (5/10)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
          <span>Haptic Delivered</span>
        </div>
      </div>

      {/* Insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="p-3 bg-muted rounded-lg">
          <h4 className="font-medium text-sm mb-1">Pattern Analysis</h4>
          <p className="text-xs text-muted-foreground">
            {averagePain > 6
              ? "High pain levels detected. Consider adjusting therapy plan."
              : averagePain > 4
                ? "Moderate pain levels. Current treatment showing progress."
                : "Pain levels well controlled. Continue current regimen."}
          </p>
        </div>

        <div className="p-3 bg-muted rounded-lg">
          <h4 className="font-medium text-sm mb-1">Haptic Effectiveness</h4>
          <p className="text-xs text-muted-foreground">
            {hapticCount > 0
              ? `${hapticCount} haptic interventions delivered. Monitoring response.`
              : "No haptic interventions needed in this period."}
          </p>
        </div>
      </div>
    </div>
  )
}
