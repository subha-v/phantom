"use client"

import { useState, useEffect } from "react"
import { Mic, MicOff } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"

interface JournalEntry {
  id: string
  timestamp: string
  transcript: string
  summary: string
  painLevel: number
  duration: number
  isRecording?: boolean
}

export function VoiceJournal() {
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [entries, setEntries] = useState<JournalEntry[]>([
    {
      id: "1",
      timestamp: "Today 6:30 PM",
      transcript:
        "Sharp pain in my right stump area, feels like burning sensation. It's been going on for about 20 minutes now. I'd rate it about 7 out of 10.",
      summary: "Sharp burning pain, 7/10, right stump, 20min duration",
      painLevel: 7,
      duration: 20,
    },
    {
      id: "2",
      timestamp: "Today 2:15 PM",
      transcript:
        "Mild tingling sensation, not too bad today. Maybe a 3 out of 10. Did my morning exercises which seemed to help.",
      summary: "Mild tingling, 3/10, improved with exercises",
      painLevel: 3,
      duration: 5,
    },
    {
      id: "3",
      timestamp: "Yesterday 8:45 PM",
      transcript:
        "Phantom limb sensation again. It feels like my missing foot is still there and cramping. Very uncomfortable, about 5 out of 10.",
      summary: "Phantom limb cramping sensation, 5/10",
      painLevel: 5,
      duration: 15,
    },
  ])

  const [currentTranscript, setCurrentTranscript] = useState("")

  // Recording timer
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isRecording) {
      interval = setInterval(() => {
        setRecordingTime((prev) => prev + 1)
      }, 1000)
    } else {
      setRecordingTime(0)
    }
    return () => clearInterval(interval)
  }, [isRecording])

  const startRecording = () => {
    setIsRecording(true)
    setCurrentTranscript("")
    // Simulate real-time transcription
    setTimeout(() => {
      setCurrentTranscript("I'm experiencing some discomfort in my...")
    }, 2000)
    setTimeout(() => {
      setCurrentTranscript("I'm experiencing some discomfort in my right stump area today...")
    }, 4000)
  }

  const stopRecording = () => {
    setIsRecording(false)
    if (currentTranscript) {
      // Simulate AI processing and summary generation
      const newEntry: JournalEntry = {
        id: Date.now().toString(),
        timestamp: new Date().toLocaleString(),
        transcript: currentTranscript + " It's about a 6 out of 10 pain level.",
        summary: "Discomfort in right stump, 6/10 pain level",
        painLevel: 6,
        duration: recordingTime,
      }
      setEntries((prev) => [newEntry, ...prev])
      setCurrentTranscript("")
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  const getPainLevelColor = (level: number) => {
    if (level <= 3) return "text-green-600"
    if (level <= 6) return "text-yellow-600"
    return "text-red-600"
  }

  return (
    <div className="space-y-4">
      {/* Recording Interface */}
      <div className="flex flex-col items-center space-y-4">
        <Button
          size="lg"
          className={`w-24 h-24 rounded-full transition-all duration-300 ${
            isRecording ? "bg-red-500 hover:bg-red-600 animate-pulse" : "bg-blue-500 hover:bg-blue-600"
          }`}
          onClick={isRecording ? stopRecording : startRecording}
        >
          {isRecording ? <MicOff className="h-8 w-8 text-white" /> : <Mic className="h-8 w-8 text-white" />}
        </Button>

        <div className="text-center">
          <p className="text-sm text-muted-foreground">
            {isRecording ? "Recording..." : "Click to record voice entry"}
          </p>
          {isRecording && (
            <Badge variant="destructive" className="mt-1">
              {formatTime(recordingTime)}
            </Badge>
          )}
        </div>
      </div>

      {/* Live Transcription */}
      {(isRecording || currentTranscript) && (
        <div className="bg-muted p-4 rounded-lg border-2 border-blue-200 dark:border-blue-800">
          <h4 className="font-medium mb-2 flex items-center space-x-2">
            <span>Live Transcription:</span>
            {isRecording && <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>}
          </h4>
          <p className="text-sm text-muted-foreground italic">{currentTranscript || "Listening..."}</p>
        </div>
      )}

      {/* Latest Entry Summary */}
      {entries.length > 0 && (
        <div className="bg-muted p-4 rounded-lg">
          <h4 className="font-medium mb-2">Latest Entry Summary:</h4>
          <p className="text-sm text-muted-foreground italic">"{entries[0].summary}"</p>
          <div className="flex items-center space-x-4 mt-2">
            <Badge variant="outline">
              Pain Level: <span className={getPainLevelColor(entries[0].painLevel)}>{entries[0].painLevel}/10</span>
            </Badge>
            <Badge variant="outline">Duration: {entries[0].duration}min</Badge>
          </div>
        </div>
      )}

      {/* Recent Entries */}
      <div className="space-y-2">
        <h4 className="font-medium">Recent Entries:</h4>
        <ScrollArea className="h-48">
          <div className="space-y-2">
            {entries.map((entry) => (
              <div key={entry.id} className="p-3 bg-muted rounded-lg">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-sm">{entry.timestamp}</span>
                  <Badge variant="outline" className={getPainLevelColor(entry.painLevel)}>
                    {entry.painLevel}/10
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">{entry.summary}</p>
                <details className="mt-2">
                  <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                    View full transcript
                  </summary>
                  <p className="text-xs text-muted-foreground mt-1 italic">"{entry.transcript}"</p>
                </details>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}
