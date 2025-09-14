"use client"

import { useState, useEffect } from "react"
import { Mic, MicOff, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useVoiceRecording } from "@/hooks/use-voice-recording"

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
  const [recordingTime, setRecordingTime] = useState(0)
  const [currentTranscript, setCurrentTranscript] = useState("")

  // Use the actual voice recording hook
  const {
    isRecording,
    isProcessing: isTranscribing,
    error: voiceError,
    recordAndTranscribe,
  } = useVoiceRecording()

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

  const handleRecordButton = async () => {
    console.log("Voice journal button clicked", { isRecording, isTranscribing })

    if (isTranscribing) {
      console.log("Already transcribing, skipping")
      return
    }

    try {
      const transcribedText = await recordAndTranscribe()
      console.log("Journal transcription result:", transcribedText)

      // If we got text back, create a new journal entry
      if (transcribedText && transcribedText.trim()) {
        // Extract pain level from the transcript (look for numbers followed by "out of 10" or "/10")
        const painMatch = transcribedText.match(/(\d+)\s*(?:out of 10|\/10)/i)
        const painLevel = painMatch ? parseInt(painMatch[1]) : 5 // Default to 5 if not mentioned

        // Generate a simple summary (first 100 characters or first sentence)
        const firstSentence = transcribedText.split(/[.!?]/)[0]
        const summary = firstSentence.length > 100
          ? firstSentence.substring(0, 97) + "..."
          : firstSentence

        const newEntry: JournalEntry = {
          id: Date.now().toString(),
          timestamp: new Date().toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
            hour12: true
          }),
          transcript: transcribedText.trim(),
          summary: summary + (painMatch ? `, ${painLevel}/10` : ""),
          painLevel: painLevel,
          duration: Math.ceil(recordingTime / 60), // Convert to minutes
        }

        // Add the new entry to the beginning of the list
        setEntries((prev) => [newEntry, ...prev])
        setCurrentTranscript("")

        console.log("New journal entry added:", newEntry)
      }
    } catch (error) {
      console.error("Voice journal recording error:", error)
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
          onClick={handleRecordButton}
          disabled={isTranscribing}
        >
          {isRecording ? <MicOff className="h-8 w-8 text-white" /> : <Mic className="h-8 w-8 text-white" />}
        </Button>

        <div className="text-center">
          <p className="text-sm text-muted-foreground">
            {isTranscribing ? "Processing speech..." : isRecording ? "Recording... Click again to stop" : "Click to record voice entry"}
          </p>
          {isRecording && (
            <Badge variant="destructive" className="mt-1">
              {formatTime(recordingTime)}
            </Badge>
          )}
          {isTranscribing && (
            <div className="flex items-center justify-center mt-2">
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
              <span className="text-sm">Transcribing...</span>
            </div>
          )}
          {voiceError && (
            <p className="text-xs text-red-500 mt-2">{voiceError}</p>
          )}
        </div>
      </div>

      {/* Recording Status */}
      {isRecording && (
        <div className="bg-muted p-4 rounded-lg border-2 border-red-200 dark:border-red-800">
          <h4 className="font-medium mb-2 flex items-center space-x-2">
            <span>Recording in Progress</span>
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
          </h4>
          <p className="text-sm text-muted-foreground italic">Speak clearly about your pain experience...</p>
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
