import { useState, useRef, useCallback } from "react"

export function useVoiceRecording() {
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  const startRecording = useCallback(async () => {
    try {
      console.log("Starting recording...")
      setError(null)

      // Request microphone permission
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      console.log("Got microphone stream")

      // Create MediaRecorder with the stream
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm'
      })
      console.log("Created MediaRecorder")

      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      // Collect audio data chunks
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      // Start recording
      mediaRecorder.start()
      setIsRecording(true)
      console.log("Recording started")
    } catch (err) {
      console.error("Error accessing microphone:", err)
      setError("Could not access microphone. Please check your permissions.")
      // Re-throw to let the caller handle it
      throw err
    }
  }, [])

  const stopRecording = useCallback(async (): Promise<Blob | null> => {
    return new Promise((resolve) => {
      if (!mediaRecorderRef.current) {
        resolve(null)
        return
      }

      const mediaRecorder = mediaRecorderRef.current

      mediaRecorder.onstop = () => {
        // Create a blob from the audio chunks
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })

        // Stop all tracks to release the microphone
        mediaRecorder.stream.getTracks().forEach(track => track.stop())

        setIsRecording(false)
        resolve(audioBlob)
      }

      // Stop recording
      mediaRecorder.stop()
      mediaRecorderRef.current = null
    })
  }, [])

  const transcribeAudio = useCallback(async (audioBlob: Blob): Promise<string | null> => {
    try {
      console.log("Starting transcription, blob size:", audioBlob.size)
      setIsProcessing(true)
      setError(null)

      // Create FormData with the audio file
      const formData = new FormData()
      formData.append('audio', audioBlob, 'recording.webm')
      console.log("Created FormData")

      // Send to our API endpoint
      console.log("Sending to /api/transcribe")
      const response = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData,
      })

      console.log("Response status:", response.status)
      if (!response.ok) {
        const errorData = await response.json()
        console.error("Transcription error response:", errorData)
        throw new Error(errorData.error || 'Transcription failed')
      }

      const data = await response.json()
      console.log("Transcription result:", data)
      return data.text || null
    } catch (err) {
      console.error("Error transcribing audio:", err)
      setError("Failed to transcribe audio. Please try again.")
      return null
    } finally {
      setIsProcessing(false)
    }
  }, [])

  const recordAndTranscribe = useCallback(async () => {
    if (isRecording) {
      // Stop recording and transcribe
      const audioBlob = await stopRecording()
      if (audioBlob) {
        const text = await transcribeAudio(audioBlob)
        return text
      }
    } else {
      // Start recording
      await startRecording()
      return null
    }
  }, [isRecording, startRecording, stopRecording, transcribeAudio])

  return {
    isRecording,
    isProcessing,
    error,
    startRecording,
    stopRecording,
    transcribeAudio,
    recordAndTranscribe,
  }
}