import { useState, useRef, useCallback } from "react"

export function useVoiceRecording() {
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  const startRecording = useCallback(async () => {
    try {
      setError(null)

      // Request microphone permission
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })

      // Create MediaRecorder with the stream
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm'
      })

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
    } catch (err) {
      console.error("Error accessing microphone:", err)
      setError("Could not access microphone. Please check your permissions.")
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
      setIsProcessing(true)
      setError(null)

      // Create FormData with the audio file
      const formData = new FormData()
      formData.append('audio', audioBlob, 'recording.webm')

      // Send to our API endpoint
      const response = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Transcription failed')
      }

      const data = await response.json()
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