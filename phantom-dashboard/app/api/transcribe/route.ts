import { NextRequest, NextResponse } from "next/server"

// Local Whisper server URL
const WHISPER_SERVER_URL = process.env.WHISPER_SERVER_URL || "http://localhost:5001"

export async function POST(req: NextRequest) {
  try {
    // Get the audio file from the form data
    const formData = await req.formData()
    const audioFile = formData.get("audio") as File

    if (!audioFile) {
      return NextResponse.json(
        { error: "No audio file provided" },
        { status: 400 }
      )
    }

    // Create new FormData for the Python server
    const whisperFormData = new FormData()
    whisperFormData.append("audio", audioFile)

    // Send to local Whisper server
    const response = await fetch(`${WHISPER_SERVER_URL}/transcribe`, {
      method: "POST",
      body: whisperFormData,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.error || `Whisper server error: ${response.status}`)
    }

    const data = await response.json()

    return NextResponse.json({
      success: true,
      text: data.text,
      language: data.language,
    })
  } catch (error) {
    console.error("Transcription error:", error)

    // Check if Whisper server is not running
    if (error instanceof Error && error.message.includes("fetch failed")) {
      return NextResponse.json(
        {
          error: "Whisper server not running",
          details: "Please start the Whisper server: python3 whisper-server.py"
        },
        { status: 503 }
      )
    }

    if (error instanceof Error) {
      return NextResponse.json(
        {
          error: "Transcription failed",
          details: error.message
        },
        { status: 500 }
      )
    }

    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}