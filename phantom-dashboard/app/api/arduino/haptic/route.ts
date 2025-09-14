import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"

const execAsync = promisify(exec)

async function sendArduinoCommand(intensity: string): Promise<{ success: boolean; message: string; response?: any }> {
  try {
    // Path to the Python script
    const scriptPath = path.join(process.cwd(), 'scripts', 'arduino_control.py')

    // Execute Python script with the command
    const { stdout, stderr } = await execAsync(`python3 ${scriptPath} ${intensity}`)

    if (stderr) {
      console.error('Python stderr:', stderr)
    }

    // Parse the JSON response from Python
    const result = JSON.parse(stdout)

    if (result.success) {
      return {
        success: true,
        message: `Command '${intensity}' sent successfully`,
        response: result
      }
    } else {
      return {
        success: false,
        message: result.error || "Failed to send command"
      }
    }
  } catch (error) {
    console.error('Error executing Python script:', error)
    return {
      success: false,
      message: error instanceof Error ? error.message : "Unknown error occurred"
    }
  }
}

export async function POST(req: NextRequest) {
  try {
    const { intensity } = await req.json()

    // Validate intensity
    const validIntensities = ["subtle", "moderate", "high"]
    if (!validIntensities.includes(intensity)) {
      return NextResponse.json(
        {
          success: false,
          error: "Invalid intensity. Must be 'subtle', 'moderate', or 'high'"
        },
        { status: 400 }
      )
    }

    // Send command to Arduino via Python script
    const result = await sendArduinoCommand(intensity)

    if (result.success) {
      return NextResponse.json({
        success: true,
        message: result.message,
        intensity: intensity,
        arduinoResponse: result.response?.response || "Command executed",
        port: result.response?.port,
        timestamp: new Date().toISOString()
      })
    } else {
      return NextResponse.json(
        {
          success: false,
          error: result.message
        },
        { status: 500 }
      )
    }
  } catch (error) {
    console.error("API route error:", error)
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Internal server error"
      },
      { status: 500 }
    )
  }
}