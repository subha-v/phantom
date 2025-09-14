import { NextRequest, NextResponse } from "next/server"

const CLAUDE_API_KEY = "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"

export async function POST(req: NextRequest) {
  try {
    const { message, conversationHistory = [] } = await req.json()

    const systemPrompt = `You are an empathetic pain assessment AI assistant for amputees experiencing phantom limb pain.

    Your task is to analyze the user's message and:
    1. Determine if they are reporting pain or discomfort
    2. Assess the pain level (mild, moderate, or severe)
    3. Suggest an appropriate haptic feedback intensity:
       - "subtle" for mild pain (1-3 on pain scale) or general comfort needs
       - "moderate" for moderate pain (4-6 on pain scale) or when they need focused relief
       - "high" for severe pain (7-10 on pain scale) or acute episodes
    4. Provide a compassionate response
    5. Ask if they would like to receive the suggested haptic feedback

    Return your response in JSON format with the following structure:
    {
      "isPainRelated": boolean,
      "painLevel": "none" | "mild" | "moderate" | "severe",
      "suggestedHaptic": "none" | "subtle" | "moderate" | "high",
      "response": "Your empathetic message to the user",
      "shouldOfferHaptic": boolean
    }

    Important guidelines:
    - Be compassionate and understanding
    - If pain is mentioned, always suggest appropriate haptic feedback
    - For general discomfort or anxiety, suggest "subtle" haptic
    - Only suggest "high" for severe pain or when explicitly requested
    - Always ask for consent before suggesting haptic activation`

    const messages = [
      ...conversationHistory.map((msg: any) => ({
        role: msg.sender === "user" ? "user" : "assistant",
        content: msg.content
      })),
      { role: "user", content: message }
    ]

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01"
      },
      body: JSON.stringify({
        model: "claude-3-5-haiku-20241022",
        max_tokens: 500,
        temperature: 0.3,
        system: systemPrompt,
        messages: messages
      })
    })

    if (!response.ok) {
      throw new Error(`Claude API error: ${response.status}`)
    }

    const data = await response.json()
    const claudeResponse = data.content[0].text

    // Parse the JSON response from Claude
    let analysis
    try {
      analysis = JSON.parse(claudeResponse)
    } catch (parseError) {
      // If Claude didn't return valid JSON, create a default response
      analysis = {
        isPainRelated: false,
        painLevel: "none",
        suggestedHaptic: "none",
        response: claudeResponse,
        shouldOfferHaptic: false
      }
    }

    return NextResponse.json({
      success: true,
      analysis: analysis,
      timestamp: new Date().toISOString()
    })
  } catch (error) {
    console.error("Error in pain analysis:", error)
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to analyze message",
        analysis: {
          isPainRelated: false,
          painLevel: "none",
          suggestedHaptic: "none",
          response: "I'm having trouble processing your message right now. Please try again.",
          shouldOfferHaptic: false
        }
      },
      { status: 500 }
    )
  }
}