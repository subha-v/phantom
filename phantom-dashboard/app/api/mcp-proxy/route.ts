import { NextRequest, NextResponse } from "next/server"

// Since the MCP server uses a complex protocol, we'll directly implement the logic here
// This is a simpler approach than trying to communicate with the MCP server

const CLAUDE_API_KEY = "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"
const EXA_API_KEY = "587da6c7-44c8-4317-a7d0-b1615e68f7ae"

// Store conversation history in memory (in production, use a database)
const conversationHistory: any[] = []

export async function POST(req: NextRequest) {
  try {
    const { tool, params } = await req.json()

    let result: any = {}

    switch (tool) {
      case "get_coaching_response":
        result = await getCoachingResponse(params.message, params.include_context)
        break

      case "search_medical_info":
        result = await searchMedicalInfo(params.query, params.num_results)
        break

      case "suggest_exercises":
        result = await suggestExercises(params.pain_type, params.intensity)
        break

      case "track_pain_level":
        result = await trackPainLevel(params.level, params.description, params.location)
        break

      case "get_coping_strategies":
        result = await getCopingStrategies(params.situation)
        break

      default:
        return NextResponse.json(
          { error: `Unknown tool: ${tool}` },
          { status: 400 }
        )
    }

    return NextResponse.json(result)
  } catch (error) {
    console.error("API route error:", error)
    return NextResponse.json(
      { error: "Internal server error", details: error },
      { status: 500 }
    )
  }
}

async function getCoachingResponse(message: string, includeContext: boolean = true) {
  try {
    const systemPrompt = `You are a compassionate and knowledgeable coach specializing in phantom limb pain management.
    You provide evidence-based advice, emotional support, and practical strategies.
    Your responses should be:
    - Empathetic and understanding
    - Based on current medical research
    - Practical and actionable
    - Encouraging but realistic
    - Safety-conscious (always recommend consulting healthcare providers for medical decisions)

    Key areas of expertise:
    - Mirror therapy techniques
    - Desensitization exercises
    - Pain management strategies
    - Mindfulness and relaxation techniques
    - Activity pacing and energy conservation
    - Prosthetic adaptation support
    - Emotional coping strategies`

    const messages = [
      { role: "user", content: message }
    ]

    // Add conversation context if requested
    if (includeContext && conversationHistory.length > 0) {
      const contextMessages = conversationHistory.slice(-10)
      messages.unshift(...contextMessages)
    }

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01"
      },
      body: JSON.stringify({
        model: "claude-3-5-haiku-20241022",
        max_tokens: 1000,
        temperature: 0.7,
        system: systemPrompt,
        messages: messages
      })
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("Claude API error:", errorText)
      throw new Error(`Claude API error: ${response.status}`)
    }

    const data = await response.json()
    const coachingResponse = data.content[0].text

    // Store in conversation history
    conversationHistory.push({ role: "user", content: message })
    conversationHistory.push({ role: "assistant", content: coachingResponse })

    return {
      success: true,
      response: coachingResponse,
      timestamp: new Date().toISOString(),
      context_used: includeContext
    }
  } catch (error) {
    console.error("Error in getCoachingResponse:", error)
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
      response: "I apologize, but I'm having trouble generating a response right now. Please try again."
    }
  }
}

async function searchMedicalInfo(query: string, numResults: number = 5) {
  try {
    const enhancedQuery = `phantom pain limb amputee ${query}`

    const response = await fetch("https://api.exa.ai/search", {
      method: "POST",
      headers: {
        "x-api-key": EXA_API_KEY,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        query: enhancedQuery,
        num_results: numResults,
        use_autoprompt: true,
        type: "neural",
        contents: {
          text: true,
          highlights: true
        }
      })
    })

    if (!response.ok) {
      throw new Error(`Exa API error: ${response.status}`)
    }

    const data = await response.json()
    const results = data.results?.map((result: any) => ({
      title: result.title || "",
      url: result.url || "",
      snippet: result.text?.substring(0, 500) || "",
      highlights: result.highlights?.slice(0, 3) || []
    })) || []

    return {
      success: true,
      query: query,
      results: results,
      total_found: results.length
    }
  } catch (error) {
    console.error("Error in searchMedicalInfo:", error)
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
      query: query,
      results: []
    }
  }
}

async function suggestExercises(painType: string, intensity: string = "moderate") {
  const exercises: Record<string, any[]> = {
    burning: [
      {
        name: "Cool Water Immersion",
        description: "Immerse the residual limb in cool (not cold) water for 5-10 minutes",
        duration: "5-10 minutes",
        frequency: "2-3 times daily",
        benefits: "Helps reduce burning sensations through temperature contrast"
      },
      {
        name: "Mirror Box Therapy",
        description: "Use mirror reflection of intact limb to perform slow, cooling movements",
        duration: "15-20 minutes",
        frequency: "Daily",
        benefits: "Visual feedback can help reduce burning phantom sensations"
      }
    ],
    cramping: [
      {
        name: "Progressive Muscle Relaxation",
        description: "Systematically tense and relax muscle groups starting from the residual limb",
        duration: "10-15 minutes",
        frequency: "2-3 times daily",
        benefits: "Reduces muscle tension and cramping sensations"
      },
      {
        name: "Gentle Stretching",
        description: "Perform gentle stretches of the residual limb and surrounding muscles",
        duration: "5-10 minutes",
        frequency: "3-4 times daily",
        benefits: "Improves flexibility and reduces cramping"
      }
    ],
    general: [
      {
        name: "Mirror Therapy",
        description: "Use mirror to reflect intact limb while performing various movements",
        duration: "15-20 minutes",
        frequency: "Daily",
        benefits: "Comprehensive approach for various phantom pain types"
      },
      {
        name: "Mindful Breathing",
        description: "Deep breathing exercises with focus on the phantom limb area",
        duration: "10-15 minutes",
        frequency: "Multiple times daily",
        benefits: "Reduces overall pain perception and promotes relaxation"
      }
    ]
  }

  const selectedExercises = exercises[painType.toLowerCase()] || exercises.general

  if (intensity.toLowerCase() === "light") {
    selectedExercises.forEach((exercise: any) => {
      exercise.modified = "Start with shorter durations and gentler movements"
    })
  } else if (intensity.toLowerCase() === "intense") {
    selectedExercises.forEach((exercise: any) => {
      exercise.modified = "Can increase duration and frequency as tolerated"
    })
  }

  return {
    success: true,
    pain_type: painType,
    intensity: intensity,
    exercises: selectedExercises,
    general_tips: [
      "Always start slowly and gradually increase intensity",
      "Stop if pain significantly worsens",
      "Consistency is more important than intensity",
      "Combine with relaxation techniques for best results"
    ],
    disclaimer: "Consult your healthcare provider before starting new exercises"
  }
}

// Simple pain tracking (in production, use a database)
const painTracking: any[] = []

async function trackPainLevel(level: number, description: string, location: string = "") {
  if (level < 1 || level > 10) {
    return {
      success: false,
      error: "Pain level must be between 1 and 10"
    }
  }

  const painEntry = {
    timestamp: new Date().toISOString(),
    level: level,
    description: description,
    location: location || "not specified"
  }

  painTracking.push(painEntry)

  const recentEntries = painTracking.slice(-10)
  const avgPain = recentEntries.reduce((sum, e) => sum + e.level, 0) / recentEntries.length

  let trend = "insufficient data"
  if (painTracking.length >= 3) {
    const recentLevels = painTracking.slice(-3).map(e => e.level)
    if (recentLevels[2] > recentLevels[0]) {
      trend = "increasing"
    } else if (recentLevels[2] < recentLevels[0]) {
      trend = "decreasing"
    } else {
      trend = "stable"
    }
  }

  return {
    success: true,
    entry_recorded: painEntry,
    insights: {
      average_recent_pain: Math.round(avgPain * 10) / 10,
      trend: trend,
      total_entries: painTracking.length,
      current_level: level
    },
    message: `Pain level ${level}/10 recorded. ${trend !== "insufficient data" ? `Your pain seems to be ${trend}.` : ""}`
  }
}

async function getCopingStrategies(situation: string) {
  try {
    const prompt = `Based on this phantom pain situation: "${situation}"

    Provide 3-5 specific, practical coping strategies that could help.
    Focus on immediate relief techniques and long-term management approaches.
    Include both physical and psychological strategies.`

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01"
      },
      body: JSON.stringify({
        model: "claude-3-5-haiku-20241022",
        max_tokens: 800,
        temperature: 0.7,
        system: "You are an expert in phantom limb pain management. Provide practical, evidence-based coping strategies.",
        messages: [{ role: "user", content: prompt }]
      })
    })

    if (!response.ok) {
      throw new Error(`Claude API error: ${response.status}`)
    }

    const data = await response.json()
    const strategies = data.content[0].text

    return {
      success: true,
      situation: situation,
      strategies: strategies,
      timestamp: new Date().toISOString()
    }
  } catch (error) {
    console.error("Error in getCopingStrategies:", error)
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
      strategies: "Unable to generate strategies at this time."
    }
  }
}