import type { Message } from "@/types/chat"

interface MCPToolCall {
  tool: string
  params: Record<string, any>
}

interface MCPResponse {
  success: boolean
  data?: any
  error?: string
}

export class MCPClient {
  private baseUrl: string

  constructor() {
    // Use environment variable or default to localhost
    this.baseUrl = process.env.NEXT_PUBLIC_MCP_SERVER_URL || "http://localhost:8000/mcp"
  }

  /**
   * Get a coaching response from the MCP server
   */
  async getCoachingResponse(
    message: string,
    conversationHistory: Message[]
  ): Promise<string> {
    try {
      // First, analyze what the user is asking about
      const analysis = this.analyzeUserIntent(message)

      // Execute MCP tools based on analysis
      const toolResults: Record<string, any> = {}

      // Track pain if mentioned
      if (analysis.painLevel !== null) {
        const trackResult = await this.callMCPTool("track_pain_level", {
          level: analysis.painLevel,
          description: message,
          location: analysis.location || ""
        })
        toolResults.track_pain = trackResult
      }

      // Search for medical information if needed
      if (analysis.needsResearch) {
        const searchResult = await this.callMCPTool("search_medical_info", {
          query: analysis.searchQuery || message,
          num_results: 3
        })
        toolResults.search = searchResult
      }

      // Get exercise suggestions if requested
      if (analysis.needsExercises) {
        const exerciseResult = await this.callMCPTool("suggest_exercises", {
          pain_type: analysis.painType || "general",
          intensity: analysis.intensity || "moderate"
        })
        toolResults.exercises = exerciseResult
      }

      // Get coping strategies if needed
      if (analysis.needsCoping) {
        const copingResult = await this.callMCPTool("get_coping_strategies", {
          situation: message
        })
        toolResults.coping = copingResult
      }

      // Get the main coaching response with context
      const coachingResult = await this.callMCPTool("get_coaching_response", {
        message: message,
        include_context: true
      })

      // Combine results into a comprehensive response
      let finalResponse = ""

      if (coachingResult.success && coachingResult.data?.response) {
        finalResponse = coachingResult.data.response
      }

      // Add relevant information from other tools
      if (toolResults.track_pain?.success && toolResults.track_pain.data?.insights) {
        const insights = toolResults.track_pain.data.insights
        if (insights.trend && insights.trend !== "insufficient data") {
          finalResponse += `\n\nI've tracked your pain level. ${toolResults.track_pain.data.message}`
        }
      }

      if (toolResults.exercises?.success && toolResults.exercises.data?.exercises) {
        const exercises = toolResults.exercises.data.exercises
        if (exercises.length > 0) {
          finalResponse += "\n\n**Recommended Exercises:**"
          exercises.slice(0, 2).forEach((ex: any) => {
            finalResponse += `\n- **${ex.name}**: ${ex.description} (${ex.duration})`
          })
        }
      }

      if (toolResults.search?.success && toolResults.search.data?.results) {
        const results = toolResults.search.data.results
        if (results.length > 0 && analysis.needsResearch) {
          finalResponse += "\n\n📚 **Research Findings:**"
          results.slice(0, 3).forEach((result: any, index: number) => {
            finalResponse += `\n\n${index + 1}. **${result.title}**`
            if (result.snippet) {
              finalResponse += `\n   ${result.snippet.substring(0, 200)}...`
            }
            if (result.url) {
              finalResponse += `\n   [Read more](${result.url})`
            }
          })
        }
      }

      return finalResponse || "I'm here to help you with your phantom pain management. Could you tell me more about what you're experiencing?"

    } catch (error) {
      console.error("Error in getCoachingResponse:", error)

      // Fallback to a simple response
      return this.getFallbackResponse(message)
    }
  }

  /**
   * Call a specific MCP tool via API route
   */
  private async callMCPTool(tool: string, params: Record<string, any>): Promise<MCPResponse> {
    try {
      // Call our Next.js API route instead of the MCP server directly
      const response = await fetch("/api/mcp-proxy", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          tool: tool,
          params: params
        }),
      })

      if (!response.ok) {
        throw new Error(`API responded with status ${response.status}`)
      }

      const data = await response.json()
      return {
        success: true,
        data: data,
      }
    } catch (error) {
      console.error(`Error calling MCP tool ${tool}:`, error)
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      }
    }
  }

  /**
   * Analyze user intent to determine which MCP tools to use
   */
  private analyzeUserIntent(message: string): {
    painLevel: number | null
    needsResearch: boolean
    needsExercises: boolean
    needsCoping: boolean
    searchQuery: string | null
    painType: string | null
    intensity: string | null
    location: string | null
  } {
    const lowerMessage = message.toLowerCase()

    // Extract pain level (e.g., "6/10" or "pain is 6")
    let painLevel: number | null = null
    const painMatch = message.match(/(\d+)\s*\/\s*10/i) || message.match(/pain\s+(?:is\s+)?(\d+)/i)
    if (painMatch) {
      painLevel = parseInt(painMatch[1])
    }

    // Detect pain types
    let painType: string | null = null
    if (lowerMessage.includes("burn")) painType = "burning"
    else if (lowerMessage.includes("cramp")) painType = "cramping"
    else if (lowerMessage.includes("stab") || lowerMessage.includes("sharp")) painType = "stabbing"
    else if (lowerMessage.includes("tingl")) painType = "tingling"
    else if (lowerMessage.includes("throb")) painType = "throbbing"

    // Detect intensity
    let intensity: string | null = null
    if (lowerMessage.includes("light") || lowerMessage.includes("mild")) intensity = "light"
    else if (lowerMessage.includes("intense") || lowerMessage.includes("severe")) intensity = "intense"
    else intensity = "moderate"

    // Detect location
    let location: string | null = null
    const locationMatch = message.match(/(?:in|at|on)\s+(?:my\s+)?(\w+\s+\w+|\w+)/i)
    if (locationMatch) {
      location = locationMatch[1]
    }

    // Determine needed tools
    const needsResearch =
      lowerMessage.includes("why") ||
      lowerMessage.includes("what causes") ||
      lowerMessage.includes("research") ||
      lowerMessage.includes("study") ||
      lowerMessage.includes("information") ||
      lowerMessage.includes("tell me about")

    const needsExercises =
      lowerMessage.includes("exercise") ||
      lowerMessage.includes("stretch") ||
      lowerMessage.includes("therapy") ||
      lowerMessage.includes("movement") ||
      lowerMessage.includes("activity")

    const needsCoping =
      lowerMessage.includes("cope") ||
      lowerMessage.includes("deal with") ||
      lowerMessage.includes("manage") ||
      lowerMessage.includes("handle") ||
      lowerMessage.includes("difficult")

    // Extract search query if needed
    let searchQuery: string | null = null
    if (needsResearch) {
      // Remove common question words to get the core topic
      searchQuery = message
        .replace(/^(what|why|how|when|where|can you tell me about|tell me about)/i, "")
        .replace(/\?/g, "")
        .trim()
    }

    return {
      painLevel,
      needsResearch,
      needsExercises,
      needsCoping,
      searchQuery,
      painType,
      intensity,
      location,
    }
  }

  /**
   * Provide a fallback response when MCP server is unavailable
   */
  private getFallbackResponse(message: string): string {
    const lowerMessage = message.toLowerCase()

    if (lowerMessage.includes("exercise") || lowerMessage.includes("stretch")) {
      return "Here are some general exercises that may help with phantom pain:\n\n" +
        "1. **Mirror Therapy**: Place a mirror to reflect your intact limb and perform movements while watching the reflection.\n" +
        "2. **Gentle Stretching**: Stretch the muscles around your residual limb regularly.\n" +
        "3. **Progressive Muscle Relaxation**: Tense and relax muscle groups systematically.\n\n" +
        "Please consult with your healthcare provider before starting any new exercise routine."
    }

    if (lowerMessage.includes("pain") && (lowerMessage.includes("bad") || lowerMessage.includes("worse"))) {
      return "I understand you're experiencing increased pain. Here are some immediate strategies:\n\n" +
        "1. Try deep breathing exercises to help relax\n" +
        "2. Apply gentle pressure or massage around the residual limb\n" +
        "3. Use distraction techniques like engaging in a favorite activity\n" +
        "4. Consider using heat or cold therapy as recommended by your doctor\n\n" +
        "If the pain persists or worsens, please contact your healthcare provider."
    }

    if (lowerMessage.includes("mirror")) {
      return "Mirror therapy is an effective technique for phantom pain:\n\n" +
        "1. Sit with a mirror positioned to reflect your intact limb\n" +
        "2. Hide your residual limb behind the mirror\n" +
        "3. Move your intact limb while watching its reflection\n" +
        "4. Try to imagine your phantom limb moving in the same way\n" +
        "5. Practice for 15-20 minutes daily\n\n" +
        "This helps retrain your brain's perception of the phantom limb."
    }

    // Generic supportive response
    return "I understand you're dealing with phantom pain, which can be challenging. " +
      "While I'm having trouble connecting to my full knowledge base right now, " +
      "I encourage you to:\n\n" +
      "- Keep track of your pain patterns\n" +
      "- Practice relaxation techniques\n" +
      "- Stay active within your comfort level\n" +
      "- Connect with your healthcare team regularly\n\n" +
      "Is there something specific about your phantom pain you'd like to discuss?"
  }
}