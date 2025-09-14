"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send, Bot, User, Loader2, Mic, MicOff } from "lucide-react"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import type { Message } from "@/types/chat"
import { MCPClient } from "@/lib/mcp-client"
import { useVoiceRecording } from "@/hooks/use-voice-recording"

export function CoachChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content:
        "Hello! I'm your amputee coach. How are you feeling today? Any phantom pain or discomfort I can help you with?",
      sender: "coach",
      timestamp: new Date(Date.now() - 300000),
    },
    {
      id: "2",
      content: "Hi coach, I've been having some phantom pain in my left leg this morning. It's about a 6/10.",
      sender: "user",
      timestamp: new Date(Date.now() - 240000),
    },
    {
      id: "3",
      content:
        "I understand that must be uncomfortable. Let's try some mirror therapy exercises. Can you position yourself in front of a mirror with your intact leg visible?",
      sender: "coach",
      timestamp: new Date(Date.now() - 180000),
    },
  ])

  const [newMessage, setNewMessage] = useState("")
  const [isProcessing, setIsProcessing] = useState(false)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const mcpClient = useRef<MCPClient | null>(null)

  // Voice recording hooks
  const {
    isRecording,
    isProcessing: isTranscribing,
    error: voiceError,
    recordAndTranscribe,
  } = useVoiceRecording()

  const handleSendMessage = async () => {
    if (newMessage.trim() && !isProcessing) {
      const userMessage: Message = {
        id: Date.now().toString(),
        content: newMessage,
        sender: "user",
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, userMessage])
      setNewMessage("")
      setIsProcessing(true)

      try {
        // Check if user wants to trigger haptic feedback
        if (newMessage.toLowerCase().includes("play haptic feedback")) {
          // Trigger haptic feedback on Arduino
          try {
            const hapticResponse = await fetch("http://localhost:3001/api/arduino/haptic", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
            })

            const hapticResult = await hapticResponse.json()

            if (hapticResult.success) {
              const feedbackMessage: Message = {
                id: (Date.now() + 1).toString(),
                content: "✓ Haptic feedback triggered successfully! The Arduino should be providing tactile feedback now.",
                sender: "coach",
                timestamp: new Date(),
              }
              setMessages((prev) => [...prev, feedbackMessage])
            } else {
              const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                content: `Unable to trigger haptic feedback: ${hapticResult.error}. Please make sure the Arduino is connected and the server is running.`,
                sender: "coach",
                timestamp: new Date(),
              }
              setMessages((prev) => [...prev, errorMessage])
            }
          } catch (error) {
            const errorMessage: Message = {
              id: (Date.now() + 1).toString(),
              content: "Unable to connect to Arduino server. Please make sure the Arduino server is running on port 3001.",
              sender: "coach",
              timestamp: new Date(),
            }
            setMessages((prev) => [...prev, errorMessage])
          }

          setIsProcessing(false)
          return
        }

        // Continue with normal message processing
        // Initialize MCP client if not already done
        if (!mcpClient.current) {
          mcpClient.current = new MCPClient()
        }

        // Analyze the message to determine which MCP tools to use
        const messageAnalysis = analyzeMessage(newMessage)

        // Show thinking indicators for each tool
        for (const tool of messageAnalysis.tools) {
          const thinkingMessage: Message = {
            id: `thinking-${Date.now()}-${tool}`,
            content: `Thinking [${tool}]...`,
            sender: "system",
            timestamp: new Date(),
            mcpTool: tool,
            isThinking: true,
          }
          setMessages((prev) => [...prev, thinkingMessage])

          // Small delay to show the thinking message
          await new Promise(resolve => setTimeout(resolve, 500))
        }

        // Get coaching response from MCP server
        const response = await mcpClient.current.getCoachingResponse(newMessage, messages)

        // Remove thinking messages
        setMessages((prev) => prev.filter(msg => !msg.isThinking))

        // Add coach response
        const coachMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: response,
          sender: "coach",
          timestamp: new Date(),
        }

        setMessages((prev) => [...prev, coachMessage])
      } catch (error) {
        console.error("Error getting coach response:", error)

        // Remove thinking messages on error
        setMessages((prev) => prev.filter(msg => !msg.isThinking))

        // Fallback response
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: "I apologize, but I'm having trouble connecting right now. Please try again in a moment.",
          sender: "coach",
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, errorMessage])
      } finally {
        setIsProcessing(false)
      }
    }
  }

  // Analyze message to determine which MCP tools might be needed
  const analyzeMessage = (message: string): { tools: string[] } => {
    const tools: string[] = []
    const lowerMessage = message.toLowerCase()

    // Check for haptic feedback request first (special handling)
    if (lowerMessage.includes("play haptic") || lowerMessage.includes("haptic feedback")) {
      tools.push("trigger_haptic")
      return { tools } // Return early for haptic feedback
    }

    // Check for different types of requests
    if (lowerMessage.includes("exercise") || lowerMessage.includes("therapy") || lowerMessage.includes("stretch")) {
      tools.push("suggest_exercises")
    }

    if (lowerMessage.includes("research") || lowerMessage.includes("study") || lowerMessage.includes("what causes") ||
        lowerMessage.includes("why") || lowerMessage.includes("information about")) {
      tools.push("search_medical_info")
    }

    if (lowerMessage.includes("pain level") || lowerMessage.includes("track") ||
        lowerMessage.match(/\b[0-9]\/10\b/) || lowerMessage.match(/\b(mild|moderate|severe)\b/)) {
      tools.push("track_pain_level")
    }

    if (lowerMessage.includes("cope") || lowerMessage.includes("deal with") || lowerMessage.includes("manage")) {
      tools.push("get_coping_strategies")
    }

    // Always include coaching response
    tools.push("get_coaching_response")

    return { tools }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleVoiceButton = async () => {
    console.log("Voice button clicked", { isRecording, isTranscribing })
    if (isTranscribing) {
      console.log("Already transcribing, skipping")
      return // Don't do anything if we're processing
    }

    try {
      const transcribedText = await recordAndTranscribe()
      console.log("Transcription result:", transcribedText)

      // If we got text back, it means we just stopped recording
      if (transcribedText && transcribedText.trim()) {
        console.log("Auto-sending transcribed message:", transcribedText)

        // Directly add the message to the chat without using the input field
        const userMessage: Message = {
          id: Date.now().toString(),
          content: transcribedText.trim(),
          sender: "user",
          timestamp: new Date(),
        }

        // Add user message to chat
        setMessages(prev => [...prev, userMessage])

        // Clear the input field
        setNewMessage("")

        // Process the message with the coach
        setIsProcessing(true)

        try {
          // Check for Arduino commands first
          if (transcribedText.toLowerCase().includes('arduino') || transcribedText.toLowerCase().includes('led')) {
            const arduinoResponse = await sendArduinoCommand(transcribedText)

            if (arduinoResponse) {
              const coachMessage: Message = {
                id: (Date.now() + 1).toString(),
                content: arduinoResponse,
                sender: "coach",
                timestamp: new Date(),
              }
              setMessages((prev) => [...prev, coachMessage])
            } else {
              const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                content: "Unable to connect to Arduino server. Please make sure the Arduino server is running on port 3001.",
                sender: "coach",
                timestamp: new Date(),
              }
              setMessages((prev) => [...prev, errorMessage])
            }

            setIsProcessing(false)
            return
          }

          // Continue with normal message processing
          if (!mcpClient.current) {
            mcpClient.current = new MCPClient()
          }

          const messageAnalysis = analyzeMessage(transcribedText)

          // Show thinking indicators
          for (const tool of messageAnalysis.tools) {
            const thinkingMessage: Message = {
              id: `thinking-${Date.now()}-${tool}`,
              content: `Thinking [${tool}]...`,
              sender: "system",
              timestamp: new Date(),
              mcpTool: tool,
              isThinking: true,
            }
            setMessages((prev) => [...prev, thinkingMessage])
            await new Promise(resolve => setTimeout(resolve, 500))
          }

          // Get coaching response
          const response = await mcpClient.current.getCoachingResponse(transcribedText, messages)

          // Remove thinking messages
          setMessages((prev) => prev.filter(msg => !msg.isThinking))

          // Add coach response
          const coachMessage: Message = {
            id: (Date.now() + 1).toString(),
            content: response,
            sender: "coach",
            timestamp: new Date(),
          }
          setMessages((prev) => [...prev, coachMessage])
        } catch (error) {
          console.error("Error processing message:", error)
          const errorMessage: Message = {
            id: (Date.now() + 1).toString(),
            content: "I'm having trouble processing your message. Please try again.",
            sender: "coach",
            timestamp: new Date(),
          }
          setMessages((prev) => [...prev, errorMessage])
        } finally {
          setIsProcessing(false)
        }
      }
    } catch (error) {
      console.error("Voice recording error:", error)
    }
  }

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  return (
    <div className="flex flex-col min-h-[400px] max-h-[600px] h-full">
      <ScrollArea className="flex-1 min-h-0 p-4 overflow-y-auto">
        <div className="space-y-4 pb-4">
          {messages.map((message) => {
            // System messages (MCP tool indicators)
            if (message.sender === "system") {
              return (
                <div key={message.id} className="flex items-center justify-center py-1">
                  <div className="flex items-center space-x-2 text-xs text-muted-foreground italic">
                    {message.isThinking && <Loader2 className="h-3 w-3 animate-spin" />}
                    <span>{message.content}</span>
                  </div>
                </div>
              )
            }

            // Regular user/coach messages
            return (
              <div
                key={message.id}
                className={`flex items-start space-x-3 ${
                  message.sender === "user" ? "flex-row-reverse space-x-reverse" : ""
                }`}
              >
                <Avatar className="h-8 w-8 flex-shrink-0">
                  <AvatarFallback
                    className={message.sender === "coach" ? "bg-primary text-primary-foreground" : "bg-muted"}
                  >
                    {message.sender === "coach" ? <Bot className="h-4 w-4" /> : <User className="h-4 w-4" />}
                  </AvatarFallback>
                </Avatar>
                <div
                  className={`flex flex-col space-y-1 max-w-[calc(100%-3rem)] sm:max-w-[80%] ${
                    message.sender === "user" ? "items-end" : ""
                  }`}
                >
                  <div
                    className={`rounded-lg px-3 py-2 text-sm break-words ${
                      message.sender === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                    }`}
                  >
                    {message.content}
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </span>
                </div>
              </div>
            )
          })}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      <div className="border-t bg-background p-4 flex-shrink-0">
        <div className="flex space-x-2 w-full">
          <Input
            placeholder={
              isRecording
                ? "Listening..."
                : isTranscribing
                ? "Processing speech..."
                : isProcessing
                ? "Coach is thinking..."
                : "Type your message to the coach..."
            }
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            onKeyDown={handleKeyPress}
            disabled={isProcessing || isRecording}
            className="flex-1 min-w-0"
          />
          <Button
            onClick={handleVoiceButton}
            size="icon"
            disabled={isProcessing || isTranscribing}
            variant={isRecording ? "destructive" : "outline"}
            className="flex-shrink-0"
            title={isRecording ? "Stop recording" : "Start voice recording"}
          >
            {isTranscribing ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : isRecording ? (
              <MicOff className="h-4 w-4" />
            ) : (
              <Mic className="h-4 w-4" />
            )}
          </Button>
          <Button
            onClick={handleSendMessage}
            size="icon"
            disabled={isProcessing || !newMessage.trim()}
            className="flex-shrink-0"
          >
            {isProcessing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </div>
        {voiceError && (
          <p className="text-xs text-destructive mt-2">{voiceError}</p>
        )}
      </div>
    </div>
  )
}
