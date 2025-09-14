"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send, Bot, User, Loader2, Mic, MicOff, Zap, AlertCircle } from "lucide-react"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/hooks/use-toast"
import type { Message } from "@/types/chat"
import { MCPClient } from "@/lib/mcp-client"
import { useVoiceRecording } from "@/hooks/use-voice-recording"

interface HapticSuggestion {
  intensity: "subtle" | "moderate" | "high"
  painLevel: string
  response: string
}

export function CoachChatEnhanced() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content: "Hello! I'm your AI-powered amputee coach. I can help you manage phantom limb pain with personalized advice and haptic feedback therapy. How are you feeling today?",
      sender: "coach",
      timestamp: new Date(Date.now() - 300000),
    }
  ])

  const [newMessage, setNewMessage] = useState("")
  const [isProcessing, setIsProcessing] = useState(false)
  const [hapticDialog, setHapticDialog] = useState(false)
  const [pendingHaptic, setPendingHaptic] = useState<HapticSuggestion | null>(null)
  const [isApplyingHaptic, setIsApplyingHaptic] = useState(false)

  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const mcpClient = useRef<MCPClient | null>(null)
  const { toast } = useToast()

  // Voice recording hooks
  const {
    isRecording,
    isProcessing: isTranscribing,
    error: voiceError,
    recordAndTranscribe,
  } = useVoiceRecording()

  const analyzeAndRespond = async (userMessage: string) => {
    try {
      // First, analyze the message for pain content
      const analysisResponse = await fetch("/api/analyze-pain", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
          conversationHistory: messages.slice(-6) // Send last 6 messages for context
        }),
      })

      const analysisData = await analysisResponse.json()

      if (!analysisData.success) {
        throw new Error("Failed to analyze message")
      }

      const analysis = analysisData.analysis

      // Add coach response
      const coachMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: analysis.response,
        sender: "coach",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, coachMessage])

      // If haptic feedback is suggested, show dialog
      if (analysis.shouldOfferHaptic && analysis.suggestedHaptic !== "none") {
        setPendingHaptic({
          intensity: analysis.suggestedHaptic,
          painLevel: analysis.painLevel,
          response: analysis.response
        })

        // Add a follow-up message about haptic suggestion
        setTimeout(() => {
          const hapticSuggestionMessage: Message = {
            id: (Date.now() + 2).toString(),
            content: `💡 Based on your ${analysis.painLevel} pain level, I recommend trying ${analysis.suggestedHaptic} haptic feedback. Would you like me to activate it?`,
            sender: "coach",
            timestamp: new Date(),
          }
          setMessages((prev) => [...prev, hapticSuggestionMessage])
          setHapticDialog(true)
        }, 1000)
      }

    } catch (error) {
      console.error("Error analyzing message:", error)

      // Fallback to regular coaching response
      if (!mcpClient.current) {
        mcpClient.current = new MCPClient()
      }

      const response = await mcpClient.current.getCoachingResponse(userMessage, messages)

      const coachMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response,
        sender: "coach",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, coachMessage])
    }
  }

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
        await analyzeAndRespond(newMessage)
      } catch (error) {
        console.error("Error processing message:", error)
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

  const applyHapticFeedback = async () => {
    if (!pendingHaptic) return

    setIsApplyingHaptic(true)

    try {
      const response = await fetch("/api/arduino/haptic", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ intensity: pendingHaptic.intensity }),
      })

      const data = await response.json()

      if (data.success) {
        toast({
          title: "Haptic Therapy Started",
          description: `${pendingHaptic.intensity.charAt(0).toUpperCase() + pendingHaptic.intensity.slice(1)} haptic feedback activated for your ${pendingHaptic.painLevel} pain.`,
        })

        const confirmMessage: Message = {
          id: Date.now().toString(),
          content: `✅ ${pendingHaptic.intensity.charAt(0).toUpperCase() + pendingHaptic.intensity.slice(1)} haptic feedback has been activated. The therapy should help alleviate your phantom pain. Let me know how you feel after a few minutes.`,
          sender: "coach",
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, confirmMessage])
      } else {
        toast({
          title: "Connection Error",
          description: "Failed to activate haptic feedback. Please check Arduino connection.",
          variant: "destructive",
        })

        const errorMessage: Message = {
          id: Date.now().toString(),
          content: "I couldn't activate the haptic feedback device. Please ensure the Arduino is connected and try again.",
          sender: "coach",
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, errorMessage])
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to send haptic command.",
        variant: "destructive",
      })
    } finally {
      setIsApplyingHaptic(false)
      setHapticDialog(false)
      setPendingHaptic(null)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleVoiceButton = async () => {
    if (isTranscribing) return

    try {
      const transcribedText = await recordAndTranscribe()

      if (transcribedText && transcribedText.trim()) {
        const userMessage: Message = {
          id: Date.now().toString(),
          content: transcribedText.trim(),
          sender: "user",
          timestamp: new Date(),
        }

        setMessages(prev => [...prev, userMessage])
        setNewMessage("")
        setIsProcessing(true)

        try {
          await analyzeAndRespond(transcribedText.trim())
        } catch (error) {
          console.error("Error processing voice message:", error)
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
    <>
      <div className="flex flex-col min-h-[400px] max-h-[600px] h-full">
        {/* Info Banner */}
        <Alert className="mb-2 border-blue-200 bg-blue-50 dark:bg-blue-950/30">
          <AlertCircle className="h-4 w-4 text-blue-600" />
          <AlertDescription className="text-xs">
            AI-powered pain assessment with haptic therapy. Describe your pain and I'll suggest appropriate haptic feedback.
          </AlertDescription>
        </Alert>

        <ScrollArea className="flex-1 min-h-0 p-4 overflow-y-auto">
          <div className="space-y-4 pb-4">
            {messages.map((message) => {
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
                  ? "Analyzing your message..."
                  : "Describe your pain or ask for help..."
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

      {/* Haptic Feedback Confirmation Dialog */}
      <Dialog open={hapticDialog} onOpenChange={setHapticDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-yellow-500" />
              Haptic Therapy Recommendation
            </DialogTitle>
            <DialogDescription className="pt-2">
              {pendingHaptic && (
                <div className="space-y-3">
                  <p>
                    Based on your <Badge variant="outline" className="mx-1">{pendingHaptic.painLevel}</Badge>
                    pain level, I recommend applying
                    <Badge
                      className="mx-1"
                      variant={
                        pendingHaptic.intensity === "high" ? "destructive" :
                        pendingHaptic.intensity === "moderate" ? "default" : "secondary"
                      }
                    >
                      {pendingHaptic.intensity}
                    </Badge>
                    haptic feedback.
                  </p>
                  <Alert className="border-blue-200 bg-blue-50 dark:bg-blue-950/30">
                    <AlertDescription className="text-xs">
                      This will activate the Arduino haptic device to provide therapeutic vibration patterns designed to help alleviate phantom limb pain.
                    </AlertDescription>
                  </Alert>
                  <p className="text-sm text-muted-foreground">
                    Would you like to start the haptic therapy now?
                  </p>
                </div>
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setHapticDialog(false)
                setPendingHaptic(null)
                const declineMessage: Message = {
                  id: Date.now().toString(),
                  content: "No problem. Let me know if you'd like to try haptic therapy later or if there's anything else I can help with.",
                  sender: "coach",
                  timestamp: new Date(),
                }
                setMessages((prev) => [...prev, declineMessage])
              }}
            >
              Not Now
            </Button>
            <Button
              onClick={applyHapticFeedback}
              disabled={isApplyingHaptic}
              className="gap-2"
            >
              {isApplyingHaptic ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Activating...
                </>
              ) : (
                <>
                  <Zap className="h-4 w-4" />
                  Apply Haptic Therapy
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}