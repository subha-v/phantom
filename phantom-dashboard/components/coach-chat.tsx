"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send, Bot, User } from "lucide-react"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"

interface Message {
  id: string
  content: string
  sender: "user" | "coach"
  timestamp: Date
}

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
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const handleSendMessage = () => {
    if (newMessage.trim()) {
      const userMessage: Message = {
        id: Date.now().toString(),
        content: newMessage,
        sender: "user",
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, userMessage])
      setNewMessage("")

      // Simulate coach response after a delay
      setTimeout(() => {
        const coachResponses = [
          "That's a great question. Let me help you with that.",
          "I understand. Have you tried the breathing exercises we discussed?",
          "That sounds challenging. Let's work through this together.",
          "Good progress! How does that feel now?",
          "Let's try a different approach. Can you describe the sensation more?",
        ]

        const coachMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: coachResponses[Math.floor(Math.random() * coachResponses.length)],
          sender: "coach",
          timestamp: new Date(),
        }

        setMessages((prev) => [...prev, coachMessage])
      }, 1000)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  return (
    <div className="flex flex-col min-h-[400px] max-h-[600px] h-full">
      <ScrollArea className="flex-1 min-h-0 p-4 overflow-y-auto">
        <div className="space-y-4 pb-4">
          {messages.map((message) => (
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
                className={`flex flex-col space-y-1 max-w-[calc(100%-3rem)] sm:max-w-[80%] ${message.sender === "user" ? "items-end" : ""}`}
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
          ))}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      <div className="border-t bg-background p-4 flex-shrink-0">
        <div className="flex space-x-2 w-full">
          <Input
            placeholder="Type your message to the coach..."
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            className="flex-1 min-w-0"
          />
          <Button onClick={handleSendMessage} size="icon" className="flex-shrink-0">
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}
