export interface Message {
  id: string
  content: string
  sender: "user" | "coach" | "system"
  timestamp: Date
  mcpTool?: string // Optional: name of MCP tool being called
  isThinking?: boolean // Optional: indicates if the system is thinking
}