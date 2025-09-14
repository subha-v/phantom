import { Settings } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { StatusIndicator } from "@/components/status-indicator"
import { HapticLog } from "@/components/haptic-log"
import { VoiceJournal } from "@/components/voice-journal"
import { CoachChatEnhanced } from "@/components/coach-chat-enhanced"
import { InsightsChart } from "@/components/insights-chart"
import { FooterActions } from "@/components/footer-actions"

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-background">
      {/* Top Navigation Bar */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            {/* Left: Logo */}
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-sm">AC</span>
              </div>
              <span className="font-semibold text-lg">Amputee Coach</span>
            </div>

            {/* Center: Page Title */}
            <h1 className="text-xl font-semibold text-foreground">Phantom</h1>

            {/* Right: Settings and Profile */}
            <div className="flex items-center space-x-3">
              <Button variant="ghost" size="icon">
                <Settings className="h-5 w-5" />
              </Button>
              <div className="flex items-center space-x-2">
                <Avatar className="h-8 w-8">
                  <AvatarImage src="/patient-profile.png" />
                  <AvatarFallback>JD</AvatarFallback>
                </Avatar>
                <span className="text-sm font-medium">John Doe</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column: Real-Time Monitoring */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Current Status</CardTitle>
              </CardHeader>
              <CardContent>
                <StatusIndicator />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Haptic Feedback Log</CardTitle>
              </CardHeader>
              <CardContent>
                <HapticLog />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Insights & Trends</CardTitle>
              </CardHeader>
              <CardContent>
                <InsightsChart />
              </CardContent>
            </Card>
          </div>

          {/* Right Column: Patient Tools */}
          <div className="space-y-6">
            {/* Pain Journal */}
            <Card>
              <CardHeader>
                <CardTitle>Pain Journal</CardTitle>
              </CardHeader>
              <CardContent>
                <VoiceJournal />
              </CardContent>
            </Card>

            {/* Coach Chat */}
            <Card>
              <CardHeader>
                <CardTitle>Coach Chat</CardTitle>
              </CardHeader>
              <CardContent>
                <CoachChatEnhanced />
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Footer Section */}
        <footer className="mt-8 pt-6 border-t">
          <FooterActions />
        </footer>
      </main>
    </div>
  )
}
