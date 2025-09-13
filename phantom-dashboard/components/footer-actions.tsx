"use client"

import { useState } from "react"
import { Download, Phone, Zap, FileText, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/hooks/use-toast"

export function FooterActions() {
  const [isHapticActive, setIsHapticActive] = useState(false)
  const [emergencyDialogOpen, setEmergencyDialogOpen] = useState(false)
  const { toast } = useToast()

  const triggerTestHaptic = () => {
    setIsHapticActive(true)
    toast({
      title: "Test Haptic Triggered",
      description: "Haptic feedback device activated for 3 seconds",
    })

    // Simulate haptic duration
    setTimeout(() => {
      setIsHapticActive(false)
      toast({
        title: "Haptic Complete",
        description: "Test haptic feedback completed successfully",
      })
    }, 3000)
  }

  const exportReport = () => {
    toast({
      title: "Generating Report",
      description: "Your comprehensive pain management report is being prepared...",
    })

    // Simulate report generation
    setTimeout(() => {
      toast({
        title: "Report Ready",
        description: "Your report has been downloaded as PDF",
      })
    }, 2000)
  }

  const contactEmergency = () => {
    toast({
      title: "Emergency Alert Sent",
      description: "Your caregiver has been notified and will contact you shortly",
      variant: "destructive",
    })
    setEmergencyDialogOpen(false)
  }

  return (
    <div className="space-y-4">
      {/* Quick Actions */}
      <div className="flex flex-wrap gap-4 justify-center">
        {/* Test Haptic Button */}
        <Button
          variant="outline"
          className={`flex items-center space-x-2 transition-all duration-300 ${
            isHapticActive ? "bg-blue-100 dark:bg-blue-950 border-blue-300 dark:border-blue-700" : "bg-transparent"
          }`}
          onClick={triggerTestHaptic}
          disabled={isHapticActive}
        >
          <div
            className={`w-2 h-2 rounded-full transition-all duration-300 ${
              isHapticActive ? "bg-blue-500 animate-pulse scale-150" : "bg-blue-500"
            }`}
          ></div>
          <Zap className={`h-4 w-4 ${isHapticActive ? "animate-bounce" : ""}`} />
          <span>{isHapticActive ? "Haptic Active..." : "Trigger Test Haptic"}</span>
          {isHapticActive && <Badge variant="secondary">3s</Badge>}
        </Button>

        {/* Export Report Button */}
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline" className="flex items-center space-x-2 bg-transparent">
              <Download className="h-4 w-4" />
              <span>Export Report</span>
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Export Pain Management Report</DialogTitle>
              <DialogDescription>
                Generate a comprehensive report including pain logs, therapy progress, and haptic feedback data.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 border rounded-lg">
                  <h4 className="font-medium text-sm mb-2">Report Contents</h4>
                  <ul className="text-xs text-muted-foreground space-y-1">
                    <li>• Pain journal entries</li>
                    <li>• Therapy completion rates</li>
                    <li>• Haptic feedback logs</li>
                    <li>• Signal analysis data</li>
                    <li>• Progress trends</li>
                  </ul>
                </div>
                <div className="p-3 border rounded-lg">
                  <h4 className="font-medium text-sm mb-2">Time Range</h4>
                  <select className="w-full text-sm border rounded px-2 py-1">
                    <option>Last 24 hours</option>
                    <option>Last week</option>
                    <option>Last month</option>
                    <option>All time</option>
                  </select>
                </div>
              </div>
              <Button onClick={exportReport} className="w-full">
                <FileText className="h-4 w-4 mr-2" />
                Generate PDF Report
              </Button>
            </div>
          </DialogContent>
        </Dialog>

        {/* Emergency Contact Button */}
        <Dialog open={emergencyDialogOpen} onOpenChange={setEmergencyDialogOpen}>
          <DialogTrigger asChild>
            <Button variant="destructive" className="flex items-center space-x-2">
              <Phone className="h-4 w-4" />
              <span>Emergency Contact</span>
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="flex items-center space-x-2">
                <AlertTriangle className="h-5 w-5 text-red-500" />
                <span>Emergency Contact</span>
              </DialogTitle>
              <DialogDescription>
                This will immediately alert your designated caregiver about a potential emergency situation.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  Your caregiver (Dr. Sarah Johnson) will be contacted via phone and SMS with your current status and
                  location.
                </AlertDescription>
              </Alert>

              <div className="p-4 bg-muted rounded-lg">
                <h4 className="font-medium text-sm mb-2">Current Status Summary</h4>
                <ul className="text-sm space-y-1">
                  <li>• Pain Level: High (7/10)</li>
                  <li>• Last Haptic: 2 minutes ago</li>
                  <li>• Location: Home</li>
                  <li>• Time: {new Date().toLocaleTimeString()}</li>
                </ul>
              </div>

              <div className="flex space-x-2">
                <Button variant="outline" onClick={() => setEmergencyDialogOpen(false)} className="flex-1">
                  Cancel
                </Button>
                <Button variant="destructive" onClick={contactEmergency} className="flex-1">
                  <Phone className="h-4 w-4 mr-2" />
                  Contact Now
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Demo Information */}
      

      {/* Sponsor Integration Indicators */}
      
    </div>
  )
}
