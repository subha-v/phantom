"use client"

import { useState } from "react"
import { CheckCircle, Circle, Clock, Play } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

interface TherapyTask {
  id: string
  name: string
  duration: number
  completed: boolean
  icon: string
  description: string
  progress: number
}

export function TherapyPlan() {
  const [tasks, setTasks] = useState<TherapyTask[]>([
    {
      id: "1",
      name: "Daily vibration therapy",
      duration: 10,
      completed: true,
      icon: "✅",
      description: "Gentle vibration to reduce phantom pain",
      progress: 100,
    },
    {
      id: "2",
      name: "Breathing exercise",
      duration: 5,
      completed: false,
      icon: "🧘",
      description: "Deep breathing for pain management",
      progress: 0,
    },
    {
      id: "3",
      name: "Mirror therapy session",
      duration: 15,
      completed: false,
      icon: "🪞",
      description: "Visual feedback therapy for phantom limb",
      progress: 0,
    },
  ])

  const [activeTask, setActiveTask] = useState<string | null>(null)

  const startTask = (taskId: string) => {
    setActiveTask(taskId)
    // Simulate task progress
    const task = tasks.find((t) => t.id === taskId)
    if (task && !task.completed) {
      let progress = 0
      const interval = setInterval(
        () => {
          progress += 10
          setTasks((prev) => prev.map((t) => (t.id === taskId ? { ...t, progress } : t)))

          if (progress >= 100) {
            clearInterval(interval)
            setTasks((prev) => prev.map((t) => (t.id === taskId ? { ...t, completed: true, progress: 100 } : t)))
            setActiveTask(null)
          }
        },
        (task.duration * 1000) / 10,
      ) // Complete over task duration
    }
  }

  const toggleTask = (taskId: string) => {
    setTasks((prev) =>
      prev.map((task) =>
        task.id === taskId ? { ...task, completed: !task.completed, progress: task.completed ? 0 : 100 } : task,
      ),
    )
  }

  const completedTasks = tasks.filter((t) => t.completed).length
  const totalTasks = tasks.length
  const overallProgress = (completedTasks / totalTasks) * 100

  return (
    <div className="space-y-4">
      {/* Overall Progress */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Daily Progress</span>
          <span className="text-sm text-muted-foreground">
            {completedTasks}/{totalTasks} completed
          </span>
        </div>
        <Progress value={overallProgress} className="h-2" />
      </div>

      {/* Task List */}
      <div className="space-y-3">
        {tasks.map((task) => (
          <div
            key={task.id}
            className={`p-4 rounded-lg border transition-all duration-200 ${
              task.completed
                ? "bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800"
                : activeTask === task.id
                  ? "bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800"
                  : "bg-muted border-border"
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <Button variant="ghost" size="sm" className="p-0 h-6 w-6" onClick={() => toggleTask(task.id)}>
                  {task.completed ? (
                    <CheckCircle className="h-5 w-5 text-green-600" />
                  ) : (
                    <Circle className="h-5 w-5 text-muted-foreground" />
                  )}
                </Button>
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm font-medium">{task.name}</span>
                    <span className="text-lg">{task.icon}</span>
                  </div>
                  <p className="text-xs text-muted-foreground">{task.description}</p>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <Badge variant={task.completed ? "secondary" : "outline"}>
                  <Clock className="h-3 w-3 mr-1" />
                  {task.duration} min
                </Badge>
                {!task.completed && activeTask !== task.id && (
                  <Button size="sm" variant="outline" onClick={() => startTask(task.id)}>
                    <Play className="h-3 w-3 mr-1" />
                    Start
                  </Button>
                )}
                {activeTask === task.id && <Badge variant="default">In Progress...</Badge>}
              </div>
            </div>

            {/* Progress Bar for Active Task */}
            {activeTask === task.id && task.progress > 0 && task.progress < 100 && (
              <div className="mt-3">
                <Progress value={task.progress} className="h-1" />
                <p className="text-xs text-muted-foreground mt-1">{Math.round(task.progress)}% complete</p>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Completion Message */}
      {completedTasks === totalTasks && (
        <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg border border-green-200 dark:border-green-800">
          <p className="text-green-700 dark:text-green-300 font-medium">
            🎉 Great job! You've completed all your therapy tasks for today.
          </p>
        </div>
      )}
    </div>
  )
}
