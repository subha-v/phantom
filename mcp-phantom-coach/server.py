#!/usr/bin/env python3
"""
MCP Server for Phantom Pain Coaching Agent
Integrates Claude API and Exa search for intelligent coaching responses
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from fastmcp import FastMCP
import httpx
from anthropic import Anthropic
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Phantom Pain Coach MCP")

# Initialize clients
anthropic_client = None
exa_api_key = None

# Store conversation context
conversation_history: List[Dict[str, str]] = []
pain_tracking: List[Dict[str, Any]] = []

def initialize_clients():
    """Initialize API clients with environment variables"""
    global anthropic_client, exa_api_key

    claude_api_key = os.getenv("CLAUDE_API_KEY")
    exa_api_key = os.getenv("EXA_API_KEY")

    if not claude_api_key:
        logger.error("CLAUDE_API_KEY not found in environment variables")
        raise ValueError("CLAUDE_API_KEY is required")

    if not exa_api_key:
        logger.error("EXA_API_KEY not found in environment variables")
        raise ValueError("EXA_API_KEY is required")

    anthropic_client = Anthropic(api_key=claude_api_key)
    logger.info("API clients initialized successfully")

@mcp.tool
async def search_medical_info(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search for phantom pain related medical information using Exa API

    Args:
        query: Search query about phantom pain or related topics
        num_results: Number of search results to return

    Returns:
        Dictionary containing search results and summaries
    """
    try:
        # Enhance query with phantom pain context
        enhanced_query = f"phantom pain limb amputee {query}"

        # Call Exa API
        headers = {
            "x-api-key": exa_api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "query": enhanced_query,
            "num_results": num_results,
            "use_autoprompt": True,
            "type": "neural",
            "contents": {
                "text": True,
                "highlights": True
            }
        }

        response = requests.post(
            "https://api.exa.ai/search",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            data = response.json()
            results = []

            for result in data.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("text", "")[:500] if result.get("text") else "",
                    "highlights": result.get("highlights", [])[:3]
                })

            return {
                "success": True,
                "query": query,
                "results": results,
                "total_found": len(results)
            }
        else:
            return {
                "success": False,
                "error": f"Exa API error: {response.status_code}",
                "query": query
            }

    except Exception as e:
        logger.error(f"Error searching medical info: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

@mcp.tool
async def get_coaching_response(message: str, include_context: bool = True) -> Dict[str, Any]:
    """
    Generate a personalized coaching response using Claude API

    Args:
        message: User's message or question
        include_context: Whether to include conversation history

    Returns:
        Dictionary containing the coaching response
    """
    try:
        # Build system prompt for phantom pain coaching
        system_prompt = """You are a compassionate and knowledgeable coach specializing in phantom limb pain management.
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
        - Emotional coping strategies"""

        # Build messages
        messages = [{"role": "user", "content": message}]

        # Add conversation context if requested
        if include_context and conversation_history:
            # Include last 5 exchanges for context
            context_messages = conversation_history[-10:]
            messages = context_messages + messages

        # Call Claude API
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.7,
            system=system_prompt,
            messages=messages
        )

        coaching_response = response.content[0].text

        # Store in conversation history
        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "assistant", "content": coaching_response})

        return {
            "success": True,
            "response": coaching_response,
            "timestamp": datetime.now().isoformat(),
            "context_used": include_context
        }

    except Exception as e:
        logger.error(f"Error generating coaching response: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "response": "I apologize, but I'm having trouble generating a response right now. Please try again."
        }

@mcp.tool
async def track_pain_level(level: int, description: str, location: str = "") -> Dict[str, Any]:
    """
    Track and log phantom pain levels for pattern analysis

    Args:
        level: Pain level on a scale of 1-10
        description: Description of the pain sensation
        location: Specific location of phantom pain

    Returns:
        Dictionary with tracking confirmation and insights
    """
    try:
        if level < 1 or level > 10:
            return {
                "success": False,
                "error": "Pain level must be between 1 and 10"
            }

        # Create pain entry
        pain_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "description": description,
            "location": location or "not specified"
        }

        # Store the entry
        pain_tracking.append(pain_entry)

        # Calculate insights
        recent_entries = pain_tracking[-10:] if len(pain_tracking) >= 10 else pain_tracking
        avg_pain = sum(e["level"] for e in recent_entries) / len(recent_entries) if recent_entries else level

        # Determine trend
        if len(pain_tracking) >= 3:
            recent_levels = [e["level"] for e in pain_tracking[-3:]]
            if recent_levels[-1] > recent_levels[0]:
                trend = "increasing"
            elif recent_levels[-1] < recent_levels[0]:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient data"

        return {
            "success": True,
            "entry_recorded": pain_entry,
            "insights": {
                "average_recent_pain": round(avg_pain, 1),
                "trend": trend,
                "total_entries": len(pain_tracking),
                "current_level": level
            },
            "message": f"Pain level {level}/10 recorded. {'Your pain seems to be ' + trend + '.' if trend != 'insufficient data' else ''}"
        }

    except Exception as e:
        logger.error(f"Error tracking pain level: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool
async def suggest_exercises(pain_type: str, intensity: str = "moderate") -> Dict[str, Any]:
    """
    Suggest appropriate exercises based on phantom pain type and intensity

    Args:
        pain_type: Type of phantom pain (burning, cramping, stabbing, tingling, etc.)
        intensity: Exercise intensity level (light, moderate, intense)

    Returns:
        Dictionary containing exercise recommendations
    """
    try:
        # Exercise database based on pain types
        exercises = {
            "burning": [
                {
                    "name": "Cool Water Immersion",
                    "description": "Immerse the residual limb in cool (not cold) water for 5-10 minutes",
                    "duration": "5-10 minutes",
                    "frequency": "2-3 times daily",
                    "benefits": "Helps reduce burning sensations through temperature contrast"
                },
                {
                    "name": "Mirror Box Therapy",
                    "description": "Use mirror reflection of intact limb to perform slow, cooling movements",
                    "duration": "15-20 minutes",
                    "frequency": "Daily",
                    "benefits": "Visual feedback can help reduce burning phantom sensations"
                }
            ],
            "cramping": [
                {
                    "name": "Progressive Muscle Relaxation",
                    "description": "Systematically tense and relax muscle groups starting from the residual limb",
                    "duration": "10-15 minutes",
                    "frequency": "2-3 times daily",
                    "benefits": "Reduces muscle tension and cramping sensations"
                },
                {
                    "name": "Gentle Stretching",
                    "description": "Perform gentle stretches of the residual limb and surrounding muscles",
                    "duration": "5-10 minutes",
                    "frequency": "3-4 times daily",
                    "benefits": "Improves flexibility and reduces cramping"
                }
            ],
            "stabbing": [
                {
                    "name": "Desensitization Tapping",
                    "description": "Gently tap around the residual limb with varying pressures",
                    "duration": "5-10 minutes",
                    "frequency": "3-4 times daily",
                    "benefits": "Helps normalize nerve signals and reduce stabbing pains"
                },
                {
                    "name": "Visualization Exercises",
                    "description": "Visualize the phantom limb moving smoothly and pain-free",
                    "duration": "10-15 minutes",
                    "frequency": "2-3 times daily",
                    "benefits": "Mental imagery can help reorganize pain signals"
                }
            ],
            "tingling": [
                {
                    "name": "Texture Stimulation",
                    "description": "Rub different textures on the residual limb (soft cloth, brush, etc.)",
                    "duration": "5-10 minutes",
                    "frequency": "3-4 times daily",
                    "benefits": "Provides varied sensory input to reduce tingling"
                },
                {
                    "name": "Bilateral Movements",
                    "description": "Perform synchronized movements with both the intact and phantom limb",
                    "duration": "10-15 minutes",
                    "frequency": "Daily",
                    "benefits": "Promotes neural reorganization and reduces abnormal sensations"
                }
            ],
            "general": [
                {
                    "name": "Mirror Therapy",
                    "description": "Use mirror to reflect intact limb while performing various movements",
                    "duration": "15-20 minutes",
                    "frequency": "Daily",
                    "benefits": "Comprehensive approach for various phantom pain types"
                },
                {
                    "name": "Mindful Breathing",
                    "description": "Deep breathing exercises with focus on the phantom limb area",
                    "duration": "10-15 minutes",
                    "frequency": "Multiple times daily",
                    "benefits": "Reduces overall pain perception and promotes relaxation"
                }
            ]
        }

        # Get exercises for specific pain type or general
        selected_exercises = exercises.get(pain_type.lower(), exercises["general"])

        # Adjust based on intensity
        if intensity.lower() == "light":
            for exercise in selected_exercises:
                exercise["modified"] = "Start with shorter durations and gentler movements"
        elif intensity.lower() == "intense":
            for exercise in selected_exercises:
                exercise["modified"] = "Can increase duration and frequency as tolerated"

        return {
            "success": True,
            "pain_type": pain_type,
            "intensity": intensity,
            "exercises": selected_exercises,
            "general_tips": [
                "Always start slowly and gradually increase intensity",
                "Stop if pain significantly worsens",
                "Consistency is more important than intensity",
                "Combine with relaxation techniques for best results"
            ],
            "disclaimer": "Consult your healthcare provider before starting new exercises"
        }

    except Exception as e:
        logger.error(f"Error suggesting exercises: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "exercises": []
        }

@mcp.tool
async def get_coping_strategies(situation: str) -> Dict[str, Any]:
    """
    Provide coping strategies for specific phantom pain situations

    Args:
        situation: Description of the challenging situation

    Returns:
        Dictionary containing relevant coping strategies
    """
    try:
        # Use Claude to generate personalized coping strategies
        prompt = f"""Based on this phantom pain situation: "{situation}"

        Provide 3-5 specific, practical coping strategies that could help.
        Focus on immediate relief techniques and long-term management approaches.
        Include both physical and psychological strategies."""

        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=800,
            temperature=0.7,
            system="You are an expert in phantom limb pain management. Provide practical, evidence-based coping strategies.",
            messages=[{"role": "user", "content": prompt}]
        )

        strategies = response.content[0].text

        return {
            "success": True,
            "situation": situation,
            "strategies": strategies,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting coping strategies: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "strategies": "Unable to generate strategies at this time."
        }

def main():
    """Main entry point for the MCP server"""
    # Initialize API clients
    initialize_clients()

    logger.info("Starting MCP server on http://localhost:8000/mcp")

    # Run the server with FastMCP
    import uvicorn
    uvicorn.run(
        mcp.http_app(),
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    # Load environment variables from .env file if it exists
    from pathlib import Path
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

    # Run the server
    main()