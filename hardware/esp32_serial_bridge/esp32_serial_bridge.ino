/*
 * ESP32 Serial Bridge for Phantom Haptic System
 * Receives commands via Serial and forwards them to Arduino UNO R4
 * Future: Will integrate Whisper API for voice command processing
 */

#include <HardwareSerial.h>

// Serial communication setup
#define BAUD_RATE 9600
#define ARDUINO_SERIAL Serial1  // Use Serial1 for communication with Arduino
#define DEBUG_SERIAL Serial     // Use Serial for debugging via USB

// Command buffer
String commandBuffer = "";
bool commandReady = false;

// Available commands that can be sent to Arduino
const char* validCommands[] = {
  "subtle", "moderate", "high",
  "soft", "modvisual", "intvisual",
  "rainbow", "binary", "heartbeat",
  "individual", "basic", "haptic",
  "proximity", "help"
};
const int numValidCommands = sizeof(validCommands) / sizeof(validCommands[0]);

void setup() {
  // Initialize debug serial (USB)
  DEBUG_SERIAL.begin(115200);
  delay(1000);
  
  // Initialize Arduino communication serial
  // ESP32C3 Serial1: TX=D6, RX=D7 (Xiao ESP32C3 pinout)
  ARDUINO_SERIAL.begin(BAUD_RATE, SERIAL_8N1, 7, 6);  // RX=D7, TX=D6
  
  DEBUG_SERIAL.println("=== ESP32 Serial Bridge Started ===");
  DEBUG_SERIAL.println("Commands will be forwarded to Arduino UNO R4");
  DEBUG_SERIAL.println("Available commands:");
  
  for (int i = 0; i < numValidCommands; i++) {
    DEBUG_SERIAL.print("  - ");
    DEBUG_SERIAL.println(validCommands[i]);
  }
  
  DEBUG_SERIAL.println("\nType commands to send to Arduino:");
  DEBUG_SERIAL.println("Format: <command>");
  DEBUG_SERIAL.println("Example: subtle");
}

void loop() {
  // Read commands from USB Serial (for testing)
  readUSBCommands();
  
  // Process and forward commands to Arduino
  if (commandReady) {
    processCommand();
    commandReady = false;
    commandBuffer = "";
  }
  
  // Read responses from Arduino and display on debug serial
  readArduinoResponse();
  
  delay(10);
}

void readUSBCommands() {
  while (DEBUG_SERIAL.available()) {
    char c = DEBUG_SERIAL.read();
    
    if (c == '\n' || c == '\r') {
      if (commandBuffer.length() > 0) {
        commandBuffer.trim();
        commandReady = true;
        return;
      }
    } else {
      commandBuffer += c;
    }
  }
}

void processCommand() {
  DEBUG_SERIAL.print("Processing command: '");
  DEBUG_SERIAL.print(commandBuffer);
  DEBUG_SERIAL.println("'");
  
  // Validate command
  bool isValidCommand = false;
  for (int i = 0; i < numValidCommands; i++) {
    if (commandBuffer.equalsIgnoreCase(validCommands[i])) {
      isValidCommand = true;
      break;
    }
  }
  
  if (isValidCommand || commandBuffer.equalsIgnoreCase("clear")) {
    // Send command to Arduino
    ARDUINO_SERIAL.println(commandBuffer);
    DEBUG_SERIAL.print("Sent to Arduino: ");
    DEBUG_SERIAL.println(commandBuffer);
  } else {
    DEBUG_SERIAL.print("Invalid command: ");
    DEBUG_SERIAL.println(commandBuffer);
    DEBUG_SERIAL.println("Use 'help' to see available commands");
  }
}

void readArduinoResponse() {
  static String responseBuffer = "";
  
  while (ARDUINO_SERIAL.available()) {
    char c = ARDUINO_SERIAL.read();
    
    if (c == '\n') {
      if (responseBuffer.length() > 0) {
        DEBUG_SERIAL.print("[Arduino]: ");
        DEBUG_SERIAL.println(responseBuffer);
        responseBuffer = "";
      }
    } else if (c != '\r') {
      responseBuffer += c;
    }
  }
}

// Function to send command programmatically (for future Whisper integration)
void sendCommandToArduino(String command) {
  DEBUG_SERIAL.print("Voice command received: ");
  DEBUG_SERIAL.println(command);
  
  // Validate and send command
  bool isValidCommand = false;
  for (int i = 0; i < numValidCommands; i++) {
    if (command.equalsIgnoreCase(validCommands[i])) {
      isValidCommand = true;
      break;
    }
  }
  
  if (isValidCommand) {
    ARDUINO_SERIAL.println(command);
    DEBUG_SERIAL.print("Forwarded to Arduino: ");
    DEBUG_SERIAL.println(command);
  } else {
    DEBUG_SERIAL.print("Invalid voice command: ");
    DEBUG_SERIAL.println(command);
  }
}

// Placeholder for future Whisper API integration
void processVoiceCommand() {
  // TODO: Implement Whisper API integration
  // This function will:
  // 1. Capture audio from microphone
  // 2. Send audio to Whisper API
  // 3. Parse the response for valid commands
  // 4. Call sendCommandToArduino() with the recognized command
}
