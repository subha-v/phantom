/*
 * Simple ESP32 Serial Test
 * Tests basic serial communication to Arduino
 */

#define ARDUINO_SERIAL Serial1  // Use Serial1 for communication with Arduino
#define DEBUG_SERIAL Serial     // Use Serial for debugging via USB

void setup() {
  DEBUG_SERIAL.begin(115200);
  delay(1000);
  
  // Initialize Arduino communication serial
  // ESP32C3 Serial1: TX=D6, RX=D7 (Xiao ESP32C3 pinout)
  ARDUINO_SERIAL.begin(9600, SERIAL_8N1, 7, 6);  // RX=D7, TX=D6w
  
  DEBUG_SERIAL.println("=== ESP32 Serial Test Started ===");
  DEBUG_SERIAL.println("Type any message to send to Arduino");
}

void loop() {
  // Read from USB Serial and send to Arduino
  if (DEBUG_SERIAL.available()) {
    String message = DEBUG_SERIAL.readStringUntil('\n');
    message.trim();
    
    if (message.length() > 0) {
      DEBUG_SERIAL.print("Sending to Arduino: ");
      DEBUG_SERIAL.println(message);
      
      ARDUINO_SERIAL.println(message);
      ARDUINO_SERIAL.flush();  // Ensure data is sent
    }
  }
  
  // Read from Arduino and display on USB Serial
  if (ARDUINO_SERIAL.available()) {
    String response = ARDUINO_SERIAL.readStringUntil('\n');
    response.trim();
    
    if (response.length() > 0) {
      DEBUG_SERIAL.print("Arduino says: ");
      DEBUG_SERIAL.println(response);
    }
  }
  
  // Send periodic test message
  static unsigned long lastTest = 0;
  if (millis() - lastTest > 5000) {
    DEBUG_SERIAL.println("Sending test message...");
    ARDUINO_SERIAL.println("test");
    ARDUINO_SERIAL.flush();
    lastTest = millis();
  }
}
