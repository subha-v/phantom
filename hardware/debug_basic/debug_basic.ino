/*
  Basic Debug Sketch - Test Serial Communication
  This sketch only tests if serial communication is working
*/

void setup() {
  Serial.begin(9600);
  
  // Wait for serial port to connect
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("=== DEBUG: Serial communication working! ===");
  Serial.println("Arduino UNO R4 WiFi - Basic Debug Test");
  Serial.println("If you see this, serial is working fine.");
  
  // Test built-in LED too
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.println("LED test starting...");
}

void loop() {
  static int counter = 0;
  
  Serial.print("Debug loop #");
  Serial.println(counter++);
  
  // Blink LED to show it's running
  digitalWrite(LED_BUILTIN, HIGH);
  Serial.println("LED ON");
  delay(500);
  
  digitalWrite(LED_BUILTIN, LOW);
  Serial.println("LED OFF");
  delay(500);
  
  if (counter >= 10) {
    Serial.println("=== 10 loops completed, continuing... ===");
    counter = 0;
    delay(2000);
  }
}
