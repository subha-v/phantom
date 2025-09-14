/*
  Stop/Empty Sketch for Arduino UNO R4 WiFi
  This sketch does nothing - use it to quickly halt any running program
*/

void setup() {
  // Initialize serial for confirmation
  Serial.begin(9600);
  Serial.println("Arduino program stopped - board is idle");
  
  // Turn off built-in LED
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
}

void loop() {
  // Do nothing - board is effectively stopped
  delay(1000);
}
