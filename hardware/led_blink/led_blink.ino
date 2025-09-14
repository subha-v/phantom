/*
  LED Blink Test for Arduino UNO R4 WiFi
  This sketch blinks the built-in LED on pin 13
*/

void setup() {
  // Initialize the built-in LED pin as an output
  pinMode(LED_BUILTIN, OUTPUT);
  
  // Initialize serial communication at 9600 bits per second
  Serial.begin(9600);
  Serial.println("Arduino UNO R4 WiFi - LED Blink Test");
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);   // Turn the LED on
  Serial.println("LED ON");
  delay(1000);                       // Wait for a second
  
  digitalWrite(LED_BUILTIN, LOW);    // Turn the LED off
  Serial.println("LED OFF");
  delay(1000);                       // Wait for a second
}
