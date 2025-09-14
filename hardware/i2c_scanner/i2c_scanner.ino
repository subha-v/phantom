/*
  I2C Scanner - Simple test to detect I2C devices
  This will help us determine if the DRV2605L is connected properly
*/

#include <Wire.h>

void setup() {
  Serial.begin(9600);
  
  // Wait for serial port to connect
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("=== I2C SCANNER TEST ===");
  Serial.println("Initializing I2C...");
  
  Wire.begin();
  
  Serial.println("I2C initialized successfully!");
  Serial.println("Scanning for I2C devices...");
  
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  byte error, address;
  int nDevices = 0;
  
  Serial.println("Scanning I2C addresses from 0x01 to 0x7F...");
  
  for(address = 1; address < 127; address++) {
    // Blink LED during scan
    digitalWrite(LED_BUILTIN, HIGH);
    
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    
    digitalWrite(LED_BUILTIN, LOW);
    
    if (error == 0) {
      Serial.print("I2C device found at address 0x");
      if (address < 16) Serial.print("0");
      Serial.print(address, HEX);
      Serial.println(" !");
      nDevices++;
      
      // Special check for DRV2605L (should be at 0x5A)
      if (address == 0x5A) {
        Serial.println("  ^ This looks like a DRV2605L haptic driver!");
      }
    }
    else if (error == 4) {
      Serial.print("Unknown error at address 0x");
      if (address < 16) Serial.print("0");
      Serial.println(address, HEX);
    }
    
    delay(10); // Small delay between addresses
  }
  
  if (nDevices == 0) {
    Serial.println("ERROR: No I2C devices found!");
    Serial.println("Possible issues:");
    Serial.println("1. Check wiring connections");
    Serial.println("2. Check power supply to device");
    Serial.println("3. Check pull-up resistors on SDA/SCL");
    Serial.println("4. Device might not be I2C compatible");
    
    // Fast blink pattern for error
    for (int i = 0; i < 10; i++) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(100);
      digitalWrite(LED_BUILTIN, LOW);
      delay(100);
    }
  } else {
    Serial.print("Scan complete. Found ");
    Serial.print(nDevices);
    Serial.println(" device(s).");
    
    // Slow blink pattern for success
    digitalWrite(LED_BUILTIN, HIGH);
    delay(500);
    digitalWrite(LED_BUILTIN, LOW);
    delay(500);
  }
  
  Serial.println("Waiting 3 seconds before next scan...\n");
  delay(3000);
}
