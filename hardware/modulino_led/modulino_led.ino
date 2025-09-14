#include <Modulino.h>

ModulinoPixels leds;

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(9600);
  delay(1000);
  Serial.println("***** MODULINO PIXELS TEST ****");

  if (!leds.begin()) {
    Serial.println("Failed to initialize Modulino Pixels!");
    while (1) delay(10);
  }
  
  Serial.println("Modulino Pixels initialized successfully!");
  leds.clear();
  leds.show();
}

void loop() {
  Serial.println("Cycling through LED colors...");
  
  // Red cycle
  for (int i = 0; i < 8; i++) {
    leds.set(i, GREEN, 50);  // Red with brightness 50
    leds.show();
    delay(500);
    Serial.print("LED ");
    Serial.print(i);
    Serial.println(" GREEN");
  }
  
  delay(500);
  
}
