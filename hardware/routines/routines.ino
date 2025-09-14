#include <Modulino.h>
#include <Wire.h>
#include "Adafruit_DRV2605.h"

#define LED_COUNT 8

ModulinoPixels leds;
ModulinoKnob knob;
Adafruit_DRV2605 drv;
ModulinoColor OFF(0, 0, 0);



inline void setOff(int i) { 
  leds.set(i, OFF, 0);   // color=0, brightness=0 = OFF
}

int brightness = 25;

  String readCommand() {
  static String buf;
  static uint32_t last = 0;

  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    last = millis();
    if (c == '\n' || c == '\r') {
      String out = buf; out.trim(); buf = "";
      if (out.length()) return out;
    } else {
      buf += c;
    }
  }
  // If user typed something but didn't send newline, accept after idle
  if (buf.length() && (millis() - last > 150)) {
    String out = buf; out.trim(); buf = "";
    return out;
  }
  return "";
}

void setup() {
  // put your setup code here, to run once:
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(9600);
  delay(1000);

  // Initialize Wire1 for Qwiic connector on R4 WiFi
  Wire1.begin();
  Wire1.setClock(100000);

  Serial.println("=== Phantom Feedback Test (Modulino + DRV2605) ===");

  Modulino.begin();

  initLeds();
  initHaptic();  


}

uint8_t effect = 1; // effect variable

void loop() {
//  String cmd = readCommand();
//   if (!cmd.length()) return;
//  if (cmd.equalsIgnoreCase("visual")) {
//     visualFeedback(VIOLET);
//   }  else if (cmd.equalsIgnoreCase("cb")){
//     clearLedsBrute();
//   } else if(cmd.equalsIgnoreCase("haptic")){
//     playHaptic(3);
//   }

// Turn all on clearly BLUE

  visualFeedback(BLUE);
  delay(800);

  // Hard clear
  clearLedsHard();
  delay(800);

}



void visualFeedback(ModulinoColor color) {
  Serial.println("Soft feedback pattern (battery charge)");
  // Blue cycle
  for (int i = 0; i < LED_COUNT; i++) {
    setPixel(i, color);
    delay(100);
  }

}




// ================ init methods ============= //

void initLeds(){
   // begin LEDs
  if (!leds.begin()) {
    Serial.println("Failed to initialize Modulino Pixels!");
    while (1) delay(10);
  }

  Serial.println("Modulino Pixels initialized successfully!");
  clearLedsHard();
}

void initHaptic() {
    if (!drv.begin(&Wire1)) {
      Serial.println("ERROR: DRV2605 not found on Wire or Wire1 (addr 0x5A). Check wiring/power.");
      while (1) { digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); delay(300); }
    }
  

  drv.selectLibrary(1);
  drv.setMode(DRV2605_MODE_INTTRIG);  // internal trigger (we call go())
  Serial.println("DRV2605 OK");
}

void initKnob() {
  
  if (!knob.begin()) {
    Serial.println("WARNING: Modulino Knob not found (continuing)");
    return;
  }
  Serial.println("Modulino Knob OK");
}

// ============ LED COMMANDS ============ //
void setPixel(int pixel, ModulinoColor color) {
  leds.set(pixel, color, brightness);
  leds.show();
}


void clearLedsBrute() {
  for (int i = 0; i < 8; i++) {
    setPixel(i, OFF);
    delay(25);
  }
}

void clearLedsHard() {
  Serial.println("CLEARING LEDS (hard)");
  // Write OFF to more pixels than you think you have, then show twice
  for (int i = 0; i < LED_COUNT; i++) setOff(i);
  leds.show();
  delayMicroseconds(300);          // WS2812 latch >50µs; be generous
  for (int i = 0; i < LED_COUNT; i++) setOff(i);
  leds.show();
}

// ============ HAPTIC COMMANDS =========== //

// -------- Haptic helper --------
void playHaptic(uint8_t fx) {
  Serial.print("Haptic effect "); Serial.println(fx);
  for (int i = 0; i < 3; i++){
    drv.setWaveform(0, fx);  // slot 0
    drv.setWaveform(1, 0);   // end
    drv.go();
    delay(300);
  }

  delay(1000);
}