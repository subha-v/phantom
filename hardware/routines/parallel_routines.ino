#include <Modulino.h>
#include <Wire.h>
#include "Adafruit_DRV2605.h"

#define LED_COUNT 8

ModulinoPixels leds;
ModulinoKnob knob;
Adafruit_DRV2605 drv;

// Color definitions
ModulinoColor OFF(0, 0, 0);
ModulinoColor VIOLETY(128, 0, 255);
ModulinoColor REDY(255, 0, 0);
ModulinoColor GREENY(0, 255, 0);
ModulinoColor BLUEY(0, 0, 255);
ModulinoColor YELLOWY(255, 255, 0);
ModulinoColor CYAN(0, 255, 255);
ModulinoColor WHITEY(255, 255, 255);
ModulinoColor PURPLE(255, 0, 255);

inline void setOff(int i) { 
  leds.set(i, OFF, 0);   // color=0, brightness=0 = OFF
}

int brightness = 5;

// Timing variables for non-blocking animations
unsigned long ledTimer = 0;
unsigned long hapticTimer = 0;

// Animation state variables
bool feedbackActive = false;
bool ledAnimationActive = false;
bool hapticAnimationActive = false;

// Animation counters
int currentLED = 0;
int currentCycle = 0;
int hapticCycle = 0;


// sate variable
int stateStep = 0;  // Used by state machine feedback


unsigned long stateTimer = 0;

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
  String cmd = readCommand();
  if (!cmd.length()) return;
  
  // Visual feedback routines
  if (cmd.equalsIgnoreCase("soft")) {
    softVisualFeedback();
  } else if (cmd.equalsIgnoreCase("modvisual")) {
    moderateVisualFeedback();
  } else if (cmd.equalsIgnoreCase("intvisual")) {
    intenseVisualFeedback();
  } else if (cmd.equalsIgnoreCase("rainbow")) {
    rainbowChase();
  } else if (cmd.equalsIgnoreCase("binary")) {
    binaryCounter();
  } else if (cmd.equalsIgnoreCase("heartbeat") || cmd.equalsIgnoreCase("heart")) {
    heartbeat();
  } 
  
  // Test routines
  else if (cmd.equalsIgnoreCase("individual") || cmd.equalsIgnoreCase("test")) {
    testIndividualLEDs();
  } else if (cmd.equalsIgnoreCase("basic") || cmd.equalsIgnoreCase("all")) {
    testBasicLEDs();
  } 
  
  // Original commands
  else if (cmd.equalsIgnoreCase("visual")) {
    testIndividualLEDs(); // Use the working function
  } else if (cmd.equalsIgnoreCase("haptic")) {
    playHaptic(3);
  } 

  // Parallel feedback commands
else if (cmd.equalsIgnoreCase("subtle") || cmd.equalsIgnoreCase("sub")) {
  parallelSimpleFeedback();
} else if (cmd.equalsIgnoreCase("moderate") || cmd.equalsIgnoreCase("mod")) {
  advancedParallelFeedback();
} else if (cmd.equalsIgnoreCase("high")) {
  synchronizedFeedback();
} 
  
  // Help and clear
  else if (cmd.equalsIgnoreCase("help")) {
    Serial.println("\n=== AVAILABLE COMMANDS ===");
    Serial.println("Visual Routines:");
    Serial.println("  'soft' - Gentle breathing effect");
    Serial.println("  'moderate' - Wave pattern");
    Serial.println("  'intense' - Rapid flashing");
    Serial.println("  'rainbow' - Rainbow chase");
    Serial.println("  'binary' - Binary counter");
    Serial.println("  'heartbeat' - Heartbeat pattern");
    Serial.println("\nTest Commands:");
    Serial.println("  'individual' - Test each LED");
    Serial.println("  'basic' - Test all LEDs at once");
    Serial.println("  'haptic' - Test haptic feedback");
    Serial.println("  'help' - Show this menu");
    Serial.println("  Any other input - Clear LEDs\n");
  } else {
    clearLedsHard();
  }
}

// ================ INIT METHODS ============= //

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

// ============ VISUAL FEEDBACK ROUTINES ============ //

// 1. SOFT FEEDBACK - Gentle breathing wave
void softVisualFeedback() {
  Serial.println("=== SOFT VISUAL FEEDBACK ===");
  
  // Breathing effect - fade in and out
  for (int cycle = 0; cycle < 2; cycle++) {
    // Fade in
    for (int bright = 1; bright <= 8; bright++) {
      for (int i = 0; i < LED_COUNT; i++) {
        leds.set(i, YELLOWY, bright);
      }
      leds.show();
      delay(150);
    }
    
    // Hold
    delay(300);
    
    // Fade out
    for (int bright = 8; bright >= 1; bright--) {
      for (int i = 0; i < LED_COUNT; i++) {
        leds.set(i, BLUEY, bright);
      }
      leds.show();
      delay(150);
    }
    
    clearLedsHard();
    delay(200);
  }
  
  Serial.println("Soft feedback complete");
}

// 2. MODERATE FEEDBACK - Wave pattern
void moderateVisualFeedback() {
  Serial.println("=== MODERATE VISUAL FEEDBACK ===");
  
  // Forward and backward wave
  for (int wave = 0; wave < 2; wave++) {
    // Forward wave
    for (int i = 0; i < LED_COUNT; i++) {
      clearLedsHard();
      
      // Current LED bright, trailing LEDs dim
      leds.set(i, VIOLETY, brightness);
      if (i > 0) leds.set(i-1, VIOLETY, brightness/2);
      if (i > 1) leds.set(i-2, VIOLETY, brightness/4);
      
      leds.show();
      delay(100);
    }
    
    // Backward wave
    for (int i = LED_COUNT-1; i >= 0; i--) {
      clearLedsHard();
      
      // Current LED bright, trailing LEDs dim
      leds.set(i, VIOLETY, brightness);
      if (i < LED_COUNT-1) leds.set(i+1, VIOLETY, brightness/2);
      if (i < LED_COUNT-2) leds.set(i+2, VIOLETY, brightness/4);
      
      leds.show();
      delay(100);
    }
  }
  
  clearLedsHard();
  Serial.println("Moderate feedback complete");
}

// 3. INTENSE FEEDBACK - Rapid flashing and pulses
void intenseVisualFeedback() {
  Serial.println("=== INTENSE VISUAL FEEDBACK ===");
  
  // Rapid flashing sequence
  for (int flash = 0; flash < 6; flash++) {
    for (int i = 0; i < LED_COUNT; i++) {
      leds.set(i, GREENY, brightness);
    }
    leds.show();
    delay(80);
    
    clearLedsHard();
    delay(80);
  }
  
  // Pulsing sequence
  for (int pulse = 0; pulse < 3; pulse++) {
    for (int bright = 1; bright <= brightness; bright++) {
      for (int i = 0; i < LED_COUNT; i++) {
        leds.set(i, GREENY, bright);
      }
      leds.show();
      delay(50);
    }
    
    for (int bright = brightness; bright >= 1; bright--) {
      for (int i = 0; i < LED_COUNT; i++) {
        leds.set(i, GREENY, bright);
      }
      leds.show();
      delay(50);
    }
    
    clearLedsHard();
    delay(100);
  }
  
  Serial.println("Intense feedback complete");
}

// 4. RAINBOW CHASE - Individual LEDs in sequence with colors
void rainbowChase() {
  Serial.println("=== RAINBOW CHASE ===");
  
  ModulinoColor colors[] = {REDY, YELLOWY, GREENY, CYAN, BLUEY, VIOLETY, PURPLE, WHITEY};
  
  // Multiple passes
  for (int pass = 0; pass < 3; pass++) {
    for (int i = 0; i < LED_COUNT; i++) {
      clearLedsHard();
      delay(50);
      
      // Light current LED with its color
      leds.set(i, colors[i], brightness);
      leds.show();
      delay(200);
    }
  }
  
  clearLedsHard();
  Serial.println("Rainbow chase complete");
}

// 5. BINARY COUNTER - Shows counting pattern
void binaryCounter() {
  Serial.println("=== BINARY COUNTER ===");
  
  for (int count = 0; count < 16; count++) {
    clearLedsHard();
    delay(100);
    
    Serial.print("Count: "); Serial.println(count);
    
    // Display binary representation
    for (int bit = 0; bit < 4 && bit < LED_COUNT; bit++) {
      if (count & (1 << bit)) {
        leds.set(bit, REDY, brightness);
      }
    }
    leds.show();
    delay(400);
  }
  
  clearLedsHard();
  Serial.println("Binary counter complete");
}

// 6. HEARTBEAT PATTERN
void heartbeat() {
  Serial.println("=== HEARTBEAT PATTERN ===");
  
  for (int beat = 0; beat < 5; beat++) {
    // First beat
    for (int i = 0; i < LED_COUNT; i++) {
      leds.set(i, REDY, brightness);
    }
    leds.show();
    delay(100);
    
    clearLedsHard();
    delay(100);
    
    // Second beat
    for (int i = 0; i < LED_COUNT; i++) {
      leds.set(i, REDY, brightness/2);
    }
    leds.show();
    delay(100);
    
    clearLedsHard();
    delay(500); // Pause between heartbeats
  }
  
  Serial.println("Heartbeat complete");
}

// Working individual test (your working function)
void testIndividualLEDs() {
  Serial.println("\n=== Testing Individual LEDs ===");
  
  for (int i = 0; i < LED_COUNT; i++) {
    // Clear all first
    clearLedsHard();
    delay(200);
    
    // Light just this one
    Serial.print("Testing LED "); Serial.print(i); Serial.print(": ");
    leds.set(i, BLUEY
    , brightness);
    leds.show();
    delay(500); // Longer to see clearly
    
    Serial.println("ON");
  }
  
  clearLedsHard();
  Serial.println("Individual test complete\n");
}

// Fixed version of your basic test
void testBasicLEDs() {
  Serial.println("=== Basic LED test - all at once ===");
  
  // Set all LEDs to same color, very low brightness
  for (int i = 0; i < LED_COUNT; i++) {
    leds.set(i, REDY, 2); // Very dim to avoid power issues
  }
  leds.show();
  delay(2000);
  
  clearLedsHard();
  Serial.println("Basic test complete\n");
}

// Your original visual feedback method (fixed)
void visualFeedback(ModulinoColor color) {
  Serial.println("=== Visual Feedback Debug ===");
  
  // First, clear everything
  clearLedsHard();
  delay(500);
  
  // Test each LED individually with status
  for (int i = 0; i < LED_COUNT; i++) {
    Serial.print("Setting LED "); Serial.print(i); Serial.print("... ");
    
    // Set this LED (keeping previous ones on)
    leds.set(i, color, brightness);
    leds.show();
    
    Serial.println("done");
    delay(30); // Longer delay for observation
  }
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

// ============ PARALLEL FEEDBACK FUNCTIONS ============ //

// Non-blocking color chase function
void updateColorChase(ModulinoColor color) {
  const int CHASE_DELAY = 100; // milliseconds between LED updates
  
  if (millis() - ledTimer >= CHASE_DELAY) {
    if (currentLED < LED_COUNT) {
      // Clear previous LED and light current one
      if (currentLED > 0) {
        leds.set(currentLED - 1, OFF, 0);
      }
      leds.set(currentLED, color, brightness);
      leds.show();
      
      currentLED++;
      ledTimer = millis();
    } else {
      // Chase cycle complete, clear all and reset
      clearLedsHard();
      currentLED = 0;
      currentCycle++;
      ledTimer = millis();
      
      // Check if all cycles are done
      if (currentCycle >= 3) {
        ledAnimationActive = false;
      }
    }
  }
}

// Non-blocking haptic function
void updateHaptic(uint8_t effect) {
  const int HAPTIC_INTERVAL = 500; // milliseconds between haptic pulses
  
  if (millis() - hapticTimer >= HAPTIC_INTERVAL) {
    if (hapticCycle < 3) {
      Serial.print("Haptic pulse "); Serial.println(hapticCycle + 1);
      drv.setWaveform(0, effect);
      drv.setWaveform(1, 0);
      drv.go();
      
      hapticCycle++;
      hapticTimer = millis();
    } else {
      // Haptic cycles complete
      hapticAnimationActive = false;
    }
  }
}

// PARALLEL VERSION: LED chase and haptic simultaneously
void parallelSimpleFeedback() {
  Serial.println("===== PARALLEL SIMPLE FEEDBACK =====");
  
  // Initialize parallel feedback
  feedbackActive = true;
  ledAnimationActive = true;
  hapticAnimationActive = true;
  
  // Reset counters
  currentLED = 0;
  currentCycle = 0;
  hapticCycle = 0;
  ledTimer = millis();
  hapticTimer = millis();
  
  // Run both animations in parallel
  while (feedbackActive && (ledAnimationActive || hapticAnimationActive)) {
    // Update LED chase
    if (ledAnimationActive) {
      updateColorChase(CYAN);
    }
    
    // Update haptic feedback
    if (hapticAnimationActive) {
      updateHaptic(1);
    }
    
    // Check if both animations are complete
    if (!ledAnimationActive && !hapticAnimationActive) {
      feedbackActive = false;
      Serial.println("Parallel feedback complete");
    }
    
    // Small delay to prevent overwhelming the processor
    delay(10);
  }
}

// ADVANCED PARALLEL: Custom timing for each
void advancedParallelFeedback() {
  Serial.println("===== ADVANCED PARALLEL FEEDBACK =====");
  
  unsigned long startTime = millis();
  unsigned long ledLastUpdate = 0;
  unsigned long hapticLastUpdate = 0;
  
  int ledStep = 0;
  int hapticStep = 0;
  
  const int LED_INTERVAL = 150;    // LED updates every 150ms
  const int HAPTIC_INTERVAL = 800; // Haptic every 800ms
  const int TOTAL_DURATION = 5000; // Total feedback duration: 5 seconds
  
  while (millis() - startTime < TOTAL_DURATION) {
    unsigned long currentTime = millis();
    
    // Handle LED animation
    if (currentTime - ledLastUpdate >= LED_INTERVAL) {
      clearLedsHard();
      
      // Create a moving wave pattern
      int centerLED = ledStep % LED_COUNT;
      leds.set(centerLED, VIOLETY, brightness);
      
      if (centerLED > 0) leds.set(centerLED - 1, VIOLETY, brightness/3);
      if (centerLED < LED_COUNT - 1) leds.set(centerLED + 1, VIOLETY, brightness/3);
      
      leds.show();
      
      ledStep++;
      ledLastUpdate = currentTime;
    }
    
    // Handle haptic feedback
    if (currentTime - hapticLastUpdate >= HAPTIC_INTERVAL) {
      Serial.print("Haptic at "); Serial.println(currentTime - startTime);
      
      // Alternating haptic effects
      uint8_t effect = (hapticStep % 2 == 0) ? 1 : 10;
      drv.setWaveform(0, effect);
      drv.setWaveform(1, 0);
      drv.go();
      
      hapticStep++;
      hapticLastUpdate = currentTime;
    }
    
    delay(10); // Small delay for stability
  }
  
  clearLedsHard();
  Serial.println("Advanced parallel feedback complete");
}

// SYNCHRONIZED VERSION: LED rainbow and haptic 
void synchronizedFeedback() {
  Serial.println("===== SYNCHRONIZED FEEDBACK WITH RAINBOW =====");
  
  const int SYNC_CYCLES = 4;
  const int CYCLE_DURATION = 1200; // Slightly longer to accommodate rainbow timing
  
  // Rainbow colors array
  ModulinoColor rainbowColors[] = {REDY, YELLOWY, GREENY, CYAN, BLUEY, VIOLETY, PURPLE, WHITEY};
  
  for (int cycle = 0; cycle < SYNC_CYCLES; cycle++) {
    Serial.print("Sync cycle "); Serial.println(cycle + 1);
    
    unsigned long cycleStart = millis();
    
    // Start haptic immediately with varying effects per cycle
    uint8_t hapticEffect = 1 + (cycle % 3); // Vary between effects 1, 2, 3
    drv.setWaveform(0, hapticEffect);
    drv.setWaveform(1, 0);
    drv.go();
    
    // Rainbow LED animation during haptic
    int colorIndex = 0;
    int ledIndex = 0;
    unsigned long lastLedUpdate = millis();
    const int LED_UPDATE_INTERVAL = 100; // Update LEDs every 100ms
    
    while (millis() - cycleStart < CYCLE_DURATION) {
      // Update rainbow chase at regular intervals
      if (millis() - lastLedUpdate >= LED_UPDATE_INTERVAL) {
        clearLedsHard();
        
        // Display current LED with current rainbow color
        leds.set(ledIndex, rainbowColors[colorIndex], brightness);
        
        // Add trailing effect for smoother visual
        if (ledIndex > 0) {
          leds.set(ledIndex - 1, rainbowColors[colorIndex], brightness/3);
        } else if (ledIndex == 0) {
          leds.set(LED_COUNT - 1, rainbowColors[colorIndex], brightness/3);
        }
        
        leds.show();
        
        // Advance to next LED
        ledIndex++;
        if (ledIndex >= LED_COUNT) {
          ledIndex = 0;
          colorIndex = (colorIndex + 1) % 8; // Cycle through rainbow colors
        }
        
        lastLedUpdate = millis();
      }
      
      delay(10); // Small delay for stability
    }
    
    clearLedsHard();
    delay(300); // Brief pause between cycles
  }
  
  Serial.println("Synchronized rainbow feedback complete");
}




// Helper function for color chase
void colorChase(ModulinoColor color) {
  for (int i = 0; i < LED_COUNT; i++) {
    clearLedsHard();
    leds.set(i, color, brightness);
    leds.show();
    delay(100);
  }
  clearLedsHard();
}