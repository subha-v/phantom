#include <Modulino.h>
#include <Wire.h>
#include "Adafruit_DRV2605.h"
#include <math.h>

#define LED_COUNT 8

// Modulino components
ModulinoPixels leds;
ModulinoKnob knob;
ModulinoDistance distance;
ModulinoBuzzer buzzer;
Adafruit_DRV2605 drv;

// Color definitions - preserving your original variables
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

// Proximity feedback parameters
const int TOUCH_THRESHOLD = 30;      // Distance threshold for "touch" detection
const int PROXIMITY_THRESHOLD = 100;  // Distance for proximity awareness
const int BASE_FREQUENCY = 200;      // Lower frequency for haptic feel
const int MAX_FREQUENCY = 800;       // Higher frequency for urgent feedback

// Timing variables for non-blocking animations
unsigned long ledTimer = 0;
unsigned long hapticTimer = 0;
unsigned long lastFeedbackTime = 0;
unsigned long feedbackInterval = 50;
int lastMeasure = 0;
bool isInContact = false;

// Animation state variables
bool feedbackActive = false;
bool ledAnimationActive = false;
bool hapticAnimationActive = false;
bool proximityFeedbackActive = true;

// Animation counters
int currentLED = 0;
int currentCycle = 0;
int hapticCycle = 0;
int stateStep = 0;
unsigned long stateTimer = 0;

// Haptic patterns from proximity sensor
enum HapticMode {
  NO_FEEDBACK,
  LIGHT_TOUCH,
  FIRM_CONTACT,
};

HapticMode currentMode = NO_FEEDBACK;
uint8_t effect = 1;

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

  Serial.println("=== Phantom Master Proximity Feedback System ===");
  Serial.println("Ready for ESP32 serial commands");

  Modulino.begin();
  distance.begin();
  buzzer.begin();

  initLeds();
  initHaptic();
  
  Serial.println("Proximity feedback system initialized");
  Serial.println("Distance ranges:");
  Serial.println("- Contact: 0-30cm");
  Serial.println("- Proximity: 31-100cm");
  Serial.println("- No feedback: >100cm");
}

void loop() {
  // Continuously monitor proximity for feedback
  updateProximityFeedback();
  
  // Handle user commands
  String cmd = readCommand();
  if (!cmd.length()) return;
  
  Serial.print("Command received: ");
  Serial.println(cmd);
  
  // Flash built-in LED to indicate command received
  digitalWrite(LED_BUILTIN, HIGH);
  delay(100);
  digitalWrite(LED_BUILTIN, LOW);
  
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
    testIndividualLEDs();
  } else if (cmd.equalsIgnoreCase("haptic")) {
    playHaptic(3);
  } 

  // Proximity-based feedback commands for whispr mcp
  else if (cmd.equalsIgnoreCase("subtle") || cmd.equalsIgnoreCase("sub")) {
    subtleFeedback();
  } else if (cmd.equalsIgnoreCase("moderate") || cmd.equalsIgnoreCase("mod")) {
    moderateFeedback();
  } else if (cmd.equalsIgnoreCase("high")) {
    highFeedback();
  }
  
  // Proximity control commands
  else if (cmd.equalsIgnoreCase("proximity") || cmd.equalsIgnoreCase("prox")) {
    toggleProximityFeedback();
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
    Serial.println("\nPain Feedback (whispr mcp):");
    Serial.println("  'subtle' - Gentle pain feedback");
    Serial.println("  'moderate' - Medium pain feedback");
    Serial.println("  'high' - Intense pain feedback");
    Serial.println("\nProximity Feedback:");
    Serial.println("  'proximity' - Toggle continuous proximity monitoring");
    Serial.println("\nTest Commands:");
    Serial.println("  'individual' - Test each LED");
    Serial.println("  'basic' - Test all LEDs at once");
    Serial.println("  'haptic' - Test haptic feedback");
    Serial.println("  'help' - Show this menu");
    Serial.println("  Any other input - Clear LEDs\n");
  } else {
    Serial.print("Unknown command: ");
    Serial.println(cmd);
    clearLedsHard();
  }
}

// ================ PROXIMITY FEEDBACK SYSTEM ============= //

void updateProximityFeedback() {
  if (!proximityFeedbackActive) return;
  
  if (distance.available()) {
    int currentMeasure = distance.get();
    
    // Only process if measurement has changed significantly
    if (abs(currentMeasure - lastMeasure) > 2) {
      processDistanceFeedback(currentMeasure);
      lastMeasure = currentMeasure;
    }
  }
  
  // Execute haptic feedback based on current mode
  executeHapticFeedback();
  
  // Execute visual feedback based on current mode
  executeVisualFeedback();
}

void processDistanceFeedback(int measure) {
  Serial.print("Distance: ");
  Serial.print(measure);
  Serial.print("cm - ");
  
  if (measure <= 10) {
    currentMode = FIRM_CONTACT;
    Serial.println("FIRM CONTACT");
  } else if (measure <= TOUCH_THRESHOLD) {
    currentMode = LIGHT_TOUCH;
    Serial.println("LIGHT TOUCH");
  } else {
    currentMode = NO_FEEDBACK;
    Serial.println("NO CONTACT");
  }
  
  isInContact = (measure <= TOUCH_THRESHOLD);
}

void executeHapticFeedback() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastFeedbackTime < feedbackInterval) {
    return;
  }
  
  lastFeedbackTime = currentTime;
  
  switch (currentMode) {
    case NO_FEEDBACK:
      buzzer.tone(0, 0);
      feedbackInterval = 100;
      break;
      
    case LIGHT_TOUCH:
      lightTouchFeedback();
      feedbackInterval = 50;
      break;
      
    case FIRM_CONTACT:
      firmContactFeedback();
      feedbackInterval = 30;
      break;
  }
}

void lightTouchFeedback() {
  buzzer.tone(BASE_FREQUENCY + 50, 200);
  
  // Add DRV2605 haptic for light touch
  drv.setWaveform(0, 1);
  drv.setWaveform(1, 0);
  drv.go();
}

void firmContactFeedback() {
  buzzer.tone(BASE_FREQUENCY + 200, 150);
  
  // Add DRV2605 haptic for firm contact
  drv.setWaveform(0, 10);
  drv.setWaveform(1, 0);
  drv.go();
}

void executeVisualFeedback() {
  static unsigned long lastVisualUpdate = 0;
  static int visualStep = 0;
  static bool waveDirection = true;
  unsigned long currentTime = millis();
  
  // Update visual feedback based on current mode
  switch (currentMode) {
    case NO_FEEDBACK:
      // Clear LEDs when no contact
      if (currentTime - lastVisualUpdate > 200) {
        clearLedsHard();
        lastVisualUpdate = currentTime;
      }
      break;
      
    case LIGHT_TOUCH:
    case FIRM_CONTACT:
      // Moderate-style wave pattern for any contact
      if (currentTime - lastVisualUpdate > 100) {
        clearLedsHard();
        
        int currentLED = visualStep % LED_COUNT;
        ModulinoColor feedbackColor = (currentMode == FIRM_CONTACT) ? REDY : VIOLETY;
        int feedbackBrightness = (currentMode == FIRM_CONTACT) ? brightness : brightness/2;
        
        // Wave pattern with trailing effect
        leds.set(currentLED, feedbackColor, feedbackBrightness);
        if (currentLED > 0) leds.set(currentLED-1, feedbackColor, feedbackBrightness/2);
        if (currentLED > 1) leds.set(currentLED-2, feedbackColor, feedbackBrightness/4);
        
        leds.show();
        
        // Visual feedback only - haptic is handled in executeHapticFeedback()
        
        visualStep++;
        lastVisualUpdate = currentTime;
      }
      break;
  }
}

void toggleProximityFeedback() {
  proximityFeedbackActive = !proximityFeedbackActive;
  Serial.print("Proximity feedback: ");
  Serial.println(proximityFeedbackActive ? "ENABLED" : "DISABLED");
  
  if (!proximityFeedbackActive) {
    buzzer.tone(0, 0); // Stop buzzer
    currentMode = NO_FEEDBACK;
  }
}

// ================ PAIN FEEDBACK COMMANDS (WHISPR MCP) ============= //

void subtleFeedback() {
  Serial.println("=== SUBTLE PAIN FEEDBACK ===");
  
  // Gentle blue breathing with light haptic
  for (int cycle = 0; cycle < 3; cycle++) {
    for (int bright = 1; bright <= brightness/2; bright++) {
      for (int i = 0; i < LED_COUNT; i++) {
        leds.set(i, BLUEY, bright);
      }
      leds.show();
      delay(100);
    }
    
    // Light haptic pulse
    drv.setWaveform(0, 1);
    drv.setWaveform(1, 0);
    drv.go();
    
    for (int bright = brightness/2; bright >= 1; bright--) {
      for (int i = 0; i < LED_COUNT; i++) {
        leds.set(i, BLUEY, bright);
      }
      leds.show();
      delay(100);
    }
    
    delay(200);
  }
  
  clearLedsHard();
  Serial.println("Subtle feedback complete");
}

void moderateFeedback() {
  Serial.println("=== MODERATE PAIN FEEDBACK ===");
  
  // Wave pattern with medium haptic
  for (int wave = 0; wave < 2; wave++) {
    for (int i = 0; i < LED_COUNT; i++) {
      clearLedsHard();
      
      leds.set(i, YELLOWY, brightness);
      if (i > 0) leds.set(i-1, YELLOWY, brightness/2);
      if (i > 1) leds.set(i-2, YELLOWY, brightness/4);
      
      leds.show();
      
      // Medium haptic every 3rd LED
      if (i % 3 == 0) {
        drv.setWaveform(0, 5);
        drv.setWaveform(1, 0);
        drv.go();
      }
      
      delay(120);
    }
  }
  
  clearLedsHard();
  Serial.println("Moderate feedback complete");
}

void highFeedback() {
  Serial.println("=== HIGH PAIN FEEDBACK ===");
  
  // Intense red flashing with strong haptic
  for (int flash = 0; flash < 5; flash++) {
    for (int i = 0; i < LED_COUNT; i++) {
      leds.set(i, REDY, brightness);
    }
    leds.show();
    
    // Strong haptic pulse
    drv.setWaveform(0, 14);
    drv.setWaveform(1, 0);
    drv.go();
    
    delay(150);
    
    clearLedsHard();
    delay(100);
  }
  
  // Additional intense pattern
  for (int i = 0; i < LED_COUNT; i++) {
    clearLedsHard();
    leds.set(i, REDY, brightness);
    leds.show();
    
    drv.setWaveform(0, 10);
    drv.setWaveform(1, 0);
    drv.go();
    
    delay(80);
  }
  
  clearLedsHard();
  Serial.println("High feedback complete");
}

// ================ INIT METHODS ============= //

void initLeds(){
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
  drv.setMode(DRV2605_MODE_INTTRIG);
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
  for (int i = 0; i < LED_COUNT; i++) setOff(i);
  leds.show();
  delayMicroseconds(300);
  for (int i = 0; i < LED_COUNT; i++) setOff(i);
  leds.show();
}

// ============ VISUAL FEEDBACK ROUTINES ============ //

void softVisualFeedback() {
  Serial.println("=== SOFT VISUAL FEEDBACK ===");
  
  for (int cycle = 0; cycle < 2; cycle++) {
    for (int bright = 1; bright <= 8; bright++) {
      for (int i = 0; i < LED_COUNT; i++) {
        leds.set(i, YELLOWY, bright);
      }
      leds.show();
      delay(150);
    }
    
    delay(300);
    
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

void moderateVisualFeedback() {
  Serial.println("=== MODERATE VISUAL FEEDBACK ===");
  
  for (int wave = 0; wave < 2; wave++) {
    for (int i = 0; i < LED_COUNT; i++) {
      clearLedsHard();
      
      leds.set(i, VIOLETY, brightness);
      if (i > 0) leds.set(i-1, VIOLETY, brightness/2);
      if (i > 1) leds.set(i-2, VIOLETY, brightness/4);
      
      leds.show();
      delay(100);
    }
    
    for (int i = LED_COUNT-1; i >= 0; i--) {
      clearLedsHard();
      
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

void intenseVisualFeedback() {
  Serial.println("=== INTENSE VISUAL FEEDBACK ===");
  
  for (int flash = 0; flash < 6; flash++) {
    for (int i = 0; i < LED_COUNT; i++) {
      leds.set(i, GREENY, brightness);
    }
    leds.show();
    delay(80);
    
    clearLedsHard();
    delay(80);
  }
  
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

void rainbowChase() {
  Serial.println("=== RAINBOW CHASE ===");
  
  ModulinoColor colors[] = {REDY, YELLOWY, GREENY, CYAN, BLUEY, VIOLETY, PURPLE, WHITEY};
  
  for (int pass = 0; pass < 3; pass++) {
    for (int i = 0; i < LED_COUNT; i++) {
      clearLedsHard();
      delay(50);
      
      leds.set(i, colors[i], brightness);
      leds.show();
      delay(200);
    }
  }
  
  clearLedsHard();
  Serial.println("Rainbow chase complete");
}

void binaryCounter() {
  Serial.println("=== BINARY COUNTER ===");
  
  for (int count = 0; count < 16; count++) {
    clearLedsHard();
    delay(100);
    
    Serial.print("Count: "); Serial.println(count);
    
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

void heartbeat() {
  Serial.println("=== HEARTBEAT PATTERN ===");
  
  for (int beat = 0; beat < 5; beat++) {
    for (int i = 0; i < LED_COUNT; i++) {
      leds.set(i, REDY, brightness);
    }
    leds.show();
    delay(100);
    
    clearLedsHard();
    delay(100);
    
    for (int i = 0; i < LED_COUNT; i++) {
      leds.set(i, REDY, brightness/2);
    }
    leds.show();
    delay(100);
    
    clearLedsHard();
    delay(500);
  }
  
  Serial.println("Heartbeat complete");
}

void testIndividualLEDs() {
  Serial.println("\n=== Testing Individual LEDs ===");
  
  for (int i = 0; i < LED_COUNT; i++) {
    clearLedsHard();
    delay(200);
    
    Serial.print("Testing LED "); Serial.print(i); Serial.print(": ");
    leds.set(i, BLUEY, brightness);
    leds.show();
    delay(500);
    
    Serial.println("ON");
  }
  
  clearLedsHard();
  Serial.println("Individual test complete\n");
}

void testBasicLEDs() {
  Serial.println("=== Basic LED test - all at once ===");
  
  for (int i = 0; i < LED_COUNT; i++) {
    leds.set(i, REDY, 2);
  }
  leds.show();
  delay(2000);
  
  clearLedsHard();
  Serial.println("Basic test complete\n");
}

// ============ HAPTIC COMMANDS =========== //

void playHaptic(uint8_t fx) {
  Serial.print("Haptic effect "); Serial.println(fx);
  for (int i = 0; i < 3; i++){
    drv.setWaveform(0, fx);
    drv.setWaveform(1, 0);
    drv.go();
    delay(300);
  }
  delay(1000);
}
