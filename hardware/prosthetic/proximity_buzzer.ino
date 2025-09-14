// Enhanced haptic feedback system for aesthetic prosthetics
// Provides variable intensity and pattern-based feedback
#include <Modulino.h>

ModulinoDistance distance;
ModulinoBuzzer buzzer;

// Feedback parameters
const int TOUCH_THRESHOLD = 30;      // Distance threshold for "touch" detection
const int PROXIMITY_THRESHOLD = 100;  // Distance for proximity awareness
const int BASE_FREQUENCY = 200;      // Lower frequency for haptic feel
const int MAX_FREQUENCY = 800;       // Higher frequency for urgent feedback

// Timing variables
unsigned long lastFeedbackTime = 0;
unsigned long feedbackInterval = 50;  // Faster response for real-time feel
int lastMeasure = 0;
bool isInContact = false;

// Haptic patterns
enum HapticMode {
  NO_FEEDBACK,
  LIGHT_TOUCH,
  FIRM_CONTACT,
};

HapticMode currentMode = NO_FEEDBACK;

void setup() {
  Serial.begin(9600);
  
  // Initialize the Modulino system
  Modulino.begin();
  distance.begin();
  buzzer.begin();
  
  Serial.println("Prosthetic Haptic Feedback System Initialized");
  Serial.println("Distance ranges:");
  Serial.println("- Contact: 0-30cm");
  Serial.println("- Proximity: 31-100cm");
  Serial.println("- No feedback: >100cm");
}

void loop() {
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
  
  delay(10);  // Small delay for system stability
}

void processDistanceFeedback(int measure) {
  Serial.print("Distance: ");
  Serial.print(measure);
  Serial.print("cm - ");
  
  if (measure <= 10) {
    // Very close contact - firm pressure sensation
    currentMode = FIRM_CONTACT;
    Serial.println("FIRM CONTACT");
    
  } else if (measure <= TOUCH_THRESHOLD) {
    // Light touch range
    currentMode = LIGHT_TOUCH;
    Serial.println("LIGHT TOUCH");
    
  }  else {
    // No feedback needed
    currentMode = NO_FEEDBACK;
    Serial.println("NO CONTACT");
  }
  
  // Update contact state
  isInContact = (measure <= TOUCH_THRESHOLD);
}

void executeHapticFeedback() {
  unsigned long currentTime = millis();
  
  // Check if it's time for feedback update
  if (currentTime - lastFeedbackTime < feedbackInterval) {
    return;
  }
  
  lastFeedbackTime = currentTime;
  
  switch (currentMode) {
    case NO_FEEDBACK:
      buzzer.tone(0, 0);  // Stop any feedback
      feedbackInterval = 100;  // Slower checking when no contact
      break;
      
    case LIGHT_TOUCH:
      // Gentle continuous feedback for light touch
      lightTouchFeedback();
      feedbackInterval = 50;   // Responsive feedback
      break;
      
    case FIRM_CONTACT:
      // Strong feedback for firm contact
      firmContactFeedback();
      feedbackInterval = 30;   // Very responsive
      break;
      
   
  }
}

void proximityPulseFeedback() {
  // Short, gentle pulse to indicate nearby object
  static bool pulseState = false;
  
  if (pulseState) {
    buzzer.tone(BASE_FREQUENCY, 100);
  } else {
    buzzer.tone(0, 50);  // Brief silence between pulses
  }
  
  pulseState = !pulseState;
}

void lightTouchFeedback() {
  // Gentle, constant low-frequency feedback
  buzzer.tone(BASE_FREQUENCY + 50, 200);
}

void firmContactFeedback() {
  // Stronger, higher frequency for firm contact
  buzzer.tone(BASE_FREQUENCY + 200, 150);
}

void pressureGradientFeedback() {
  // Map distance to frequency for gradient feedback
  int mappedFreq = map(lastMeasure, 0, TOUCH_THRESHOLD, MAX_FREQUENCY, BASE_FREQUENCY);
  buzzer.tone(mappedFreq, 100);
}

// Optional: Function to adjust sensitivity (could be called from external input)
void adjustSensitivity(int newThreshold) {
  if (newThreshold > 5 && newThreshold < 200) {
    // Update threshold within reasonable bounds
    Serial.print("Sensitivity adjusted to: ");
    Serial.println(newThreshold);
    // You could update TOUCH_THRESHOLD here if made non-const
  }
}

// Optional: Function to change haptic mode manually
void setHapticMode(HapticMode mode) {
  currentMode = mode;
  Serial.print("Haptic mode changed to: ");
  Serial.println(mode);
}
