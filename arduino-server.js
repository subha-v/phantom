const express = require('express');
const cors = require('cors');
const { SerialPort } = require('serialport');
const { ReadlineParser } = require('@serialport/parser-readline');

const app = express();
app.use(cors());
app.use(express.json());

// Add timeout middleware
app.use((req, res, next) => {
  res.setTimeout(5000, () => {
    console.log('Request timeout');
    res.status(408).send('Request timeout');
  });
  next();
});

let port = null;
let parser = null;

// Auto-detect Arduino port
async function findArduinoPort() {
  const ports = await SerialPort.list();
  console.log('Available ports:', ports.map(p => p.path));

  // Look for Arduino - check both tty and cu devices
  const arduinoPort = ports.find(p =>
    (p.manufacturer && p.manufacturer.includes('Arduino')) ||
    p.path.includes('usbmodem') ||
    p.path.includes('usbserial') ||
    p.path.includes('tty.usbmodem') ||
    p.path.includes('cu.usbmodem')
  );

  // Prefer tty over cu for better compatibility
  if (arduinoPort && arduinoPort.path.includes('cu.')) {
    const ttyPath = arduinoPort.path.replace('/dev/cu.', '/dev/tty.');
    const ttyExists = ports.find(p => p.path === ttyPath);
    if (ttyExists) {
      return ttyPath;
    }
  }

  return arduinoPort ? arduinoPort.path : null;
}

// Initialize serial connection
async function initSerial() {
  try {
    const arduinoPath = await findArduinoPort();
    if (!arduinoPath) {
      console.error('Arduino not found. Please connect your Arduino.');
      return false;
    }

    console.log(`Found Arduino at ${arduinoPath}`);

    port = new SerialPort({
      path: arduinoPath,
      baudRate: 9600,
      autoOpen: false
    });

    parser = port.pipe(new ReadlineParser({ delimiter: '\n' }));

    return new Promise((resolve) => {
      port.open((err) => {
        if (err) {
          console.error('Error opening port:', err.message);
          resolve(false);
        } else {
          console.log('Serial port opened successfully');

          parser.on('data', (data) => {
            console.log('Arduino:', data.toString().trim());
          });

          resolve(true);
        }
      });
    });
  } catch (error) {
    console.error('Serial initialization error:', error);
    return false;
  }
}

// API endpoint to trigger haptic feedback
app.post('/api/arduino/haptic', async (req, res) => {
  console.log('Received haptic feedback request');

  if (!port || !port.isOpen) {
    console.log('Port not open, initializing...');
    const initialized = await initSerial();
    if (!initialized) {
      console.log('Failed to initialize serial port');
      return res.status(503).json({
        success: false,
        error: 'Arduino not connected'
      });
    }
  }

  try {
    console.log('Sending HAPTIC command to Arduino...');

    // Send command and immediately respond
    const command = 'HAPTIC\n';
    port.write(command);

    console.log('Haptic feedback command sent successfully');

    // Don't wait for write callback, just respond
    res.json({
      success: true,
      message: 'Haptic feedback triggered'
    });

  } catch (error) {
    console.error('Error sending command:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Health check endpoint
app.get('/api/arduino/status', async (req, res) => {
  const isConnected = port && port.isOpen;
  res.json({
    connected: isConnected,
    port: isConnected ? port.path : null
  });
});

// Start server
const PORT = process.env.PORT || 3001;
app.listen(PORT, async () => {
  console.log(`Arduino server running on port ${PORT}`);
  await initSerial();
});

// Cleanup on exit
process.on('SIGINT', () => {
  if (port && port.isOpen) {
    port.close(() => {
      console.log('Serial port closed');
      process.exit();
    });
  } else {
    process.exit();
  }
});