# Wrestling Stance Pose Detection App

A real-time pose detection application that uses machine learning to analyze and provide feedback on wrestling stances. Built with TensorFlow.js and Teachable Machine.

Play around with it [here!](https://tarmacjupiter.github.io/wrestling-coach/)

## Setup

### Prerequisites
- Modern web browser (Chrome, Firefox, Edge, or Brave)
- Webcam
- Local web server (for file:// protocol compatibility)

### Installation

1. Clone or download the repository
2. Ensure you have the following file structure:
```
project/
├── index.html
├── app.js
├── styles.css
├── model.json
├── metadata.json
├── weights.bin
└── sounds/
    ├── badtstance.mp3
    ├── goodstance.mp3
    ├── tooclose.mp3
    └── toolow.mp3
```

3. Start a local web server:
```bash
# Using Python 3
python -m http.server 8000

# Using Node.js http-server
npx http-server

# Using PHP
php -S localhost:8000
```

4. Open browser and navigate to `http://localhost:8000`

## Usage

### Starting a Session
1. Click **"Start Camera"** button
2. Allow camera permissions when prompted
3. Wait for model to load
4. Position yourself in front of the camera

### During a Session
- **Pause Recording**: Temporarily stop tracking (camera stays on)
- **🔊 Audio On/Off**: Toggle audio feedback
- **Fullscreen**: Hover over video and click fullscreen icon
- **Reset Statistics**: Clear all data and start fresh
- **Export Data**: Download session statistics as JSON

### Stopping a Session
- Click **"Stop Camera"** to end session
- All statistics are preserved until reset

## Pose Classes

The model detects the following wrestling stances:
- **Good Stance**: Proper wrestling position
- **Bad Stance**: Incorrect positioning
- **Too Low**: Stance is too low to ground
- **Too Close**: Too close to camera/opponent

## Audio Files

### Creating Custom Audio
Audio files can be generated using text-to-speech tools:
- [TTS Tool](https://ttstool.com/index.html) - Free online TTS
- [ElevenLabs](https://elevenlabs.io) - High-quality AI voices
- Record your own voice instructions

### Audio File Requirements
- Format: MP3 (recommended) or WAV
- Location: `/sounds/` directory
- Naming: Must match pose class names in `poseAudioMap` (app.js)

### Updating Audio Mapping
Edit the `poseAudioMap` in `app.js`:
```javascript
const poseAudioMap = {
  "Bad Stance": "sounds/badtstance.mp3",
  "Good Stance": "sounds/goodstance.mp3",
  "Too Close": "sounds/tooclose.mp3",
  "Too Low": "sounds/toolow.mp3"
};
```

## Configuration

### Adjusting Detection Sensitivity
In `app.js`, modify:
```javascript
const CONFIDENCE_THRESHOLD = 0.6; // 0.0 to 1.0 (default: 0.6)
```
- Lower value = more sensitive (may detect poses with less certainty)
- Higher value = less sensitive (only detects very confident poses)

### Changing History Display Limit
```javascript
const MAX_HISTORY_DISPLAY = 20; // Number of poses to show in history
```

## Training Your Own Model

To create a custom pose detection model:

1. Visit [Teachable Machine](https://teachablemachine.withgoogle.com/train/pose)
2. Create pose classes and record training samples
3. Train the model
4. Export as "TensorFlow.js" format
5. Replace `model.json`, `metadata.json`, and `weights.bin`
6. Update audio files and mappings to match your class names

## Browser Compatibility

### Fully Supported
- ✅ Chrome/Edge (v90+)
- ✅ Firefox (v88+)
- ✅ Brave (with audio files)
- ✅ Safari (v14+)

### Known Issues
- Speech synthesis may be blocked in privacy-focused browsers (Brave)
- Older browsers may not support required APIs
- Mobile browsers may have limited camera access

## File Structure

```
├── index.html         # Main HTML structure
├── app.js             # Application logic and pose detection
├── styles.css         # Styling and layout
├── model.json         # Teachable Machine model architecture
├── metadata.json      # Model metadata and class labels
├── weights.bin        # Model weights
├── sounds/            # Audio feedback files
│   ├── badtstance.mp3
│   ├── goodstance.mp3
│   ├── tooclose.mp3
│   └── toolow.mp3
└── README.md          
```

## Technologies Used

- [TensorFlow.js](https://www.tensorflow.org/js) - Machine learning in the browser
- [Teachable Machine](https://teachablemachine.withgoogle.com/) - Model training
- [PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet) - Pose estimation
- HTML5 Canvas - Video rendering
- Web Audio API - Audio playback

## License

This project is open source and available for educational and personal use.
