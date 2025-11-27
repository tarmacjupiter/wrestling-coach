const MODEL_URL = "./";
let model, webcam, ctx, labelContainer, maxPredictions;
let isRunning = false;
let isPaused = false;
let animationId = null;

let poseHistory = [];
let poseStats = {};
let currentPose = null;
let currentPoseStartTime = null;
let sessionStartTime = null;
let statsUpdateInterval = null;
let pausedTime = 0;
let pauseStartTime = null;

const CONFIDENCE_THRESHOLD = 0.6;
const MAX_HISTORY_DISPLAY = 20;

let audioEnabled = true;
const audioCache = {};
const poseAudioMap = {
  "Bad Stance": "sounds/badtstance.mp3",
  "Good Stance": "sounds/goodstance.mp3",
  "Too Close": "sounds/tooclose.mp3",
  "Too Low": "sounds/toolow.mp3",
};

async function init() {
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const resetBtn = document.getElementById("resetBtn");
  const exportBtn = document.getElementById("exportBtn");
  const errorMessage = document.getElementById("errorMessage");
  const loadingIndicator = document.getElementById("loadingIndicator");
  const mainContent = document.getElementById("mainContent");
  const canvas = document.getElementById("canvas");

  try {
    if (webcam) {
      webcam.stop();
      webcam = null;
    }
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }

    startBtn.disabled = true;
    errorMessage.style.display = "none";
    loadingIndicator.classList.add("active");

    const modelURL = MODEL_URL + "model.json";
    const metadataURL = MODEL_URL + "metadata.json";

    console.log("Loading model...");
    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();
    console.log("Model loaded successfully");

    for (let i = 0; i < maxPredictions; i++) {
      const className = model.getClassLabels()[i];
      if (!poseStats[className]) {
        poseStats[className] = {
          count: 0,
          totalDuration: 0,
          avgDuration: 0,
        };
      }
    }

    initAudio();

    const size = 300;
    const flip = true;
    webcam = new tmPose.Webcam(size, size, flip);

    console.log("Setting up webcam...");
    await webcam.setup();
    console.log("Webcam setup complete");

    canvas.width = size;
    canvas.height = size;
    ctx = canvas.getContext("2d");

    labelContainer = document.getElementById("label-container");
    labelContainer.innerHTML = "";

    for (let i = 0; i < maxPredictions; i++) {
      labelContainer.appendChild(document.createElement("div"));
    }

    console.log("Starting webcam playback...");
    try {
      await webcam.play();
    } catch (playError) {
      console.log("Initial play failed, retrying...");
      await new Promise((resolve) => setTimeout(resolve, 100));
      await webcam.play();
    }

    isRunning = true;
    isPaused = false;
    sessionStartTime = Date.now();
    pausedTime = 0;
    pauseStartTime = null;

    mainContent.style.display = "flex";
    loadingIndicator.classList.remove("active");

    startBtn.style.display = "none";
    stopBtn.style.display = "inline-block";
    document.getElementById("recordingControls").style.display = "flex";
    document.getElementById("dataControls").style.display = "flex";

    loop();

    statsUpdateInterval = setInterval(updateStatsDisplay, 100);

    console.log("Initialization complete");
  } catch (error) {
    console.error("Error initializing:", error);

    let errorMsg = `Error: ${error.message}`;
    if (error.message.includes("Failed to fetch")) {
      errorMsg =
        "Error: Model files not found. Make sure model.json, metadata.json, and weights.bin are in the same directory as this HTML file.";
    } else if (
      error.message.includes("Permission denied") ||
      error.message.includes("NotAllowedError")
    ) {
      errorMsg =
        "Error: Camera access denied. Please allow camera access and try again.";
    } else if (error.message.includes("NotFoundError")) {
      errorMsg =
        "Error: No camera found. Please make sure you have a webcam connected.";
    }

    errorMessage.textContent = errorMsg;
    errorMessage.style.display = "block";
    startBtn.disabled = false;
    startBtn.style.display = "inline-block";
    loadingIndicator.classList.remove("active");

    if (webcam) {
      try {
        webcam.stop();
      } catch (e) {}
      webcam = null;
    }
  }
}

function loop() {
  if (isRunning && webcam && webcam.canvas) {
    webcam.update();
    predict().then(() => {
      animationId = requestAnimationFrame(loop);
    });
  }
}

async function predict() {
  if (!webcam || !webcam.canvas || !model) return;

  try {
    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);

    const prediction = await model.predict(posenetOutput);

    let maxProb = 0;
    let detectedPose = null;

    for (let i = 0; i < maxPredictions; i++) {
      if (labelContainer.childNodes[i]) {
        const classPrediction =
          prediction[i].className +
          ": " +
          (prediction[i].probability * 100).toFixed(1) +
          "%";
        labelContainer.childNodes[i].innerHTML = classPrediction;

        if (prediction[i].probability > 0.7) {
          labelContainer.childNodes[i].style.backgroundColor = "#90EE90";
          labelContainer.childNodes[i].style.color = "#006400";
        } else {
          labelContainer.childNodes[i].style.backgroundColor = "#f8f8f8";
          labelContainer.childNodes[i].style.color = "#333";
        }
      }

      if (prediction[i].probability > maxProb) {
        maxProb = prediction[i].probability;
        detectedPose = prediction[i].className;
      }
    }

    if (maxProb >= CONFIDENCE_THRESHOLD && detectedPose && !isPaused) {
      recordPose(detectedPose, maxProb);
    } else {
      if (currentPose && !isPaused) {
        finalizePose();
      }
    }

    drawPose(pose);
  } catch (error) {
    console.error("Prediction error:", error);
  }
}

function recordPose(poseName, confidence) {
  const now = Date.now();

  if (currentPose && currentPose !== poseName) {
    finalizePose();
  }

  if (currentPose !== poseName) {
    currentPose = poseName;
    currentPoseStartTime = now;
    updateCurrentPoseDisplay(poseName, 0);

    playPoseSound(poseName);
  } else {
    const duration = (now - currentPoseStartTime) / 1000;
    updateCurrentPoseDisplay(poseName, duration);
  }
}

function finalizePose() {
  if (!currentPose || !currentPoseStartTime) return;

  const now = Date.now();
  const duration = now - currentPoseStartTime;

  poseHistory.push({
    pose: currentPose,
    duration: duration,
    timestamp: currentPoseStartTime,
    endTime: now,
  });

  updateStats(currentPose, duration);

  updateHistoryDisplay();
  updateStatsDisplay();

  currentPose = null;
  currentPoseStartTime = null;
}

function updateStats(poseName, duration) {
  if (!poseStats[poseName]) {
    poseStats[poseName] = {
      count: 0,
      totalDuration: 0,
      avgDuration: 0,
    };
  }

  poseStats[poseName].count++;
  poseStats[poseName].totalDuration += duration;
  poseStats[poseName].avgDuration =
    poseStats[poseName].totalDuration / poseStats[poseName].count;
}

function updateCurrentPoseDisplay(poseName, duration) {
  const display = document.getElementById("currentPoseDisplay");
  display.querySelector(".current-pose-name").textContent = poseName;
  document.getElementById("poseTimer").textContent = duration.toFixed(1) + "s";
}

function updateStatsDisplay() {
  const statsContainer = document.getElementById("statsContainer");
  const barChart = document.getElementById("barChart");

  if (sessionStartTime) {
    const currentPausedTime =
      isPaused && pauseStartTime ? Date.now() - pauseStartTime : 0;
    const totalPausedTime = pausedTime + currentPausedTime;
    const sessionDuration = Math.floor(
      (Date.now() - sessionStartTime - totalPausedTime) / 1000
    );

    const pauseIndicator = isPaused
      ? ' <span style="color: #f44336;">(PAUSED)</span>'
      : "";
    document.getElementById(
      "sessionInfo"
    ).innerHTML = `Session Duration: <strong>${formatDuration(
      sessionDuration * 1000
    )}</strong>${pauseIndicator}`;
  }

  const hasStats = Object.keys(poseStats).some(
    (key) => poseStats[key].count > 0
  );

  if (!hasStats) {
    statsContainer.innerHTML =
      '<div class="empty-state">Start moving to see statistics</div>';
    barChart.innerHTML = '<div class="empty-state">No data yet</div>';
    return;
  }

  let statsHTML = "";
  let maxDuration = 0;

  Object.keys(poseStats).forEach((poseName) => {
    if (poseStats[poseName].totalDuration > maxDuration) {
      maxDuration = poseStats[poseName].totalDuration;
    }
  });

  Object.keys(poseStats)
    .filter((key) => poseStats[key].count > 0)
    .forEach((poseName) => {
      const stat = poseStats[poseName];
      statsHTML += `
        <div class="stat-item">
          <div class="stat-label">${poseName}</div>
          <div class="stat-value">${stat.count}x (${formatDuration(
        stat.totalDuration
      )})</div>
        </div>
      `;
    });

  statsContainer.innerHTML = statsHTML;

  let chartHTML = "";
  Object.keys(poseStats)
    .filter((key) => poseStats[key].count > 0)
    .sort((a, b) => poseStats[b].totalDuration - poseStats[a].totalDuration)
    .forEach((poseName) => {
      const stat = poseStats[poseName];
      const percentage =
        maxDuration > 0 ? (stat.totalDuration / maxDuration) * 100 : 0;
      chartHTML += `
        <div class="bar-item">
          <div class="bar-label">
            <span>${poseName}</span>
            <span>${formatDuration(stat.totalDuration)}</span>
          </div>
          <div class="bar-bg">
            <div class="bar-fill" style="width: ${percentage}%">
              ${percentage > 15 ? stat.count + "x" : ""}
            </div>
          </div>
        </div>
      `;
    });

  barChart.innerHTML = chartHTML;
}

function updateHistoryDisplay() {
  const historyLog = document.getElementById("historyLog");

  if (poseHistory.length === 0) {
    historyLog.innerHTML =
      '<div class="empty-state">No poses detected yet</div>';
    return;
  }

  const recentHistory = poseHistory.slice(-MAX_HISTORY_DISPLAY).reverse();

  let historyHTML = "";
  recentHistory.forEach((item, index) => {
    const time = new Date(item.timestamp).toLocaleTimeString();
    historyHTML += `
      <div class="history-item">
        <div>
          <span class="history-pose">${item.pose}</span>
          <div class="history-duration">${time}</div>
        </div>
        <div class="history-duration">
          ${formatDuration(item.duration)}
        </div>
      </div>
    `;
  });

  historyLog.innerHTML = historyHTML;
}

function formatDuration(milliseconds) {
  const seconds = milliseconds / 1000;
  if (seconds < 60) {
    return seconds.toFixed(1) + "s";
  } else {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  }
}

function drawPose(pose) {
  if (webcam && webcam.canvas && ctx) {
    ctx.drawImage(webcam.canvas, 0, 0);
    if (pose) {
      const minPartConfidence = 0.5;
      tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
      tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
    }
  }
}

function stop() {
  if (currentPose && !isPaused) {
    finalizePose();
  }

  isRunning = false;
  isPaused = false;

  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  if (statsUpdateInterval) {
    clearInterval(statsUpdateInterval);
    statsUpdateInterval = null;
  }

  if (webcam) {
    webcam.stop();
    webcam = null;
  }

  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const mainContent = document.getElementById("mainContent");

  startBtn.style.display = "inline-block";
  startBtn.disabled = false;
  stopBtn.style.display = "none";
  document.getElementById("recordingControls").style.display = "none";
  document.getElementById("dataControls").style.display = "none";

  mainContent.style.display = "none";

  if (ctx) {
    const canvas = document.getElementById("canvas");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  document.querySelector(".current-pose-name").textContent = "Stopped";
  document.getElementById("poseTimer").textContent = "0.0s";
}

function resetStats() {
  if (
    !confirm(
      "Are you sure you want to reset all statistics? This cannot be undone."
    )
  ) {
    return;
  }

  if (currentPose && !isPaused) {
    finalizePose();
  }

  poseHistory = [];
  poseStats = {};

  if (model) {
    for (let i = 0; i < maxPredictions; i++) {
      const className = model.getClassLabels()[i];
      poseStats[className] = {
        count: 0,
        totalDuration: 0,
        avgDuration: 0,
      };
    }
  }

  sessionStartTime = Date.now();
  pausedTime = 0;
  pauseStartTime = null;

  updateStatsDisplay();
  updateHistoryDisplay();

  console.log("Statistics reset");
}

function togglePause() {
  const pauseBtn = document.getElementById("pauseBtn");

  if (!isPaused) {
    isPaused = true;
    pauseStartTime = Date.now();

    if (currentPose) {
      finalizePose();
    }

    pauseBtn.textContent = "Resume Recording";
    pauseBtn.classList.remove("secondary");
    pauseBtn.classList.add("warning");

    document.querySelector(".current-pose-name").textContent = "PAUSED";

    console.log("Recording paused");
  } else {
    isPaused = false;

    if (pauseStartTime) {
      pausedTime += Date.now() - pauseStartTime;
      pauseStartTime = null;
    }

    pauseBtn.textContent = "Pause Recording";
    pauseBtn.classList.remove("warning");
    pauseBtn.classList.add("secondary");

    document.querySelector(".current-pose-name").textContent = "Waiting...";

    console.log("Recording resumed");
  }
}

function toggleFullscreen() {
  const videoSection = document.querySelector(".video-section");

  if (!document.fullscreenElement) {
    if (videoSection.requestFullscreen) {
      videoSection.requestFullscreen();
    } else if (videoSection.webkitRequestFullscreen) {
      videoSection.webkitRequestFullscreen();
    } else if (videoSection.msRequestFullscreen) {
      videoSection.msRequestFullscreen();
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    } else if (document.msExitFullscreen) {
      document.msExitFullscreen();
    }
  }
}

function updateFullscreenButton() {
  const fullscreenIcon = document.getElementById("fullscreenIcon");
  if (fullscreenIcon) {
    if (document.fullscreenElement) {
      fullscreenIcon.setAttribute(
        "d",
        "M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"
      );
    } else {
      fullscreenIcon.setAttribute(
        "d",
        "M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"
      );
    }
  }
}

function initAudio() {
  try {
    Object.keys(poseAudioMap).forEach((poseName) => {
      const audioPath = poseAudioMap[poseName];
      const audio = new Audio(audioPath);
      audio.preload = "auto";
      audio.volume = 1.0;
      audioCache[poseName] = audio;

      audio.load();
    });

    console.log("Audio files preloaded:", Object.keys(audioCache));
  } catch (error) {
    console.error("Audio initialization error:", error);
    audioEnabled = false;
  }
}

function playPoseSound(poseName) {
  if (!audioEnabled) {
    return;
  }

  const audio = audioCache[poseName];

  if (!audio) {
    console.warn("No audio file found for pose:", poseName);
    return;
  }

  try {
    audio.currentTime = 0;

    audio.play().catch((error) => {
      console.error("Error playing audio for", poseName, ":", error);
    });

    console.log("Playing audio for:", poseName);
  } catch (error) {
    console.error("Error playing audio:", error);
  }
}

function toggleAudio() {
  audioEnabled = !audioEnabled;
  const audioBtn = document.getElementById("audioBtn");
  if (audioBtn) {
    audioBtn.textContent = audioEnabled ? "🔊 Audio On" : "🔇 Audio Off";
  }

  if (audioEnabled && Object.keys(audioCache).length > 0) {
    const firstPose = Object.keys(audioCache)[0];
    const testAudio = audioCache[firstPose];
    if (testAudio) {
      testAudio.currentTime = 0;
      testAudio.play().catch((error) => {
        console.warn("Audio test failed:", error);
      });
      console.log("Audio test: playing", firstPose);
    }
  }

  console.log("Audio " + (audioEnabled ? "enabled" : "disabled"));
}

function exportData() {
  const exportData = {
    sessionStart: sessionStartTime
      ? new Date(sessionStartTime).toISOString()
      : null,
    sessionEnd: new Date().toISOString(),
    sessionDuration: sessionStartTime ? Date.now() - sessionStartTime : 0,
    statistics: poseStats,
    history: poseHistory.map((item) => ({
      pose: item.pose,
      duration: item.duration,
      timestamp: new Date(item.timestamp).toISOString(),
      endTime: new Date(item.endTime).toISOString(),
    })),
    totalPoses: poseHistory.length,
  };

  const dataStr = JSON.stringify(exportData, null, 2);
  const dataBlob = new Blob([dataStr], { type: "application/json" });
  const blobUrl = window.URL.createObjectURL(dataBlob);

  const downloadLink = document.createElement("a");
  downloadLink.href = blobUrl;
  downloadLink.download = `pose-statistics-${Date.now()}.json`;
  document.body.appendChild(downloadLink);
  downloadLink.click();
  document.body.removeChild(downloadLink);

  window.URL.revokeObjectURL(blobUrl);

  console.log("Data exported successfully");
}

window.addEventListener("beforeunload", () => {
  if (webcam) {
    webcam.stop();
  }
  if (statsUpdateInterval) {
    clearInterval(statsUpdateInterval);
  }
});

document.addEventListener("fullscreenchange", updateFullscreenButton);
document.addEventListener("webkitfullscreenchange", updateFullscreenButton);
document.addEventListener("mozfullscreenchange", updateFullscreenButton);
document.addEventListener("MSFullscreenChange", updateFullscreenButton);
