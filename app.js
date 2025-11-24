// More API functions here:
// https://github.com/googlecreativelab/teachablemachine-community/tree/master/libraries/pose

// Model and webcam variables
const MODEL_URL = "./";
let model, webcam, ctx, labelContainer, maxPredictions;
let isRunning = false;
let animationId = null;

// Statistics tracking variables
let poseHistory = []; // Array of {pose, duration, timestamp, endTime}
let poseStats = {}; // Object with pose names as keys
let currentPose = null;
let currentPoseStartTime = null;
let sessionStartTime = null;
let statsUpdateInterval = null;

// Configuration
const CONFIDENCE_THRESHOLD = 0.6; // Minimum confidence to record a pose
const MAX_HISTORY_DISPLAY = 20; // Maximum history items to display

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
    // Reset any previous state
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

    // Load the model
    console.log("Loading model...");
    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();
    console.log("Model loaded successfully");

    // Initialize stats for all classes
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

    // Setup webcam
    const size = 300;
    const flip = true;
    webcam = new tmPose.Webcam(size, size, flip);

    console.log("Setting up webcam...");
    await webcam.setup();
    console.log("Webcam setup complete");

    // Setup canvas
    canvas.width = size;
    canvas.height = size;
    ctx = canvas.getContext("2d");

    // Setup label container
    labelContainer = document.getElementById("label-container");
    labelContainer.innerHTML = "";

    for (let i = 0; i < maxPredictions; i++) {
      labelContainer.appendChild(document.createElement("div"));
    }

    // Start playing webcam
    console.log("Starting webcam playback...");
    try {
      await webcam.play();
    } catch (playError) {
      console.log("Initial play failed, retrying...");
      await new Promise((resolve) => setTimeout(resolve, 100));
      await webcam.play();
    }

    isRunning = true;
    sessionStartTime = Date.now();

    // Show UI elements
    mainContent.style.display = "flex";
    loadingIndicator.classList.remove("active");

    // Update buttons
    startBtn.style.display = "none";
    stopBtn.style.display = "inline-block";
    resetBtn.style.display = "inline-block";
    exportBtn.style.display = "inline-block";

    // Start the prediction loop
    loop();

    // Start updating stats display
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

    // Clean up on error
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
    // Prediction #1: run input through posenet
    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);

    // Prediction 2: run input through teachable machine classification model
    const prediction = await model.predict(posenetOutput);

    // Find the pose with highest confidence
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

        // Highlight the most probable prediction
        if (prediction[i].probability > 0.7) {
          labelContainer.childNodes[i].style.backgroundColor = "#90EE90";
          labelContainer.childNodes[i].style.color = "#006400";
        } else {
          labelContainer.childNodes[i].style.backgroundColor = "#f8f8f8";
          labelContainer.childNodes[i].style.color = "#333";
        }
      }

      // Track highest probability
      if (prediction[i].probability > maxProb) {
        maxProb = prediction[i].probability;
        detectedPose = prediction[i].className;
      }
    }

    // Record pose if confidence is high enough
    if (maxProb >= CONFIDENCE_THRESHOLD && detectedPose) {
      recordPose(detectedPose, maxProb);
    } else {
      // If no confident pose detected, end current pose tracking
      if (currentPose) {
        finalizePose();
      }
    }

    // Draw the pose
    drawPose(pose);
  } catch (error) {
    console.error("Prediction error:", error);
  }
}

function recordPose(poseName, confidence) {
  const now = Date.now();

  // If pose changed, save the previous one
  if (currentPose && currentPose !== poseName) {
    finalizePose();
  }

  // Start tracking new pose
  if (currentPose !== poseName) {
    currentPose = poseName;
    currentPoseStartTime = now;
    updateCurrentPoseDisplay(poseName, 0);
  } else {
    // Update timer for current pose
    const duration = (now - currentPoseStartTime) / 1000;
    updateCurrentPoseDisplay(poseName, duration);
  }
}

function finalizePose() {
  if (!currentPose || !currentPoseStartTime) return;

  const now = Date.now();
  const duration = now - currentPoseStartTime;

  // Add to history
  poseHistory.push({
    pose: currentPose,
    duration: duration,
    timestamp: currentPoseStartTime,
    endTime: now,
  });

  // Update statistics
  updateStats(currentPose, duration);

  // Update displays
  updateHistoryDisplay();
  updateStatsDisplay();

  // Reset current pose
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

  // Update session duration
  if (sessionStartTime) {
    const sessionDuration = Math.floor((Date.now() - sessionStartTime) / 1000);
    document.getElementById(
      "sessionInfo"
    ).innerHTML = `Session Duration: <strong>${formatDuration(
      sessionDuration * 1000
    )}</strong>`;
  }

  // Check if we have any stats
  const hasStats = Object.keys(poseStats).some(
    (key) => poseStats[key].count > 0
  );

  if (!hasStats) {
    statsContainer.innerHTML =
      '<div class="empty-state">Start moving to see statistics</div>';
    barChart.innerHTML = '<div class="empty-state">No data yet</div>';
    return;
  }

  // Build stats display
  let statsHTML = "";
  let maxDuration = 0;

  // Find max duration for chart scaling
  Object.keys(poseStats).forEach((poseName) => {
    if (poseStats[poseName].totalDuration > maxDuration) {
      maxDuration = poseStats[poseName].totalDuration;
    }
  });

  // Create stat items
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

  // Build bar chart
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

  // Show last N items, most recent first
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
  // Finalize current pose before stopping
  if (currentPose) {
    finalizePose();
  }

  isRunning = false;

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

  // Hide main content
  mainContent.style.display = "none";

  // Clear canvas
  if (ctx) {
    const canvas = document.getElementById("canvas");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  // Reset current pose display
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

  // Finalize current pose
  if (currentPose) {
    finalizePose();
  }

  // Reset all data
  poseHistory = [];
  poseStats = {};

  // Reinitialize stats for all classes
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

  // Update displays
  updateStatsDisplay();
  updateHistoryDisplay();

  console.log("Statistics reset");
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

  // Create downloadable JSON file
  const dataStr = JSON.stringify(exportData, null, 2);
  const dataBlob = new Blob([dataStr], { type: "application/json" });
  const blobUrl = window.URL.createObjectURL(dataBlob);

  const downloadLink = document.createElement("a");
  downloadLink.href = blobUrl;
  downloadLink.download = `pose-statistics-${Date.now()}.json`;
  document.body.appendChild(downloadLink);
  downloadLink.click();
  document.body.removeChild(downloadLink);

  // Clean up the blob URL
  window.URL.revokeObjectURL(blobUrl);

  console.log("Data exported successfully");
}

// Clean up on page unload
window.addEventListener("beforeunload", () => {
  if (webcam) {
    webcam.stop();
  }
  if (statsUpdateInterval) {
    clearInterval(statsUpdateInterval);
  }
});
