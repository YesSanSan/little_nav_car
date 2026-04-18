const state = {
  mapPayload: null,
  mapBaseCanvas: null,
  logLines: [],
};

const els = {
  slamStatus: document.getElementById("slamStatus"),
  navStatus: document.getElementById("navStatus"),
  stackSummary: document.getElementById("stackSummary"),
  visionStatusPill: document.getElementById("visionStatusPill"),
  visionServiceStatus: document.getElementById("visionServiceStatus"),
  trackingFlagStatus: document.getElementById("trackingFlagStatus"),
  visionHint: document.getElementById("visionHint"),
  returnStatusPill: document.getElementById("returnStatusPill"),
  returnPoseText: document.getElementById("returnPoseText"),
  robotPoseText: document.getElementById("robotPoseText"),
  mapStatusPill: document.getElementById("mapStatusPill"),
  mapCanvas: document.getElementById("mapCanvas"),
  mapOverlay: document.getElementById("mapOverlay"),
  messageLog: document.getElementById("messageLog"),
  startSlamButton: document.getElementById("startSlamButton"),
  stopSlamButton: document.getElementById("stopSlamButton"),
  startNavButton: document.getElementById("startNavButton"),
  stopNavButton: document.getElementById("stopNavButton"),
  startTrackingButton: document.getElementById("startTrackingButton"),
  stopTrackingButton: document.getElementById("stopTrackingButton"),
  markCurrentButton: document.getElementById("markCurrentButton"),
  clearReturnButton: document.getElementById("clearReturnButton"),
  returnHomeButton: document.getElementById("returnHomeButton"),
};

function logMessage(message) {
  const timestamp = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  state.logLines.unshift(`[${timestamp}] ${message}`);
  state.logLines = state.logLines.slice(0, 40);
  els.messageLog.textContent = state.logLines.join("\n");
}

function setPill(element, text, tone = "neutral") {
  element.textContent = text;
  element.classList.remove("good", "warn", "neutral");
  element.classList.add(tone);
}

function formatPose(pose) {
  if (!pose) {
    return "未知";
  }
  return `x=${pose.x.toFixed(2)} m, y=${pose.y.toFixed(2)} m, yaw=${(pose.yaw * 180 / Math.PI).toFixed(1)}°`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.message || `${url} request failed`);
  }
  return payload;
}

async function postJson(url, body = {}) {
  return fetchJson(url, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

function worldToCanvas(point, payload) {
  const dx = point.x - payload.origin.x;
  const dy = point.y - payload.origin.y;
  const cosYaw = Math.cos(payload.origin.yaw);
  const sinYaw = Math.sin(payload.origin.yaw);
  const localX = cosYaw * dx + sinYaw * dy;
  const localY = -sinYaw * dx + cosYaw * dy;
  return {
    x: localX / payload.resolution,
    y: payload.height - localY / payload.resolution,
  };
}

function canvasToWorld(canvasX, canvasY, payload) {
  const localX = canvasX * payload.resolution;
  const localY = (payload.height - canvasY) * payload.resolution;
  const cosYaw = Math.cos(payload.origin.yaw);
  const sinYaw = Math.sin(payload.origin.yaw);
  return {
    x: payload.origin.x + cosYaw * localX - sinYaw * localY,
    y: payload.origin.y + sinYaw * localX + cosYaw * localY,
  };
}

function worldYawToCanvasAngle(yaw, payload) {
  return -(yaw - payload.origin.yaw);
}

function ensureBaseCanvas(payload) {
  if (
    state.mapBaseCanvas &&
    state.mapBaseCanvas.width === payload.width &&
    state.mapBaseCanvas.height === payload.height
  ) {
    return;
  }

  state.mapBaseCanvas = document.createElement("canvas");
  state.mapBaseCanvas.width = payload.width;
  state.mapBaseCanvas.height = payload.height;
}

function rebuildBaseMap(payload) {
  ensureBaseCanvas(payload);
  const ctx = state.mapBaseCanvas.getContext("2d");
  const imageData = ctx.createImageData(payload.width, payload.height);

  for (let y = 0; y < payload.height; y += 1) {
    const sourceY = payload.height - 1 - y;
    for (let x = 0; x < payload.width; x += 1) {
      const sourceIndex = sourceY * payload.width + x;
      const value = payload.data[sourceIndex];
      const pixelIndex = (y * payload.width + x) * 4;

      let r = 220;
      let g = 214;
      let b = 203;
      if (value >= 0) {
        const shade = Math.max(24, 255 - Math.round(value * 2.2));
        r = shade;
        g = shade;
        b = shade;
      }

      imageData.data[pixelIndex] = r;
      imageData.data[pixelIndex + 1] = g;
      imageData.data[pixelIndex + 2] = b;
      imageData.data[pixelIndex + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

function drawDirectionalMarker(ctx, point, color, angle, shape = "circle") {
  ctx.save();
  ctx.translate(point.x, point.y);
  ctx.rotate(angle);

  if (shape === "diamond") {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(0, -10);
    ctx.lineTo(10, 0);
    ctx.lineTo(0, 10);
    ctx.lineTo(-10, 0);
    ctx.closePath();
    ctx.fill();
  } else if (shape === "square") {
    ctx.fillStyle = color;
    ctx.fillRect(-7, -7, 14, 14);
  } else {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(0, 0, 8, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(14, 0);
  ctx.stroke();
  ctx.restore();
}

function drawPoint(ctx, point, color, radius = 6) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawMap(payload) {
  state.mapPayload = payload;
  rebuildBaseMap(payload);

  const canvas = els.mapCanvas;
  canvas.width = payload.width;
  canvas.height = payload.height;
  canvas.style.aspectRatio = `${payload.width} / ${payload.height}`;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.mapBaseCanvas, 0, 0);

  if (payload.vision_target) {
    drawPoint(ctx, worldToCanvas(payload.vision_target, payload), "#0f766e", 7);
  }

  if (payload.vision_goal) {
    drawDirectionalMarker(
      ctx,
      worldToCanvas(payload.vision_goal, payload),
      "#d38e00",
      worldYawToCanvasAngle(payload.vision_goal.yaw, payload),
      "square"
    );
  }

  if (payload.return_pose) {
    drawDirectionalMarker(
      ctx,
      worldToCanvas(payload.return_pose, payload),
      "#cf4e2f",
      worldYawToCanvasAngle(payload.return_pose.yaw, payload),
      "diamond"
    );
  }

  if (payload.robot_pose) {
    drawDirectionalMarker(
      ctx,
      worldToCanvas(payload.robot_pose, payload),
      "#2f6df6",
      worldYawToCanvasAngle(payload.robot_pose.yaw, payload),
      "circle"
    );
  }
}

function updateStatusUi(payload) {
  const slamRunning = payload.stacks.slam;
  const navRunning = payload.stacks.nav;

  els.slamStatus.textContent = slamRunning ? "运行中" : "未运行";
  els.navStatus.textContent = navRunning ? "运行中" : "未运行";

  if (slamRunning && navRunning) {
    setPill(els.stackSummary, "SLAM 与导航同时运行", "warn");
  } else if (slamRunning) {
    setPill(els.stackSummary, "SLAM 运行中", "good");
  } else if (navRunning) {
    setPill(els.stackSummary, "导航运行中", "good");
  } else {
    setPill(els.stackSummary, "系统待机", "neutral");
  }

  if (payload.vision.available) {
    const enabled = payload.vision.tracking_enabled && payload.vision.goal_enabled;
    setPill(els.visionStatusPill, enabled ? "视觉追踪开启" : "视觉追踪关闭", enabled ? "good" : "neutral");
    els.visionServiceStatus.textContent = "可用";
    els.trackingFlagStatus.textContent = enabled ? "开启" : "关闭";
    els.visionHint.textContent = payload.vision.last_error || "可直接切换 lc_vision 的运行时追踪开关。";
  } else {
    setPill(els.visionStatusPill, "lc_vision 未连接", "warn");
    els.visionServiceStatus.textContent = "不可用";
    els.trackingFlagStatus.textContent = "未知";
    els.visionHint.textContent = payload.vision.last_error || "等待 lc_vision 参数服务。";
  }

  if (payload.return_pose) {
    setPill(els.returnStatusPill, "返航点已设置", "good");
    els.returnPoseText.textContent = formatPose(payload.return_pose);
  } else {
    setPill(els.returnStatusPill, "未设置返航点", "neutral");
    els.returnPoseText.textContent = "未设置";
  }

  els.robotPoseText.textContent = payload.robot_pose ? formatPose(payload.robot_pose) : "不可用";
}

async function refreshStatus() {
  try {
    const payload = await fetchJson("/api/status");
    updateStatusUi(payload);
  } catch (error) {
    setPill(els.stackSummary, "状态刷新失败", "warn");
    setPill(els.visionStatusPill, "状态刷新失败", "warn");
    logMessage(`状态读取失败: ${error.message}`);
  }
}

async function refreshMap() {
  try {
    const payload = await fetchJson("/api/map");
    if (!payload.available) {
      state.mapPayload = null;
      els.mapOverlay.textContent = payload.message || "等待 /map 话题...";
      els.mapOverlay.classList.remove("hidden");
      setPill(els.mapStatusPill, "等待地图", "neutral");
      return;
    }

    drawMap(payload);
    els.mapOverlay.classList.add("hidden");
    setPill(els.mapStatusPill, `地图已更新 ${payload.width}x${payload.height}`, "good");
  } catch (error) {
    els.mapOverlay.textContent = `地图刷新失败: ${error.message}`;
    els.mapOverlay.classList.remove("hidden");
    setPill(els.mapStatusPill, "地图刷新失败", "warn");
  }
}

async function runAction(button, label, action) {
  button.disabled = true;
  try {
    const payload = await action();
    logMessage(`${label}: ${payload.message || "执行成功"}`);
    await refreshStatus();
    await refreshMap();
  } catch (error) {
    logMessage(`${label}失败: ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

function installButtonHandlers() {
  els.startSlamButton.addEventListener("click", () =>
    runAction(els.startSlamButton, "启动 SLAM", () => postJson("/api/control/start_slam"))
  );
  els.stopSlamButton.addEventListener("click", () =>
    runAction(els.stopSlamButton, "停止 SLAM", () => postJson("/api/control/stop_slam"))
  );
  els.startNavButton.addEventListener("click", () =>
    runAction(els.startNavButton, "启动导航", () => postJson("/api/control/start_nav"))
  );
  els.stopNavButton.addEventListener("click", () =>
    runAction(els.stopNavButton, "停止导航", () => postJson("/api/control/stop_nav"))
  );
  els.startTrackingButton.addEventListener("click", () =>
    runAction(els.startTrackingButton, "开启视觉追踪", () => postJson("/api/control/start_tracking"))
  );
  els.stopTrackingButton.addEventListener("click", () =>
    runAction(els.stopTrackingButton, "停止视觉追踪", () => postJson("/api/control/stop_tracking"))
  );
  els.markCurrentButton.addEventListener("click", () =>
    runAction(els.markCurrentButton, "标记当前位置为返航点", () => postJson("/api/return_point/current"))
  );
  els.clearReturnButton.addEventListener("click", () =>
    runAction(els.clearReturnButton, "清除返航点", () => postJson("/api/return_point/clear"))
  );
  els.returnHomeButton.addEventListener("click", () =>
    runAction(els.returnHomeButton, "返航", () => postJson("/api/control/return_home"))
  );
}

function installMapHandler() {
  els.mapCanvas.addEventListener("click", async (event) => {
    if (!state.mapPayload || !state.mapPayload.available) {
      logMessage("当前没有地图，暂时无法通过点击地图标记返航点。");
      return;
    }

    const rect = els.mapCanvas.getBoundingClientRect();
    const scaleX = state.mapPayload.width / rect.width;
    const scaleY = state.mapPayload.height / rect.height;
    const canvasX = (event.clientX - rect.left) * scaleX;
    const canvasY = (event.clientY - rect.top) * scaleY;
    const worldPoint = canvasToWorld(canvasX, canvasY, state.mapPayload);

    try {
      const payload = await postJson("/api/return_point", {
        x: worldPoint.x,
        y: worldPoint.y,
        frame_id: state.mapPayload.frame_id,
      });
      logMessage(`返航点已标记: ${formatPose(payload.return_pose)}`);
      await refreshStatus();
      await refreshMap();
    } catch (error) {
      logMessage(`标记返航点失败: ${error.message}`);
    }
  });
}

async function bootstrap() {
  installButtonHandlers();
  installMapHandler();
  await refreshStatus();
  await refreshMap();
  logMessage("页面已连接，可通过控制面板启动系统和标记返航点。");
  window.setInterval(refreshStatus, 1000);
  window.setInterval(refreshMap, 3000);
}

bootstrap();
