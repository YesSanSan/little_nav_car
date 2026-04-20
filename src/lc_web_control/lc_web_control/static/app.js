const state = {
  mapPayload: null,
  mapBaseCanvas: null,
  logLines: [],
  view: {
    scale: 1,
    minScale: 1,
    maxScale: 8,
    translateX: 0,
    translateY: 0,
    initializedFor: null,
    isInteracting: false,
    pointerStartDistance: 0,
    pinchStartScale: 1,
    pinchAnchorScreen: null,
    lastTapAt: 0,
  },
  pointers: new Map(),
  interaction: {
    primaryPointerId: null,
    moved: false,
    startScreenX: 0,
    startScreenY: 0,
    startTranslateX: 0,
    startTranslateY: 0,
    clickThresholdPx: 6,
  },
};

const els = {
  slamStatus: document.getElementById("slamStatus"),
  mapSummaryText: document.getElementById("mapSummaryText"),
  stackSummary: document.getElementById("stackSummary"),
  visionStatusPill: document.getElementById("visionStatusPill"),
  visionServiceStatus: document.getElementById("visionServiceStatus"),
  trackingFlagStatus: document.getElementById("trackingFlagStatus"),
  visionHint: document.getElementById("visionHint"),
  returnStatusPill: document.getElementById("returnStatusPill"),
  returnPoseText: document.getElementById("returnPoseText"),
  robotPoseText: document.getElementById("robotPoseText"),
  mapStatusPill: document.getElementById("mapStatusPill"),
  mapViewport: document.getElementById("mapViewport"),
  mapCanvas: document.getElementById("mapCanvas"),
  markerCanvas: document.getElementById("markerCanvas"),
  mapOverlay: document.getElementById("mapOverlay"),
  mapClickToggle: document.getElementById("mapClickToggle"),
  messageLog: document.getElementById("messageLog"),
  startSlamButton: document.getElementById("startSlamButton"),
  stopSlamButton: document.getElementById("stopSlamButton"),
  startTrackingButton: document.getElementById("startTrackingButton"),
  stopTrackingButton: document.getElementById("stopTrackingButton"),
  returnHomeButton: document.getElementById("returnHomeButton"),
  markCurrentButton: document.getElementById("markCurrentButton"),
  markStartupButton: document.getElementById("markStartupButton"),
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
  return `x=${pose.x.toFixed(2)} m, y=${pose.y.toFixed(2)} m, yaw=${((pose.yaw * 180) / Math.PI).toFixed(1)}°`;
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

function resizeCanvasToViewport(canvas) {
  const rect = els.mapViewport.getBoundingClientRect();
  const devicePixelRatio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  canvas.width = Math.max(1, Math.round(width * devicePixelRatio));
  canvas.height = Math.max(1, Math.round(height * devicePixelRatio));
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  return { ctx, width, height };
}

function getViewportSize() {
  const rect = els.mapViewport.getBoundingClientRect();
  return {
    width: Math.max(1, rect.width),
    height: Math.max(1, rect.height),
  };
}

function getMapSignature(payload) {
  return `${payload.width}x${payload.height}@${payload.resolution}`;
}

function clampTranslation() {
  if (!state.mapPayload) {
    return;
  }

  const { width: viewportWidth, height: viewportHeight } = getViewportSize();
  const scaledWidth = state.mapPayload.width * state.view.scale;
  const scaledHeight = state.mapPayload.height * state.view.scale;

  if (scaledWidth <= viewportWidth) {
    state.view.translateX = (viewportWidth - scaledWidth) / 2;
  } else {
    const minTranslateX = viewportWidth - scaledWidth;
    state.view.translateX = Math.min(0, Math.max(minTranslateX, state.view.translateX));
  }

  if (scaledHeight <= viewportHeight) {
    state.view.translateY = (viewportHeight - scaledHeight) / 2;
  } else {
    const minTranslateY = viewportHeight - scaledHeight;
    state.view.translateY = Math.min(0, Math.max(minTranslateY, state.view.translateY));
  }
}

function resetView(force = false) {
  if (!state.mapPayload) {
    return;
  }

  const { width: viewportWidth, height: viewportHeight } = getViewportSize();
  const baseScale = Math.min(
    viewportWidth / state.mapPayload.width,
    viewportHeight / state.mapPayload.height
  );

  if (!Number.isFinite(baseScale) || baseScale <= 0) {
    return;
  }

  state.view.minScale = baseScale;
  state.view.maxScale = baseScale * 8;
  state.view.scale = baseScale;
  state.view.translateX = (viewportWidth - state.mapPayload.width * baseScale) / 2;
  state.view.translateY = (viewportHeight - state.mapPayload.height * baseScale) / 2;

  if (force) {
    state.view.initializedFor = getMapSignature(state.mapPayload);
  }
}

function ensureViewInitialized(force = false) {
  if (!state.mapPayload) {
    return;
  }

  const signature = getMapSignature(state.mapPayload);
  if (force || state.view.initializedFor !== signature) {
    resetView(true);
    return;
  }

  if (state.view.minScale <= 0 || !Number.isFinite(state.view.scale)) {
    resetView(true);
  }
}

function screenToCanvasPoint(screenX, screenY) {
  if (!state.mapPayload || state.view.scale <= 0) {
    return null;
  }

  return {
    x: (screenX - state.view.translateX) / state.view.scale,
    y: (screenY - state.view.translateY) / state.view.scale,
  };
}

function screenToWorld(screenX, screenY) {
  const canvasPoint = screenToCanvasPoint(screenX, screenY);
  if (!canvasPoint) {
    return null;
  }
  return canvasToWorld(canvasPoint.x, canvasPoint.y, state.mapPayload);
}

function worldToScreen(point) {
  if (!state.mapPayload) {
    return null;
  }

  const canvasPoint = worldToCanvas(point, state.mapPayload);
  return {
    x: canvasPoint.x * state.view.scale + state.view.translateX,
    y: canvasPoint.y * state.view.scale + state.view.translateY,
  };
}

function zoomAt(screenX, screenY, nextScale) {
  if (!state.mapPayload) {
    return;
  }

  const previousScale = state.view.scale;
  const clampedScale = Math.min(state.view.maxScale, Math.max(state.view.minScale, nextScale));
  if (!Number.isFinite(clampedScale) || clampedScale <= 0 || clampedScale === previousScale) {
    return;
  }

  const anchorCanvasX = (screenX - state.view.translateX) / previousScale;
  const anchorCanvasY = (screenY - state.view.translateY) / previousScale;
  state.view.scale = clampedScale;
  state.view.translateX = screenX - anchorCanvasX * clampedScale;
  state.view.translateY = screenY - anchorCanvasY * clampedScale;
  clampTranslation();
  renderMap();
}

function renderBaseMap() {
  const { ctx, width, height } = resizeCanvasToViewport(els.mapCanvas);
  ctx.clearRect(0, 0, width, height);

  if (!state.mapPayload || !state.mapBaseCanvas) {
    return;
  }

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(
    state.mapBaseCanvas,
    state.view.translateX,
    state.view.translateY,
    state.mapPayload.width * state.view.scale,
    state.mapPayload.height * state.view.scale
  );
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

function renderMarkers() {
  const { ctx, width, height } = resizeCanvasToViewport(els.markerCanvas);
  ctx.clearRect(0, 0, width, height);

  if (!state.mapPayload) {
    return;
  }

  if (state.mapPayload.vision_target) {
    drawPoint(ctx, worldToScreen(state.mapPayload.vision_target), "#0f766e", 7);
  }

  if (state.mapPayload.vision_goal) {
    drawDirectionalMarker(
      ctx,
      worldToScreen(state.mapPayload.vision_goal),
      "#d38e00",
      worldYawToCanvasAngle(state.mapPayload.vision_goal.yaw, state.mapPayload),
      "square"
    );
  }

  if (state.mapPayload.return_pose) {
    drawDirectionalMarker(
      ctx,
      worldToScreen(state.mapPayload.return_pose),
      "#cf4e2f",
      worldYawToCanvasAngle(state.mapPayload.return_pose.yaw, state.mapPayload),
      "diamond"
    );
  }

  if (state.mapPayload.robot_pose) {
    drawDirectionalMarker(
      ctx,
      worldToScreen(state.mapPayload.robot_pose),
      "#2f6df6",
      worldYawToCanvasAngle(state.mapPayload.robot_pose.yaw, state.mapPayload),
      "circle"
    );
  }
}

function renderMap() {
  renderBaseMap();
  renderMarkers();
}

function updateStatusUi(payload) {
  const slamRunning = payload.stacks.slam;

  els.slamStatus.textContent = slamRunning ? "运行中" : "未运行";
  setPill(els.stackSummary, slamRunning ? "SLAM 运行中" : "系统待机", slamRunning ? "good" : "neutral");

  if (payload.map.available) {
    els.mapSummaryText.textContent = payload.map.frame_id || "地图可用";
  } else {
    els.mapSummaryText.textContent = "等待地图";
  }

  if (!payload.vision.available && payload.vision.pending_apply) {
    setPill(els.visionStatusPill, "已记录，等待 lc_vision", "warn");
    els.visionServiceStatus.textContent = "等待中";
    els.trackingFlagStatus.textContent = payload.vision.desired_tracking_enabled ? "待同步开启" : "待同步关闭";
    els.visionHint.textContent = payload.vision.last_error || "等待 lc_vision 参数服务可用后自动同步。";
  } else if (payload.vision.available && payload.vision.pending_apply) {
    setPill(els.visionStatusPill, "参数同步失败", "warn");
    els.visionServiceStatus.textContent = "可用";
    els.trackingFlagStatus.textContent = payload.vision.desired_tracking_enabled ? "同步失败，目标开启" : "同步失败，目标关闭";
    els.visionHint.textContent = payload.vision.last_error || "正在重试同步视觉追踪参数。";
  } else if (payload.vision.available) {
    const enabled = payload.vision.effective_tracking_enabled;
    setPill(els.visionStatusPill, enabled ? "视觉追踪开启" : "视觉追踪关闭", enabled ? "good" : "neutral");
    els.visionServiceStatus.textContent = "可用";
    els.trackingFlagStatus.textContent = enabled ? "已同步开启" : "已同步关闭";
    els.visionHint.textContent = payload.vision.last_error || " ";
  } else {
    setPill(els.visionStatusPill, "lc_vision 未连接", "neutral");
    els.visionServiceStatus.textContent = "不可用";
    els.trackingFlagStatus.textContent = payload.vision.desired_tracking_enabled ? "待同步开启" : "待同步关闭";
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
      renderMap();
      return;
    }

    const previousSignature = state.mapPayload ? getMapSignature(state.mapPayload) : null;
    state.mapPayload = payload;
    rebuildBaseMap(payload);
    ensureViewInitialized(previousSignature !== getMapSignature(payload));
    clampTranslation();
    renderMap();
    els.mapOverlay.classList.add("hidden");
    setPill(els.mapStatusPill, `地图 ${payload.width}x${payload.height}`, "good");
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

function getRelativePointerPosition(event) {
  const rect = els.mapViewport.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

function getPointerPair() {
  return Array.from(state.pointers.values()).slice(0, 2);
}

function getDistance(pointA, pointB) {
  return Math.hypot(pointA.x - pointB.x, pointA.y - pointB.y);
}

function getMidpoint(pointA, pointB) {
  return {
    x: (pointA.x + pointB.x) / 2,
    y: (pointA.y + pointB.y) / 2,
  };
}

async function setReturnPointFromScreen(screenX, screenY) {
  if (!els.mapClickToggle.checked) {
    logMessage("地图点击设置返航点未开启。");
    return;
  }

  if (!state.mapPayload || !state.mapPayload.available) {
    logMessage("当前没有地图，暂时无法通过点击地图标记返航点。");
    return;
  }

  const canvasPoint = screenToCanvasPoint(screenX, screenY);
  if (
    !canvasPoint ||
    canvasPoint.x < 0 ||
    canvasPoint.y < 0 ||
    canvasPoint.x > state.mapPayload.width ||
    canvasPoint.y > state.mapPayload.height
  ) {
    return;
  }

  const worldPoint = screenToWorld(screenX, screenY);
  if (!worldPoint) {
    return;
  }

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
}

function installButtonHandlers() {
  els.startSlamButton.addEventListener("click", () =>
    runAction(els.startSlamButton, "启动 SLAM", () => postJson("/api/control/start_slam"))
  );
  els.stopSlamButton.addEventListener("click", () =>
    runAction(els.stopSlamButton, "停止 SLAM", () => postJson("/api/control/stop_slam"))
  );
  els.startTrackingButton.addEventListener("click", () =>
    runAction(els.startTrackingButton, "开启视觉追踪", () => postJson("/api/control/start_tracking"))
  );
  els.stopTrackingButton.addEventListener("click", () =>
    runAction(els.stopTrackingButton, "停止视觉追踪", () => postJson("/api/control/stop_tracking"))
  );
  els.returnHomeButton.addEventListener("click", () =>
    runAction(els.returnHomeButton, "立即返航", () => postJson("/api/control/return_home"))
  );
  els.markCurrentButton.addEventListener("click", () =>
    runAction(els.markCurrentButton, "标记当前位置为返航点", () => postJson("/api/return_point/current"))
  );
  els.markStartupButton.addEventListener("click", () =>
    runAction(els.markStartupButton, "标记启动位置为返航点", () => postJson("/api/return_point/startup"))
  );
}

function installMapHandlers() {
  els.mapViewport.addEventListener("wheel", (event) => {
    if (!state.mapPayload) {
      return;
    }

    event.preventDefault();
    const point = getRelativePointerPosition(event);
    const zoomFactor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    zoomAt(point.x, point.y, state.view.scale * zoomFactor);
  });

  els.mapViewport.addEventListener("dblclick", (event) => {
    event.preventDefault();
    ensureViewInitialized(true);
    renderMap();
  });

  els.mapViewport.addEventListener("pointerdown", (event) => {
    if (!state.mapPayload) {
      return;
    }

    const point = getRelativePointerPosition(event);
    state.pointers.set(event.pointerId, point);
    state.view.isInteracting = true;
    els.mapViewport.setPointerCapture(event.pointerId);

    if (state.pointers.size === 1) {
      state.interaction.primaryPointerId = event.pointerId;
      state.interaction.moved = false;
      state.interaction.startScreenX = point.x;
      state.interaction.startScreenY = point.y;
      state.interaction.startTranslateX = state.view.translateX;
      state.interaction.startTranslateY = state.view.translateY;
    } else if (state.pointers.size === 2) {
      const [firstPoint, secondPoint] = getPointerPair();
      state.view.pointerStartDistance = getDistance(firstPoint, secondPoint);
      state.view.pinchStartScale = state.view.scale;
      state.view.pinchAnchorScreen = getMidpoint(firstPoint, secondPoint);
    }
  });

  els.mapViewport.addEventListener("pointermove", (event) => {
    if (!state.pointers.has(event.pointerId) || !state.mapPayload) {
      return;
    }

    const point = getRelativePointerPosition(event);
    state.pointers.set(event.pointerId, point);

    if (state.pointers.size >= 2) {
      const [firstPoint, secondPoint] = getPointerPair();
      const distance = getDistance(firstPoint, secondPoint);
      const midpoint = getMidpoint(firstPoint, secondPoint);
      if (state.view.pointerStartDistance > 0) {
        const scaleRatio = distance / state.view.pointerStartDistance;
        zoomAt(
          state.view.pinchAnchorScreen ? state.view.pinchAnchorScreen.x : midpoint.x,
          state.view.pinchAnchorScreen ? state.view.pinchAnchorScreen.y : midpoint.y,
          state.view.pinchStartScale * scaleRatio
        );
      }
      return;
    }

    if (state.interaction.primaryPointerId !== event.pointerId) {
      return;
    }

    const deltaX = point.x - state.interaction.startScreenX;
    const deltaY = point.y - state.interaction.startScreenY;
    if (Math.hypot(deltaX, deltaY) > state.interaction.clickThresholdPx) {
      state.interaction.moved = true;
    }

    state.view.translateX = state.interaction.startTranslateX + deltaX;
    state.view.translateY = state.interaction.startTranslateY + deltaY;
    clampTranslation();
    renderMap();
  });

  const releasePointer = async (event) => {
    const point = getRelativePointerPosition(event);
    const wasPrimaryTap =
      state.interaction.primaryPointerId === event.pointerId && !state.interaction.moved && state.pointers.size === 1;

    state.pointers.delete(event.pointerId);
    if (els.mapViewport.hasPointerCapture(event.pointerId)) {
      els.mapViewport.releasePointerCapture(event.pointerId);
    }

    if (state.pointers.size === 0) {
      state.view.isInteracting = false;
      state.interaction.primaryPointerId = null;
      state.view.pointerStartDistance = 0;
      state.view.pinchAnchorScreen = null;
    } else if (state.pointers.size === 1) {
      const remainingPoint = getPointerPair()[0];
      state.interaction.primaryPointerId = Number.parseInt(
        Array.from(state.pointers.keys())[0],
        10
      );
      state.interaction.startScreenX = remainingPoint.x;
      state.interaction.startScreenY = remainingPoint.y;
      state.interaction.startTranslateX = state.view.translateX;
      state.interaction.startTranslateY = state.view.translateY;
      state.interaction.moved = true;
      state.view.pointerStartDistance = 0;
      state.view.pinchAnchorScreen = null;
    }

    if (!wasPrimaryTap) {
      return;
    }

    const now = Date.now();
    if (now - state.view.lastTapAt < 280) {
      state.view.lastTapAt = 0;
      ensureViewInitialized(true);
      renderMap();
      return;
    }

    state.view.lastTapAt = now;
    await setReturnPointFromScreen(point.x, point.y);
  };

  els.mapViewport.addEventListener("pointerup", releasePointer);
  els.mapViewport.addEventListener("pointercancel", releasePointer);
  window.addEventListener("resize", () => {
    if (!state.mapPayload) {
      renderMap();
      return;
    }

    ensureViewInitialized(true);
    renderMap();
  });

  if (typeof ResizeObserver !== "undefined") {
    const observer = new ResizeObserver(() => {
      if (!state.mapPayload) {
        renderMap();
        return;
      }

      ensureViewInitialized(true);
      renderMap();
    });
    observer.observe(els.mapViewport);
  }
}

async function bootstrap() {
  installButtonHandlers();
  installMapHandlers();
  renderMap();
  await refreshStatus();
  await refreshMap();
  logMessage("页面已连接，可通过控制面板启动 SLAM、切换视觉追踪并标记返航点。");
  window.setInterval(refreshStatus, 1000);
  window.setInterval(refreshMap, 3000);
}

bootstrap();
