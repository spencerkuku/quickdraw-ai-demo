import React, { useCallback, useEffect, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

function pickRandomLabel(labels, previousLabel = "") {
  const choices = labels.filter((label) => label !== previousLabel);
  const pool = choices.length ? choices : labels;
  return pool[Math.floor(Math.random() * pool.length)] ?? "";
}

function Icon({ name, size = 20 }) {
  const paths = {
    sparkles: (
      <>
        <path d="m12 3-1.2 3.3L7.5 7.5l3.3 1.2L12 12l1.2-3.3 3.3-1.2-3.3-1.2L12 3Z" />
        <path d="m5 13-.8 2.2L2 16l2.2.8L5 19l.8-2.2L8 16l-2.2-.8L5 13Z" />
        <path d="m18 14-.7 1.8-1.8.7 1.8.7L18 19l.7-1.8 1.8-.7-1.8-.7L18 14Z" />
      </>
    ),
    trash: (
      <>
        <path d="M4 7h16" />
        <path d="M9 7V4h6v3M18 7l-1 13H7L6 7" />
        <path d="M10 11v5M14 11v5" />
      </>
    ),
    arrow: (
      <>
        <path d="M5 12h14" />
        <path d="m13 6 6 6-6 6" />
      </>
    ),
    pen: (
      <>
        <path d="m4 20 4.2-1 10.6-10.6a2 2 0 0 0-2.8-2.8L5.4 16.2 4 20Z" />
        <path d="m14.5 7.1 2.8 2.8" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
  };

  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
    >
      <g stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8">
        {paths[name]}
      </g>
    </svg>
  );
}

function DrawingCanvas({ brushSize, onStroke, canvasRef, disabled }) {
  const drawing = useRef(false);
  const lastPoint = useRef(null);

  const prepareCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const snapshot = canvas.width ? canvas.toDataURL() : null;
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);

    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, rect.width, rect.height);
    context.lineCap = "round";
    context.lineJoin = "round";

    if (snapshot) {
      const image = new Image();
      image.onload = () => context.drawImage(image, 0, 0, rect.width, rect.height);
      image.src = snapshot;
    }
  }, [canvasRef]);

  useEffect(() => {
    prepareCanvas();
    window.addEventListener("resize", prepareCanvas);
    return () => window.removeEventListener("resize", prepareCanvas);
  }, [prepareCanvas]);

  const getPoint = (event) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  const startDrawing = (event) => {
    if (disabled) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    drawing.current = true;
    lastPoint.current = getPoint(event);
  };

  const draw = (event) => {
    if (!drawing.current) return;
    const nextPoint = getPoint(event);
    const context = canvasRef.current.getContext("2d");
    context.strokeStyle = "#111827";
    context.lineWidth = brushSize;
    context.beginPath();
    context.moveTo(lastPoint.current.x, lastPoint.current.y);
    context.lineTo(nextPoint.x, nextPoint.y);
    context.stroke();
    lastPoint.current = nextPoint;
    onStroke();
  };

  const stopDrawing = () => {
    drawing.current = false;
    lastPoint.current = null;
  };

  return (
    <canvas
      aria-label="塗鴉畫布"
      className={`drawing-canvas ${disabled ? "disabled" : ""}`}
      onPointerDown={startDrawing}
      onPointerMove={draw}
      onPointerUp={stopDrawing}
      onPointerCancel={stopDrawing}
      onPointerLeave={stopDrawing}
      ref={canvasRef}
    />
  );
}

function App() {
  const canvasRef = useRef(null);
  const requestRef = useRef(null);
  const [brushSize, setBrushSize] = useState(16);
  const [hasDrawing, setHasDrawing] = useState(false);
  const [drawingVersion, setDrawingVersion] = useState(0);
  const [classes, setClasses] = useState([]);
  const [target, setTarget] = useState("");
  const [roundStatus, setRoundStatus] = useState("drawing");
  const [predictions, setPredictions] = useState([]);
  const [gradcam, setGradcam] = useState(null);
  const [analyzedImage, setAnalyzedImage] = useState("");
  const [latency, setLatency] = useState(null);
  const [status, setStatus] = useState("checking");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API_URL}/api/health`)
      .then(async (response) => {
        if (!response.ok) throw new Error();
        const data = await response.json();
        setClasses(data.classes);
        setTarget(pickRandomLabel(data.classes));
        setStatus("ready");
      })
      .catch(() => setStatus("offline"));
  }, []);

  const clearCanvas = useCallback(() => {
    requestRef.current?.abort();
    requestRef.current = null;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    const ratio = window.devicePixelRatio || 1;
    context.save();
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.restore();
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    setHasDrawing(false);
    setPredictions([]);
    setGradcam(null);
    setAnalyzedImage("");
    setLatency(null);
    setError("");
    setIsLoading(false);
    setRoundStatus("drawing");
  }, []);

  const startNewRound = useCallback(() => {
    setTarget((currentTarget) => pickRandomLabel(classes, currentTarget));
    setRoundStatus("drawing");
    requestRef.current?.abort();
    clearCanvas();
  }, [classes, clearCanvas]);

  const markDrawing = useCallback(() => {
    setHasDrawing(true);
    setDrawingVersion((version) => version + 1);
  }, []);

  const predict = useCallback(async () => {
    if (!hasDrawing || roundStatus === "won" || status !== "ready") return;

    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setIsLoading(true);
    setError("");
    const canvasImage = canvasRef.current.toDataURL("image/png");

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: canvasImage }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error("辨識服務暫時無法使用");
      const data = await response.json();
      setPredictions(data.predictions);
      setGradcam(data.gradcam);
      setAnalyzedImage(canvasImage);
      setLatency(data.latency_ms);
      setStatus("ready");
      if (data.predictions[0]?.label === target) {
        setRoundStatus("won");
      }
    } catch (requestError) {
      if (requestError.name === "AbortError") return;
      setError(requestError.message || "無法連接辨識服務");
      setStatus("offline");
    } finally {
      if (requestRef.current === controller) {
        setIsLoading(false);
      }
    }
  }, [hasDrawing, roundStatus, status, target]);

  useEffect(() => {
    if (!hasDrawing || roundStatus === "won") return undefined;
    const timer = window.setTimeout(predict, 400);
    return () => window.clearTimeout(timer);
  }, [drawingVersion, hasDrawing, predict, roundStatus]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        roundStatus === "won" ? startNewRound() : predict();
      }
      if (event.key === "Escape") clearCanvas();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [clearCanvas, predict, roundStatus, startNewRound]);

  const topPrediction = predictions[0];

  return (
    <div className="app-shell">
      <main>
        <section className="hero">
          <div className="eyebrow">DEEP LEARNING FINAL PROJECT</div>
          <p>系統會從模型類別中隨機出題，畫圖過程將自動進行即時辨識。</p>
        </section>

        <section className={`challenge-card ${roundStatus}`}>
          <div>
            <span className="challenge-label">本回合題目</span>
            <p>請畫一個</p>
          </div>
          <strong>{target || "載入中..."}</strong>
          <button
            disabled={!classes.length}
            onClick={startNewRound}
            type="button"
          >
            換一題
          </button>
        </section>

        <section className="workspace">
          <div className="canvas-panel panel">
            <div className="panel-heading">
              <div>
                <span className="step">01</span>
                <h2>開始創作</h2>
              </div>
              <button className="text-button" onClick={clearCanvas} type="button">
                <Icon name="trash" size={17} /> 清除畫布
              </button>
            </div>

            <div className="canvas-wrap">
              {!hasDrawing && (
                <div className="canvas-hint">
                  <span><Icon name="pen" size={28} /></span>
                  <strong>從這裡開始畫</strong>
                  <small>支援滑鼠、觸控板與觸控螢幕</small>
                </div>
              )}
              <DrawingCanvas
                brushSize={brushSize}
                canvasRef={canvasRef}
                disabled={roundStatus === "won"}
                onStroke={markDrawing}
              />
            </div>

            <div className="canvas-toolbar">
              <label htmlFor="brush-size">筆刷粗細</label>
              <input
                id="brush-size"
                max="30"
                min="6"
                onChange={(event) => setBrushSize(Number(event.target.value))}
                type="range"
                value={brushSize}
              />
              <span className="brush-preview" style={{ width: brushSize, height: brushSize }} />
              <span className="brush-value">{brushSize}px</span>
            </div>
          </div>

          <aside className="result-panel panel">
            <div className="panel-heading">
              <div>
                <span className="step">02</span>
                <h2>AI 辨識結果</h2>
              </div>
              {latency !== null && <span className="latency">{latency} ms</span>}
            </div>

            <div className={`result-content ${topPrediction ? "has-result" : ""}`}>
              {roundStatus === "won" ? (
                <div className="success-result">
                  <span className="success-icon"><Icon name="check" size={34} /></span>
                  <span className="success-caption">辨識成功</span>
                  <strong>我知道了！<br />這是「{target}」</strong>
                  <p>模型以 {topPrediction?.confidence.toFixed(1)}% 的信心辨識出正確答案。</p>
                </div>
              ) : topPrediction ? (
                <>
                  <div className="live-indicator">
                    <span /> {isLoading ? "正在更新猜測..." : "即時辨識中"}
                  </div>
                  <div className="top-result">
                    <span className="result-caption"><Icon name="sparkles" size={16} /> AI 猜是</span>
                    <strong>{topPrediction.label}</strong>
                    <div className="confidence-row">
                      <span>信心分數</span>
                      <b>{topPrediction.confidence.toFixed(1)}%</b>
                    </div>
                    <div className="main-progress">
                      <span style={{ width: `${topPrediction.confidence}%` }} />
                    </div>
                  </div>
                  <div className="alternatives">
                    <p>其他可能</p>
                    {predictions.slice(1).map((prediction, index) => (
                      <div className="alternative" key={prediction.label}>
                        <span className="rank">0{index + 2}</span>
                        <span className="alternative-label">{prediction.label}</span>
                        <div className="mini-progress">
                          <span style={{ width: `${prediction.confidence}%` }} />
                        </div>
                        <b>{prediction.confidence.toFixed(1)}%</b>
                      </div>
                    ))}
                  </div>
                </>
              ) : isLoading ? (
                <div className="result-empty">
                  <span className="loader" />
                  <h3>模型分析中</h3>
                  <p>正在讀取你的筆跡特徵...</p>
                </div>
              ) : (
                <div className="result-empty">
                  <span className="empty-icon"><Icon name="sparkles" size={30} /></span>
                  <h3>開始畫「{target}」</h3>
                  <p>你一開始畫，AI 就會自動猜測，不需要按下辨識按鈕。</p>
                  {error && <div className="error-message">{error}</div>}
                </div>
              )}
            </div>

            <button
              className="predict-button"
              disabled={roundStatus !== "won" && (!hasDrawing || isLoading)}
              onClick={roundStatus === "won" ? startNewRound : predict}
              type="button"
            >
              {roundStatus === "won" ? "下一題" : isLoading ? "即時辨識中..." : "立即辨識"}
              {!isLoading && <Icon name="arrow" size={20} />}
            </button>
            <span className="shortcut">每次停筆約 0.4 秒後自動更新猜測</span>
          </aside>
        </section>

        <section className="explainability-panel panel">
          <div className="explainability-heading">
            <div>
              <span className="step">03</span>
              <div>
                <span className="analysis-kicker">MODEL EXPLAINABILITY</span>
                <h2>Grad-CAM 模型關注區域</h2>
              </div>
            </div>
            <span className="method-badge">
              <span /> 真實梯度反向傳播
            </span>
          </div>

          {gradcam ? (
            <div className="gradcam-content">
              <div className="gradcam-comparison">
                <figure>
                  <div className="analysis-image">
                    <img alt="送入模型的原始塗鴉" src={analyzedImage} />
                  </div>
                  <figcaption>原始塗鴉</figcaption>
                </figure>
                <figure>
                  <div className="analysis-image heatmap-image">
                    <img
                      alt={`模型辨識「${gradcam.target_label}」時的 Grad-CAM 熱力圖`}
                      src={gradcam.image}
                    />
                  </div>
                  <figcaption>Grad-CAM 疊圖</figcaption>
                </figure>
              </div>

              <div className="gradcam-details">
                <span className="detail-label">本次解釋目標</span>
                <strong>{gradcam.target_label}</strong>
                <div className="detail-row">
                  <span>目標信心</span>
                  <b>{gradcam.target_confidence.toFixed(1)}%</b>
                </div>
                <div className="detail-row">
                  <span>分析方法</span>
                  <b>{gradcam.method}</b>
                </div>
                <div className="detail-row">
                  <span>特徵層</span>
                  <b>{gradcam.layer}</b>
                </div>

                <div className="heat-legend">
                  <div className="legend-bar" />
                  <div>
                    <span>低關注</span>
                    <span>高關注</span>
                  </div>
                </div>

                <p>
                  紅色與黃色區域對目前預測的貢獻較高；藍色區域影響較低。
                  熱力圖由卷積特徵與分類梯度計算，並以筆跡鄰域抑制遠端空白雜訊。
                </p>
              </div>
            </div>
          ) : (
            <div className="gradcam-empty">
              <span><Icon name="sparkles" size={26} /></span>
              <div>
                <strong>等待第一次模型分析</strong>
                <p>開始作畫後，這裡會顯示 AI 做出判斷時真正關注的筆跡區域。</p>
              </div>
            </div>
          )}
        </section>

        <section className="classes-section">
          <div>
            <span className="section-label">SUPPORTED CLASSES</span>
            <h2>模型能辨識的 10 種物件</h2>
          </div>
          <div className="class-list">
            {classes.map((label, index) => (
              <span key={label}><b>{String(index + 1).padStart(2, "0")}</b>{label}</span>
            ))}
          </div>
        </section>
      </main>

      <footer>
        <span>SketchSense © 2026</span>
        <span>MobileNetV2 · PyTorch · React</span>
      </footer>
    </div>
  );
}

export default App;
