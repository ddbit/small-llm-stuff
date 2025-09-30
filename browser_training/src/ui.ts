export interface UIHandlers {
  onPredict: (text: string) => Promise<{ label: string; probs: number[] }>;
  onTrain: (label: number) => Promise<void>;
  onSave: () => Promise<void>;
  onLoad: () => Promise<void>;
  onSendDelta: () => Promise<void>;
  onApplyAggregate: () => Promise<void>;
  onSwitchModel: (backend: string, modelName: string) => Promise<void>;
  onClearCache: () => Promise<void>;
}

interface UIElements {
  status: HTMLElement;
  textInput: HTMLTextAreaElement;
  predictBtn: HTMLButtonElement;
  results: HTMLElement;
  correctLabel: HTMLSelectElement;
  trainBtn: HTMLButtonElement;
  modelSelect: HTMLSelectElement;
  backendSelect: HTMLSelectElement;
  switchModelBtn: HTMLButtonElement;
  modelInfo: HTMLElement;
  saveBtn: HTMLButtonElement;
  loadBtn: HTMLButtonElement;
  clearCacheBtn: HTMLButtonElement;
  sendDeltaBtn: HTMLButtonElement;
  applyAggregateBtn: HTMLButtonElement;
  backend: HTMLElement;
  latency: HTMLElement;
  memory: HTMLElement;
  avgLoss: HTMLElement;
  probsChart: HTMLCanvasElement;
}

let elements: UIElements | null = null;
let handlers: UIHandlers | null = null;
let metrics = {
  backend: '',
  latency: 0,
  memory: 0,
  avgLoss: 0,
  totalTrainingSteps: 0
};

export async function initUI(): Promise<UIHandlers> {
  // Get DOM elements
  elements = {
    status: document.getElementById('status')!,
    textInput: document.getElementById('textInput') as HTMLTextAreaElement,
    predictBtn: document.getElementById('predictBtn') as HTMLButtonElement,
    results: document.getElementById('results')!,
    correctLabel: document.getElementById('correctLabel') as HTMLSelectElement,
    trainBtn: document.getElementById('trainBtn') as HTMLButtonElement,
    modelSelect: document.getElementById('modelSelect') as HTMLSelectElement,
    backendSelect: document.getElementById('backendSelect') as HTMLSelectElement,
    switchModelBtn: document.getElementById('switchModelBtn') as HTMLButtonElement,
    modelInfo: document.getElementById('modelInfo')!,
    saveBtn: document.getElementById('saveBtn') as HTMLButtonElement,
    loadBtn: document.getElementById('loadBtn') as HTMLButtonElement,
    clearCacheBtn: document.getElementById('clearCacheBtn') as HTMLButtonElement,
    sendDeltaBtn: document.getElementById('sendDeltaBtn') as HTMLButtonElement,
    applyAggregateBtn: document.getElementById('applyAggregateBtn') as HTMLButtonElement,
    backend: document.getElementById('backend')!,
    latency: document.getElementById('latency')!,
    memory: document.getElementById('memory')!,
    avgLoss: document.getElementById('avgLoss')!,
    probsChart: document.getElementById('probsChart') as HTMLCanvasElement,
  };
  
  if (!elements.status) {
    throw new Error('Required UI elements not found');
  }
  
  // Create handlers object that will be populated by main.ts
  handlers = {
    onPredict: async () => ({ label: '', probs: [] }),
    onTrain: async () => {},
    onSave: async () => {},
    onLoad: async () => {},
    onSendDelta: async () => {},
    onApplyAggregate: async () => {},
    onSwitchModel: async () => {},
    onClearCache: async () => {}
  };
  
  // Bind event listeners
  bindEventListeners();
  
  return handlers;
}

function bindEventListeners(): void {
  if (!elements || !handlers) return;
  
  // Predict button
  elements.predictBtn.addEventListener('click', async () => {
    const text = elements!.textInput.value.trim();
    if (!text) return;
    
    try {
      elements!.predictBtn.disabled = true;
      elements!.predictBtn.textContent = 'Predicting...';
      
      const result = await handlers!.onPredict(text);
      displayPredictionResult(result);
      
    } catch (error) {
      console.error('Prediction failed:', error);
      updateStatus(`Prediction failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      elements!.predictBtn.disabled = false;
      elements!.predictBtn.textContent = 'Predict';
    }
  });
  
  // Train button
  elements.trainBtn.addEventListener('click', async () => {
    const labelValue = elements!.correctLabel.value;
    if (!labelValue) {
      alert('Please select the correct label first');
      return;
    }
    
    try {
      elements!.trainBtn.disabled = true;
      elements!.trainBtn.textContent = 'Training...';
      
      await handlers!.onTrain(parseInt(labelValue));
      metrics.totalTrainingSteps++;
      updateStatus('Training step completed');
      
      // Clear selection after training
      elements!.correctLabel.value = '';
      
    } catch (error) {
      console.error('Training failed:', error);
      updateStatus(`Training failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      elements!.trainBtn.disabled = false;
      elements!.trainBtn.textContent = 'Train on Feedback';
    }
  });
  
  // Save button
  elements.saveBtn.addEventListener('click', async () => {
    try {
      elements!.saveBtn.disabled = true;
      elements!.saveBtn.textContent = 'Saving...';
      
      await handlers!.onSave();
      
    } catch (error) {
      console.error('Save failed:', error);
    } finally {
      elements!.saveBtn.disabled = false;
      elements!.saveBtn.textContent = 'Save Model';
    }
  });
  
  // Load button
  elements.loadBtn.addEventListener('click', async () => {
    try {
      elements!.loadBtn.disabled = true;
      elements!.loadBtn.textContent = 'Loading...';
      
      await handlers!.onLoad();
      
    } catch (error) {
      console.error('Load failed:', error);
    } finally {
      elements!.loadBtn.disabled = false;
      elements!.loadBtn.textContent = 'Load Model';
    }
  });
  
  // Send delta button
  elements.sendDeltaBtn.addEventListener('click', async () => {
    try {
      elements!.sendDeltaBtn.disabled = true;
      elements!.sendDeltaBtn.textContent = 'Sending...';
      
      await handlers!.onSendDelta();
      updateStatus('Delta sent to FL server');
      
    } catch (error) {
      console.error('Send delta failed:', error);
      updateStatus(`Send delta failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      elements!.sendDeltaBtn.disabled = false;
      elements!.sendDeltaBtn.textContent = 'Send Delta';
    }
  });
  
  // Apply aggregate button
  elements.applyAggregateBtn.addEventListener('click', async () => {
    try {
      elements!.applyAggregateBtn.disabled = true;
      elements!.applyAggregateBtn.textContent = 'Applying...';
      
      await handlers!.onApplyAggregate();
      updateStatus('Aggregate applied');
      
    } catch (error) {
      console.error('Apply aggregate failed:', error);
      updateStatus(`Apply aggregate failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      elements!.applyAggregateBtn.disabled = false;
      elements!.applyAggregateBtn.textContent = 'Apply Aggregate';
    }
  });
  
  // Switch model button
  elements.switchModelBtn.addEventListener('click', async () => {
    const backend = elements!.backendSelect.value;
    const modelName = elements!.modelSelect.value;
    
    if (!modelName) {
      alert('Please select a model first');
      return;
    }
    
    try {
      elements!.switchModelBtn.disabled = true;
      elements!.switchModelBtn.textContent = 'Switching...';
      
      await handlers!.onSwitchModel(backend, modelName);
      
    } catch (error) {
      console.error('Switch model failed:', error);
      updateStatus(`Switch model failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      elements!.switchModelBtn.disabled = false;
      elements!.switchModelBtn.textContent = 'Switch Model';
    }
  });
  
  // Clear cache button
  elements.clearCacheBtn.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to clear the model cache? This will require re-downloading models.')) {
      return;
    }
    
    try {
      elements!.clearCacheBtn.disabled = true;
      elements!.clearCacheBtn.textContent = 'Clearing...';
      
      await handlers!.onClearCache();
      updateStatus('Model cache cleared');
      
    } catch (error) {
      console.error('Clear cache failed:', error);
      updateStatus(`Clear cache failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      elements!.clearCacheBtn.disabled = false;
      elements!.clearCacheBtn.textContent = 'Clear Model Cache';
    }
  });
  
  // Model selection change handler
  elements.modelSelect.addEventListener('change', updateModelInfo);
  
  // Enable buttons after initialization
  elements.predictBtn.disabled = false;
  elements.trainBtn.disabled = false;
  elements.saveBtn.disabled = false;
  elements.loadBtn.disabled = false;
  elements.sendDeltaBtn.disabled = false;
  elements.applyAggregateBtn.disabled = false;
  elements.switchModelBtn.disabled = false;
  elements.clearCacheBtn.disabled = false;
}

function displayPredictionResult(result: { label: string; probs: number[] }): void {
  if (!elements) return;
  
  const labels = ['Dev', 'Meeting', 'Email'];
  
  // Show results div
  elements.results.style.display = 'block';
  
  // Create results HTML
  const probsHtml = labels.map((label, idx) => 
    `<div><strong>${label}:</strong> ${(result.probs[idx] * 100).toFixed(1)}%</div>`
  ).join('');
  
  elements.results.innerHTML = `
    <div><strong>Predicted Label:</strong> ${result.label}</div>
    <div style="margin-top: 10px;"><strong>Probabilities:</strong></div>
    ${probsHtml}
  `;
  
  // Draw probability chart
  drawProbabilityChart(result.probs, labels);
}

function drawProbabilityChart(probs: number[], labels: string[]): void {
  if (!elements) return;
  
  const canvas = elements.probsChart;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  // Show canvas
  canvas.style.display = 'block';
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Chart dimensions
  const padding = 40;
  const chartWidth = canvas.width - 2 * padding;
  const chartHeight = canvas.height - 2 * padding;
  const barWidth = chartWidth / labels.length;
  
  // Colors for bars
  const colors = ['#2196f3', '#4caf50', '#ff9800'];
  
  // Draw bars
  probs.forEach((prob, idx) => {
    const barHeight = prob * chartHeight;
    const x = padding + idx * barWidth + barWidth * 0.1;
    const y = padding + chartHeight - barHeight;
    const width = barWidth * 0.8;
    
    // Draw bar
    ctx.fillStyle = colors[idx] || '#666';
    ctx.fillRect(x, y, width, barHeight);
    
    // Draw label
    ctx.fillStyle = '#333';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(labels[idx], x + width / 2, padding + chartHeight + 15);
    
    // Draw percentage
    ctx.fillStyle = '#666';
    ctx.font = '10px Arial';
    ctx.fillText(`${(prob * 100).toFixed(1)}%`, x + width / 2, y - 5);
  });
  
  // Draw axes
  ctx.strokeStyle = '#ccc';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, padding + chartHeight);
  ctx.lineTo(padding + chartWidth, padding + chartHeight);
  ctx.stroke();
}

export function updateStatus(message: string): void {
  if (elements) {
    elements.status.textContent = message;
    console.log('Status:', message);
  }
}

export function updateMetrics(newMetrics: Partial<typeof metrics>): void {
  if (!elements) return;
  
  Object.assign(metrics, newMetrics);
  
  if (newMetrics.backend) {
    elements.backend.textContent = newMetrics.backend.toUpperCase();
  }
  
  if (typeof newMetrics.latency === 'number') {
    elements.latency.textContent = newMetrics.latency.toFixed(1);
  }
  
  if (typeof newMetrics.memory === 'number') {
    elements.memory.textContent = newMetrics.memory.toString();
  }
  
  if (typeof newMetrics.avgLoss === 'number') {
    // Keep running average of loss
    const alpha = 0.1;
    metrics.avgLoss = metrics.avgLoss === 0 ? 
      newMetrics.avgLoss : 
      alpha * newMetrics.avgLoss + (1 - alpha) * metrics.avgLoss;
    
    elements.avgLoss.textContent = metrics.avgLoss.toFixed(4);
  }
  
  // Update memory periodically
  if ((performance as any).memory?.usedJSHeapSize) {
    const memoryMB = Math.round((performance as any).memory.usedJSHeapSize / 1024 / 1024);
    elements.memory.textContent = memoryMB.toString();
  }
}

export function getMetrics() {
  return { ...metrics };
}

// Utility function to show toast notifications
export function populateModelSelect(models: Array<{name: string; description: string}>): void {
  if (!elements) return;
  
  // Clear existing options
  elements.modelSelect.innerHTML = '';
  
  // Add model options
  models.forEach(model => {
    const option = document.createElement('option');
    option.value = model.name;
    option.textContent = model.description;
    elements!.modelSelect.appendChild(option);
  });
  
  // Set default selection and update info
  if (models.length > 1) {
    elements.modelSelect.value = models[1].name; // Select quantized model by default
  }
  updateModelInfo();
}

export function setCurrentBackend(backend: string): void {
  if (!elements) return;
  elements.backendSelect.value = backend;
}

function updateModelInfo(): void {
  if (!elements) return;
  
  const selectedModel = elements.modelSelect.value;
  if (!selectedModel) {
    elements.modelInfo.textContent = '';
    return;
  }
  
  // This would normally come from the model loader
  let info = '';
  switch (selectedModel) {
    case 'distilbert-base-uncased-onnx':
      info = 'Full precision model (~250MB) - Best quality, slower inference';
      break;
    case 'distilbert-base-uncased-quantized':
      info = 'Quantized model (~65MB) - Good quality, faster inference';
      break;
    case 'local-model':
      info = 'Local model files - Custom model from public/ directory';
      break;
    default:
      info = 'Model information not available';
  }
  
  elements.modelInfo.textContent = info;
}

export function showToast(message: string, type: 'success' | 'error' | 'info' = 'info'): void {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 12px 20px;
    border-radius: 4px;
    color: white;
    font-weight: bold;
    z-index: 1000;
    opacity: 0;
    transition: opacity 0.3s ease;
  `;
  
  const colors = {
    success: '#4caf50',
    error: '#f44336',
    info: '#2196f3'
  };
  
  toast.style.backgroundColor = colors[type];
  toast.textContent = message;
  
  document.body.appendChild(toast);
  
  // Fade in
  requestAnimationFrame(() => {
    toast.style.opacity = '1';
  });
  
  // Remove after 3 seconds
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => {
      document.body.removeChild(toast);
    }, 300);
  }, 3000);
}