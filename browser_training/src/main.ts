import { initModel, getAvailableModels, switchModel, getCurrentBackend, type BackendType } from './model.js';
import { clearModelCache } from './model-loader.js';
import { initStorage } from './storage.js';
import { initUI, updateStatus, updateMetrics, populateModelSelect, setCurrentBackend } from './ui.js';
import { ClassificationHead } from './head.js';
import { initFederatedLearning } from './fl.js';

let backend: BackendType | null = null;
let classificationHead: ClassificationHead | null = null;
let currentText = '';

async function detectBestBackend(): Promise<BackendType> {
  // Check for WebGPU support first
  if ('gpu' in navigator) {
    try {
      const gpu = navigator.gpu as any;
      const adapter = await gpu.requestAdapter();
      if (adapter) {
        console.log('WebGPU support detected');
        return 'webgpu';
      }
    } catch (error) {
      console.warn('WebGPU detection failed:', error);
    }
  }
  
  console.log('WebGPU not available, using WASM');
  return 'wasm';
}

async function predict(text: string): Promise<{ label: string; probs: number[] }> {
  if (!backend || !classificationHead) {
    throw new Error('Model not initialized');
  }
  
  const startTime = performance.now();
  
  try {
    // Get CLS embeddings from the model
    const { embedCLS } = await import('./model.js');
    const embeddings = await embedCLS(text);
    
    // Run through classification head
    const logits = classificationHead.forward(embeddings);
    const probs = softmax(logits);
    
    const maxProb = Math.max(...probs);
    const predictedClass = probs.indexOf(maxProb);
    
    const labels = ['Dev', 'Meeting', 'Email'];
    const label = maxProb < 0.55 ? 'Undetermined' : labels[predictedClass];
    
    const latency = performance.now() - startTime;
    updateMetrics({ latency });
    
    return { label, probs };
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
}

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map(x => Math.exp(x - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  return expLogits.map(x => x / sumExp);
}

async function trainStep(text: string, label: number) {
  if (!backend || !classificationHead) {
    throw new Error('Model not initialized');
  }
  
  try {
    const { embedCLS } = await import('./model.js');
    const embeddings = await embedCLS(text);
    
    const loss = classificationHead.trainStep(embeddings, label);
    updateMetrics({ avgLoss: loss });
    
    console.log(`Training step completed, loss: ${loss.toFixed(4)}`);
  } catch (error) {
    console.error('Training error:', error);
    throw error;
  }
}

async function saveModel() {
  if (!classificationHead) {
    throw new Error('Model not initialized');
  }
  
  try {
    const { saveHead } = await import('./storage.js');
    await saveHead(classificationHead.getWeights());
    updateStatus('Model saved successfully');
  } catch (error) {
    console.error('Save error:', error);
    updateStatus('Failed to save model');
  }
}

async function loadModel() {
  if (!classificationHead) {
    throw new Error('Model not initialized');
  }
  
  try {
    const { loadHead } = await import('./storage.js');
    const weights = await loadHead();
    if (weights) {
      classificationHead.setWeights(weights);
      updateStatus('Model loaded successfully');
    } else {
      updateStatus('No saved model found');
    }
  } catch (error) {
    console.error('Load error:', error);
    updateStatus('Failed to load model');
  }
}

async function bootstrap() {
  try {
    updateStatus('Detecting backend capabilities...');
    backend = await detectBestBackend();
    
    updateStatus('Loading model...');
    const models = getAvailableModels();
    const selectedModel = models[1]; // Start with quantized model
    
    await initModel(backend, selectedModel, (progress, stage) => {
      updateStatus(`${stage} (${progress.toFixed(0)}%)`);
    });
    
    updateStatus('Initializing classification head...');
    classificationHead = new ClassificationHead(768, 3);
    
    updateStatus('Initializing storage...');
    await initStorage();
    
    updateStatus('Initializing federated learning...');
    await initFederatedLearning();
    
    updateStatus('Setting up UI...');
    const ui = await initUI();
    
    // Populate model selection UI
    const availableModels = getAvailableModels();
    populateModelSelect(availableModels);
    setCurrentBackend(backend);
    
    // Bind event handlers
    ui.onPredict = async (text: string) => {
      currentText = text;
      const result = await predict(text);
      return result;
    };
    
    ui.onTrain = async (label: number) => {
      if (!currentText) {
        throw new Error('No text to train on');
      }
      await trainStep(currentText, label);
    };
    
    ui.onSave = saveModel;
    ui.onLoad = loadModel;
    
    ui.onSwitchModel = async (backendType: string, modelName: string) => {
      const selectedModel = availableModels.find(m => m.name === modelName);
      if (!selectedModel) {
        throw new Error('Selected model not found');
      }
      
      updateStatus('Switching model...');
      
      await switchModel(backendType as BackendType, selectedModel, (progress, stage) => {
        updateStatus(`${stage} (${progress.toFixed(0)}%)`);
      });
      
      // Update backend info
      backend = getCurrentBackend();
      updateMetrics({ backend });
      
      updateStatus('Model switched successfully');
    };
    
    ui.onClearCache = async () => {
      await clearModelCache();
    };
    
    ui.onSendDelta = async () => {
      if (!classificationHead) return;
      const { getDelta, flClientRound } = await import('./fl.js');
      const delta = getDelta(classificationHead.getWeights());
      await flClientRound(delta);
    };
    
    ui.onApplyAggregate = async () => {
      if (!classificationHead) return;
      // This would be called after receiving aggregated weights from server
      console.log('Apply aggregate not implemented - would receive weights from FL server');
    };
    
    updateMetrics({ 
      backend,
      memory: (performance as any).memory?.usedJSHeapSize ? 
        Math.round((performance as any).memory.usedJSHeapSize / 1024 / 1024) : 
        undefined
    });
    
    updateStatus('Ready! Enter text to classify or train the model.');
    
  } catch (error) {
    console.error('Bootstrap error:', error);
    updateStatus(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

// Start the application
bootstrap();