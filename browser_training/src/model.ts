import * as ort from 'onnxruntime-web';
import { MockDistilBERT } from './mock-model.js';
import { downloadModel, cacheModel, loadCachedModel, AVAILABLE_MODELS, type ModelSource } from './model-loader.js';

export type BackendType = 'webgpu' | 'wasm' | 'mock';

let session: ort.InferenceSession | null = null;
let tokenizer: any = null;
let mockModel: MockDistilBERT | null = null;
let currentBackend: BackendType = 'mock';
let currentModel: ModelSource | null = null;

export async function initModel(
  backend: BackendType, 
  modelSource?: ModelSource,
  onProgress?: (progress: number, stage: string) => void
): Promise<void> {
  currentBackend = backend;
  currentModel = modelSource || AVAILABLE_MODELS[1]; // Default to quantized model
  
  try {
    if (backend === 'mock') {
      // Use mock model for testing
      onProgress?.(50, 'Initializing mock model...');
      mockModel = new MockDistilBERT();
      await mockModel.initialize();
      console.log('Mock model initialized for testing');
      
      // Load tokenizer for mock mode too
      const tokenizerResponse = await fetch('/public/tokenizer.json');
      if (!tokenizerResponse.ok) {
        throw new Error('Failed to load tokenizer');
      }
      tokenizer = await tokenizerResponse.json();
      onProgress?.(100, 'Mock model ready');
      return;
    }
    
    // Configure ONNX Runtime for real models
    if (backend === 'webgpu') {
      ort.env.wasm.numThreads = 1;
      ort.env.webgpu.powerPreference = 'high-performance';
    } else {
      ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    }
    
    // Set execution providers based on backend
    const providers = backend === 'webgpu' 
      ? ['webgpu', 'wasm'] 
      : ['wasm'];
    
    onProgress?.(10, 'Checking for cached model...');
    
    // Try to load from cache first
    let modelData = await loadCachedModel(currentModel.name);
    
    if (!modelData) {
      onProgress?.(20, `Downloading ${currentModel.name}...`);
      
      // Download the model
      modelData = await downloadModel(currentModel, (progress) => {
        onProgress?.(20 + (progress * 0.6), `Downloading model: ${progress.toFixed(1)}%`);
      });
      
      onProgress?.(80, 'Caching model...');
      
      // Cache for future use
      await cacheModel(currentModel.name, modelData.onnxBuffer, modelData.tokenizerData);
    }
    
    onProgress?.(85, 'Loading ONNX session...');
    
    try {
      // Create session from buffer
      session = await ort.InferenceSession.create(modelData.onnxBuffer, {
        executionProviders: providers,
        graphOptimizationLevel: 'all',
        logSeverityLevel: 2
      });
      
      console.log(`Model loaded with ${backend} backend`);
      console.log('Model:', currentModel.name);
      console.log('Input names:', session.inputNames);
      console.log('Output names:', session.outputNames);
      
      // Log input/output metadata for debugging
      session.inputNames.forEach((name, idx) => {
        console.log(`Input ${idx}: ${name}`, (session as any).inputMetadata[name]);
      });
      session.outputNames.forEach((name, idx) => {
        console.log(`Output ${idx}: ${name}`, (session as any).outputMetadata[name]);
      });
      
      // Set tokenizer data
      tokenizer = modelData.tokenizerData;
      
      onProgress?.(100, `${currentModel.name} ready`);
      
    } catch (modelError) {
      console.warn(`Failed to load ONNX model, falling back to mock model:`, modelError);
      
      onProgress?.(90, 'Falling back to mock model...');
      
      // Fall back to mock model
      mockModel = new MockDistilBERT();
      await mockModel.initialize();
      currentBackend = 'mock';
      
      // Load local tokenizer for fallback
      const tokenizerResponse = await fetch('/public/tokenizer.json');
      if (tokenizerResponse.ok) {
        tokenizer = await tokenizerResponse.json();
      } else {
        tokenizer = modelData.tokenizerData; // Use downloaded tokenizer
      }
      
      onProgress?.(100, 'Mock model ready (ONNX failed)');
      console.log('Switched to mock model due to ONNX loading failure');
    }
    
  } catch (error) {
    console.error('Model initialization error:', error);
    
    // Last resort: try mock model
    console.log('Attempting fallback to mock model...');
    try {
      mockModel = new MockDistilBERT();
      await mockModel.initialize();
      currentBackend = 'mock';
      
      const tokenizerResponse = await fetch('/public/tokenizer.json');
      if (tokenizerResponse.ok) {
        tokenizer = await tokenizerResponse.json();
      }
      
      onProgress?.(100, 'Mock model ready (fallback)');
      console.log('Successfully fell back to mock model');
    } catch (fallbackError) {
      throw new Error(`Failed to initialize any model: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
}

export async function embedCLS(text: string): Promise<Float32Array> {
  if (currentBackend === 'mock') {
    if (!mockModel) {
      throw new Error('Mock model not initialized');
    }
    return mockModel.embedCLS(text);
  }
  
  if (!session) {
    throw new Error('Model not initialized');
  }
  
  try {
    // Tokenize the input text
    const tokens = await tokenize(text);
    
    // Determine the expected input data type from model metadata
    const inputIdName = session.inputNames.find(name => 
      name.includes('input_ids') || name.includes('input') 
    ) || session.inputNames[0];
    
    const inputMetadata = (session as any).inputMetadata?.[inputIdName];
    const expectedType = inputMetadata?.type || 'tensor(int64)';
    const isInt64 = expectedType.includes('int64');
    
    console.log(`Using input type: ${expectedType} (int64: ${isInt64})`);
    
    // Pad or truncate to max sequence length (256)
    const maxLen = 256;
    const paddedTokens = tokens.slice(0, maxLen);
    while (paddedTokens.length < maxLen) {
      paddedTokens.push(0); // PAD token
    }
    
    // Create tensors with appropriate data type
    let inputTensor: ort.Tensor;
    let maskTensor: ort.Tensor;
    
    if (isInt64) {
      const inputIds = new BigInt64Array(paddedTokens.map(t => BigInt(t)));
      const attentionMask = new BigInt64Array(paddedTokens.map((_, i) => 
        i < tokens.length ? BigInt(1) : BigInt(0)
      ));
      
      inputTensor = new ort.Tensor('int64', inputIds, [1, maxLen]);
      maskTensor = new ort.Tensor('int64', attentionMask, [1, maxLen]);
    } else {
      const inputIds = new Int32Array(paddedTokens);
      const attentionMask = new Int32Array(paddedTokens.map((_, i) => 
        i < tokens.length ? 1 : 0
      ));
      
      inputTensor = new ort.Tensor('int32', inputIds, [1, maxLen]);
      maskTensor = new ort.Tensor('int32', attentionMask, [1, maxLen]);
    }
    
    // Determine input names dynamically
    const feeds: Record<string, ort.Tensor> = {};
    
    // Map input_ids
    if (session.inputNames.includes('input_ids')) {
      feeds['input_ids'] = inputTensor;
    } else if (session.inputNames.includes('inputs')) {
      feeds['inputs'] = inputTensor;
    } else if (session.inputNames.length > 0) {
      feeds[session.inputNames[0]] = inputTensor;
      console.log(`Mapped input_ids to: ${session.inputNames[0]}`);
    }
    
    // Map attention_mask
    if (session.inputNames.includes('attention_mask')) {
      feeds['attention_mask'] = maskTensor;
    } else if (session.inputNames.includes('mask')) {
      feeds['mask'] = maskTensor;
    } else if (session.inputNames.length > 1) {
      feeds[session.inputNames[1]] = maskTensor;
      console.log(`Mapped attention_mask to: ${session.inputNames[1]}`);
    }
    
    console.log('Feeding inputs:', Object.keys(feeds));
    console.log('Input shapes:', Object.entries(feeds).map(([name, tensor]) => 
      `${name}: [${tensor.dims.join(', ')}]`
    ));
    
    // Run inference
    const results = await session.run(feeds);
    
    // Extract CLS token embedding (first token of last hidden state)
    const outputName = session.outputNames[0];
    const hiddenStates = results[outputName];
    
    if (!hiddenStates || !hiddenStates.data) {
      throw new Error('Invalid model output');
    }
    
    // Extract CLS embedding [batch_size=1, seq_len, hidden_size=768]
    const hiddenSize = 768;
    const clsEmbedding = new Float32Array(hiddenSize);
    
    // Copy the first token's embeddings (CLS token)
    for (let i = 0; i < hiddenSize; i++) {
      clsEmbedding[i] = (hiddenStates.data as Float32Array)[i];
    }
    
    return clsEmbedding;
    
  } catch (error) {
    console.error('Embedding error:', error);
    console.error('Session inputs expected:', session?.inputNames);
    console.error('Session inputs metadata:', (session as any)?.inputMetadata);
    
    // Provide more specific error messages
    if (error instanceof Error) {
      if (error.message.includes('data type')) {
        throw new Error(`Data type mismatch: ${error.message}. Check model input requirements.`);
      } else if (error.message.includes('shape')) {
        throw new Error(`Shape mismatch: ${error.message}. Check input tensor dimensions.`);
      } else {
        throw new Error(`ONNX Runtime error: ${error.message}`);
      }
    }
    
    throw new Error(`Failed to get embeddings: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

async function tokenize(text: string): Promise<number[]> {
  if (!tokenizer) {
    throw new Error('Tokenizer not loaded');
  }
  
  // Simple tokenization using the loaded tokenizer
  // This is a simplified implementation - in practice you'd use a proper tokenizer
  const worker = new Worker(new URL('./tokenizer-worker.ts', import.meta.url), { type: 'module' });
  
  return new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      if (e.data.error) {
        reject(new Error(e.data.error));
      } else {
        resolve(e.data.tokens);
      }
      worker.terminate();
    };
    
    worker.onerror = (error) => {
      reject(error);
      worker.terminate();
    };
    
    worker.postMessage({ text, tokenizer });
  });
}

export function getAvailableModels(): ModelSource[] {
  return AVAILABLE_MODELS;
}

export function getCurrentModel(): ModelSource | null {
  return currentModel;
}

export function getCurrentBackend(): BackendType {
  return currentBackend;
}

export async function switchModel(backend: BackendType, modelSource: ModelSource, onProgress?: (progress: number, stage: string) => void): Promise<void> {
  // Clean up current session
  if (session) {
    session.release();
    session = null;
  }
  mockModel = null;
  tokenizer = null;
  
  // Initialize with new model
  await initModel(backend, modelSource, onProgress);
}