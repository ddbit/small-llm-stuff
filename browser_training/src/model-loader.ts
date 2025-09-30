// Runtime model loader for DistilBERT ONNX models

export interface ModelSource {
  name: string;
  onnxUrl: string;
  tokenizerUrl: string;
  description: string;
}

export const AVAILABLE_MODELS: ModelSource[] = [
  {
    name: 'distilbert-base-uncased-onnx',
    onnxUrl: 'https://huggingface.co/Xenova/distilbert-base-uncased/resolve/main/onnx/model.onnx',
    tokenizerUrl: 'https://huggingface.co/Xenova/distilbert-base-uncased/resolve/main/tokenizer.json',
    description: 'DistilBERT base uncased (Xenova/transformers.js compatible)'
  },
  {
    name: 'distilbert-base-uncased-quantized',
    onnxUrl: 'https://huggingface.co/Xenova/distilbert-base-uncased/resolve/main/onnx/model_quantized.onnx',
    tokenizerUrl: 'https://huggingface.co/Xenova/distilbert-base-uncased/resolve/main/tokenizer.json',
    description: 'DistilBERT base uncased quantized (smaller, faster)'
  },
  {
    name: 'local-model',
    onnxUrl: '/public/distilbert-int8.onnx',
    tokenizerUrl: '/public/tokenizer.json',
    description: 'Local model files'
  }
];

export async function downloadModel(source: ModelSource, onProgress?: (progress: number) => void): Promise<{ onnxBuffer: ArrayBuffer; tokenizerData: any }> {
  console.log(`Starting download of ${source.name}...`);
  
  // Download tokenizer first (smaller)
  const tokenizerResponse = await fetch(source.tokenizerUrl);
  if (!tokenizerResponse.ok) {
    throw new Error(`Failed to fetch tokenizer: ${tokenizerResponse.statusText}`);
  }
  const tokenizerData = await tokenizerResponse.json();
  console.log('Tokenizer downloaded successfully');
  
  // Download ONNX model with progress tracking
  const onnxResponse = await fetch(source.onnxUrl);
  if (!onnxResponse.ok) {
    throw new Error(`Failed to fetch ONNX model: ${onnxResponse.statusText}`);
  }
  
  const contentLength = onnxResponse.headers.get('content-length');
  const totalSize = contentLength ? parseInt(contentLength, 10) : 0;
  
  if (!onnxResponse.body) {
    throw new Error('Response body is null');
  }
  
  const reader = onnxResponse.body.getReader();
  const chunks: Uint8Array[] = [];
  let receivedLength = 0;
  
  while (true) {
    const { done, value } = await reader.read();
    
    if (done) break;
    
    chunks.push(value);
    receivedLength += value.length;
    
    if (onProgress && totalSize > 0) {
      onProgress((receivedLength / totalSize) * 100);
    }
  }
  
  // Combine chunks into single ArrayBuffer
  const onnxBuffer = new ArrayBuffer(receivedLength);
  const uint8View = new Uint8Array(onnxBuffer);
  let offset = 0;
  
  for (const chunk of chunks) {
    uint8View.set(chunk, offset);
    offset += chunk.length;
  }
  
  console.log(`Model downloaded: ${receivedLength} bytes`);
  
  return { onnxBuffer, tokenizerData };
}

export async function cacheModel(name: string, onnxBuffer: ArrayBuffer, tokenizerData: any): Promise<void> {
  if ('caches' in window) {
    try {
      const cache = await caches.open('tinybert-models');
      
      // Cache ONNX model
      const onnxResponse = new Response(onnxBuffer, {
        headers: { 'Content-Type': 'application/octet-stream' }
      });
      await cache.put(`/model/${name}/model.onnx`, onnxResponse);
      
      // Cache tokenizer
      const tokenizerResponse = new Response(JSON.stringify(tokenizerData), {
        headers: { 'Content-Type': 'application/json' }
      });
      await cache.put(`/model/${name}/tokenizer.json`, tokenizerResponse);
      
      console.log(`Model ${name} cached successfully`);
    } catch (error) {
      console.warn('Failed to cache model:', error);
    }
  }
}

export async function loadCachedModel(name: string): Promise<{ onnxBuffer: ArrayBuffer; tokenizerData: any } | null> {
  if ('caches' in window) {
    try {
      const cache = await caches.open('tinybert-models');
      
      const onnxResponse = await cache.match(`/model/${name}/model.onnx`);
      const tokenizerResponse = await cache.match(`/model/${name}/tokenizer.json`);
      
      if (onnxResponse && tokenizerResponse) {
        const onnxBuffer = await onnxResponse.arrayBuffer();
        const tokenizerData = await tokenizerResponse.json();
        
        console.log(`Loaded cached model ${name}`);
        return { onnxBuffer, tokenizerData };
      }
    } catch (error) {
      console.warn('Failed to load cached model:', error);
    }
  }
  
  return null;
}

export async function clearModelCache(): Promise<void> {
  if ('caches' in window) {
    try {
      await caches.delete('tinybert-models');
      console.log('Model cache cleared');
    } catch (error) {
      console.warn('Failed to clear model cache:', error);
    }
  }
}

export function getModelSizeEstimate(source: ModelSource): string {
  switch (source.name) {
    case 'distilbert-base-uncased-onnx':
      return '~250MB';
    case 'distilbert-base-uncased-quantized':
      return '~65MB';
    case 'local-model':
      return 'Unknown';
    default:
      return 'Unknown';
  }
}