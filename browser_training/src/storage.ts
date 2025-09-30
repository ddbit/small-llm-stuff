import type { HeadWeights } from './head.js';

const DB_NAME = 'TinyBertStorage';
const DB_VERSION = 1;
const STORE_NAME = 'modelWeights';

export interface StoredModel {
  id: string;
  modelId: string;
  headVersion: number;
  weights: HeadWeights;
  timestamp: number;
  metadata?: {
    inputSize: number;
    numClasses: number;
    trainingSteps: number;
  };
}

let db: IDBDatabase | null = null;

export async function initStorage(): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    
    request.onerror = () => {
      console.error('Failed to open IndexedDB:', request.error);
      reject(new Error('Failed to initialize storage'));
    };
    
    request.onsuccess = () => {
      db = request.result;
      console.log('IndexedDB initialized successfully');
      resolve();
    };
    
    request.onupgradeneeded = (event) => {
      const database = (event.target as IDBOpenDBRequest).result;
      
      // Create object store
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        const store = database.createObjectStore(STORE_NAME, { keyPath: 'id' });
        
        // Create indices
        store.createIndex('modelId', 'modelId', { unique: false });
        store.createIndex('timestamp', 'timestamp', { unique: false });
        store.createIndex('headVersion', 'headVersion', { unique: false });
        
        console.log('Created object store and indices');
      }
    };
  });
}

export async function saveHead(
  weights: HeadWeights, 
  modelId = 'distilbert-base',
  metadata?: Partial<StoredModel['metadata']>
): Promise<string> {
  if (!db) {
    throw new Error('Storage not initialized');
  }
  
  // Generate unique ID for this save
  const id = `${modelId}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  // Get current head version (increment from latest)
  const latestModel = await getLatestModel(modelId);
  const headVersion = latestModel ? latestModel.headVersion + 1 : 1;
  
  const storedModel: StoredModel = {
    id,
    modelId,
    headVersion,
    weights: {
      W: new Float32Array(weights.W),
      b: new Float32Array(weights.b)
    },
    timestamp: Date.now(),
    metadata: metadata ? {
      inputSize: metadata.inputSize || 768,
      numClasses: metadata.numClasses || 3,
      trainingSteps: metadata.trainingSteps || 0
    } : undefined
  };
  
  return new Promise((resolve, reject) => {
    const transaction = db!.transaction([STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    
    transaction.onerror = () => {
      console.error('Save transaction failed:', transaction.error);
      reject(new Error('Failed to save model'));
    };
    
    transaction.oncomplete = () => {
      console.log(`Model saved with ID: ${id}, version: ${headVersion}`);
      resolve(id);
    };
    
    store.add(storedModel);
  });
}

export async function loadHead(modelId = 'distilbert-base'): Promise<HeadWeights | null> {
  if (!db) {
    throw new Error('Storage not initialized');
  }
  
  const latestModel = await getLatestModel(modelId);
  return latestModel ? latestModel.weights : null;
}

export async function loadHeadById(id: string): Promise<StoredModel | null> {
  if (!db) {
    throw new Error('Storage not initialized');
  }
  
  return new Promise((resolve, reject) => {
    const transaction = db!.transaction([STORE_NAME], 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.get(id);
    
    request.onerror = () => {
      console.error('Load by ID failed:', request.error);
      reject(new Error('Failed to load model by ID'));
    };
    
    request.onsuccess = () => {
      resolve(request.result || null);
    };
  });
}

async function getLatestModel(modelId: string): Promise<StoredModel | null> {
  if (!db) {
    throw new Error('Storage not initialized');
  }
  
  return new Promise((resolve, reject) => {
    const transaction = db!.transaction([STORE_NAME], 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const index = store.index('modelId');
    const request = index.openCursor(IDBKeyRange.only(modelId), 'prev');
    
    request.onerror = () => {
      console.error('Get latest model failed:', request.error);
      reject(new Error('Failed to get latest model'));
    };
    
    request.onsuccess = () => {
      const cursor = request.result;
      const models: StoredModel[] = [];
      if (cursor) {
        // Find the model with the highest headVersion
        do {
          models.push(cursor.value);
        } while (cursor.continue() !== null);
      }
      
      if (models.length > 0) {
        const latest = models.reduce((prev, current) => 
          current.headVersion > prev.headVersion ? current : prev
        );
        resolve(latest);
      } else {
        resolve(null);
      }
    };
  });
}

export async function listModels(modelId?: string): Promise<StoredModel[]> {
  if (!db) {
    throw new Error('Storage not initialized');
  }
  
  return new Promise((resolve, reject) => {
    const transaction = db!.transaction([STORE_NAME], 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    
    const request = modelId 
      ? store.index('modelId').getAll(modelId)
      : store.getAll();
    
    request.onerror = () => {
      console.error('List models failed:', request.error);
      reject(new Error('Failed to list models'));
    };
    
    request.onsuccess = () => {
      const models = request.result as StoredModel[];
      // Sort by timestamp (newest first)
      models.sort((a, b) => b.timestamp - a.timestamp);
      resolve(models);
    };
  });
}

export async function deleteModel(id: string): Promise<void> {
  if (!db) {
    throw new Error('Storage not initialized');
  }
  
  return new Promise((resolve, reject) => {
    const transaction = db!.transaction([STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    
    transaction.onerror = () => {
      console.error('Delete transaction failed:', transaction.error);
      reject(new Error('Failed to delete model'));
    };
    
    transaction.oncomplete = () => {
      console.log(`Model deleted: ${id}`);
      resolve();
    };
    
    store.delete(id);
  });
}

export async function clearAllModels(): Promise<void> {
  if (!db) {
    throw new Error('Storage not initialized');
  }
  
  return new Promise((resolve, reject) => {
    const transaction = db!.transaction([STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    
    transaction.onerror = () => {
      console.error('Clear all transaction failed:', transaction.error);
      reject(new Error('Failed to clear all models'));
    };
    
    transaction.oncomplete = () => {
      console.log('All models cleared');
      resolve();
    };
    
    store.clear();
  });
}

export async function getStorageInfo(): Promise<{
  totalModels: number;
  estimatedSize: number;
  oldestTimestamp?: number;
  newestTimestamp?: number;
}> {
  const models = await listModels();
  
  let estimatedSize = 0;
  let oldestTimestamp: number | undefined;
  let newestTimestamp: number | undefined;
  
  for (const model of models) {
    // Estimate size: Float32Array size + JSON overhead
    const weightsSize = model.weights.W.length * 4 + model.weights.b.length * 4;
    const jsonOverhead = JSON.stringify(model).length * 2; // rough estimate
    estimatedSize += weightsSize + jsonOverhead;
    
    if (!oldestTimestamp || model.timestamp < oldestTimestamp) {
      oldestTimestamp = model.timestamp;
    }
    if (!newestTimestamp || model.timestamp > newestTimestamp) {
      newestTimestamp = model.timestamp;
    }
  }
  
  return {
    totalModels: models.length,
    estimatedSize,
    oldestTimestamp,
    newestTimestamp
  };
}