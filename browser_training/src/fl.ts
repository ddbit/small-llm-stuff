import type { HeadWeights } from './head.js';

export interface QuantizedDelta {
  W: Int8Array;
  b: Int8Array;
  scaleW: number;
  scaleb: number;
  metadata: {
    clientId: string;
    round: number;
    timestamp: number;
    numSamples: number;
  };
}

export interface FLAggregateResponse {
  success: boolean;
  aggregatedWeights?: HeadWeights;
  nextRound?: number;
  message?: string;
}

let clientId: string = '';
let currentRound = 0;
let baselineWeights: HeadWeights | null = null;

export async function initFederatedLearning(): Promise<void> {
  // Generate or retrieve client ID
  clientId = localStorage.getItem('fl-client-id') || generateClientId();
  localStorage.setItem('fl-client-id', clientId);
  
  console.log(`FL Client initialized with ID: ${clientId}`);
}

function generateClientId(): string {
  return 'client_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
}

export function setBaseline(weights: HeadWeights): void {
  baselineWeights = {
    W: new Float32Array(weights.W),
    b: new Float32Array(weights.b)
  };
  console.log('FL baseline weights set');
}

export function getDelta(currentWeights: HeadWeights, numSamples = 1): QuantizedDelta {
  if (!baselineWeights) {
    console.warn('No baseline weights set, using zero delta');
    return quantizeDelta({
      W: new Float32Array(currentWeights.W.length),
      b: new Float32Array(currentWeights.b.length)
    }, numSamples);
  }
  
  // Compute weight differences (delta)
  const deltaW = new Float32Array(currentWeights.W.length);
  const deltab = new Float32Array(currentWeights.b.length);
  
  for (let i = 0; i < deltaW.length; i++) {
    deltaW[i] = currentWeights.W[i] - baselineWeights.W[i];
  }
  
  for (let i = 0; i < deltab.length; i++) {
    deltab[i] = currentWeights.b[i] - baselineWeights.b[i];
  }
  
  return quantizeDelta({ W: deltaW, b: deltab }, numSamples);
}

function quantizeDelta(delta: HeadWeights, numSamples: number): QuantizedDelta {
  // Find scaling factors for quantization
  const maxW = Math.max(...Array.from(delta.W).map(Math.abs));
  const maxb = Math.max(...Array.from(delta.b).map(Math.abs));
  
  const scaleW = maxW > 0 ? maxW / 127 : 1e-8; // Map to [-127, 127]
  const scaleb = maxb > 0 ? maxb / 127 : 1e-8;
  
  // Quantize to int8
  const quantizedW = new Int8Array(delta.W.length);
  const quantizedb = new Int8Array(delta.b.length);
  
  for (let i = 0; i < delta.W.length; i++) {
    quantizedW[i] = Math.round(Math.max(-127, Math.min(127, delta.W[i] / scaleW)));
  }
  
  for (let i = 0; i < delta.b.length; i++) {
    quantizedb[i] = Math.round(Math.max(-127, Math.min(127, delta.b[i] / scaleb)));
  }
  
  return {
    W: quantizedW,
    b: quantizedb,
    scaleW,
    scaleb,
    metadata: {
      clientId,
      round: currentRound,
      timestamp: Date.now(),
      numSamples
    }
  };
}

function dequantizeDelta(quantized: QuantizedDelta): HeadWeights {
  const W = new Float32Array(quantized.W.length);
  const b = new Float32Array(quantized.b.length);
  
  for (let i = 0; i < W.length; i++) {
    W[i] = quantized.W[i] * quantized.scaleW;
  }
  
  for (let i = 0; i < b.length; i++) {
    b[i] = quantized.b[i] * quantized.scaleb;
  }
  
  return { W, b };
}

export function applyAggregate(
  currentWeights: HeadWeights, 
  aggregatedDelta: QuantizedDelta | HeadWeights,
  learningRate = 1.0
): HeadWeights {
  let delta: HeadWeights;
  
  if ('scaleW' in aggregatedDelta) {
    // It's quantized, need to dequantize first
    delta = dequantizeDelta(aggregatedDelta as QuantizedDelta);
  } else {
    delta = aggregatedDelta as HeadWeights;
  }
  
  // Apply aggregated update: w_new = w_current + lr * delta_aggregated
  const newWeights: HeadWeights = {
    W: new Float32Array(currentWeights.W.length),
    b: new Float32Array(currentWeights.b.length)
  };
  
  for (let i = 0; i < newWeights.W.length; i++) {
    newWeights.W[i] = currentWeights.W[i] + learningRate * delta.W[i];
  }
  
  for (let i = 0; i < newWeights.b.length; i++) {
    newWeights.b[i] = currentWeights.b[i] + learningRate * delta.b[i];
  }
  
  return newWeights;
}

// HTTP-based federated learning client
export async function flClientRound(
  delta: QuantizedDelta, 
  serverUrl = 'http://localhost:8080/fl'
): Promise<FLAggregateResponse> {
  try {
    console.log('Sending delta to FL server...');
    
    const response = await fetch(`${serverUrl}/update`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        clientId: delta.metadata.clientId,
        round: delta.metadata.round,
        delta: {
          W: Array.from(delta.W),
          b: Array.from(delta.b),
          scaleW: delta.scaleW,
          scaleb: delta.scaleb
        },
        metadata: delta.metadata
      })
    });
    
    if (!response.ok) {
      throw new Error(`FL server error: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    if (result.success && result.aggregatedWeights) {
      currentRound = result.nextRound || currentRound + 1;
      console.log(`FL round ${currentRound} completed successfully`);
    }
    
    return result;
    
  } catch (error) {
    console.error('FL client round failed:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Unknown FL error'
    };
  }
}

// WebRTC-based federated learning (stub implementation)
export class WebRTCFLClient {
  private peerConnection: RTCPeerConnection | null = null;
  private isConnected = false;
  private rtcConfig: RTCConfiguration; // Used for WebRTC connection configuration
  
  constructor(config?: RTCConfiguration) {
    // Default STUN servers for WebRTC
    this.rtcConfig = config || {
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
      ]
    };
  }
  
  async connect(_signalServer = 'ws://localhost:8081'): Promise<void> {
    console.log('WebRTC FL connection not implemented - this is a stub');
    console.log('In production, this would:');
    console.log('1. Connect to signaling server');
    console.log('2. Exchange SDP offers/answers');
    console.log('3. Establish P2P data channel');
    console.log('4. Implement secure aggregation protocol');
    console.log('Using RTC config:', this.rtcConfig);
    
    // Simulate connection
    this.isConnected = true;
  }
  
  async sendDelta(delta: QuantizedDelta): Promise<void> {
    if (!this.isConnected) {
      throw new Error('WebRTC not connected');
    }
    
    console.log('Sending delta via WebRTC (stub):', delta.metadata);
    
    // TODO: Implement actual WebRTC data channel communication
    // This would include:
    // - Serialize delta to binary format
    // - Send via data channel with reliability
    // - Implement secure aggregation protocols
    // - Handle peer coordination for aggregation
  }
  
  disconnect(): void {
    if (this.peerConnection) {
      this.peerConnection.close();
      this.peerConnection = null;
    }
    this.isConnected = false;
    console.log('WebRTC FL disconnected');
  }
}

// Utility functions for security enhancements (TODO)
export function clipDelta(delta: HeadWeights, clipNorm = 1.0): HeadWeights {
  // Gradient clipping for privacy
  let norm = 0;
  
  for (const w of delta.W) {
    norm += w * w;
  }
  for (const b of delta.b) {
    norm += b * b;
  }
  norm = Math.sqrt(norm);
  
  if (norm > clipNorm) {
    const scale = clipNorm / norm;
    return {
      W: new Float32Array(delta.W.map(w => w * scale)),
      b: new Float32Array(delta.b.map(b => b * scale))
    };
  }
  
  return delta;
}

export function addNoise(delta: HeadWeights, noiseScale = 0.01): HeadWeights {
  // Differential privacy noise addition
  const noisyW = new Float32Array(delta.W.length);
  const noisyb = new Float32Array(delta.b.length);
  
  for (let i = 0; i < delta.W.length; i++) {
    // Add Gaussian noise
    noisyW[i] = delta.W[i] + (Math.random() - 0.5) * 2 * noiseScale;
  }
  
  for (let i = 0; i < delta.b.length; i++) {
    noisyb[i] = delta.b[i] + (Math.random() - 0.5) * 2 * noiseScale;
  }
  
  return { W: noisyW, b: noisyb };
}

export function getCurrentRound(): number {
  return currentRound;
}

export function getClientId(): string {
  return clientId;
}