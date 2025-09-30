export interface HeadWeights {
  W: Float32Array; // [inputSize, numClasses]
  b: Float32Array; // [numClasses]
}

export class ClassificationHead {
  private W: Float32Array;
  private b: Float32Array;
  private inputSize: number;
  private numClasses: number;
  private learningRate: number;
  
  // Adam optimizer parameters
  private mW: Float32Array;
  private vW: Float32Array;
  private mb: Float32Array;
  private vb: Float32Array;
  private beta1 = 0.9;
  private beta2 = 0.999;
  private epsilon = 1e-8;
  private t = 0; // time step for Adam
  
  constructor(inputSize: number, numClasses: number, learningRate = 1e-2) {
    this.inputSize = inputSize;
    this.numClasses = numClasses;
    this.learningRate = learningRate;
    
    // Initialize weights with Xavier/Glorot initialization
    const limit = Math.sqrt(6 / (inputSize + numClasses));
    this.W = new Float32Array(inputSize * numClasses);
    this.b = new Float32Array(numClasses);
    
    // Initialize weights
    for (let i = 0; i < this.W.length; i++) {
      this.W[i] = (Math.random() - 0.5) * 2 * limit;
    }
    
    // Initialize biases to zero
    this.b.fill(0);
    
    // Initialize Adam optimizer moments
    this.mW = new Float32Array(this.W.length);
    this.vW = new Float32Array(this.W.length);
    this.mb = new Float32Array(this.b.length);
    this.vb = new Float32Array(this.b.length);
  }
  
  forward(input: Float32Array): number[] {
    if (input.length !== this.inputSize) {
      throw new Error(`Expected input size ${this.inputSize}, got ${input.length}`);
    }
    
    const output = new Array(this.numClasses);
    
    // Linear transformation: output = input * W + b
    for (let j = 0; j < this.numClasses; j++) {
      let sum = this.b[j];
      for (let i = 0; i < this.inputSize; i++) {
        sum += input[i] * this.W[i * this.numClasses + j];
      }
      output[j] = sum;
    }
    
    return output;
  }
  
  trainStep(input: Float32Array, targetLabel: number, lr?: number): number {
    const currentLr = lr ?? this.learningRate;
    this.t++; // Increment time step for Adam
    
    // Forward pass
    const logits = this.forward(input);
    
    // Softmax
    const probs = this.softmax(logits);
    
    // Cross-entropy loss
    const loss = -Math.log(Math.max(probs[targetLabel], 1e-15));
    
    // Backward pass - compute gradients
    const dLogits = new Array(this.numClasses);
    for (let i = 0; i < this.numClasses; i++) {
      dLogits[i] = probs[i] - (i === targetLabel ? 1 : 0);
    }
    
    // Compute gradients for W and b
    const dW = new Float32Array(this.W.length);
    const db = new Float32Array(this.b.length);
    
    // dW = input^T * dLogits (outer product)
    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.numClasses; j++) {
        dW[i * this.numClasses + j] = input[i] * dLogits[j];
      }
    }
    
    // db = dLogits
    for (let j = 0; j < this.numClasses; j++) {
      db[j] = dLogits[j];
    }
    
    // Adam optimizer update
    const beta1Power = Math.pow(this.beta1, this.t);
    const beta2Power = Math.pow(this.beta2, this.t);
    
    // Update weights
    for (let i = 0; i < this.W.length; i++) {
      // Update biased first moment estimate
      this.mW[i] = this.beta1 * this.mW[i] + (1 - this.beta1) * dW[i];
      // Update biased second raw moment estimate
      this.vW[i] = this.beta2 * this.vW[i] + (1 - this.beta2) * dW[i] * dW[i];
      
      // Compute bias-corrected moments
      const mHat = this.mW[i] / (1 - beta1Power);
      const vHat = this.vW[i] / (1 - beta2Power);
      
      // Update weights
      this.W[i] -= currentLr * mHat / (Math.sqrt(vHat) + this.epsilon);
    }
    
    // Update biases
    for (let j = 0; j < this.b.length; j++) {
      // Update biased first moment estimate
      this.mb[j] = this.beta1 * this.mb[j] + (1 - this.beta1) * db[j];
      // Update biased second raw moment estimate
      this.vb[j] = this.beta2 * this.vb[j] + (1 - this.beta2) * db[j] * db[j];
      
      // Compute bias-corrected moments
      const mHat = this.mb[j] / (1 - beta1Power);
      const vHat = this.vb[j] / (1 - beta2Power);
      
      // Update biases
      this.b[j] -= currentLr * mHat / (Math.sqrt(vHat) + this.epsilon);
    }
    
    return loss;
  }
  
  private softmax(logits: number[]): number[] {
    // Subtract max for numerical stability
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(x => Math.exp(x - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map(x => x / sumExp);
  }
  
  getWeights(): HeadWeights {
    return {
      W: new Float32Array(this.W),
      b: new Float32Array(this.b)
    };
  }
  
  setWeights(weights: HeadWeights): void {
    if (weights.W.length !== this.W.length) {
      throw new Error(`Weight matrix size mismatch: expected ${this.W.length}, got ${weights.W.length}`);
    }
    if (weights.b.length !== this.b.length) {
      throw new Error(`Bias vector size mismatch: expected ${this.b.length}, got ${weights.b.length}`);
    }
    
    this.W.set(weights.W);
    this.b.set(weights.b);
    
    // Reset optimizer state when loading new weights
    this.mW.fill(0);
    this.vW.fill(0);
    this.mb.fill(0);
    this.vb.fill(0);
    this.t = 0;
  }
  
  resetOptimizer(): void {
    this.mW.fill(0);
    this.vW.fill(0);
    this.mb.fill(0);
    this.vb.fill(0);
    this.t = 0;
  }
  
  setLearningRate(lr: number): void {
    this.learningRate = lr;
  }
}