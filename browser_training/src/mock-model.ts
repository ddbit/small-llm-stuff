// Mock implementation for testing without real ONNX model

export class MockDistilBERT {
  private initialized = false;
  
  async initialize(): Promise<void> {
    // Simulate model loading time
    await new Promise(resolve => setTimeout(resolve, 500));
    this.initialized = true;
    console.log('Mock DistilBERT initialized');
  }
  
  async embedCLS(text: string): Promise<Float32Array> {
    if (!this.initialized) {
      throw new Error('Model not initialized');
    }
    
    // Generate deterministic but realistic embeddings based on text
    const embedding = new Float32Array(768);
    
    // Simple hash-based embedding generation
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      hash = ((hash << 5) - hash + text.charCodeAt(i)) & 0xffffffff;
    }
    
    // Fill embedding with pseudo-random values based on text content
    for (let i = 0; i < 768; i++) {
      const seed = hash + i;
      const x = Math.sin(seed) * 10000;
      embedding[i] = (x - Math.floor(x)) * 2 - 1; // Range [-1, 1]
    }
    
    // Add some structure based on keywords
    const devKeywords = ['code', 'programming', 'bug', 'fix', 'develop', 'software', 'function', 'api', 'debug'];
    const meetingKeywords = ['meeting', 'call', 'discussion', 'agenda', 'schedule', 'appointment', 'conference'];
    const emailKeywords = ['email', 'message', 'send', 'reply', 'inbox', 'subject', 'urgent', 'forward'];
    
    const lowerText = text.toLowerCase();
    
    // Bias embeddings based on content
    let devScore = 0, meetingScore = 0, emailScore = 0;
    
    devKeywords.forEach(keyword => {
      if (lowerText.includes(keyword)) devScore += 1;
    });
    
    meetingKeywords.forEach(keyword => {
      if (lowerText.includes(keyword)) meetingScore += 1;
    });
    
    emailKeywords.forEach(keyword => {
      if (lowerText.includes(keyword)) emailScore += 1;
    });
    
    // Apply bias to specific dimensions
    if (devScore > 0) {
      for (let i = 0; i < 50; i++) {
        embedding[i] += devScore * 0.3;
      }
    }
    
    if (meetingScore > 0) {
      for (let i = 50; i < 100; i++) {
        embedding[i] += meetingScore * 0.3;
      }
    }
    
    if (emailScore > 0) {
      for (let i = 100; i < 150; i++) {
        embedding[i] += emailScore * 0.3;
      }
    }
    
    // Normalize to reasonable range
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (norm > 0) {
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] = embedding[i] / norm * 10; // Scale to reasonable magnitude
      }
    }
    
    return embedding;
  }
}