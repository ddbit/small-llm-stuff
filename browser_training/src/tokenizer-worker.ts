import { WordPieceTokenizer } from './tokenizer.js';

self.onmessage = async (e) => {
  try {
    const { text, tokenizer: tokenizerConfig } = e.data;
    
    if (!tokenizerConfig) {
      throw new Error('Tokenizer configuration not provided');
    }
    
    const tokenizer = new WordPieceTokenizer(tokenizerConfig);
    const tokens = tokenizer.tokenize(text, 256);
    
    self.postMessage({ tokens });
    
  } catch (error) {
    console.error('Tokenizer worker error:', error);
    self.postMessage({ 
      error: error instanceof Error ? error.message : 'Unknown tokenization error' 
    });
  }
};