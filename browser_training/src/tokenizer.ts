export interface TokenizerConfig {
  vocab: Record<string, number>;
  model: {
    type: string;
    vocab: Record<string, number>;
  };
  normalizer?: any;
  pre_tokenizer?: any;
  post_processor?: any;
  decoder?: any;
}

export class WordPieceTokenizer {
  private vocab: Record<string, number>;
  private idToToken: Record<number, string>;
  private unkToken = '[UNK]';
  private clsToken = '[CLS]';
  private sepToken = '[SEP]';
  private padToken = '[PAD]';
  
  constructor(config: TokenizerConfig) {
    this.vocab = config.model?.vocab || config.vocab || {};
    this.idToToken = {};
    for (const [token, id] of Object.entries(this.vocab)) {
      this.idToToken[id] = token;
    }
  }
  
  tokenize(text: string, maxLength = 256): number[] {
    if (!text || text.trim().length === 0) {
      return [this.getTokenId(this.clsToken), this.getTokenId(this.sepToken)];
    }
    
    // Simple preprocessing
    text = text.toLowerCase().trim();
    
    // Split into words
    const words = text.split(/\s+/);
    const tokens: number[] = [this.getTokenId(this.clsToken)];
    
    for (const word of words) {
      if (tokens.length >= maxLength - 1) break; // Reserve space for [SEP]
      
      const wordTokens = this.tokenizeWord(word);
      tokens.push(...wordTokens);
      
      if (tokens.length >= maxLength - 1) break;
    }
    
    // Add [SEP] token
    if (tokens.length < maxLength) {
      tokens.push(this.getTokenId(this.sepToken));
    }
    
    // Truncate if too long
    if (tokens.length > maxLength) {
      tokens.splice(maxLength - 1, tokens.length - maxLength, this.getTokenId(this.sepToken));
    }
    
    return tokens;
  }
  
  private tokenizeWord(word: string): number[] {
    if (this.vocab.hasOwnProperty(word)) {
      return [this.getTokenId(word)];
    }
    
    const tokens: number[] = [];
    let start = 0;
    
    while (start < word.length) {
      let end = word.length;
      let foundToken = false;
      
      // Try to find the longest matching subword
      while (start < end) {
        const substr = start === 0 ? word.substring(start, end) : '##' + word.substring(start, end);
        
        if (this.vocab.hasOwnProperty(substr)) {
          tokens.push(this.getTokenId(substr));
          start = end;
          foundToken = true;
          break;
        }
        end--;
      }
      
      if (!foundToken) {
        tokens.push(this.getTokenId(this.unkToken));
        start++;
      }
    }
    
    return tokens;
  }
  
  private getTokenId(token: string): number {
    return this.vocab[token] ?? this.vocab[this.unkToken] ?? 0;
  }
  
  decode(tokenIds: number[]): string {
    const tokens = tokenIds
      .map(id => this.idToToken[id] || this.unkToken)
      .filter(token => token !== this.clsToken && token !== this.sepToken && token !== this.padToken);
    
    return tokens
      .join(' ')
      .replace(/ ##/g, '')
      .trim();
  }
}

export async function loadTokenizer(url: string): Promise<WordPieceTokenizer> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load tokenizer: ${response.statusText}`);
    }
    
    const config = await response.json();
    return new WordPieceTokenizer(config);
  } catch (error) {
    console.error('Tokenizer loading error:', error);
    throw error;
  }
}