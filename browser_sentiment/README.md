# Browser Sentiment Analysis

A client-side sentiment analysis tool that runs entirely in the browser using WebAssembly and WebGPU acceleration via Transformers.js.

## Features

- **No Server Required**: Runs completely in the browser
- **DistilBERT Model**: Uses `distilbert-base-uncased-finetuned-sst-2-english` for accurate sentiment classification
- **WebAssembly/WebGPU**: Leverages browser optimization for fast inference
- **Real-time Analysis**: Instant sentiment scoring with confidence levels
- **Responsive Design**: Works on desktop and mobile devices

## Usage

1. Open `index.html` in a modern web browser
2. Wait for the model to load (first time may take a few moments)
3. Enter text in the textarea
4. Click "Analyze Sentiment" or press Ctrl+Enter
5. View sentiment results with confidence scores

## Model Details

- **Model**: DistilBERT (distilled BERT)
- **Task**: Binary sentiment classification (Positive/Negative)
- **Framework**: Transformers.js (ONNX.js backend)
- **Size**: ~250MB (downloaded once and cached)
- **Performance**: Runs at near real-time speeds in modern browsers

## Browser Compatibility

- Chrome/Edge 88+ (WebAssembly SIMD)
- Firefox 89+ (WebAssembly SIMD)
- Safari 14+ (WebAssembly)
- Mobile browsers with WebAssembly support

## Technical Implementation

The application uses:
- **Transformers.js**: Hugging Face transformers in the browser
- **ONNX.js**: Optimized model execution
- **WebAssembly**: High-performance computation
- **WebGPU**: GPU acceleration when available
- **Service Worker**: Model caching for offline use

## Examples

Try these sample texts:
- "I absolutely love this product!"
- "This is the worst experience ever."
- "The weather is okay today."
- "I'm feeling neutral about this decision."