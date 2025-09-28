# DistilBERT Multi-Class Text Classifier

A client-side text classification tool that runs entirely in the browser using WebAssembly and WebGPU acceleration via Transformers.js. Classifies text into 6 categories: Sport, Technology, Business, Politics, Music, and Cinema.

## Features

- **Zero-Shot Classification**: Uses DistilBERT-MNLI for zero-shot text classification
- **No Server Required**: Runs completely in the browser
- **6 Categories**: Sport, Technology, Business, Politics, Music, Cinema
- **WebAssembly/WebGPU**: Leverages browser optimization for fast inference
- **Real-time Analysis**: Instant classification with confidence scores for all categories
- **Interactive Examples**: Pre-built example texts to test each category
- **Responsive Design**: Works on desktop and mobile devices

## Usage

1. Open `index.html` in a modern web browser
2. Wait for the model to load (first time may take a few moments)
3. Enter text in the textarea or click example buttons
4. Click "Classify Text" or press Ctrl+Enter
5. View classification results with confidence scores for each category

## Model Details

- **Model**: DistilBERT Base Uncased (MNLI fine-tuned)
- **Task**: Zero-shot multi-class text classification
- **Categories**: 6 predefined classes
- **Framework**: Transformers.js (ONNX.js backend)
- **Size**: ~250MB (downloaded once and cached)
- **Performance**: Real-time classification in modern browsers

## Classification Categories

1. **Sport** ⚽ - Sports, athletics, games, competitions
2. **Technology** 💻 - Tech news, gadgets, software, innovation
3. **Business** 💼 - Finance, companies, markets, economics
4. **Politics** 🏛️ - Government, elections, policies, political events
5. **Music** 🎵 - Artists, albums, concerts, musical content
6. **Cinema** 🎬 - Movies, actors, film industry, entertainment

## Browser Compatibility

- Chrome/Edge 88+ (WebAssembly SIMD + WebGPU)
- Firefox 89+ (WebAssembly SIMD)
- Safari 14+ (WebAssembly)
- Mobile browsers with WebAssembly support

## Technical Implementation

The application uses:
- **Transformers.js**: Hugging Face transformers in the browser
- **DistilBERT-MNLI**: Zero-shot classification model
- **ONNX.js**: Optimized model execution
- **WebAssembly**: High-performance computation
- **WebGPU**: GPU acceleration when available
- **Zero-shot Learning**: No training required for new categories

## Performance

- **Model Loading**: ~10-30 seconds (first time only)
- **Inference Time**: ~100-500ms per classification
- **Memory Usage**: ~500MB during inference
- **Accuracy**: High accuracy for general domain classification

## Example Classifications

Try these sample texts:

**Technology:**
- "The latest iPhone features an amazing camera system with computational photography"
- "Artificial intelligence is transforming software development"

**Business:**
- "Apple stock prices surged after quarterly earnings exceeded expectations"
- "The startup raised $50M in Series A funding"

**Sport:**
- "The World Cup final was incredible with amazing goals"
- "The basketball team won the championship game"

**Cinema:**
- "The new Marvel movie features stunning visual effects"
- "The director won an Oscar for best picture"

**Politics:**
- "The presidential election results sparked debates"
- "Congress passed the new healthcare bill"

**Music:**
- "Beyoncé's new album showcases incredible vocal range"
- "The concert featured amazing guitar solos"

## How Zero-Shot Classification Works

The model uses Natural Language Inference (NLI) to determine if a text entails each category hypothesis:
1. Input: "Apple released a new iPhone"
2. Hypotheses: "This text is about technology", "This text is about business", etc.
3. The model scores how well each hypothesis fits the input
4. Results are normalized to show confidence percentages

## Limitations

- Performance depends on browser and device capabilities
- Categories are predefined and cannot be changed without code modification
- Works best with English text
- Requires internet connection for initial model download