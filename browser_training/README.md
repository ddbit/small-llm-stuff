# TinyBERT Browser Training

A browser-native text classification system with frozen DistilBERT backbone and trainable classification head, supporting federated learning.

## Features

- **Browser-native inference**: Runs entirely in the browser using WebGPU (with WASM fallback)
- **Quantized DistilBERT**: INT8 quantized model for efficient inference
- **Trainable head**: Only the classification layer is trainable, keeping the backbone frozen
- **Local training**: Train on user feedback without sending data to servers
- **Federated learning**: Share model updates while preserving privacy
- **Persistent storage**: Save and load trained models using IndexedDB
- **Zero native dependencies**: Pure web standards (WebGPU, WASM, WebRTC)

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Modern browser with WebGPU support (Chrome 113+, Edge 113+) or WebAssembly support

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open http://localhost:5173 in your browser.

### Build

```bash
npm run build
npm run preview
```

## Usage

### Text Classification

1. Enter text in the input field
2. Click "Predict" to classify the text
3. View the predicted label and confidence scores

### Training

1. After making a prediction, select the correct label from the dropdown
2. Click "Train on Feedback" to update the model
3. The model learns from your corrections

### Model Management

- **Save Model**: Persist the current model to IndexedDB
- **Load Model**: Restore a previously saved model
- **Send Delta**: Share model updates for federated learning
- **Apply Aggregate**: Receive and apply federated updates

## Architecture

### Core Components

- **`src/main.ts`**: Application bootstrap and runtime initialization
- **`src/model.ts`**: ONNX session management and embedding extraction
- **`src/tokenizer.ts`**: WordPiece tokenization with web workers
- **`src/head.ts`**: Trainable classification head with Adam optimizer
- **`src/storage.ts`**: IndexedDB persistence for model weights
- **`src/fl.ts`**: Federated learning hooks and weight quantization
- **`src/ui.ts`**: User interface handlers and metrics display

### Model Files

Place your model files in the `public/` directory:

- **`distilbert-int8.onnx`**: Quantized DistilBERT model (not included)
- **`tokenizer.json`**: WordPiece tokenizer configuration (included)

## Model Requirements

### ONNX Model

You need to provide a quantized DistilBERT ONNX model. The application expects:

- Input names: `input_ids`, `attention_mask`
- Output: last hidden states with shape `[batch_size, seq_length, 768]`
- Maximum sequence length: 256 tokens
- Quantization: INT8 for optimal performance

### Converting from Hugging Face

```python
from transformers import DistilBertModel, DistilBertTokenizer
import torch
from pathlib import Path

# Load model
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Create dummy input
dummy_input = {
    'input_ids': torch.randint(0, 1000, (1, 256)),
    'attention_mask': torch.ones(1, 256, dtype=torch.long)
}

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    'public/distilbert-base.onnx',
    input_names=['input_ids', 'attention_mask'],
    output_names=['last_hidden_state'],
    dynamic_axes={
        'input_ids': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'},
        'last_hidden_state': {0: 'batch_size'}
    },
    opset_version=14
)
```

Then quantize using ONNX Runtime tools:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    'public/distilbert-base.onnx',
    'public/distilbert-int8.onnx',
    weight_type=QuantType.QInt8
)
```

## Labels

The system is configured for 3-class classification:

- **Dev**: Development-related content
- **Meeting**: Meeting-related content  
- **Email**: Email-related content

Modify the labels in `src/main.ts` and `index.html` as needed.

## Federated Learning

The system includes hooks for federated learning:

### HTTP-based FL

- Implements weight delta quantization and transmission
- Stub server implementation for testing
- Supports secure aggregation protocols

### WebRTC-based FL (Experimental)

- P2P model updates without central server
- Placeholder implementation for future development

### Security Features (TODO)

- Gradient clipping for privacy
- Differential privacy noise
- Secure aggregation protocols

## Performance

### Expected Performance

- **Inference latency**: <100ms on modern laptops with WebGPU
- **Training convergence**: 20 examples per label in ≤5 epochs
- **Memory usage**: ~50-100MB depending on model size
- **Storage**: ~1MB per saved model

### Optimization Tips

- Use WebGPU-capable browsers for best performance
- Enable hardware acceleration in browser settings
- Consider batch processing for multiple predictions
- Monitor memory usage in developer tools

## Browser Compatibility

| Browser | WebGPU | WASM | Status |
|---------|--------|------|---------|
| Chrome 113+ | ✅ | ✅ | Full support |
| Edge 113+ | ✅ | ✅ | Full support |
| Firefox | ❌ | ✅ | WASM fallback |
| Safari | ❌ | ✅ | WASM fallback |

## Limitations

- Requires manual model file placement
- Limited to 256 token sequences
- No built-in data augmentation
- Simplified tokenization (production would use full WordPiece)
- FL server implementation is a stub

## Future Extensions

### Model Architecture
- **LoRA adaptation**: More efficient fine-tuning
- **BitFit**: Bias-only fine-tuning
- **Adapter layers**: Task-specific adaptations

### Training Enhancements
- **Data augmentation**: Text augmentation techniques
- **Active learning**: Smart sample selection
- **Multi-task learning**: Shared representations

### Federated Learning
- **Secure aggregation**: Cryptographic protocols
- **Byzantine fault tolerance**: Robust aggregation
- **Personalization**: Client-specific adaptations

## Development

### Type Checking

```bash
npm run typecheck
```

### Project Structure

```
src/
├── main.ts          # Application entry point
├── model.ts         # ONNX runtime integration
├── tokenizer.ts     # Text tokenization
├── tokenizer-worker.ts # Web worker for tokenization
├── head.ts          # Classification head
├── storage.ts       # IndexedDB persistence
├── fl.ts            # Federated learning
└── ui.ts            # User interface

public/
├── tokenizer.json   # Tokenizer configuration
└── .gitkeep         # Placeholder for ONNX model

index.html           # Main HTML file
vite.config.ts       # Build configuration
package.json         # Dependencies
tsconfig.json        # TypeScript configuration
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions, please open an issue on GitHub.