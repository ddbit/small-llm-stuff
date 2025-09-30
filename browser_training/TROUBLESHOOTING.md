# Troubleshooting Guide

## Common Issues and Solutions

### 1. Data Type Mismatch Error
```
ERROR_MESSAGE: Unexpected input data type. Actual: (tensor(int32)) , expected: (tensor(int64))
```

**Solution**: Fixed automatically! The system now detects the expected data type from model metadata and creates the appropriate tensor type (int32 vs int64).

**What was changed**:
- Dynamic tensor type detection based on model metadata
- Automatic conversion between Int32Array and BigInt64Array
- Better error messaging for data type issues

### 2. Model Download Issues

**Symptoms**:
- "Failed to fetch" errors
- Network timeout
- CORS errors

**Solutions**:
1. **Check Internet Connection**: Ensure stable internet for large model downloads
2. **Try Different Model**: Switch to quantized model (~65MB) instead of full model (~250MB)
3. **Use Mock Model**: Falls back automatically, or manually select "Mock Model" from dropdown
4. **CORS Issues**: Some networks block Hugging Face downloads - use local model files instead

### 3. WebGPU Issues

**Symptoms**:
- "WebGPU not supported" warnings
- Falls back to WASM automatically

**Solutions**:
1. **Enable WebGPU**: In Chrome/Edge, go to `chrome://flags` and enable WebGPU
2. **Update Browser**: Ensure Chrome 113+ or Edge 113+
3. **Hardware Acceleration**: Enable in browser settings
4. **Use WASM**: Select "WebAssembly" backend as alternative

### 4. Memory Issues

**Symptoms**:
- Page crashes
- "Out of memory" errors
- Slow performance

**Solutions**:
1. **Use Quantized Model**: Much smaller memory footprint (~65MB vs ~250MB)
2. **Close Other Tabs**: Free up browser memory
3. **Clear Cache**: Use "Clear Model Cache" button
4. **Refresh Page**: Restart the application

### 5. Tokenization Issues

**Symptoms**:
- Text not being tokenized correctly
- Unexpected classification results

**Solutions**:
1. **Check Text Length**: Keep text under 256 tokens for best results
2. **Use English Text**: Model trained primarily on English
3. **Avoid Special Characters**: Stick to regular text for testing

### 6. Model Loading Stuck

**Symptoms**:
- Progress bar stops
- "Loading..." never completes

**Solutions**:
1. **Wait**: Large models can take 2-4 minutes to download
2. **Check Network**: Ensure stable connection
3. **Refresh Page**: Sometimes helps with stuck downloads
4. **Clear Cache**: Remove corrupted cached models
5. **Try Different Model**: Switch to smaller quantized version

### 7. Cache Issues

**Symptoms**:
- Models re-download every time
- Cache not working

**Solutions**:
1. **Check Browser Storage**: Ensure adequate space (>500MB recommended)
2. **Enable Cache**: Some browsers disable cache in private mode
3. **Manual Clear**: Use "Clear Model Cache" and try again

## Debug Console Output

The system provides detailed logging. Open browser developer tools (F12) and check the Console tab for:

- Model metadata (input/output types and shapes)
- Tensor creation details
- Download progress
- Error details with specific causes

## Performance Tips

1. **First Run**: Download quantized model, cache it
2. **Subsequent Runs**: Will load from cache instantly
3. **Text Length**: Shorter text = faster inference
4. **Backend Choice**: WebGPU > WebAssembly > Mock (in performance order)
5. **Model Choice**: Quantized > Full precision (for speed vs quality tradeoff)

## Known Limitations

1. **Model Size**: Hugging Face models are large (65-250MB)
2. **Internet Required**: For initial model download (cached afterwards)
3. **Browser Compatibility**: WebGPU requires modern browsers
4. **Text Length**: Limited to 256 tokens maximum
5. **Languages**: Best performance on English text

## Getting Help

1. **Check Console**: F12 → Console tab for detailed error messages
2. **Try Mock Model**: Always works as fallback
3. **Network Issues**: Try different internet connection
4. **Browser Issues**: Try different browser (Chrome/Edge recommended)

## Success Indicators

✅ **Everything Working**:
- Model downloads and caches successfully
- Real-time inference (<2 seconds)
- Accurate text classification
- Training and saving works
- No console errors

The system is designed with multiple fallbacks, so it should always work in some capacity!