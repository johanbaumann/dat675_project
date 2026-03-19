# Model Information Script

This script provides a comprehensive breakdown of the CVAE model architecture and parameters.

## Usage

Run the script from the project root directory:

```bash
python model_info_script.py
```

## What It Does

1. **Loads Configuration**: Reads `fold_pipeline/fold_pipeline_config.example.json` to extract model configuration
2. **Initializes Model**: Creates a CVAE model instance with the configuration
3. **Prints Detailed Parameter Information**:
   - Per-layer parameter counts organized by component (Embedding, Encoder, Decoder, Output, Label Head)
   - Total and trainable parameter counts
   - Architecture configuration details
   - Transformer-specific settings (if applicable)
   - Label prediction head settings (if enabled)

## Output

The script produces structured output showing:

- **Model Configuration**: Display of all loaded hyperparameters
- **Layer-by-Layer Breakdown**: Parameters grouped by functional component
- **Overall Statistics**: Total, trainable, and non-trainable parameters
- **Architecture Config**: Mode, sizes, device, optimizer settings
- **Special Modules**: Transformer settings and label prediction head info

## Example Output

```
Total Parameters:         4,332,502
Trainable Parameters:     4,332,502
Non-trainable:                    0

Architecture Configuration:
  Model Mode:          transformer
  Vocabulary Size:     36
  Latent Size:         200
  Unit/Hidden Size:    256
  ...
```

## Alternative: Direct Method Call in Python

You can also call the parameter info method directly in Python:

```python
from model_labels import CVAE
import json

# Load your config
with open('your_config.json') as f:
    config = json.load(f)

# Create model
model = CVAE(vocab_size=36, args=config)

# Print detailed parameter information
model.print_parameters_info()
```

## Notes

- The script automatically handles missing values in the config (e.g., None values are replaced with defaults)
- If `predict_labels=True` but `label_dim` is not specified, it defaults to `num_prop`
- Both LSTM and Transformer modes are supported
