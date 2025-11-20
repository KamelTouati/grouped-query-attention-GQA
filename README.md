# Grouped Query Attention (GQA) Visualization

This project demonstrates and visualizes Grouped Query Attention (GQA) using the Meta-Llama-3-8B model. GQA is an attention mechanism that groups multiple query heads to share the same key and value heads, improving efficiency while maintaining model performance.

## Overview

The notebook (`GQA_notebook.ipynb`) provides a hands-on implementation that:
- Loads the Meta-Llama-3-8B model from Hugging Face
- Extracts and visualizes key matrices from the attention mechanism
- Demonstrates how query heads are grouped to share key-value pairs
- Creates heatmap visualizations showing the key matrix structure

## Features

- **Model Loading**: Utilizes the Hugging Face Transformers library to load Meta-Llama-3-8B
- **Attention Visualization**: Captures and visualizes key projections from specific transformer layers
- **GQA Mapping**: Shows how 32 query heads are grouped to share 8 key-value heads
- **Heatmap Generation**: Creates visual representations of key matrices for better understanding

## Requirements

See `requirements.txt` for the complete list of dependencies. Key requirements include:
- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Matplotlib
- Seaborn
- NumPy

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Hugging Face Token**:
   - You need a Hugging Face account and access token
   - Request access to the Meta-Llama-3-8B model on Hugging Face
   - Set up your token (the notebook uses Google Colab's userdata, but you can modify this)

3. **GPU Recommended**:
   - The model uses float16 precision and requires significant GPU memory
   - A GPU with at least 16GB VRAM is recommended

## Usage

### Running in Google Colab

1. Upload the notebook to Google Colab
2. Set your Hugging Face token in Colab secrets as `HF_TOKEN`
3. Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)
4. Run all cells

### Running Locally

1. Modify the authentication section to use your HF token directly:
   ```python
   from huggingface_hub import login
   login(token="your_hf_token_here")
   ```

2. Run the notebook using Jupyter:
   ```bash
   jupyter notebook GQA_notebook.ipynb
   ```

## How It Works

### Grouped Query Attention

In standard Multi-Head Attention (MHA):
- Each query head has its own key and value heads
- For 32 query heads, you need 32 key heads and 32 value heads

In Grouped Query Attention (GQA):
- Multiple query heads share the same key and value heads
- Llama-3-8B uses 32 query heads but only 8 key-value heads
- Each group of 4 query heads shares 1 key-value head

### Visualization

The notebook creates a grid visualization where:
- Each row represents a KV group (8 groups total)
- Each column shows a query head within that group (4 heads per group)
- Heatmaps display the first 20 dimensions of the key matrix for the first token
- All heads in the same row share identical key values (demonstrating GQA)

## Output

The visualization generates:
- A comprehensive heatmap showing all 32 query heads organized by their KV groups
- PNG file: `llama3_layer_{layer_idx}_gqa_keys.png`
- Clear demonstration of which query heads share key-value pairs

## Model Architecture

**Meta-Llama-3-8B Configuration**:
- Total Parameters: ~8 billion
- Query Heads: 32
- Key-Value Heads: 8
- Head Dimension: 128
- Attention Type: Grouped Query Attention (GQA)

## Notes

- The model requires acceptance of Meta's license agreement on Hugging Face
- Processing requires significant computational resources
- The visualization focuses on a single layer and token for clarity
- You can modify `layer_idx` to visualize different transformer layers

## References

- [Meta Llama 3 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)

## License

This project is for educational purposes. The Meta-Llama-3-8B model has its own license terms that must be accepted on Hugging Face.
