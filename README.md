# ARM Neural Network Optimization

This repository contains sample code and scripts for optimizing neural networks for Arm Cortex-A and Cortex-M processors. The optimizations include 8-bit quantization techniques and custom NEON-optimized kernels for convolution operations.

## Repository Structure
- `quantization/`: Scripts for quantizing neural network models.
- `neon_kernels/`: NEON-optimized assembly kernels and Python integration.

## Usage

### Quantization
1. Train your model using `example_model.py`.
2. Quantize the trained model using `quantize_model.py`.

### NEON-Optimized Kernels
1. Use the `conv_neon.s` assembly file for optimized convolution operations.
2. Integrate the NEON kernels with your neural network using `neon_optimized_convolution.py`.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
