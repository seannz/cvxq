<h1> LLM Compression with Convex Optimization—Part 1: Weight Quantization

[![arXiv](https://img.shields.io/badge/arXiv-2312.03102-b31b1b.svg)](https://arxiv.org/abs/)
[![License:MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<img width="783" alt="Screenshot 2024-09-04 at 9 15 57 AM" src="https://github.com/user-attachments/assets/6a3385e2-9c43-425d-b9db-914c11c85648">
</h1>

This is the official repo for the paper "LLM Compression with Convex Optimization—Part 1: Weight Quantization".

<br/>
<img width="783" alt="Screenshot 2024-09-04 at 9 09 48 AM" src="https://github.com/user-attachments/assets/ef9f6f0c-f32d-4f13-a951-36a7a043d974">

<h2>Pre-requisites</h2>

All pre-requisite python packages are listed in `pytorch_2.2.1.yml`. Run `conda env create -f pytorch_2.2.1.yml`.</br>


<h2>Quantizing Models</h2>

Run `scripts/opt_all.sh` to quantize OPT models.</br>
Run `scripts/llama_all.sh` to quantize Llama-2 models.</br>
