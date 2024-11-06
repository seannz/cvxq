<h1> LLM Compression with Convex Optimization—Part 1: Weight Quantization

[![arXiv](https://img.shields.io/badge/arXiv-2312.03102-b31b1b.svg)](https://arxiv.org/abs/2409.02026)
[![License:MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<img width="783" alt="Screenshot 2024-09-04 at 9 15 57 AM" src="https://github.com/user-attachments/assets/6a3385e2-9c43-425d-b9db-914c11c85648">
</h1>

This is the official repo for the paper "Foundations of LLM Compression—Part 1: Weight Quantization".

<br/>
<img width="783" alt="Screenshot 2024-11-05 at 9 46 02 PM" src="https://github.com/user-attachments/assets/d8e93e81-ba51-4cb8-a046-d49c55c606ac">
<img width="783" alt="Screenshot 2024-11-05 at 9 45 50 PM" src="https://github.com/user-attachments/assets/6c83b8f3-01d2-4aff-932b-e955a1f90f35">
<img width="783" alt="Screenshot 2024-11-05 at 9 45 31 PM" src="https://github.com/user-attachments/assets/9b6dd284-a04e-45ff-9d7f-9d01c01e7a2c">


<h2>Pre-requisites</h2>

All pre-requisite python packages are listed in `pytorch_2.2.1.yml`. Run `conda env create -f pytorch_2.2.1.yml`.</br>


<h2>Quantizing Models</h2>

Run `scripts/opt_all.sh` to quantize OPT models.</br>
Run `scripts/llama_all.sh` to quantize Llama-2 models.</br>
