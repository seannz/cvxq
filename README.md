<h1> LLM Compression with Convex Optimization—Part 1: Weight Quantization

[![arXiv](https://img.shields.io/badge/arXiv-2312.03102-b31b1b.svg)](https://arxiv.org/abs/2409.02026)
[![License:MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<img width="783" alt="Screenshot 2024-09-04 at 9 15 57 AM" src="https://github.com/user-attachments/assets/6a3385e2-9c43-425d-b9db-914c11c85648">
</h1>

This is the official repo for the paper "Foundations of LLM Compression—Part 1: Weight Quantization".

<br/>
<img width="781" alt="Screenshot 2024-10-31 at 8 12 14 PM" src="https://github.com/user-attachments/assets/53ca4ed4-b4b1-4b6e-b06c-3f1b84990b89">
<img width="781" alt="Screenshot 2024-10-31 at 8 12 29 PM" src="https://github.com/user-attachments/assets/902dcadd-5fdc-4573-99e8-56bc5f48c22f">
<img width="781" alt="Screenshot 2024-10-31 at 8 13 05 PM" src="https://github.com/user-attachments/assets/c3346cde-2e9e-40fc-82cf-c158461e0e1b">
<img width="781" alt="Screenshot 2024-10-31 at 8 13 21 PM" src="https://github.com/user-attachments/assets/73e42292-bcc2-451c-ba10-87d8638e9d9d">


<h2>Pre-requisites</h2>

All pre-requisite python packages are listed in `pytorch_2.2.1.yml`. Run `conda env create -f pytorch_2.2.1.yml`.</br>


<h2>Quantizing Models</h2>

Run `scripts/opt_all.sh` to quantize OPT models.</br>
Run `scripts/llama_all.sh` to quantize Llama-2 models.</br>
