## De-mark: Watermark Removal in Large Language Models (ICML 2025)
Ruibo Chen*, Yihan Wu*, Junfeng Guo, Heng Huang

## Introduction \[[Paper](https://arxiv.org/pdf/2410.13808)\]

<p align="center">
<img src=images/demark_intro.png  width="95%" height="95%">
</p>

<p align="center">
<img src=images/demark_methods.png  width="95%" height="95%">
</p>


We propose **De-mark**, a framework designed for the removal of n-gram-based watermarks. Our framework introduces a provable unbiased estimator to assess the strength of watermarks and offers theoretical guarantees about the gap between the original and the post-removal distributions of the language model. Essentially, **De-mark** operates without requiring prior knowledge of the n-gram watermark parameters.

Our proposed method can work for both watermark removal and watermark exploitation.

<p align="center">
<img src=images/demark_results.png  width="95%" height="95%">
</p>


## Quick Start

Prepare the environment:

```
conda create -n demark python=3.11
conda activate demark
pip install -r requirements.txt
```

## Run Watermark Removal

To remove the KGW watemark: 

```bash scripts/watermark_removal/run_exp_watermark_removal_KGW.sh```

To remove Dip-mark: 

```bash scripts/watermark_removal/run_exp_watermark_removal_Dipmark.sh```

## Run Watermark Exploitation

Run watermark exploitations on the KGW watermark:

```bash scripts/watermark_exploitation/run_exp_watermark_exploitation.sh```

## Evaluation

Evaluate TPR@FPR, median p-value and the GPT score:

```bash scripts/evaluations/evaluate.sh```

## Citation

If you find our work useful for your research and applications, please consider citing:

```
@article{chen2024mark,
  title={De-mark: Watermark Removal in Large Language Models},
  author={Chen, Ruibo and Wu, Yihan and Guo, Junfeng and Huang, Heng},
  journal={arXiv preprint arXiv:2410.13808},
  year={2024}
}
```
