# Reproduction Source Notes

The runnable code in this folder uses a unified PyTorch benchmark harness so every method shares the same dataset splits, optimizer, metrics, and output format. The model modules follow the corresponding paper/open-source architecture families below.

| Method | Paper/open-source reference used for reproduction |
|---|---|
| 3D-CNN | DeepHyperX/eecn PyTorch HSI classification implementations of 3-D CNN baselines: https://github.com/eecn/Hyperspectral-Classification |
| FDSSC | Official project page and repository: https://github.com/shuguang-52/FDSSC |
| HybridSN | Official repository: https://github.com/gokriznastic/HybridSN |
| SpectralFormer | Official repository: https://github.com/danfenghong/IEEE_TGRS_SpectralFormer |
| SSDGL | Official repository listed by the paper: https://github.com/dengweihuan/SSDGL |
| SSFTT | Official PyTorch demo repository: https://github.com/zgr6010/HSI_SSFTT |
| SSRN | Official repository: https://github.com/zilongzhong/SSRN |
| VIT | Official Google Research Vision Transformer repository: https://github.com/google-research/vision_transformer |

Run all selected methods through:

```powershell
E:\anaconda\envs\LDX\python.exe C:\Users\PC\Desktop\cvoca-feinfn\other-Method\run_comparison.py --methods HybridSN --datasets IndianPines --repeats 1
```
