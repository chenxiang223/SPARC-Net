# SPARC-Net Training Code Rename Guide

This document is for the Codex agent on the training machine. The training code still uses older internal names, while the current paper uses the SPARC-Net naming system. Please rename code symbols, comments, logs, config labels, and ablation labels according to the mapping below. Keep backward-compatible aliases when possible so old checkpoints and old experiment scripts can still load.

## 1. Core Class Name Mapping

| Old name in training code | New name to use | Meaning in paper |
|---|---|---|
| `CVOCAFeINFNFusion` | `SPARCNet` | Full proposed network |
| `CVOCAFeatureExtractor` | `AnalyticComplexSpectralEncoder` | ACSE, analytic complex spectral encoder |
| `ComplexChannelAttention` | `AmplitudePhaseChannelRecalibration` | APCR, amplitude-phase channel recalibration |
| `CVOCAFeINFNAdapter` | `AmplitudePhaseTokenAdapter` | APTA, amplitude-phase token adapter |
| `FrequencyAmplitudePhaseBranch` | `DualAxisAmplitudePhaseFrequencyBranch` | DAF Branch, dual-axis amplitude-phase frequency branch |
| `SFIDInteraction` | `TokenGuidedSpatialFrequencyInteraction` | TSFI, token-guided spatial-frequency interaction |
| `FeINFNCore` | `SpatialFrequencyFusionCore` | Spatial-frequency fusion core |
| `RepresentationPreservingFusionGate` | `SpectralFidelityFusionGate` | SFF Gate, spectral-fidelity fusion gate |
| `SpectralResidualBypassGate` | `SpectralAnchorBypass` | SAB, spectral anchor bypass |

## 2. Recommended Attribute / Variable Renames

These are not always required for running the code, but they make the implementation match the paper.

| Old attribute / variable | New attribute / variable |
|---|---|
| `cvoca` | `acse` |
| `adapter` | `apta` |
| `feinfn` | `fusion_core` or `sffc` |
| `sfid` | `tsfi` |
| `fusion_stabilizer` | `sff_gate` |
| `spectral_bypass` | `spectral_anchor_bypass` or `sab` |
| `use_sfid` | `use_tsfi` |
| `use_complex_attention` | `use_complex_attention` can stay, but comments should say APCR |
| `frequency_branch` | `daf_branch` where appropriate |

Keep `use_sfid` as an optional backward-compatible argument if old scripts use it, but internally map it to `use_tsfi`.

## 3. Ablation Name Mapping

Use these labels in logs, configs, result JSON files, and paper tables.

| Old ablation label | New label |
|---|---|
| `w/o CVOCAFeINFN` | `w/o SPAP Backbone` |
| `w/o FeINFN` | `w/o Spatial-Frequency Fusion Core` |
| `w/o FrequencyAmplitudePhaseBranch` | `w/o DAF Branch` |
| `w/o SFID` | `w/o TSFI` |
| `w/o RepresentationPreservingFusionGate` | `w/o SFF Gate` |
| `w/o SpectralResidualBypassGate` | `w/o SAB` |
| `w/o ComplexChannelAttention` | `w/o APCR` |

For frequency ablation, use:

- `w/o DAF Branch`: remove the whole `DualAxisAmplitudePhaseFrequencyBranch`.
- `w/o Spa-Freq`: remove only the spatial-frequency FFT path.
- `w/o Spe-Freq`: remove only the spectral-frequency FFT path.
- `w/o Amp`: remove amplitude update in the DAF Branch.
- `w/o Phase`: remove phase update in the DAF Branch.

If the training code currently has only `use_frequency_branch`, then `use_frequency_branch=False` corresponds to `w/o DAF Branch`.

## 4. Module Meaning To Preserve

Do not change the actual behavior unless explicitly requested. This task is mainly renaming.

- `AnalyticComplexSpectralEncoder` builds real and imaginary complex spectral features.
- `AmplitudePhaseChannelRecalibration` recalibrates amplitude and phase channels.
- `AmplitudePhaseTokenAdapter` produces the base feature map, spectral token, and spatial token.
- `SpatialHighFrequencyBranch` enhances spatial local details and boundaries.
- `DualAxisAmplitudePhaseFrequencyBranch` contains spatial-frequency FFT and spectral-frequency FFT paths, both with amplitude and phase updates.
- `TokenGuidedSpatialFrequencyInteraction` fuses spatial and frequency features under token guidance.
- `SpectralFidelityFusionGate` stabilizes enhanced features using base features and raw spectral hints.
- `SpectralAnchorBypass` preserves raw spectral signature through gated bypass.
- `SPARCNet` is the full model.

## 5. Compatibility Aliases

After renaming classes, add aliases at the bottom of the model file so old scripts/checkpoints still work:

```python
ComplexChannelAttention = AmplitudePhaseChannelRecalibration
CVOCAFeatureExtractor = AnalyticComplexSpectralEncoder
CVOCAFeINFNAdapter = AmplitudePhaseTokenAdapter
FrequencyAmplitudePhaseBranch = DualAxisAmplitudePhaseFrequencyBranch
SFIDInteraction = TokenGuidedSpatialFrequencyInteraction
FeINFNCore = SpatialFrequencyFusionCore
RepresentationPreservingFusionGate = SpectralFidelityFusionGate
SpectralResidualBypassGate = SpectralAnchorBypass
CVOCAFeINFNFusion = SPARCNet
```

## 6. Files To Check On Training Machine

Search and update these likely files:

```bash
rg -n "CVOCA|FeINFN|SFID|ComplexChannelAttention|FrequencyAmplitudePhase|RepresentationPreserving|SpectralResidual|use_sfid"
```

Likely targets:

- `models/cvoca_feinfn_fusion.py`
- `models/__init__.py`
- `main.py`
- `main2.py`
- `run_custom_experiment.py`
- `run_ablation_benchmark.py`
- config files
- log/report generation code

## 7. Prompt For Codex On Training Computer

Please rename the old module/class/config names in my training code to match the current SPARC-Net paper naming. Follow `RENAME_GUIDE_FOR_TRAINING_CODEX.md`. Keep backward-compatible aliases for old names so old checkpoints and scripts still work. Do not change model behavior except for name compatibility. After editing, run a quick import/smoke test and report which files changed.

