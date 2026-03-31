# NGIML Model Architecture

This diagram describes the current `HybridNGIML` forward path implemented under `src/model`, with the branch selection and stage widths matching the effective config used by the training helpers and Colab notebook.

Included in the current effective model path:
- EfficientNet low-level backbone
- Swin-Tiny context backbone
- Residual/SRM backbone
- Residual-guided attention on the low-level branch
- 4-stage adaptive feature fusion
- U-Net decoder with per-stage heads
- Edge guidance
- Dropout on the highest-resolution logits
- Detail refinement
- Boundary refinement

Not drawn because they are disabled or runtime-only in the current path:
- Context residual attention
- Joint fusion gating
- Late residual boost
- Gradient checkpointing
- Flash attention and xformers toggles
- Final output resize to `target_size`
- Optional sigmoid postprocessing

```mermaid
flowchart TB
    I["Input RGB image x"]
    HP["On-the-fly residual_noise tensor hp<br/>computed upstream from x"]

    subgraph B["Backbones"]
        direction LR
        E["EfficientNet-B0<br/>features_only<br/>out_indices: 1,2,3,4"]
        S["Swin-Tiny<br/>features_only<br/>out_indices: 0,1,2,3"]
        SRM["Fixed 5x5 SRM filter bank<br/>3 kernels per input channel"]
        MIX["Average SRM(x) and SRM(hp)"]
        RS["Learnable residual_scale"]
        R["Residual CNN backbone<br/>4 stages<br/>channels: 32,64,128,256"]
    end

    I --> E
    I --> S
    I --> SRM
    HP --> MIX
    SRM --> MIX --> RS --> R

    E --> L["Low-level pyramid L1-L4"]
    S --> C["Context pyramid C1-C4"]
    R --> N["Residual pyramid R1-R4"]

    N --> RA["Residual-guided attention on low-level branch<br/>per stage: 1x1 residual projection -> 2*sigmoid-1<br/>zero-init scale keeps identity at start"]
    L --> RA
    RA --> A["Attended low-level pyramid A1-A4"]

    subgraph F["4-stage adaptive feature fusion (stage 1 = highest resolution)"]
        direction TB
        FNOTE["For each stage i:<br/>take matching A_i, C_i, R_i features<br/>1x1 project to stage width<br/>bilinear align spatial size<br/>generate per-branch sigmoid gates<br/>weighted average fusion<br/>3x3 refine<br/>extra 3x3 refine"]
        F1["F1 from stage 1 -> 64 ch"]
        F2["F2 from stage 2 -> 128 ch"]
        F3["F3 from stage 3 -> 192 ch"]
        F4["F4 from stage 4 -> 256 ch"]
    end

    A --> FNOTE
    C --> FNOTE
    N --> FNOTE
    FNOTE --> F1
    FNOTE --> F2
    FNOTE --> F3
    FNOTE --> F4

    subgraph D["U-Net decoder"]
        direction TB
        P1["Projected F1<br/>1x1 + norm + activation"]
        P2["Projected F2<br/>1x1 + norm + activation"]
        P3["Projected F3<br/>1x1 + norm + activation"]
        P4["Projected F4<br/>1x1 + norm + activation"]
        EG["Sobel(grayscale image) -> edge_proj<br/>added only to projected F1"]
        B4["Bottleneck on projected F4"]
        D3["Upsample + concat projected F3 -> decode block"]
        D2["Upsample + concat projected F2 -> decode block"]
        D1["Upsample + concat projected F1 -> decode block"]
        H4["Head 4 from bottleneck<br/>coarsest auxiliary logits"]
        H3["Head 3 from decoder stage 3<br/>auxiliary logits"]
        H2["Head 2 from decoder stage 2<br/>coarse logits used by refinement"]
        H1["Head 1 from decoder stage 1<br/>main logits"]
        DO["Dropout2d on current H1 logits"]
        DR["Detail refinement head<br/>concat final decoder feature + projected F1 + upsampled H2<br/>predict residual correction added to current H1"]
        BR["Boundary refinement head<br/>concat logits + Sobel magnitude(sigmoid(logits))<br/>predict residual correction added to logits"]
        PM["Primary prediction mask path<br/>final H1 logits -> sigmoid at inference"]
        AUX["Auxiliary heads for deep supervision<br/>H2, H3, H4"]
    end

    F1 --> P1
    F2 --> P2
    F3 --> P3
    F4 --> P4
    I --> EG --> P1

    P4 --> B4
    B4 --> H4
    B4 --> D3
    P3 --> D3
    D3 --> H3
    D3 --> D2
    P2 --> D2
    D2 --> H2
    D2 --> D1
    P1 --> D1
    D1 --> H1

    H1 --> DO
    D1 --> DR
    P1 --> DR
    H2 --> DR
    DO --> DR
    DR --> BR

    BR --> PM
    H2 --> AUX
    H3 --> AUX
    H4 --> AUX
```

Key implementation details:
- `decoder_channels=None`, so decoder widths follow the fusion widths: 64, 128, 192, 256.
- The residual branch participates in all four fusion stages because `noise_skip_stage=None` and `noise_decay=1.0`.
- Residual attention is applied only to the low-level branch in the current path because low-level residual attention is enabled and context residual attention is disabled.
- In the current training and inference path, `hp` is the on-the-fly `residual_noise` tensor computed before the model call by `_compute_residual_noise(image)`, which reflect-pads the RGB image, subtracts a 5x5 average blur, and standardizes each channel.
- The model still always computes fixed SRM responses from `x` internally. In the current path, the residual branch also computes SRM on `hp`, averages `SRM(x)` and `SRM(hp)`, then sends the result through the learnable residual CNN.
- Fusion uses feature-conditioned per-branch gates. With the current config, joint cross-branch gating is off, and late residual boost is off.
- Detail refinement does not concatenate the main logits into its refinement-head input. It uses the final decoder feature, the finest projected skip, and upsampled coarse logits, then adds the predicted residual back onto the current highest-resolution logits.
- Boundary refinement is applied only to the highest-resolution prediction path. Auxiliary heads `H2`, `H3`, and `H4` bypass the final refinement heads and exist for deep supervision.
- The model return value is still a list when `per_stage_heads=True`: `[final H1, H2, H3, H4]`. But inference code selects only index `0` as the mask logits, then applies `sigmoid`, so the practical inference output is a single predicted mask.
- `HybridNGIML.forward` can optionally resize every returned head to `target_size` after decoding. That resize step is intentionally omitted from the graph because it is post-decoder bookkeeping rather than a learned module.
