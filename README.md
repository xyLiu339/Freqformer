# Freqformer: Image-Demoiréing Transformer via Efficient Frequency Decomposition

[Xiaoyang Liu](https://xyliu339.github.io/), [Bolin Qiu](https://github.com/BolinQiu), [Jiezhang Cao](https://www.jiezhangcao.com/), [Zheng Chen](https://zhengchen1999.github.io/), [Yulun Zhang](http://yulunzhang.com/) and [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ), "Freqformer: Image-Demoiréing Transformer via Efficient Frequency Decomposition", arxiv, 2025.

[![releases](https://img.shields.io/github/downloads/xyLiu339/Freqformer/total)](https://github.com/xyLiu339/Freqformer/releases) 
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=xyLiu339.Freqformer)](https://github.com/xyLiu339/Freqformer) 
[![GitHub Stars](https://img.shields.io/github/stars/xyLiu339/Freqformer)](https://github.com/xyLiu339/Freqformer) 

[[arXiv]()] [[supplementary material]()]

---

> **Abstract:** Image demoiréing remains a challenging task due to the complex interplay between texture corruption and color distortions caused by moiré patterns. Existing methods, especially those relying on direct image-to-image restoration, often fail to disentangle these intertwined artifacts effectively. While wavelet-based frequency-aware approaches offer a promising direction, their potential remains underexplored. In this paper, we present Freqformer, a Transformer-based framework specifically designed for image demoiréing through targeted frequency separation. Our method performs an effective frequency decomposition that explicitly splits moiré patterns into high-frequency spatially-localized textures and low-frequency scale-robust color distortions, which are then handled by a dual-branch architecture tailored to their distinct characteristics. We further propose a learnable Frequency Composition Transform (FCT) module to adaptively fuse the frequency-specific outputs, enabling consistent and high-fidelity reconstruction. To better aggregate the spatial dependencies and the inter-channel complementary information, we introduce a Spatial-Aware Channel Attention (SA-CA) module that refines moiré-sensitive regions without incurring high computational cost. Extensive experiments on various demoiréing benchmarks demonstrate that Freqformer achieves state-of-the-art performance with a compact model size.

---

### 🔖 TODO
* [ ] Release code and pretrained models.

## 🔗 Contents

- [ ] Models
- [ ] Training
- [ ] Testing
- [ ] [Results](#results)
- [ ] [Citation](#citation)
- [ ] [Acknowledgements](#acknowledgements)



## <a name="results"></a>🔎 Results
<details close>
<summary>Quantitative Results (click to expand) </summary>


</details>


<details open>
<summary>Visual Results on FHDMi and UHDM </summary>

</details>



### <a name="citation"></a>📎 Citation
```

```

### <a name="acknowledgements"></a>💡 Acknowledgements
[TBD]
