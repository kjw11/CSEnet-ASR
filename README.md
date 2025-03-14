# Cross-Speaker Encoding Network for Multi-Talker Speech Recognition
><em> In International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2024 
<br> Authors: Jiawen Kang, Lingwei Meng, Mingyu Cui, Haohan Guo, Xixin Wu, Xunying Liu, Helen Meng</em>

[Paper Poster](assets/poster.pdf) | [PDF Paper](https://arxiv.org/pdf/2401.04152v1.pdf) | [HTML Paper](https://arxiv.org/html/2401.04152v1) | [Citation](#citation)

This repository contains the core implementation of the <em>Cross-Speakre Encoding (CSE)</em> and <em>CSE-SOT </em> network.

**<font color=red>News</font>**: [Paper Poster](assets/poster.pdf) uploaded!

## Architecture at a glance
<img src="assets/img.png" width=100%>

- (a) Standard single-input-multi-output (SIMO) architecture encodes speakers in separate branches.
- (b) The proposed cross-speaker encoding (CSE) architecture allows mutual attention across speakers.
- (c) CSE network can further integrate with serialized output training (SOT), which takes advantage of both approaches.
- (d) Details of the cross encoder module
 

## Requirments
- ESPnet and its required dependencies
- Additional packages used for scoring can be found in `./scoring/requirements.txt`


## Usage
To use this code, please:
1. Replace the original ESPnet code with code under the `./espnet2-patch` directory
2. Run ESPnet ASR recipe (refer to [Librispeech recipe](https://github.com/espnet/espnet/tree/master/egs2/librispeech/asr1)) using configurations under the `./config` directory
3. After ESPnet scoring, additionally perform permutation-invariant scoring with `./scoring/run_pi_scoring.sh`

`./run.sh` provides a running demo for more useful details. Please note that this code was developed under <b>ESPnet 202209 version</b> and could be incompatible with later versions. 


## Related works
-  SA-CTC: A **speaker-aware CTC** for multi-talker overlapped speech recognition [![arXiv](https://img.shields.io/badge/arXiv-2409.12388-b31b1b.svg?style=flat)](https://arxiv.org/abs/2409.12388)
 [![Static Badge](https://img.shields.io/badge/Github-SACTC-blue)](https://github.com/kjw11/Speaker-Aware-CTC)
- Sidecar Separator: Convert **single-talker** speech recognition systems to **multi-talker** [![arXiv](https://img.shields.io/badge/arXiv-2302.09908-b31b1b.svg?style=flat)](https://arxiv.org/abs/2302.09908)
- Unified modeling of **multi-talker** speech recognition and **diarization** [![arXiv](https://img.shields.io/badge/arXiv-2305.16263-b31b1b.svg?style=flat)](https://arxiv.org/abs/2305.16263)
- Empowering **Whisper** for multi-talker and target-talker speech recognition [![arXiv](https://img.shields.io/badge/arXiv-2407.09817-b31b1b.svg?style=flat)](https://arxiv.org/abs/2407.09817)
 [![Static Badge](https://img.shields.io/badge/Github-Whisper--Sidecar-blue)](https://github.com/LingweiMeng/Whisper-Sidecar)
- **Large language model** for multi-talker speech recognition [![arXiv](https://img.shields.io/badge/arXiv-2409.08596-b31b1b.svg?style=flat)](https://arxiv.org/abs/2409.08596)


## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@article{kang2024cross,
  title={Cross-Speaker Encoding Network for Multi-Talker Speech Recognition},
  author={Kang, Jiawen and Meng, Lingwei and Cui, Mingyu and Guo, Haohan and Wu, Xixin and Liu, Xunying and Meng, Helen},
  journal={arXiv preprint arXiv:2401.04152},
  year={2024}
}
```

## Contact
Feel free to contact [me](https://kjw11.github.io/) if you have any question.

## Acknowledgements
This repository is based on [ESPnet speech processing toolkit](https://github.com/espnet/espnet), version 202209.
