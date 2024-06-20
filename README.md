## Effective Targeted Attacks for Adversarial Self-Supervised Learning
This is the official PyTorch implementation for the paper Effective Targeted Attacks for Adversarial Self-Supervised Learning, NeurIPS 2023: [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b28ae1166e1035c26b89d20f0286c9eb-Abstract-Conference.html)

## Abstract
Recently, unsupervised adversarial training (AT) has been highlighted as a means of achieving robustness in models without any label information. Previous studies in unsupervised AT have mostly focused on implementing self-supervised learning (SSL) frameworks, which maximize the instance-wise classification loss to generate adversarial examples. However, we observe that simply maximizing the self-supervised training loss with an untargeted adversarial attack often results in generating ineffective adversaries that may not help improve the robustness of the trained model, especially for non-contrastive SSL frameworks without negative examples. To tackle this problem, we propose a novel positive mining for targeted adversarial attack to generate effective adversaries for adversarial SSL frameworks. Specifically, we introduce an algorithm that selects the most confusing yet similar target example for a given instance based on entropy and similarity, and subsequently perturbs the given instance towards the selected target. Our method demonstrates significant enhancements in robustness when applied to non-contrastive SSL frameworks, and less but consistent robustness improvements with contrastive SSL frameworks, on the benchmark datasets.


## Pretrain
```
$ CUDA_VISIBLE_DEVICES=0,1 python taro_pretrain.py 
```

## Linear evaluation
```
$ sh run_lineval.sh ckpt_path epoch
```

## Citation
If you found the provided code useful, please cite our work.
```
@article{kim2024effective,
  title={Effective Targeted Attacks for Adversarial Self-Supervised Learning},
  author={Kim, Minseon and Ha, Hyeonjeong and Son, Sooel and Hwang, Sung Ju},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

