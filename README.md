## MIMEx: Intrinsic Rewards from Masked Input Modeling

This is the anonymized code for ICML 2023 submission "MIMEx: Intrinsic Rewards from Masked Input Modeling". It contains the training code to obtain results on ALE or VizDoom environment.

### Example training commands

Train ALE `PRIVATE EYE` with ICM:

```
python train.py -e pe -a ICM
```

Train sparse `VizDoom` with MIMEx:

```
python train.py -e doom -a MIMEx
```