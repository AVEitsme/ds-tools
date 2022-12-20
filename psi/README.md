# Popultion stability index (PSI)
## Call examples
### Data frames
```python
PSI(test_df, noised).eval_psi(10, 1e-4)
```
### Single feature
```python
PSI.psi(test_df.to_numpy(), noised.to_numpy(), 10, 1e-4)
```
## References
- https://github.com/mwburke/population-stability-index/blob/master/psi.py
- https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf
- https://towardsdatascience.com/checking-model-stability-and-population-shift-with-psi-and-csi-6d12af008783
