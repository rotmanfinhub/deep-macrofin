# Tips and Additional Considerations

- Watch out for constraints - adding too many can cause issues with learning.
- Try to simplify the constraints, making them in a linear form. Reducing the number of divisions, especially divisions with derivatives, help. This is probably because of numerical stability issues.
- Parametrizing too many variables with neural networks may not always help learning.
- Using ReLU as activation might be better for functions with discontinuity in first order derivatives. 
