# Setup 

```python
python3.12 -m venv .venv
source .venv/bin/activate

pip install \
    jax \
    jaxlib \
    numpy \
    scipy \
    matplotlib \
    pytest

pip install -U "jax[cuda12]"    
```



# Performance on A100

```text
(venv) [dc-kili1@mad06 benchmark]$ for d in jax_reduction jax_vmap jax_matmul; do     echo "Running $d";     (cd $d && python3 main.py); done
Running jax_reduction

Matrix size = 128
MatVec time : 0.389 ms
CG time     : 0.0159 s

Matrix size = 256
MatVec time : 0.432 ms
CG time     : 0.0158 s

Matrix size = 512
MatVec time : 0.449 ms
CG time     : 0.0163 s

Matrix size = 1024
MatVec time : 0.440 ms
CG time     : 0.0161 s
Running jax_vmap

Matrix size = 128
MatVec time : 0.705 ms
CG time     : 0.0207 s

Matrix size = 256
MatVec time : 0.689 ms
CG time     : 0.0190 s

Matrix size = 512
MatVec time : 0.680 ms
CG time     : 0.0214 s

Matrix size = 1024
MatVec time : 0.678 ms
CG time     : 0.0209 s
Running jax_matmul

Matrix size = 128
MatVec time : 0.404 ms
CG time     : 0.0159 s

Matrix size = 256
MatVec time : 0.377 ms
CG time     : 0.0159 s

Matrix size = 512
MatVec time : 0.465 ms
CG time     : 0.0162 s

Matrix size = 1024
MatVec time : 0.427 ms
CG time     : 0.0161 s
(venv) [dc-kili1@mad06 benchmark]$ 
```