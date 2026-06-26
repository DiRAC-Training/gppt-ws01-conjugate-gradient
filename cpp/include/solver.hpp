#pragma once

void matvec(float *y, const float *A, const float *x, const int n);
float dot(const float *a, const float *b, int n);
void axpby(float *y, const float *x, float alpha, float beta, int n);
int cg_solve(float *x, const float *A, const float *b, const int n,
             int max_iter);
