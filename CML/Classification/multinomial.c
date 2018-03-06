/**
 *  file name:  multinomial.c
 *  function:   Multinomial(Softmax) Classification for Machine Learning
 *  author:     JungJaeJoon(rgbi3307@nate.com) on the www.kernel.bz
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef unsigned int    u32;
typedef int             i32;
typedef float           f32;
typedef double          d64;

#define ARRAY_CNT(a)    sizeof(a) / sizeof(a[0])

///matrix count
#define MATC    3
#define LEARNING_RATE 0.01

static inline f32 tahnh(f32 h)
{
    return (exp(h) - exp(-h)) / (exp(h) + exp(-h));
}

///hypothesis: h = w1*x1 + w2*x2 + w3*x3 ... + wn*xn
static inline f32 hypothesis(f32 *w, f32 *x, u32 n)
{
    u32 i;
    f32 h = 0.0;

    for(i=0; i < n; i++)
        h += w[i] * x[i];
    ///return tahnh(h);
    return h;
}

static inline f32 softmax_sum(u32 n, f32 (*wm)[MATC], f32 *x)
{
    u32 i;
    f32 h, sms=0.0;

    for (i=0; i<n; i++) {
        h = hypothesis(wm[i], x, n);
        sms += exp(h);
    }
    return sms;
}

static inline f32 softmax_smv(f32 *w, f32 *x, u32 n, f32 sms)
{
    f32 smv;
    smv = (exp(hypothesis(w, x, n)) / sms);  ///softmax value
    ///printf("softmax=%f\n", sm);
    return smv;
}

static void softmax_out(u32 n, f32 (*wm)[MATC], f32 *x)
{
    u32 i, imax=0;
    f32 sms, smv, smax=0.0;
    sms = softmax_sum(n, wm, x);
    printf("Answer Softmax [ ");
    for (i=0; i < n; i++) {
        smv = softmax_smv(wm[i], x, n, sms);
        printf("%f, ", smv);
        if (smv > smax) {
            smax = smv;
            imax = i;
        }
    }
    printf("]\n");
    printf("Answer Argmax [%d]\n", imax);
}

static inline f32 cost(f32 *w, f32 *x, u32 n, f32 y, f32 sms)
{
    f32 c;

    ///base-e logarithm
    ///c = (-y * log(softmax_smv(w,x,n,sms)) - (1-y) * log(1 - softmax_smv(w,x,n,sms)));
    ///c = (y==0) ? -log(1 - softmax_smv(w,x,n,sms)) : -log(softmax_smv(w,x,n,sms));
    c = (-y * log(softmax_smv(w,x,n,sms)));
    return c;
}

static f32 descent_error(u32 i, f32 *w, f32 *x, f32 *y, u32 n, f32 alpha, f32 sms)
{
    f32 py, err;

    py = softmax_smv(w,x,n,sms);
    err = py - y[i];

    return err;
}

static f32 mult_descent_mv(f32 (*wm)[MATC], f32 (*xm)[MATC], f32 (*yd)[MATC], u32 n, u32 m, f32 alpha)
{
    u32 j, k, l;
    f32 sms, c = 0.0;

    k = n * n * sizeof(f32);
    f32 *d = alloca(k);
    f32 *dp = d;
    memset(d, 0.0, k);

    for (j=0; j < m; j++) {     ///data number
        sms = softmax_sum(n, wm, xm[j]);

        for(l=0; l < n; l++) {
            for (k=0; k < n; k++) {
                *dp += descent_error(k, wm[l], xm[j], yd[j], n, alpha, sms);
                dp++;
            }
        }
        dp = d;
    }

    ///descent
    for(l=0; l < n; l++) {
        for (k=0; k < n; k++) {
            wm[l][k] = wm[l][k] - alpha * (*dp / m);  ///average descent(w)
            dp++;
        }
    }

    ///cost
    for (j=0; j < m; j++) {
        sms = softmax_sum(n, wm, xm[j]);
        for(k=0; k < n; k++)
            c += cost(wm[k], xm[j], n, yd[j][k], sms);
    }
    c /= m;    ///average cost(w)
    return c;
}

static void mat_output(u32 n, f32 (*mat)[MATC])
{
    u32 j, k;

    printf("[\n");
    for (j=0; j < n; j++) {
        printf("[");
        for (k=0; k < MATC; k++)
            printf("%f, ", mat[j][k]);
        printf("], \n");
    }
    printf("]\n");
}

/**
    Multinomial Classification multi-variables for learning
    @xm:    x_matrix
    @yd:    y_data
    @w:     first weight
    @alpha: learning rate
    @cnt:   count for loop
    @n:     row count for x_matrix
    @m:     col count for y_data
    loop:   O(cnt x m x w#)
    return: learning weight
*/
void mult_learning_mv(f32 (*xm)[MATC], f32 (*yd)[MATC], f32 (*wm)[MATC], f32 alpha, u32 cnt, u32 n, u32 m)
{
    u32 i;
    f32 c;

    printf("\n----------------------------------\n");
    printf("Learning...\n");

    mat_output(n, wm);

    for (i=1; i <= cnt; i++){
        c = mult_descent_mv(wm, xm, yd, n, m, alpha);
        printf("%d: cost=%f:\n", i, c);
        mat_output(n, wm);
    }
}

void mult_test(f32 (*wm)[MATC], f32 alpha, u32 cnt)
{
    ///Transpose
    f32 x_mat_t[][MATC] = {
        {1.0, 2.0, 1.0}, {1.0, 3.0, 2.0}, {1.0, 3.0, 4.0},
        {1.0, 5.0, 5.0}, {1.0, 7.0, 5.0}, {1.0, 2.0, 5.0},
        {1.0, 6.0, 6.0}, {1.0, 7.0, 7.0}
    };
    f32 y_data[][MATC] = {
        {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}
    };
    u32 n, m;

    n = ARRAY_CNT(x_mat_t[0]);  ///column
    m = ARRAY_CNT(y_data);      ///data num
    mult_learning_mv(x_mat_t, y_data, wm, alpha, cnt, n, m);
}

void mult_answer(f32 (*wm)[MATC], f32 *x, u32 n)
{
    printf("----------------------------------\n");
    printf("Answer for x[ %f, %f, %f ]\n", x[0], x[1], x[2]);
    softmax_out(n, wm, x);
}

int main(void)
{
    f32 w[][MATC] = {
        {-2.0, 0.0, 0.1}, {0.0, 0.2, 0.0}, {0.4, 0.0, -2.0} };  ///init important!
    u32 cnt;

    ///learning  loop count
    cnt = 500;
    mult_test(w, LEARNING_RATE, cnt);

    f32 x1[MATC] = {1.0, 2.0, 2.0};
    mult_answer(w, x1, MATC);

    f32 x2[MATC] = {1.0, 7.0, 7.0};
    mult_answer(w, x2, MATC);

    f32 x3[MATC] = {1.0, 8.0, 9.0};
    mult_answer(w, x3, MATC);

    return 0;
}
