/**
 *  file name:  logistic.c
 *  function:   Logistic Classification for Machine Learning
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

static inline f32 sigmoid(f32 h)
{
    return (1 / (1 + exp(-h)));
}

///hypothesis: h = w1*x1 + w2*x2 + w3*x3 ... + wn*xn
static inline f32 hypothesis(f32 *w, f32 *x, u32 n)
{
    u32 i;
    f32 sum = 0.0;

    for(i=0; i < n; i++)
        sum += w[i] * x[i];
    return sigmoid(sum);
}

static inline f32 cost(f32 *w, f32 *x, u32 n, f32 y)
{
    f32 c;
    /**
    c += (hypothesis(w, xm[j], n) - yd[j])
                        * (hypothesis(w, xm[j], n) - yd[j]);
    */
    ///base-e logarithm
    c = (-y * log(hypothesis(w,x,n)) - (1-y) * log(1 - hypothesis(w,x,n)));
    ///c = (y==0) ? -log(1 - hypothesis(w,x,n)) : -log(hypothesis(w,x,n));
    return c;
}

static f32 descent(u32 i, f32 *w, f32 *x, f32 y, u32 n, f32 alpha)
{
    f32 c1, c2, d;

    d = w[i];
    w[i] = d - alpha;
    c1 = cost(w, x, n, y);
    w[i] = d + alpha;
    c2 = cost(w, x, n, y);

    w[i] = d;
    d = (c2 - c1) / (alpha * 2);
    return d;
}

static inline void logistic_descent_mv(f32 *w, f32 (*xm)[MATC], f32 *yd, u32 n, u32 m, f32 alpha)
{
    u32 j, k;
    f32 d[MATC] = {0.0,};
    ///f32 *d = alloca(n * sizeof(f32));
    ///memset(d, 0.0, n * sizeof(f32));

    for (j=0; j < m; j++)
        for (k=0; k < n; k++)
            ///d[k] += ((hypothesis(w, xm[j], n) - yd[j]) * xm[j][k]);  ///cost(w) = d/dw
            d[k] += descent(k, w, xm[j], yd[j], n, alpha);

    for (k=0; k < n; k++)
        w[k] = w[k] - alpha * (d[k] / m);  ///average descent(w)
}

/**
    Logistic Classification multi-variables for learning
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
void logistic_learning_mv(f32 (*xm)[MATC], f32 *yd, f32 *w, f32 alpha, u32 cnt, u32 n, u32 m)
{
    u32 i, j;
    f32 c;

    printf("\n----------------------------------\n");
    printf("Learning...\n");
    printf("0: w1=%f, w2=%f, w3=%f\n", w[0], w[1], w[2]);

    for (i=1; i <= cnt; i++)
    {
        logistic_descent_mv(w, xm, yd, n, m, alpha);

        c=0.0;
        for (j=0; j < m; j++)
            c += cost(w, xm[j], n, yd[j]);

        c /= m;    ///average cost(w)
        printf("%d: cost=%f, w1=%f, w2=%f, w3=%f\n", i, c, w[0], w[1], w[2]);
    }
}

void logistic_test(f32 *w, f32 alpha, u32 cnt)
{
    ///Transpose
    f32 x_mat_t[][MATC] = {
        {1.0, 2.0, 1.0}, {1.0, 3.0, 2.0}, {1.0, 3.0, 3.0},
        {1.0, 5.0, 5.0}, {1.0, 7.0, 5.0}, {1.0, 2.0, 5.0}
    };
    f32 y_data[] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    u32 n, m;

    n = ARRAY_CNT(x_mat_t[0]);
    m = ARRAY_CNT(y_data);
    logistic_learning_mv(x_mat_t, y_data, w, alpha, cnt, n, m);
}

void logistic_answer(f32 *w, f32 *x, u32 n)
{
    f32 y;

    y = hypothesis(w, x, n);

    printf("----------------------------------\n");
    printf("Answer for %f, %f, %f: %f\n", x[0], x[1], x[2], y);
}

int main(void)
{
    f32 w[MATC] = {1.0, 1.0, 1.0};
    f32 alpha = 0.1;
    u32 cnt;

    ///learning  loop count
    cnt = 2000;
    logistic_test(w, alpha, cnt);

    f32 x1[MATC] = {1.0, 2.0, 2.0};
    logistic_answer(w, x1, MATC);

    f32 x2[MATC] = {1.0, 7.0, 7.0};
    logistic_answer(w, x2, MATC);

    return 0;
}
