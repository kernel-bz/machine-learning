/**
 *  file name:  linear_mat.c
 *  function:   Linear Regression Multi-Variables for Machine Learning
 *  author:     JungJaeJoon(rgbi3307@nate.com) on the www.kernel.bz
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
///#include <alloca.h>

typedef unsigned int    u32;
typedef int             i32;
typedef float           f32;
typedef double          d64;

#define ARRAY_CNT(a)    sizeof(a) / sizeof(a[0])

///matrix count
#define MATC    2

///hypothesis: h = w1*x1 + w2*x2 + w3*x3 ... + wn*xn
static inline f32 hypothesis(f32 *w, f32 *x, u32 n)
{
    u32 i;
    f32 sum = 0.0;

    for(i=0; i < n; i++)
        sum += w[i] * x[i];
    return sum;
}

static inline void linear_descent_mv(f32 *w, f32 (*xm)[MATC], f32 *yd, u32 n, u32 m, f32 alpha)
{
    u32 j, k;
    f32 d[MATC] = {0.0,};
    ///f32 *d = alloca(n * sizeof(f32));
    ///memset(d, 0.0, n * sizeof(f32));

    for (j=0; j < m; j++)
        for (k=0; k < n; k++)
            d[k] += ((hypothesis(w, xm[j], n) - yd[j]) * xm[j][k]);  ///cost(w) = d/dw

    for (k=0; k < n; k++)
        w[k] = w[k] - alpha * (d[k] / m);  ///average descent(w)
}

/**
    linear regression multi-variables for learning
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
void linear_learning_mv(f32 (*xm)[MATC], f32 *yd, f32 *w, f32 alpha, u32 cnt, u32 n, u32 m)
{
    u32 i, j;
    f32 cost;

    printf("\n----------------------------------\n");
    printf("Learning...\n");
    for (i=0; i < cnt; i++)
    {
        linear_descent_mv(w, xm, yd, n, m, alpha);

        cost=0.0;
        for (j=0; j < m; j++)
            cost += (hypothesis(w, xm[j], n) - yd[j])
                        * (hypothesis(w, xm[j], n) - yd[j]);

        cost /= m;    ///average cost(w)
        printf("%d: cost=%f, w1=%f, w2=%f\n", i, cost, w[0], w[1]);
    }
}

void linear_test1(f32 *w, f32 alpha, u32 cnt)
{
    /**
    f32 x_mat[][5] = {
            {1.0, 0.0, 3.0, 0.0, 5.0}
           ,{0.0, 2.0, 0.0, 4.0, 0.0}
    };
    Transpose
    */
    f32 x_mat_t[][MATC] = {
            {1.0, 0.0}, {0.0, 2.0}, {3.0, 0.0}, {0.0, 4.0}, {5.0, 0.0}
    };
    f32 y_data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    u32 n, m;

    n = sizeof(x_mat_t[0]) / sizeof(x_mat_t[0][0]);
    m = sizeof(y_data) / sizeof(y_data[0]);
    linear_learning_mv(x_mat_t, y_data, w, alpha, cnt, n, m);
}

void linear_test2(f32 *w, f32 alpha, u32 cnt)
{
    /**
    f32 x_mat[][5] = {
            {1.0, 2.0, 3.0, 4.0, 5.0}
           ,{1.0, 2.0, 3.0, 4.0, 5.0}
    };
    Transpose
    */
    f32 x_mat_t[][MATC] = {
            {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0}
    };
    f32 y_data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    u32 n, m;

    n = sizeof(x_mat_t[0]) / sizeof(x_mat_t[0][0]);
    m = sizeof(y_data) / sizeof(y_data[0]);
    linear_learning_mv(x_mat_t, y_data, w, alpha, cnt, n, m);
}


void linear_test3(f32 *w, f32 alpha, u32 cnt)
{
    /**
    f32 x_mat[][5] = {
            {1.0, 2.0, 3.0, 4.0, 5.0}
           ,{4.0, 5.0, 6.0, 7.0, 8.0}
    };
    Transpose
    */
    f32 x_mat_t[][MATC] = {
            {1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}, {4.0, 7.0}, {5.0, 8.0}
    };
    f32 y_data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    u32 n, m;

    n = sizeof(x_mat_t[0]) / sizeof(x_mat_t[0][0]);
    m = sizeof(y_data) / sizeof(y_data[0]);
    linear_learning_mv(x_mat_t, y_data, w, alpha, cnt, n, m);
}

void linear_answer(f32 *w, f32 *x, u32 n)
{
    f32 y;

    y = hypothesis(w, x, n);

    printf("----------------------------------\n");
    printf("Answer for %f, %f: %f\n", x[0], x[1], y);
}

int main(void)
{
    f32 w[MATC], x[MATC], alpha;
    u32 cnt;

    w[0] = 5.0;
    w[1] = 5.0;
    x[0] = 0.0;
    x[1] = 8.0;
    ///learning rate: Watch Out for Overfitting, Underfitting
    alpha = 0.1;
    ///learning  loop count
    cnt = 30;
    linear_test1(w, alpha, cnt);
    linear_answer(w, x, 2);

    w[0] = 5.0;
    w[1] = 5.0;
    x[0] = 8.0;
    x[1] = 8.0;
    ///learning rate: Watch Out for Overfitting, Underfitting
    alpha = 0.01;
    ///learning  loop count
    cnt = 60;
    linear_test2(w, alpha, cnt);
    linear_answer(w, x, 2);

    alpha = 0.01;
    ///learning  loop count
    cnt = 2500;
    linear_test3(w, alpha, cnt);
    linear_answer(w, x, 2);

    return 0;
}
