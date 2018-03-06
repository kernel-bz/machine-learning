/**
 *  file name:  linear_mv.c
 *  function:   Linear Regression Multi-Variables for Machine Learning
 *  author:     JungJaeJoon(rgbi3307@nate.com) on the www.kernel.bz
 */

#include <stdio.h>
#include <stdlib.h>

typedef unsigned int    u32;
typedef int             i32;
typedef float           f32;
typedef double          d64;

#define ARRAY_CNT(a)    sizeof(a) / sizeof(a[0])

///hypothesis: h = w1*x1 + w2*x2
#define H2(w1,x1, w2,x2)  w1*x1 + w2*x2

static inline void linear_descent_mv(f32 *w1, f32 *w2, f32 *xd1, f32 *xd2, f32 *yd, u32 m, f32 alpha)
{
    f32 x1, x2, y;
    f32 d1=0.0, d2=0.0;
    u32 j;

    for (j=0; j < m; j++) {
        x1 = *(xd1 + j);
        x2 = *(xd2 + j);
        y = *(yd + j);
        d1 += ((H2(*w1,x1, *w2,x2) - y) * x1);  ///cost(w) = d/dw
        d2 += ((H2(*w1,x1, *w2,x2) - y) * x2);  ///cost(w) = d/dw
    }
    *w1 = *w1 - alpha * (d1 / m);  ///average descent(w)
    *w2 = *w2 - alpha * (d2 / m);  ///average descent(w)
}

/**
    linear regression multi-variables for learning
    @dx:    x_data
    @dy:    y_data
    @w:     first weight
    @alpha: learning rate
    @cnt:   count for loop
    @m:     count for data
    loop:   O(cnt x m x w#)
    return: learning weight
*/
void linear_learning_mv(f32 *xd1, f32 *xd2, f32 *yd, f32 *w1, f32 *w2, f32 alpha, u32 cnt, u32 m)
{
    u32 i, j;
    f32 x1, x2, y, cost;

    printf("\n----------------------------------\n");
    printf("Learning...\n");
    for (i=0; i < cnt; i++)
    {
        linear_descent_mv(w1, w2, xd1, xd2, yd, m, alpha);

        cost=0.0;
        for (j=0; j < m; j++) {
            x1 = *(xd1 + j);
            x2 = *(xd2 + j);
            y = *(yd + j);
            cost += (H2(*w1,x1, *w2,x2) - y) * (H2(*w1,x1, *w2,x2) - y);
        }
        cost = cost / m;    ///average cost(w)
        printf("%d: cost=%f, w1=%f, w2=%f\n", i, cost, *w1, *w2);
    }
}

void linear_test1(f32 *w1, f32 *w2, f32 alpha, u32 cnt)
{
    f32 x1_data[] = {1.0, 0.0, 3.0, 0.0, 5.0};
    f32 x2_data[] = {0.0, 2.0, 0.0, 4.0, 0.0};
    f32 y_data[]  = {1.0, 2.0, 3.0, 4.0, 5.0};
    u32 m;

    m = sizeof(y_data) / sizeof(y_data[0]);
    linear_learning_mv(x1_data, x2_data, y_data, w1, w2, alpha, cnt, m);
}

void linear_test2(f32 *w1, f32 *w2, f32 alpha, u32 cnt)
{
    f32 x1_data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    f32 x2_data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    f32 y_data[]  = {1.0, 2.0, 3.0, 4.0, 5.0};
    u32 m;

    m = sizeof(y_data) / sizeof(y_data[0]);
    linear_learning_mv(x1_data, x2_data, y_data, w1, w2, alpha, cnt, m);
}

void linear_test3(f32 *w1, f32 *w2, f32 alpha, u32 cnt)
{
    f32 x1_data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    f32 x2_data[] = {4.0, 5.0, 6.0, 7.0, 8.0};
    f32 y_data[]  = {1.0, 2.0, 3.0, 4.0, 5.0};
    u32 m;

    m = sizeof(y_data) / sizeof(y_data[0]);
    linear_learning_mv(x1_data, x2_data, y_data, w1, w2, alpha, cnt, m);
}

void linear_answer(f32 w1, f32 w2, f32 x1, f32 x2)
{
    f32 y;

    y = H2(w1,x1, w2,x2);

    printf("----------------------------------\n");
    printf("Answer for %f, %f: %f\n", x1, x2, y);
}

int main(void)
{
    f32 w1, w2, x1, x2, alpha;
    u32 cnt;

    ///hypothesis to learn for w
    ///h = w * x;
    w1 = 5.0;
    w2 = 5.0;
    x1 = 0.0;
    x2 = 8.0;
    ///learning rate: Watch Out for Overfitting, Underfitting
    alpha = 0.1;
    ///learning  loop count
    cnt = 30;

    linear_test1(&w1, &w2, alpha, cnt);
    linear_answer(w1, w2, x1, x2);

    w1 = 5.0;
    w2 = 5.0;
    x1 = 8.0;
    x2 = 8.0;
    ///alpha = 0.1;     ///overfitting
    ///alpha = 0.001;   ///underfitting
    alpha = 0.01;
    linear_test2(&w1, &w2, alpha, cnt);
    linear_answer(w1, w2, x1, x2);

    alpha = 0.01;
    cnt = 1600;
    linear_test3(&w1, &w2, alpha, cnt);
    linear_answer(w1, w2, x1, x2);

    return 0;
}
