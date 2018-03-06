/**
 *  file name:  cost_minimize.c
 *  function:   Linear Regression for Machine Learning
 *  author:     JungJaeJoon(rgbi3307@nate.com) on the www.kernel.bz
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef unsigned int    u32;
typedef int             i32;
typedef float           f32;
typedef double          d64;

#define ARRAY_CNT(a)    sizeof(a) / sizeof(a[0])

///hypothesis: h = w * x
///#define H(w,x)  w * x
inline f32 H(f32 w, f32 x)
{
    return w * x;
}

inline f32 cost(f32 w, f32 x, f32 y)
{
    return (H(w,x) - y) * (H(w,x) - y);
}

inline f32 cost_minimize(f32 w, f32 x, f32 y, f32 alpha)
{
    f32 c;
    const f32 target = 0.00001;

    while ((c = cost(w, x, y)) > target) {
        w = w - alpha * c;
        if (w < 0.0) {
            w = 0.0;
            break;
        }
        ///printf("cost=%f\n", c);
    }
    return w;
}


///((H(w,x) - y) * x);
///gradient_descent
inline float descent(f32 x, f32 y, f32 w, f32 alpha)
{
    /**
    f32 c1, c2;
    c1 = cost(w - alpha, x, y);
    c2 = cost(w + alpha, x, y);
    w = (c2 - c1) / (alpha * 2);
    */
    w = cost_minimize(w, x, y, alpha);

    return w;
}

/**
    linear regression for learning
    @dx:    x_data
    @dy:    y_data
    @w:     first weight
    @alpha: learning rate
    @cnt:   count for loop
    @m:     count for data
    loop:   O(cnt x m)
    return: learning weight
*/
f32 linear_learning(f32 *xd, f32 *yd, f32 w, f32 alpha, u32 cnt, u32 m)
{
    f32 x, y, c, d;
    u32 i, j;

    printf("\n----------------------------------\n");
    printf("Learning...\n");
    for (i=0; i < cnt; i++)
    {
        d = 0.0;
        c = 0.0;
        for (j=0; j < m; j++) {
            x = *(xd + j);
            y = *(yd + j);
            c += cost(w, x, y);
            d += descent(x, y, w, alpha);
        }
        c = c / m;                ///average cost(w)
        w = w - alpha * (d / m);  ///average descent(w)
        printf("%4d: cost=%f, w=%f\n", i, c, w);
    }

    return w;
}

f32 linear_test1(f32 w, f32 alpha, u32 cnt)
{
    f32 x_data[] = {1.0, 2.0, 3.0};
    f32 y_data[] = {1.0, 2.0, 3.0};
    u32 m;

    m = sizeof(y_data) / sizeof(y_data[0]);
    w = linear_learning(x_data, y_data, w, alpha, cnt, m);
    return w;
}

f32 linear_test2(f32 w, f32 alpha, u32 cnt)
{
    f32 x_data[] = {1.0, 2.0, 3.0};
    f32 y_data[] = {2.0, 4.0, 6.0};
    u32 m;

    m = sizeof(y_data) / sizeof(y_data[0]);
    w = linear_learning(x_data, y_data, w, alpha, cnt, m);
    return w;
}

f32 linear_test3(f32 w, f32 alpha, u32 cnt)
{
    f32 x_data[] = {1.0, 5.0, 10.0};
    f32 y_data[] = {1.0, 5.0, 10.0};
    u32 m;

    m = sizeof(y_data) / sizeof(y_data[0]);
    w = linear_learning(x_data, y_data, w, alpha, cnt, m);
    return w;
}

void linear_answer(f32 w, f32 x)
{
    f32 y;

    y = H(w,x);

    printf("----------------------------------\n");
    printf("Answer for %f: %f\n", x, y);
}

int main(void)
{
    f32 w, x, alpha;
    u32 cnt;

    ///hypothesis to learn for w
    ///h = w * x;
    w = 5.0;
    x = 8.0;
    ///learning rate: Watch Out for Overfitting, Underfitting
    alpha = 0.1;
    ///learning  loop count
    cnt = 200;

    w = linear_test1(w, alpha, cnt);
    linear_answer(w, x);

    w = 5.0;
    w = linear_test2(w, alpha, cnt);
    linear_answer(w, x);

    w = 5.0;
    ///alpha = 0.1;    ///overfitting!!!!
    alpha = 0.1;
    w = linear_test3(w, alpha, cnt);
    linear_answer(w, x);

    return 0;
}
