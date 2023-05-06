#include "pch.h"
#include "darknet.h"
#include "image.h"
#include "image_opencv.h"
#include "DNLIB_Utilities.h"


image load_image_color(char* filename, int w, int h)
{
    return load_image(filename, w, h, 3);
}


image load_image(char* filename, int w, int h, int c)
{
#ifdef OPENCV
    //image out = load_image_stb(filename, c);
    image out = load_image_cv(filename, c);
#else
    image out = load_image_stb(filename, c);    // without OpenCV
#endif  // OPENCV

    if ((h && w) && (h != out.h || w != out.w)) {
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}


static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w&& y < m.h&& c < m.c);
    return m.data[c * m.h * m.w + y * m.w + x];
}


static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w&& y < m.h&& c < m.c);
    m.data[c * m.h * m.w + y * m.w + x] = val;
}


static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w&& y < m.h&& c < m.c);
    m.data[c * m.h * m.w + y * m.w + x] += val;
}


image resize_image(image im, int w, int h)
{
    if (im.w == w && im.h == h) return copy_image(im);

    image resized = make_image(w, h, im.c);
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < im.h; ++r) {
            for (c = 0; c < w; ++c) {
                float val = 0;
                if (c == w - 1 || im.w == 1) {
                    val = get_pixel(im, im.w - 1, r, k);
                }
                else {
                    float sx = c * w_scale;
                    int ix = (int)sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < h; ++r) {
            float sy = r * h_scale;
            int iy = (int)sy;
            float dy = sy - iy;
            for (c = 0; c < w; ++c) {
                float val = (1 - dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if (r == h - 1 || im.h == 1) continue;
            for (c = 0; c < w; ++c) {
                float val = dy * get_pixel(part, c, iy + 1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}


image copy_image(image p)
{
    image copy = p;
    copy.data = (float*)xcalloc(p.h * p.w * p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h * p.w * p.c * sizeof(float));
    return copy;
}


void free_image(image m)
{
    if (m.data) {
        free(m.data);
    }
}


image make_image(int w, int h, int c)
{
    image out = make_empty_image(w, h, c);
    out.data = (float*)xcalloc(h * w * c, sizeof(float));
    return out;
}


image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}


image resize_min(image im, int min)
{
    int w = im.w;
    int h = im.h;
    if (w < h) {
        h = (h * min) / w;
        w = min;
    }
    else {
        w = (w * min) / h;
        h = min;
    }
    if (w == im.w && h == im.h) return im;
    image resized = resize_image(im, w, h);
    return resized;
}


image crop_image(image im, int dx, int dy, int w, int h)
{
    image cropped = make_image(w, h, im.c);
    int i, j, k;
    for (k = 0; k < im.c; ++k) {
        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                int r = j + dy;
                int c = i + dx;
                float val = 0;
                r = constrain_int(r, 0, im.h - 1);
                c = constrain_int(c, 0, im.w - 1);
                if (r >= 0 && r < im.h && c >= 0 && c < im.w) {
                    val = get_pixel(im, c, r, k);
                }
                set_pixel(cropped, i, j, k, val);
            }
        }
    }
    return cropped;
}