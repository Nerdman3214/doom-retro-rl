/*
 * frame_processor.c
 *
 * C extension for fast frame operations:
 *   - Nearest-neighbour resize (BGRA -> BGR, BGR -> BGR)
 *   - Per-channel mean difference  (replaces np.mean(ch_a - ch_b))
 *   - Mean brightness              (replaces np.mean(frame))
 *   - Mean absolute difference     (replaces np.mean(np.abs(a - b)))
 *   - Centre-box channel diff      (replaces sliced np operations)
 *   - Centre-box pixel count       (replaces (mask).sum())
 *   - Centre-box x-centroid        (replaces x_positions.mean())
 *
 * All functions accept any object that implements the Python buffer
 * protocol (bytes, bytearray, memoryview, numpy uint8 arrays).
 *
 * Build:
 *   python setup.py build_ext --inplace
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* helpers                                                              */
/* ------------------------------------------------------------------ */

static int
get_buf(PyObject *obj, Py_buffer *view)
{
    return PyObject_GetBuffer(obj, view, PyBUF_SIMPLE);
}

/* ------------------------------------------------------------------ */
/* resize_bgra_to_bgr(buf, in_w, in_h, out_w, out_h) -> bytes          */
/*   Nearest-neighbour downsample.  Input stride = 4 (BGRA).           */
/*   Output stride = 3 (BGR).  Returns a bytes object.                 */
/* ------------------------------------------------------------------ */
static PyObject *
fp_resize_bgra_to_bgr(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int in_w, in_h, out_w, out_h;

    if (!PyArg_ParseTuple(args, "Oiiii", &obj, &in_w, &in_h, &out_w, &out_h))
        return NULL;

    Py_buffer view;
    if (get_buf(obj, &view) < 0)
        return NULL;

    Py_ssize_t out_size = (Py_ssize_t)out_h * out_w * 3;
    PyObject *result = PyBytes_FromStringAndSize(NULL, out_size);
    if (!result) {
        PyBuffer_Release(&view);
        return NULL;
    }

    const uint8_t *data = (const uint8_t *)view.buf;
    uint8_t       *out  = (uint8_t *)PyBytes_AS_STRING(result);

    for (int y = 0; y < out_h; y++) {
        int src_y = y * in_h / out_h;
        for (int x = 0; x < out_w; x++) {
            int src_x = x * in_w / out_w;
            const uint8_t *p = data + ((Py_ssize_t)src_y * in_w + src_x) * 4;
            uint8_t       *q = out  + (y * out_w + x) * 3;
            q[0] = p[0]; /* B */
            q[1] = p[1]; /* G */
            q[2] = p[2]; /* R */
        }
    }

    PyBuffer_Release(&view);
    return result;
}

/* ------------------------------------------------------------------ */
/* resize_bgr(buf, in_w, in_h, out_w, out_h) -> bytes                  */
/*   Nearest-neighbour downsample.  Input stride = 3 (BGR).            */
/* ------------------------------------------------------------------ */
static PyObject *
fp_resize_bgr(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int in_w, in_h, out_w, out_h;

    if (!PyArg_ParseTuple(args, "Oiiii", &obj, &in_w, &in_h, &out_w, &out_h))
        return NULL;

    Py_buffer view;
    if (get_buf(obj, &view) < 0)
        return NULL;

    Py_ssize_t out_size = (Py_ssize_t)out_h * out_w * 3;
    PyObject *result = PyBytes_FromStringAndSize(NULL, out_size);
    if (!result) {
        PyBuffer_Release(&view);
        return NULL;
    }

    const uint8_t *data = (const uint8_t *)view.buf;
    uint8_t       *out  = (uint8_t *)PyBytes_AS_STRING(result);

    for (int y = 0; y < out_h; y++) {
        int src_y = y * in_h / out_h;
        for (int x = 0; x < out_w; x++) {
            int src_x = x * in_w / out_w;
            const uint8_t *p = data + ((Py_ssize_t)src_y * in_w + src_x) * 3;
            uint8_t       *q = out  + (y * out_w + x) * 3;
            q[0] = p[0]; q[1] = p[1]; q[2] = p[2];
        }
    }

    PyBuffer_Release(&view);
    return result;
}

/* ------------------------------------------------------------------ */
/* channel_diff_mean(buf, stride, ch_a, ch_b) -> float                 */
/*   mean( data[i*stride + ch_a] - data[i*stride + ch_b] )            */
/*   stride = 3 for BGR, 4 for BGRA.                                   */
/* ------------------------------------------------------------------ */
static PyObject *
fp_channel_diff_mean(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int stride, ch_a, ch_b;

    if (!PyArg_ParseTuple(args, "Oiii", &obj, &stride, &ch_a, &ch_b))
        return NULL;

    Py_buffer view;
    if (get_buf(obj, &view) < 0)
        return NULL;

    const uint8_t *data    = (const uint8_t *)view.buf;
    Py_ssize_t     n_pixels = view.len / stride;
    double         sum      = 0.0;

    for (Py_ssize_t i = 0; i < n_pixels; i++)
        sum += (int)data[i * stride + ch_a] - (int)data[i * stride + ch_b];

    PyBuffer_Release(&view);
    return PyFloat_FromDouble(n_pixels > 0 ? sum / (double)n_pixels : 0.0);
}

/* ------------------------------------------------------------------ */
/* mean_brightness(buf) -> float                                        */
/*   mean of every byte in buf.                                         */
/* ------------------------------------------------------------------ */
static PyObject *
fp_mean_brightness(PyObject *self, PyObject *args)
{
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;

    Py_buffer view;
    if (get_buf(obj, &view) < 0)
        return NULL;

    const uint8_t *data = (const uint8_t *)view.buf;
    double         sum  = 0.0;

    for (Py_ssize_t i = 0; i < view.len; i++)
        sum += data[i];

    PyBuffer_Release(&view);
    return PyFloat_FromDouble(view.len > 0 ? sum / (double)view.len : 0.0);
}

/* ------------------------------------------------------------------ */
/* mean_abs_diff(buf_a, buf_b) -> float                                 */
/*   mean |a[i] - b[i]|  over min(len_a, len_b) bytes.                */
/* ------------------------------------------------------------------ */
static PyObject *
fp_mean_abs_diff(PyObject *self, PyObject *args)
{
    PyObject *obj_a, *obj_b;
    if (!PyArg_ParseTuple(args, "OO", &obj_a, &obj_b))
        return NULL;

    Py_buffer va, vb;
    if (get_buf(obj_a, &va) < 0)
        return NULL;
    if (get_buf(obj_b, &vb) < 0) {
        PyBuffer_Release(&va);
        return NULL;
    }

    const uint8_t *a = (const uint8_t *)va.buf;
    const uint8_t *b = (const uint8_t *)vb.buf;
    Py_ssize_t     n = va.len < vb.len ? va.len : vb.len;
    double         s = 0.0;

    for (Py_ssize_t i = 0; i < n; i++) {
        int d = (int)a[i] - (int)b[i];
        s += d < 0 ? -d : d;
    }

    PyBuffer_Release(&va);
    PyBuffer_Release(&vb);
    return PyFloat_FromDouble(n > 0 ? s / (double)n : 0.0);
}

/* ------------------------------------------------------------------ */
/* centre_channel_diff_mean(buf, w, h, stride, box_half, ch_a, ch_b)   */
/*   mean(ch_a - ch_b) inside [cy±box_half, cx±box_half].             */
/* ------------------------------------------------------------------ */
static PyObject *
fp_centre_channel_diff_mean(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int w, h, stride, box_half, ch_a, ch_b;

    if (!PyArg_ParseTuple(args, "Oiiiiii", &obj, &w, &h, &stride, &box_half, &ch_a, &ch_b))
        return NULL;

    Py_buffer view;
    if (get_buf(obj, &view) < 0)
        return NULL;

    const uint8_t *data = (const uint8_t *)view.buf;
    int cy = h / 2, cx = w / 2;
    int y0 = cy - box_half > 0      ? cy - box_half : 0;
    int y1 = cy + box_half < h      ? cy + box_half : h;
    int x0 = cx - box_half > 0      ? cx - box_half : 0;
    int x1 = cx + box_half < w      ? cx + box_half : w;

    double sum   = 0.0;
    long   count = (long)(y1 - y0) * (x1 - x0);

    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            const uint8_t *p = data + ((Py_ssize_t)y * w + x) * stride;
            sum += (int)p[ch_a] - (int)p[ch_b];
        }
    }

    PyBuffer_Release(&view);
    return PyFloat_FromDouble(count > 0 ? sum / (double)count : 0.0);
}

/* ------------------------------------------------------------------ */
/* centre_channel_count(buf, w, h, stride, box_half, ch_a, ch_b, thr)  */
/*   count pixels in centre box where ch_a - ch_b > thr.              */
/* ------------------------------------------------------------------ */
static PyObject *
fp_centre_channel_count(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int w, h, stride, box_half, ch_a, ch_b, thr;

    if (!PyArg_ParseTuple(args, "Oiiiiiii", &obj, &w, &h, &stride, &box_half, &ch_a, &ch_b, &thr))
        return NULL;

    Py_buffer view;
    if (get_buf(obj, &view) < 0)
        return NULL;

    const uint8_t *data = (const uint8_t *)view.buf;
    int cy = h / 2, cx = w / 2;
    int y0 = cy - box_half > 0 ? cy - box_half : 0;
    int y1 = cy + box_half < h ? cy + box_half : h;
    int x0 = cx - box_half > 0 ? cx - box_half : 0;
    int x1 = cx + box_half < w ? cx + box_half : w;

    long count = 0;
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            const uint8_t *p = data + ((Py_ssize_t)y * w + x) * stride;
            if ((int)p[ch_a] - (int)p[ch_b] > thr)
                count++;
        }
    }

    PyBuffer_Release(&view);
    return PyLong_FromLong(count);
}

/* ------------------------------------------------------------------ */
/* centre_channel_x_mean(buf, w, h, stride, box_half, ch_a, ch_b, thr) */
/*   x-centroid of pixels in centre box where ch_a - ch_b > thr.      */
/*   Returns float or None if no qualifying pixel found.               */
/* ------------------------------------------------------------------ */
static PyObject *
fp_centre_channel_x_mean(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int w, h, stride, box_half, ch_a, ch_b, thr;

    if (!PyArg_ParseTuple(args, "Oiiiiiii", &obj, &w, &h, &stride, &box_half, &ch_a, &ch_b, &thr))
        return NULL;

    Py_buffer view;
    if (get_buf(obj, &view) < 0)
        return NULL;

    const uint8_t *data = (const uint8_t *)view.buf;
    int cy = h / 2, cx = w / 2;
    int y0 = cy - box_half > 0 ? cy - box_half : 0;
    int y1 = cy + box_half < h ? cy + box_half : h;
    int x0 = cx - box_half > 0 ? cx - box_half : 0;
    int x1 = cx + box_half < w ? cx + box_half : w;

    long x_sum = 0, count = 0;
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            const uint8_t *p = data + ((Py_ssize_t)y * w + x) * stride;
            if ((int)p[ch_a] - (int)p[ch_b] > thr) {
                x_sum += x;
                count++;
            }
        }
    }

    PyBuffer_Release(&view);
    if (count == 0)
        Py_RETURN_NONE;
    return PyFloat_FromDouble((double)x_sum / (double)count);
}

/* ------------------------------------------------------------------ */
/* method table                                                         */
/* ------------------------------------------------------------------ */

static PyMethodDef FpMethods[] = {
    {"resize_bgra_to_bgr",       fp_resize_bgra_to_bgr,       METH_VARARGS,
     "resize_bgra_to_bgr(buf, in_w, in_h, out_w, out_h) -> bytes\n"
     "Nearest-neighbour resize BGRA buffer to BGR bytes."},

    {"resize_bgr",               fp_resize_bgr,               METH_VARARGS,
     "resize_bgr(buf, in_w, in_h, out_w, out_h) -> bytes\n"
     "Nearest-neighbour resize BGR buffer to BGR bytes."},

    {"channel_diff_mean",        fp_channel_diff_mean,        METH_VARARGS,
     "channel_diff_mean(buf, stride, ch_a, ch_b) -> float\n"
     "Mean of (ch_a - ch_b) over all pixels."},

    {"mean_brightness",          fp_mean_brightness,          METH_VARARGS,
     "mean_brightness(buf) -> float\n"
     "Mean of all bytes in buf."},

    {"mean_abs_diff",            fp_mean_abs_diff,            METH_VARARGS,
     "mean_abs_diff(buf_a, buf_b) -> float\n"
     "Mean |a[i]-b[i]| between two same-size buffers."},

    {"centre_channel_diff_mean", fp_centre_channel_diff_mean, METH_VARARGS,
     "centre_channel_diff_mean(buf, w, h, stride, box_half, ch_a, ch_b) -> float\n"
     "Mean (ch_a - ch_b) inside the centre box."},

    {"centre_channel_count",     fp_centre_channel_count,     METH_VARARGS,
     "centre_channel_count(buf, w, h, stride, box_half, ch_a, ch_b, threshold) -> int\n"
     "Count pixels in centre box where ch_a - ch_b > threshold."},

    {"centre_channel_x_mean",    fp_centre_channel_x_mean,    METH_VARARGS,
     "centre_channel_x_mean(buf, w, h, stride, box_half, ch_a, ch_b, threshold) -> float|None\n"
     "X-centroid of pixels in centre box where ch_a - ch_b > threshold."},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fpmodule = {
    PyModuleDef_HEAD_INIT, "frame_processor",
    "Fast C routines for DOOM RL frame processing.",
    -1, FpMethods
};

PyMODINIT_FUNC
PyInit_frame_processor(void)
{
    return PyModule_Create(&fpmodule);
}
