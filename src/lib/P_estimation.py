# coding: utf-8
import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count

from lib.parzen import estimate_scipy_gaussian_kernel_density, estimate_one_class_parzen_width


def estimate_P_on_B_segmentation(sess, classifier, B, Xtr, ytr, dy, seg_size=40, multiprocess=True):

    yt = np.argmax(ytr, axis=1)

    B_seg_ind, g_seg_ind = [], []
    B_seg = np.array([np.argsort([np.sum(np.square(br - x))
                     for x in Xtr])[:seg_size] for br in B])  # B * seg
    for Bsi in B_seg:
        OB_ind = Bsi[0]  # Centered sample (as on-boundary sample)
        OB_yt = ytr[OB_ind]  # Segment class y
        OB_gs_ind_oh = sess.run(classifier.gs_ind_oh,
                                feed_dict={classifier.x: [Xtr[OB_ind]], classifier.y: [ytr[OB_ind]]})[0]
        if np.count_nonzero(OB_gs_ind_oh * OB_yt) == 0:
            continue
        OB_gs_ind = sess.run(classifier.gs_ind, feed_dict={classifier.x: [
                             Xtr[OB_ind]], classifier.y: [ytr[OB_ind]]})[0]
        Bsi_gs_ind_oh = sess.run(classifier.gs_ind_oh,
                                 feed_dict={classifier.x: Xtr[Bsi], classifier.y: ytr[Bsi]})

        new_B = []
        new_yt = []
        for b, Bgio in zip(Bsi, Bsi_gs_ind_oh):
            if (Bgio == OB_gs_ind_oh).all():
                new_B.append(b)
                new_yt.append(yt[b])
        # for estimate gaussian kde (more than 1 sample)
        if len(new_B) > 1:
            # for estimate multiple class
            if len(np.unique(new_yt)) > 1:
                # for estimate gaussian kde (do not exist same point sample only; void singular matrix)
                if not all([e == dy[new_B][0] for e in dy[new_B][1:]]):
                    B_seg_ind.append(new_B)
                    g_seg_ind.append(OB_gs_ind.tolist())

    S = list(set(sum(B_seg_ind, [])))

    seg_args = [(S, Bsi, gsi, dy[Bsi], yt)
                for Bsi, gsi in zip(B_seg_ind, g_seg_ind)]

    seg_map_h, seg_map_y, seg_map_g = [], [], []
    cnt = 0
    thread = 1
    if multiprocess:
        print("Run multiprocess by {} cores".format(cpu_count()))
        thread = cpu_count()

    with Pool(thread) as p:
        for res_h, res_y, res_g in p.imap_unordered(run_sub_process_on_multiclass, seg_args):
            cnt += 1
            print("{} / {}".format(cnt, len(seg_args)))
            seg_map_g.append(res_g)
            seg_map_h.append(res_h)
            seg_map_y.append(res_y)

    S = np.reshape(np.array(S), (-1, 1))  # S * 1
    seg_map_g = np.array(seg_map_g)  # seg * 2
    seg_map_h = np.array(seg_map_h)  # seg * S
    seg_map_y = np.array(seg_map_y)  # seg * S

    return S, seg_map_g, seg_map_h, seg_map_y


def run_sub_process_on_multiclass(args):
    return estimate_segment_probability_on_multiclass(*args)


def estimate_segment_probability_on_multiclass(S, Bsi, gsi, Bsi_dy, yt):
    seg_h = np.ones(len(S), dtype=float) * np.nan
    seg_y = np.ones(len(S), dtype=float) * np.nan
    seg_g = np.ones(2, dtype=float) * np.nan
    S_inds = [S.index(x) for x in Bsi]

    # set segment y
    seg_y[S_inds] = yt[Bsi]
    # set segment g1 and g2
    seg_g[0] = gsi[0]
    seg_g[1] = gsi[1]
    # estimate h by parzen
    h = estimate_scipy_gaussian_kernel_density(Bsi_dy)  # scipy-parzen
    # h = estimate_one_class_parzen_width(Bsi_dy)  # loo-parzen
    seg_h[S_inds] = h.flatten()

    return seg_h, seg_y, seg_g
