'''
Descripttion: This code is used for the pre-processing of the RWMD dataset
version: 1.0
Author: duany
Date: 2024-07-29 03:47:03
LastEditors: xuzhen
LastEditTime: 2024-07-29 22:29:50
'''
import os
from typing import Optional

import cv2
from tqdm import tqdm
import shutil
import random
import numpy as np
import glob
import json
from base64 import b64encode, b64decode
from io import BytesIO
from PIL import Image

def _quantize_instance_mask_for_png(mask_u8: np.ndarray) -> np.ndarray:
    """
    Map contiguous instance ids ``1..K`` to spread grayscale ``1..255`` (``0`` stays background).

    Raw ids are hard to see in viewers (e.g. a single sheet is id ``1``, looks black). Scaling
    preserves order so :func:`np.unique` / ``max`` still identify the top-sheet instance.
    """
    mid = int(mask_u8.max())
    if mid <= 0:
        return mask_u8
    out = np.zeros_like(mask_u8)
    for k in range(1, mid + 1):
        v = int(round(255.0 * float(k) / float(mid)))
        v = max(1, min(255, v))
        out[mask_u8 == k] = np.uint8(v)
    return out


def _out_png_from_src_rel(src_root: str, img_p: str) -> str:
    """
    Unique flat ``*.png`` filename from ``img_p`` relative to ``src_root``.
    Subdirectories become ``__`` (e.g. ``receipt/a.jpg`` → ``receipt__a.png``).
    """
    root = os.path.abspath(os.path.normpath(src_root))
    abs_img = os.path.abspath(os.path.normpath(img_p))
    rel = os.path.relpath(abs_img, root)
    rel_norm = rel.replace("\\", "/")
    stem = os.path.splitext(rel_norm)[0]
    safe = stem.replace("/", "__")
    for c in '<>:"|?*':
        safe = safe.replace(c, "_")
    if not safe or set(safe) <= {".", "_"}:
        safe = "unnamed"
    return safe + ".png"


def change_max_size(src_dir):
    dir_match = [("IMG", "IMG_RESIZE"), ("GT", "GT_RESIZE")]
    for src_name, dst_name in dir_match:
        src_path = os.path.join(src_dir, src_name)
        dst_path = os.path.join(src_dir, dst_name)
        
        for file_name in tqdm([name for name in os.listdir(src_path) if name.endswith(".png") or name.endswith(".jpg")]):
            src_file = os.path.join(src_path, file_name)
            dst_file = os.path.join(dst_path, file_name)
            
            if "GT" in src_name:
                read_flag = cv2.IMREAD_GRAYSCALE
                resize_flag = cv2.INTER_NEAREST
            else:
                read_flag = cv2.IMREAD_COLOR
                resize_flag = cv2.INTER_AREA
            img = cv2.imread(src_file, read_flag)
            # resize the max size to 1500
            max_size = max(img.shape[0], img.shape[1])
            if max_size > 1500:
                scale = 1500 / max_size
                img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=resize_flag)
                cv2.imwrite(dst_file, img)

def choise_datas(src_dir, dst_dir, num=100):
    img_dir = os.path.join(src_dir, "img")
    gt_dir = os.path.join(src_dir, "mask")
    img_name = [name for name in os.listdir(img_dir) if not name.endswith("_4.png") and not name.endswith("_1.png") and not name.endswith("_2.png") and not name.endswith("_3.png")]
    random.shuffle(img_name)
    img_name = img_name[:num]
    for name in tqdm(img_name):
        img_path = os.path.join(img_dir, name)
        mask_path = os.path.join(gt_dir, name)
        shutil.copy(img_path, os.path.join(dst_dir, "img"))
        shutil.copy(mask_path, os.path.join(dst_dir, "mask"))

def check_labels(src_dir):
    gt_dir = os.path.join(src_dir, "mask")
    gt_names = [name for name in os.listdir(gt_dir)]
    label_set = set()
    for gt_n in tqdm(gt_names):
        gt_path = os.path.join(gt_dir, gt_n)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        labels = np.unique(gt)
        label_set.update(labels)
    print(label_set)
    


def cp2all(src_dir_list, dst_dir):
    for src_dir in src_dir_list:
        img_dir = os.path.join(src_dir, "img")
        gt_dir = os.path.join(src_dir, "mask")
        img_name = [name for name in os.listdir(img_dir)]
        # img_name = [name for name in os.listdir(img_dir) if not name.endswith("_4.png") and not name.endswith("_1.png") and not name.endswith("_2.png") and not name.endswith("_3.png")]
        for name in tqdm(img_name):
            img_path = os.path.join(img_dir, name)
            mask_path = os.path.join(gt_dir, name)
            shutil.copy(img_path, os.path.join(dst_dir, "img"))
            shutil.copy(mask_path, os.path.join(dst_dir, "mask"))

def resize(src_dir, dst_dir):
    imgs_paths = glob.glob(src_dir + "/img/*png")
    for img_p in tqdm(imgs_paths):
        img = cv2.imread(img_p)
        img_n = os.path.basename(img_p)
        label_p = os.path.join(src_dir, "mask", img_n)
        mask = cv2.imread(label_p, cv2.IMREAD_GRAYSCALE)
        # 最大边缩放到1024, scale the longest edge of the image to 1024
        max_size = max(img.shape[0], img.shape[1])
        if max_size > 1024:
            scale = 1024 / max_size
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (int(mask.shape[1] * scale), int(mask.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(os.path.join(dst_dir, "img", img_n), img)
        cv2.imwrite(os.path.join(dst_dir, "mask", img_n), mask)
        
def savePoints(src_dir):
    savePointsWithResize(src_dir, None)

def savePointsWithResize(src_dir, edge_limit=1024):
    label_points = {}
    imgs_path = glob.glob(src_dir + "/*[jp][pn]g")
    for img_p in tqdm(imgs_path):
        img = cv2.imread(img_p)
        img_n = os.path.basename(img_p)
        label_p = img_p.replace(".jpg", ".json") if "jpg" in img_n else  img_p.replace(".png", ".json")
        with open(label_p, "r", encoding="utf-8") as _lf:
            label = json.load(_lf)
        
        for l in label['shapes']:
            label_symbol = l['label']      # str
            if "foreground_doc" == label_symbol:
                points = np.asarray(l['points']).astype(np.float32)
                if edge_limit is not None:
                    # 最大边缩放到1024, scale the longest edge of the image to 1024
                    max_size = max(img.shape[0], img.shape[1])
                    if max_size > 1024:
                        scale = 1024 / max_size
                    else:
                        scale = 1.0
                    points = points * scale
                points = points.astype(np.int32)           # (N, 2)
                img_k = os.path.splitext(img_n)[0]
                label_points[img_k] = points.tolist()
                break
    json.dump(label_points, open("label_points.json", "w"), indent=4, ensure_ascii=False)
 
def test_rotate3D():
    from datas_augment.tools.math_utils import PerspectiveTransform, cliped_rand_norm
    x = cliped_rand_norm(0, 180 / 4)
    y = cliped_rand_norm(0, 180 / 4)
    z = cliped_rand_norm(0, 180 / 4)
    trans = PerspectiveTransform(x, y, z, 1.0, 50)
    img = cv2.imread("/data3/duanyong2/datas/competation/FloodNet/val/IMG/10833.jpg")
    h, w, _ = img.shape
    points_src = [[10, 10], [w - 100, 100], [w - 400, h - 400], [100, h - 200]]
    img_src = img.copy()
    # 画点,draw points
    for p in points_src:
        cv2.circle(img_src, p, 50, (0, 0, 255), -1)
    cv2.imwrite("img_src.png", img_src)
    # dst, M33, ptsOut = trans.transform_image(img)
    heatmap = np.zeros((h, w), dtype=np.uint8)
    dst, M33, ptsOut, heatmap = trans.transform_image_with_heatmap(img, heatmap)
    points_dst = trans.transform_pnts([points_src], M33)
    
    left = min([int(points[0]) for points in ptsOut])
    top = min([int(points[1]) for points in ptsOut])
    right = max([int(points[0]) for points in ptsOut])
    bottom = max([int(points[1]) for points in ptsOut])
    result_img = dst[top:bottom, left:right, :]
    
    if points_dst is not None and len(points_dst) > 0:
        points_dst[:, :, 0] -= left
        points_dst[:, :, 1] -= top

    for p in points_dst[0]:
        cv2.circle(result_img, list(map(int, p)), 50, (0, 0, 255), -1)
    cv2.imwrite("img_dst.png", result_img)


def increase_exampaper(src_dir, dst_dir, expand=7):
    name_exampaper = ['edge_4', 'edge_5', 'edge_6', 'edge_7', 'edge_8', 'edge_15', 'edge_16', 'edge_35', 'edge_39', 'edge_455']
    paths_img = glob.glob(src_dir + "/img/*[jp][pn]g")
    path_label = os.path.join(src_dir, "label_points.json")
    with open(path_label, "r", encoding="utf-8") as _lf:
        label = json.load(_lf)
    for path in tqdm(paths_img):
        name = os.path.split(path)[-1]
        path_img = os.path.join(src_dir, "img", name)
        name_k = os.path.splitext(name)[0]
        path_mask = os.path.join(src_dir, "mask", name_k+".png")
        if name_k in name_exampaper:
            for i in range(expand):
                name_expand_k = name_k + "_" + str(i)
                name_expand = name_expand_k + ".png"
                shutil.copy(path_img, os.path.join(dst_dir, "img", name_expand))
                shutil.copy(path_mask, os.path.join(dst_dir, "mask", name_expand))
                assert name_expand_k not in label, f"{name_expand_k} should not in label keys"
                label[name_expand_k] = label[name_k]
    json.dump(label, open(os.path.join(dst_dir, "label_points.json"), "w"), indent=4, ensure_ascii=False)

def test_shape(mask_path1, mask_path2):
    img1_cv = cv2.imread(mask_path1, cv2.IMREAD_GRAYSCALE)
    img2_cv = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)
    print(img1_cv.shape, img2_cv.shape)
    # for name in os.listdir(mask_dir):
    #     mask_p = os.path.join(mask_dir, name)
    #     img = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
    #     if len(img.shape) == 3:
    #         print(img.shape)
    #     else:
    #         print(img.shape)
    from detectron2.data import detection_utils as utils
    img1 = utils.read_image(mask_path1, "L")
    img1 = np.squeeze(img1, -1)
    img2 = utils.read_image(mask_path2)
    print(img1.shape, img2.shape)


def statistics_label_v2(mask_dir):
    max_num_dict = {}
    for mask_name in tqdm(os.listdir(mask_dir)):
        mask_p = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        labels = list(np.unique(mask))
        if 0 in labels:
            labels.remove(0)
        for l in labels:
            if l not in max_num_dict:
                max_num_dict[l] = 1
            else:
                max_num_dict[l] += 1
    print(max_num_dict)


def genarate_label_from_ori(src_dir, dst_dir):
    """
    Walk ``src_dir`` recursively; for each image with a sidecar LabelMe ``.json``, emit flat
    ``img/*.png``, ``mask/*.png``, and ``label_points.json`` keys matching those basenames.
    Output names encode relative paths (``subdir__file.png``) so nested files do not overwrite.

    Instance ids ``1..K`` in ``mask`` are scaled to spread grayscale ``1..255`` (``K==1`` → 255)
    so PNGs are visible; order is preserved for downstream loaders that use ``max(unique)``.
    """
    label_mapper = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
                     "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18}

    src_root = os.path.abspath(os.path.normpath(src_dir))

    points_json = {}
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    if not os.path.exists(dst_dir + "/img"):
        os.mkdir(dst_dir+"/img")
    if not os.path.exists(dst_dir + "/mask"):
        os.mkdir(dst_dir+"/mask")


    # for img_p in tqdm(imgs_path):
    for root, dirs, files in os.walk(src_root):
        for file in tqdm(files):
            # TODO delete
            # if os.path.splitext(file)[0] in ['sanxingS21ultra-1984', 'sanxingS21ultra-1949', 'sanxingS21ultra-1466', 'sanxingS21ultra-1950']:
            #     continue

            if file.endswith(".json"):
                continue
            img_p = os.path.join(root, file)
            img = cv2.imread(img_p)
            if img is None:
                continue
            img_n = os.path.basename(img_p)
            if "jpg" in img_n:
                label_p = img_p.replace(".jpg", ".json")
            elif "png" in img_n:
                label_p = img_p.replace(".png", ".json")
            elif "jpeg" in img_n:
                label_p = img_p.replace(".jpeg", ".json")
            elif "JPG" in img_n:
                label_p = img_p.replace(".JPG", ".json")
            elif "PNG" in img_n:
                label_p = img_p.replace(".PNG", ".json")
            elif "JPEG" in img_n:
                label_p = img_p.replace(".JPEG", ".json")
            with open(label_p, "r", encoding="utf-8") as _lf:
                label = json.load(_lf)

            label_infos = []
            foreground_quad: Optional[np.ndarray] = None
            for l in label['shapes']:
                label_symbol = l['label']      # str
                if label_symbol not in label_mapper:
                    if label_symbol == "foreground_doc":
                        foreground_quad = np.asarray(l['points']).astype(np.int32)
                    continue
                label_idx = label_mapper[label_symbol]
                points = np.asarray(l['points']).astype(np.int32)           # (N, 2)
                label_infos.append((label_idx, points))

            if not label_infos:
                continue

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            label_infos_sorted = sorted(label_infos, key=lambda x: x[0])
            # 找到前景id, find the foreground document id
            fore_idx = label_infos_sorted[-1][0]
            # 用1临时填充背景mask, temporarily fill the background mask with 1
            for label_idx, points in label_infos_sorted[:-1]:
                cv2.fillPoly(mask, [points], 1)
            # 重新给背景文档赋值, reassign the background document
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            too_small_contours = 0
            new_label_infos = []
            new_label_idx = 0
            for i in range(len(contours)):
                hull = cv2.convexHull(contours[i])
                area_cur = cv2.contourArea(hull)
                if area_cur < 100:
                    too_small_contours += 1
                    continue
                new_label_idx = i + 1
                new_label_infos.append((new_label_idx, contours[i]))
            # 重新给前景文档赋值, reassign the foreground document
            fore_idx = new_label_idx + 1
            new_label_infos.append((fore_idx, label_infos_sorted[-1][1]))

            mask_new = np.zeros(img.shape[:2], dtype=np.uint8)
            new_label_infos_sorted = sorted(new_label_infos, key=lambda x: x[0])
            for label_idx, points in new_label_infos_sorted:
                points = np.asarray(points).astype(np.int32)
                cv2.fillPoly(mask_new, [points], label_idx)

            out_png = _out_png_from_src_rel(src_root, img_p)
            if too_small_contours > 0:
                print(out_png, too_small_contours)
            if foreground_quad is not None and foreground_quad.size > 0:
                points_json[out_png] = foreground_quad.tolist()
            else:
                points_json[out_png] = []

            mask_png = _quantize_instance_mask_for_png(mask_new)

            shutil.copy(img_p, os.path.join(dst_dir, "img", out_png))
            cv2.imwrite(os.path.join(dst_dir, "mask", out_png), mask_png)
    with open(f"{dst_dir}/label_points.json", "w", encoding="utf-8") as f:
        json.dump(points_json, f, indent=4, ensure_ascii=False)


def rotate_img(root_dir, out_dir):
    # TODO 若旋转原始图片，清晰度会略微提升
    def save_img(json_path):
        with open(json_path, "r", encoding="utf-8") as _lf:
            label = json.load(_lf)
        data_bytes = b64decode(label["imageData"])
        bytes_stream = BytesIO(data_bytes)
        image = Image.open(bytes_stream)
        
        new_img_name = os.path.split(json_path)[-1].replace(".json", ".png")
        new_img_path = os.path.join(out_dir, new_img_name)
        image.save(new_img_path)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for root, dirs, files in os.walk(root_dir):
        for file in tqdm(files):
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                save_img(json_path)
            
def split_data(root_dir: str, train_ratio: float = 0.75, seed: int = 42) -> None:
    """Split ``img``/``mask`` and ``label_points.json`` (keys = PNG basenames) into ``train/`` and ``test/``."""

    def split_copy(img_dir: str, mask_dir: str, img_names: list, out_dir: str) -> None:
        os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "mask"), exist_ok=True)
        for img_name in tqdm(img_names):
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)
            shutil.copy2(img_path, os.path.join(out_dir, "img", img_name))
            shutil.copy2(mask_path, os.path.join(out_dir, "mask", img_name))

    img_dir = os.path.join(root_dir, "img")
    mask_dir = os.path.join(root_dir, "mask")

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)

    random.seed(seed)
    img_name_list = [n for n in os.listdir(img_dir) if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(img_name_list)
    train_num = int(len(img_name_list) * train_ratio)

    img_names_train = img_name_list[:train_num]
    img_names_test = img_name_list[train_num:]

    pts_path = os.path.join(root_dir, "label_points.json")
    pts_all = {}
    if os.path.isfile(pts_path):
        with open(pts_path, "r", encoding="utf-8") as f:
            pts_all = json.load(f)

    train_root = os.path.join(root_dir, "train")
    test_root = os.path.join(root_dir, "test")
    split_copy(img_dir, mask_dir, img_names_train, train_root)
    split_copy(img_dir, mask_dir, img_names_test, test_root)

    train_pts = {k: pts_all[k] for k in img_names_train if k in pts_all}
    test_pts = {k: pts_all[k] for k in img_names_test if k in pts_all}
    with open(os.path.join(train_root, "label_points.json"), "w", encoding="utf-8") as f:
        json.dump(train_pts, f, indent=4, ensure_ascii=False)
    with open(os.path.join(test_root, "label_points.json"), "w", encoding="utf-8") as f:
        json.dump(test_pts, f, indent=4, ensure_ascii=False)

def _lookup_label_points(json_label: dict, img_n: str):
    """Resolve ``img_n`` key; allow stem match if legacy keys differ."""
    if img_n in json_label:
        return img_n
    stem = os.path.splitext(img_n)[0]
    for k in json_label:
        if os.path.splitext(k)[0] == stem:
            return k
    return None


def resize_customdata(
    src_dir: str,
    dst_dir: str,
    json_path: str,
    json_save: Optional[dict] = None,
    max_edge: int = 1024,
):
    """Resize image/mask by longest edge to ``max_edge``; scale ``label_points`` the same way; write ``label_points_resize.json``."""
    if json_save is None:
        json_save = {}
    os.makedirs(os.path.join(dst_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, "mask"), exist_ok=True)

    imgs_paths = sorted(glob.glob(os.path.join(src_dir, "img", "*.png")))
    with open(json_path, "r", encoding="utf-8") as f:
        json_label = json.load(f)

    for img_p in tqdm(imgs_paths):
        img = cv2.imread(img_p)
        if img is None:
            continue
        img_n = os.path.basename(img_p)
        label_p = os.path.join(src_dir, "mask", img_n)
        mask = cv2.imread(label_p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        max_size = max(img.shape[0], img.shape[1])
        scale = 1.0
        if max_size > max_edge:
            scale = max_edge / max_size
            img = cv2.resize(
                img,
                (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
            mask = cv2.resize(
                mask,
                (int(mask.shape[1] * scale), int(mask.shape[0] * scale)),
                interpolation=cv2.INTER_NEAREST,
            )

        cv2.imwrite(os.path.join(dst_dir, "img", img_n), img)
        cv2.imwrite(os.path.join(dst_dir, "mask", img_n), mask)

        key = _lookup_label_points(json_label, img_n)
        if key is None or not json_label[key]:
            json_save[img_n] = []
            continue
        point_scale = np.asarray(json_label[key], dtype=np.float64) * scale
        json_save[img_n] = point_scale.tolist()

    out_json = os.path.join(dst_dir, "label_points_resize.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_save, f, indent=4, ensure_ascii=False)
    return json_save


def run_rwmd_preprocess(
    src_dir: str,
    out_dir: str,
    *,
    train_ratio: float = 0.75,
    seed: int = 42,
    max_edge: int = 1024,
    keep_work_flat: bool = False,
) -> None:
    """
    Full pipeline: ``genarate_label_from_ori`` (recursive under ``src_dir``; unique flat names) →
    ``split_data`` (+ split ``label_points``) → ``resize_customdata`` for train and test →
    ``out_dir/train_resize`` and ``out_dir/test_resize``.
    Intermediate flat export lives in ``out_dir/_work_flat`` (removed unless ``keep_work_flat``).
    """
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, "_work_flat")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)

    genarate_label_from_ori(src_dir, work)
    split_data(work, train_ratio=train_ratio, seed=seed)

    train_dir = os.path.join(work, "train")
    test_dir = os.path.join(work, "test")
    train_resize = os.path.join(out_dir, "train_resize")
    test_resize = os.path.join(out_dir, "test_resize")

    resize_customdata(
        train_dir,
        train_resize,
        os.path.join(train_dir, "label_points.json"),
        None,
        max_edge=max_edge,
    )
    resize_customdata(
        test_dir,
        test_resize,
        os.path.join(test_dir, "label_points.json"),
        None,
        max_edge=max_edge,
    )

    if not keep_work_flat:
        shutil.rmtree(work)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "RWMD: LabelMe → instance masks → train/test split → resize (longest edge). "
            "Recursively scans --src at any depth; output PNG names encode relative paths (subdir__file.png)."
        )
    )
    ap.add_argument(
        "--src",
        type=str,
        required=True,
        help="Root folder to walk recursively; each image needs a sidecar .json next to it (LabelMe).",
    )
    ap.add_argument("--out", type=str, required=True, help="Output root: creates train_resize/ and test_resize/")
    ap.add_argument("--train-ratio", type=float, default=0.75, help="Train fraction (default 0.75)")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed for split")
    ap.add_argument("--max-edge", type=int, default=1024, help="Longest edge after resize (default 1024)")
    ap.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep intermediate _work_flat under --out (default: delete after pipeline)",
    )
    args = ap.parse_args()
    run_rwmd_preprocess(
        args.src,
        args.out,
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_edge=args.max_edge,
        keep_work_flat=args.keep_work,
    )
    print(f"Done. Preprocessed splits: {os.path.join(args.out, 'train_resize')}, {os.path.join(args.out, 'test_resize')}")

