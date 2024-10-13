import argparse
import copy
import json
import pathlib

import joblib
import lpips
import numpy as np
import PIL.Image
import scipy.signal
import tqdm

import torch

TENSOIR_SCENES = ["armadillo", "ficus", "hotdog", "lego"]
TENSOIR_TEST_LIGHTS = ["bridge", "city", "fireplace", "forest", "night"]


DEBUG = False


def list_of_dicts_to_dict_of_lists(lst):
    """
    x = [
        {'foo': 3, 'bar': 1},
        {'foo': 4, 'bar': 2},
        {'foo': 5, 'bar': 3},
    ]
    ppp.list_of_dicts__to__dict_of_lists(x)
    # Output:
    # {'foo': [3, 4, 5], 'bar': [1, 2, 3]}
    """
    assert isinstance(lst, (list, tuple)), type(lst)
    if len(lst) == 0:
        return {}
    keys = lst[0].keys()
    output_dict = dict()
    for d in lst:
        assert set(d.keys()) == set(keys), (d.keys(), keys)
        for k in keys:
            if k not in output_dict:
                output_dict[k] = []
            output_dict[k].append(d[k])
    return output_dict


def compute_lpips(lpips_func_dict, np_gt, np_im, net_name, device):
    # Ref: https://github.com/Haian-Jin/TensoIR/blob/8f7b8dd87b5960847264536bb969c93dadeac1b5/utils.py#L76
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return lpips_func_dict[net_name](gt, im, normalize=True).item()


def compute_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    # Ref: https://github.com/Haian-Jin/TensoIR/blob/8f7b8dd87b5960847264536bb969c93dadeac1b5/utils.py#L93

    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode="valid")

    filt_fn = lambda z: np.stack(
        [
            convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
            for i in range(z.shape[-1])
        ],
        -1,
    )
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0.0, sigma00)
    sigma11 = np.maximum(0.0, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


def rescale_img(pred, mask, rescale_ratio=None):
    # Ref: https://github.com/Haian-Jin/TensoIR/blob/8f7b8dd87b5960847264536bb969c93dadeac1b5/renderer.py#L289
    assert (mask.ndim == 3) and (mask.shape[2] == 1), f"{mask.shape=}"
    pred_vals = pred[mask[..., 0] > 0.5, :]
    pred_vals_rescaled = np.clip(pred_vals * rescale_ratio, 0.0, 1.0)
    pred_vals_rescaled = (pred_vals_rescaled * 255).astype(np.uint8).astype(
        np.float32
    ) / 255.0  # ensure images are the same as those loaded from saved ones
    new_pred = copy.deepcopy(pred)
    new_pred[mask[..., 0] > 0.5, :] = pred_vals_rescaled
    return new_pred


def compute_metrics_single_data(
    *,
    gt_path,
    pred_path,
    gt_root,
    pred_root,
    device,
    lpips_func_dict,
    rescale_ratio_dict,
):
    # e.g., /root/armadillo/test_000/rgba_city.png
    gt_path = pathlib.Path(gt_path)
    scene_id = gt_path.parent.parent.stem
    img_idx = gt_path.parent.stem
    light_name = gt_path.stem.split("_")[1]

    gt_rgba = np.array(PIL.Image.open(gt_path)).astype(np.float32) / 255.0  # [H, W, 4]
    gt_mask = gt_rgba[..., 3:]
    gt_rgb = gt_rgba[..., :3] * gt_mask + (1 - gt_mask)

    pred_rgb = np.array(PIL.Image.open(pred_path)).astype(np.float32) / 255.0

    if rescale_ratio_dict is not None:
        rescale_ratio = rescale_ratio_dict[scene_id][light_name]
        pred_rgb = rescale_img(pred_rgb, gt_mask, rescale_ratio=rescale_ratio)

    # Ref: https://github.com/Haian-Jin/TensoIR/blob/8f7b8dd87b5960847264536bb969c93dadeac1b5/renderer.py#L302
    loss_rgb = np.mean((pred_rgb - gt_rgb) ** 2)
    metric_psnr = -10.0 * np.log(loss_rgb) / np.log(10.0)

    # Ref: https://github.com/Haian-Jin/TensoIR/blob/8f7b8dd87b5960847264536bb969c93dadeac1b5/renderer.py#L306
    metric_ssim = compute_ssim(pred_rgb, gt_rgb, 1.0)

    # Ref: https://github.com/Haian-Jin/TensoIR/blob/8f7b8dd87b5960847264536bb969c93dadeac1b5/renderer.py#L308
    metric_lpips_vgg = compute_lpips(lpips_func_dict, gt_rgb, pred_rgb, "vgg", device)

    metric_dict = {
        "gt_f": str(gt_path.relative_to(gt_root)),
        "pred_f": str(pred_path.relative_to(pred_root)),
        "psnr": metric_psnr,
        "ssim": metric_ssim,
        "lpips_vgg": metric_lpips_vgg,
    }

    torch.cuda.empty_cache()

    return {scene_id: {light_name: {img_idx: metric_dict}}}


def compute_rescale_ratios(result_path, tensoir_data_path, sampled_num=20):
    # Ref: https://github.com/Haian-Jin/TensoIR/blob/8f7b8dd87b5960847264536bb969c93dadeac1b5/renderer.py#L12

    eps = np.finfo(np.float32).eps

    rescale_ratio_dict = {}

    for tmp_scene in tqdm.tqdm(TENSOIR_SCENES, desc="computing rescale ratios"):
        rescale_ratio_dict[tmp_scene] = {}

        # e.g., /root/tensoir_raw_data/armadillo/test_000
        tmp_test_idx_dir_list = list(
            sorted((tensoir_data_path / tmp_scene).glob("test_*"))
        )

        tmp_interval = len(tmp_test_idx_dir_list) // sampled_num
        tmp_sampled_test_idx_dir_list = [
            tmp_test_idx_dir_list[i * tmp_interval] for i in range(sampled_num)
        ]

        # print(f"\n\n{len(tmp_sampled_test_idx_dir_list)=}\n\n")

        for tmp_light in TENSOIR_TEST_LIGHTS:
            tmp_all_ratio_list = []

            for tmp_test_idx_dir in tmp_sampled_test_idx_dir_list:
                tmp_test_idx = int(tmp_test_idx_dir.stem.split("_")[1])

                tmp_gt_path = tmp_test_idx_dir / f"rgba_{tmp_light}.png"

                tmp_pred_path = (
                    result_path
                    / tmp_scene
                    / tmp_light
                    / "latent_all_zeros"
                    / f"color_raw_{tmp_test_idx:03d}.png"
                )

                tmp_gt_rgba = (
                    np.array(PIL.Image.open(tmp_gt_path)).astype(np.float32) / 255.0
                )  # [H, W, 4]
                tmp_gt_mask = tmp_gt_rgba[..., 3:]
                tmp_gt_rgb = tmp_gt_rgba[..., :3] * tmp_gt_mask + (1 - tmp_gt_mask)

                tmp_pred_rgb = (
                    np.array(PIL.Image.open(tmp_pred_path)).astype(np.float32) / 255.0
                )

                tmp_pred_vals = tmp_pred_rgb[tmp_gt_mask[..., 0] > 0.5, :]
                tmp_gt_vals = tmp_gt_rgb[tmp_gt_mask[..., 0] > 0.5, :]
                tmp_ratio = tmp_gt_vals / np.clip(tmp_pred_vals, eps, 1.0)
                tmp_all_ratio_list.append(tmp_ratio)

            tmp_all_ratios = np.concatenate(tmp_all_ratio_list, axis=0)
            # Ref: https://github.com/Haian-Jin/TensoIR/blob/8f7b8dd87b5960847264536bb969c93dadeac1b5/renderer.py#L50
            tmp_ratio = np.median(tmp_all_ratios, axis=0)

            rescale_ratio_dict[tmp_scene][tmp_light] = tmp_ratio

    return rescale_ratio_dict


def main(args):
    device = args.device

    lpips_func_dict = {}
    for net_name in ["alex", "vgg"]:
        lpips_func_dict[net_name] = (
            lpips.LPIPS(net=net_name, version="0.1").eval().to(device)
        )

    result_path = pathlib.Path(args.results_path)
    tensoir_data_path = pathlib.Path(args.tensoir_data_path)

    if args.result_type == "illuminerf" and (not args.result_already_rescaled):
        rescale_ratio_dict = compute_rescale_ratios(result_path, tensoir_data_path)

        print(f"\n\n{rescale_ratio_dict=}\n\n")
    else:
        rescale_ratio_dict = None

    all_path_pairs = []

    for tmp_scene in tqdm.tqdm(TENSOIR_SCENES):
        # e.g., /root/tensoir_raw_data/armadillo/test_000
        tmp_test_idx_dir_list = list(
            sorted((tensoir_data_path / tmp_scene).glob("test_*"))
        )

        for tmp_test_idx_dir in tmp_test_idx_dir_list:
            tmp_test_idx = int(tmp_test_idx_dir.stem.split("_")[1])

            for tmp_light in TENSOIR_TEST_LIGHTS:
                tmp_gt_path = tmp_test_idx_dir / f"rgba_{tmp_light}.png"

                if args.result_type == "illuminerf":
                    if args.result_already_rescaled:
                        tmp_fname = f"color_global_rescaled_{tmp_test_idx:03d}.png"
                    else:
                        tmp_fname = f"color_raw_{tmp_test_idx:03d}.png"
                    tmp_pred_path = (
                        result_path
                        / tmp_scene
                        / tmp_light
                        / "latent_all_zeros"
                        / tmp_fname
                    )
                elif args.result_type == "tensoir":
                    tmp_pred_path = (
                        result_path
                        / f"{tmp_scene}_single"
                        / f"test_{tmp_test_idx:03d}"
                        / "relighting_without_bg"
                        / f"{tmp_light}.png"
                    )
                else:
                    raise ValueError(f"{args.result_type=}")
                assert tmp_gt_path.exists(), f"{str(tmp_gt_path)=}"
                assert tmp_pred_path.exists(), f"{str(tmp_pred_path)=}"
                all_path_pairs.append((tmp_gt_path, tmp_pred_path))

    rng = np.random.Generator(np.random.Philox(seed=42, counter=0))
    rng.shuffle(all_path_pairs)

    if DEBUG:
        all_path_pairs = all_path_pairs[:10]

    print(f"\n\n\n{len(all_path_pairs)=}\n")
    print(f"\n{all_path_pairs[0]=}\n\n\n")

    n_data = len(all_path_pairs)

    all_rets = joblib.Parallel(n_jobs=args.nproc)(
        joblib.delayed(compute_metrics_single_data)(
            gt_path=tmp_gt_path,
            pred_path=tmp_pred_path,
            gt_root=tensoir_data_path,
            pred_root=result_path,
            device=device,
            lpips_func_dict=lpips_func_dict,
            rescale_ratio_dict=rescale_ratio_dict,
        )
        for tmp_gt_path, tmp_pred_path in tqdm.tqdm(all_path_pairs)
    )

    n_rets = len(all_rets)

    assert n_data == n_rets, f"{n_data=}, {n_rets=}"

    agg_dict = {}
    for tmp_dict in all_rets:
        for tmp_scene in tmp_dict:
            if tmp_scene not in agg_dict:
                agg_dict[tmp_scene] = {}
            for tmp_light in tmp_dict[tmp_scene]:
                if tmp_light not in agg_dict[tmp_scene]:
                    agg_dict[tmp_scene][tmp_light] = {}
                agg_dict[tmp_scene][tmp_light].update(tmp_dict[tmp_scene][tmp_light])

    agg_dict = json.loads(json.dumps(agg_dict, sort_keys=True))

    n_agg_elems = sum(
        [
            sum(
                [
                    len(agg_dict[tmp_scene][tmp_light])
                    for tmp_light in agg_dict[tmp_scene]
                ]
            )
            for tmp_scene in agg_dict
        ]
    )

    final_dict = {
        "final": {},
        "aggregate_each_scene": {},
        "aggregate_each_scene_light": {},
        "info": agg_dict,
    }

    assert n_rets == n_agg_elems, f"{n_rets=}, {n_agg_elems=}"

    # print("\n\n", final_dict, "\n\n")

    if args.result_type == "tensoir":
        f_suffix = ""
    elif args.result_type == "illuminerf":
        if args.result_already_rescaled:
            f_suffix = "_from_rescaled"
        else:
            f_suffix = "_from_raw"
    else:
        raise ValueError(f"{args.result_type=}")

    final_save_f = result_path / f"metrics{f_suffix}.json"

    with open(final_save_f, "w") as f:
        json.dump(json.loads(json.dumps(final_dict)), f, indent=4, sort_keys=True)

    # aggregated for each light
    for tmp_scene in sorted(agg_dict.keys()):
        if tmp_scene not in final_dict["aggregate_each_scene_light"]:
            final_dict["aggregate_each_scene_light"][tmp_scene] = {}
        for tmp_light in sorted(agg_dict[tmp_scene].keys()):
            tmp_dict_of_list = list_of_dicts_to_dict_of_lists(
                list(agg_dict[tmp_scene][tmp_light].values())
            )
            # print(f"{tmp_scene=}, {tmp_light=}, {len(tmp_dict_of_list)=}")

            final_dict["aggregate_each_scene_light"][tmp_scene][tmp_light] = {
                k: np.mean(v)
                for k, v in tmp_dict_of_list.items()
                if not isinstance(v[0], str)
            }

    with open(final_save_f, "w") as f:
        json.dump(json.loads(json.dumps(final_dict)), f, indent=4, sort_keys=True)

    # aggregated for each scene
    for tmp_scene in sorted(final_dict["aggregate_each_scene_light"].keys()):
        tmp_dict_of_list = list_of_dicts_to_dict_of_lists(
            list(final_dict["aggregate_each_scene_light"][tmp_scene].values())
        )
        final_dict["aggregate_each_scene"][tmp_scene] = {
            k: np.mean(v)
            for k, v in tmp_dict_of_list.items()
            if not isinstance(v[0], str)
        }

    with open(final_save_f, "w") as f:
        json.dump(json.loads(json.dumps(final_dict)), f, indent=4)

    # aggregated across scenes
    tmp_dict_of_list = list_of_dicts_to_dict_of_lists(
        list(final_dict["aggregate_each_scene"].values())
    )
    final_dict["final"] = {
        k: np.mean(v) for k, v in tmp_dict_of_list.items() if not isinstance(v[0], str)
    }

    # final_dict["info"] = json.loads(json.dumps(final_dict["info"], sort_keys=True))

    with open(final_save_f, "w") as f:
        json.dump(json.loads(json.dumps(final_dict)), f, indent=4)

    print(f"\n\nComputed metrics have been saved to {str(final_save_f)}.\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--result_type",
        type=str,
        default="illuminerf",
        choices=["illuminerf", "tensoir"],
    )
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--tensoir_data_path", type=str, default=None)
    parser.add_argument(
        "--result_already_rescaled", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
