import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def random_unit_vector(rng):
    v = rng.normal(size=3)
    norm = np.linalg.norm(v)

    while norm < 1e-12:
        v = rng.normal(size=3)
        norm = np.linalg.norm(v)

    return v / norm


def axis_angle_to_matrix(axis, angle_rad):
    """
    Rodrigues' rotation formula.

    axis:
        unit 3D rotation axis

    angle_rad:
        rotation angle in radians
    """

    x, y, z = axis

    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )

    I = np.eye(3, dtype=np.float64)

    R = (
        I
        + np.sin(angle_rad) * K
        + (1.0 - np.cos(angle_rad)) * (K @ K)
    )

    return R


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dst",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--translation-ratio",
        type=float,
        default=0.0,
        help="Translation error magnitude / mean camera radius",
    )

    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=0.0,
        help="Rotation error magnitude in degrees",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.translation_ratio < 0:
        raise ValueError("--translation-ratio must be >= 0")

    if args.rotation_deg < 0:
        raise ValueError("--rotation-deg must be >= 0")

    if args.translation_ratio == 0 and args.rotation_deg == 0:
        raise ValueError(
            "At least one perturbation must be non-zero."
        )

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().absolute()

    if dst.exists() or dst.is_symlink():
        raise FileExistsError(
            f"{dst} already exists. "
            "Remove it manually if you really want to regenerate it."
        )

    if not (src / 'transforms_train.json').is_file():
        parser.error('Missing training camera metadata: ' + str(src / 'transforms_train.json'))
    dst.mkdir(parents=True)

    # ============================================================
    # Load TRAIN poses
    # ============================================================

    train_data = load_json(
        src / "transforms_train.json"
    )

    frames = train_data["frames"]

    positions = []

    for frame in frames:
        T = np.asarray(
            frame["transform_matrix"],
            dtype=np.float64,
        )

        positions.append(T[:3, 3])

    positions = np.stack(positions)

    # ============================================================
    # Scene / camera scale
    # ============================================================

    camera_center = positions.mean(axis=0)

    camera_radius = np.linalg.norm(
        positions - camera_center,
        axis=1,
    ).mean()

    translation_magnitude = (
        args.translation_ratio * camera_radius
    )

    rotation_rad = np.deg2rad(
        args.rotation_deg
    )

    print("================================================")
    print("Camera Pose Perturbation")
    print("================================================")
    print(f"Training cameras       : {len(frames)}")
    print(f"Mean camera radius     : {camera_radius:.8f}")
    print(
        f"Translation ratio      : "
        f"{args.translation_ratio:.8f}"
    )
    print(
        f"Translation magnitude  : "
        f"{translation_magnitude:.8f}"
    )
    print(
        f"Rotation magnitude     : "
        f"{args.rotation_deg:.8f} deg"
    )
    print(f"Seed                    : {args.seed}")
    print("================================================")

    # Separate RNGs make translation and rotation experiments
    # independently reproducible.
    rng_translation = np.random.default_rng(
        args.seed
    )

    rng_rotation = np.random.default_rng(
        args.seed + 1
    )

    perturbations = []

    max_rotation_orthogonality_error = 0.0
    min_rotation_det = float("inf")
    max_rotation_det = float("-inf")

    # ============================================================
    # Perturb cameras
    # ============================================================

    for i, frame in enumerate(frames):

        T = np.asarray(
            frame["transform_matrix"],
            dtype=np.float64,
        ).copy()

        R_gt = T[:3, :3].copy()
        t_gt = T[:3, 3].copy()

        # --------------------------------------------------------
        # Translation
        # --------------------------------------------------------

        delta_t = np.zeros(3)

        if args.translation_ratio > 0:

            direction = random_unit_vector(
                rng_translation
            )

            delta_t = (
                direction * translation_magnitude
            )

            T[:3, 3] = t_gt + delta_t

        # --------------------------------------------------------
        # Rotation
        #
        # Left multiplication means the perturbation axis is
        # represented in world coordinates.
        #
        # R_noisy = R_delta @ R_gt
        # --------------------------------------------------------

        rotation_axis = np.zeros(3)

        if args.rotation_deg > 0:

            rotation_axis = random_unit_vector(
                rng_rotation
            )

            R_delta = axis_angle_to_matrix(
                rotation_axis,
                rotation_rad,
            )

            R_noisy = R_delta @ R_gt

            T[:3, :3] = R_noisy

        else:

            R_noisy = R_gt

        # --------------------------------------------------------
        # Validate rotation matrix
        # --------------------------------------------------------

        orthogonality_error = np.max(
            np.abs(
                R_noisy.T @ R_noisy
                - np.eye(3)
            )
        )

        det = np.linalg.det(R_noisy)

        max_rotation_orthogonality_error = max(
            max_rotation_orthogonality_error,
            orthogonality_error,
        )

        min_rotation_det = min(
            min_rotation_det,
            det,
        )

        max_rotation_det = max(
            max_rotation_det,
            det,
        )

        frame["transform_matrix"] = T.tolist()

        perturbations.append(
            {
                "frame_index": i,
                "file_path": frame["file_path"],
                "delta_translation": delta_t.tolist(),
                "translation_norm": float(
                    np.linalg.norm(delta_t)
                ),
                "rotation_axis": rotation_axis.tolist(),
                "rotation_deg": float(
                    args.rotation_deg
                ),
            }
        )

    # ============================================================
    # Save TRAIN poses
    # ============================================================

    save_json(
        train_data,
        dst / "transforms_train.json",
    )

    shutil.copy2(
        src / "transforms_train.json",
        dst / "transforms_train_gt.json",
    )

    # ============================================================
    # VAL + TEST remain GT
    # ============================================================

    for split in ["val", "test"]:

        filename = f"transforms_{split}.json"

        source = src / filename

        if source.exists():

            shutil.copy2(
                source,
                dst / filename,
            )

    # ============================================================
    # Image directories: symlink only
    # ============================================================

    for split in ["train", "val", "test"]:

        source_dir = src / split
        target_dir = dst / split

        if source_dir.exists():

            os.symlink(
                source_dir.resolve(),
                target_dir,
                target_is_directory=True,
            )

    # ============================================================
    # Metadata
    # ============================================================

    if (
        args.translation_ratio > 0
        and args.rotation_deg > 0
    ):
        noise_type = "translation_and_rotation"

    elif args.translation_ratio > 0:
        noise_type = "translation_only"

    else:
        noise_type = "rotation_only"

    metadata = {
        "source_dataset": str(src.resolve()),
        "noise_type": noise_type,
        "translation_ratio": float(
            args.translation_ratio
        ),
        "translation_magnitude": float(
            translation_magnitude
        ),
        "rotation_deg": float(
            args.rotation_deg
        ),
        "camera_radius": float(camera_radius),
        "seed": args.seed,
        "num_training_cameras": len(frames),
        "rotation_validation": {
            "max_orthogonality_error": float(
                max_rotation_orthogonality_error
            ),
            "min_determinant": float(
                min_rotation_det
            ),
            "max_determinant": float(
                max_rotation_det
            ),
        },
        "perturbations": perturbations,
    }

    save_json(
        metadata,
        dst / "pose_noise_metadata.json",
    )

    print()
    print("DONE")
    print(f"Dataset: {dst}")
    print()
    print(
        "Max rotation orthogonality error:",
        f"{max_rotation_orthogonality_error:.3e}",
    )
    print(
        "Rotation determinant range:",
        f"[{min_rotation_det:.12f}, "
        f"{max_rotation_det:.12f}]",
    )
    print()
    print("TRAIN poses : perturbed")
    print("VAL poses   : GT")
    print("TEST poses  : GT")


if __name__ == "__main__":
    main()
