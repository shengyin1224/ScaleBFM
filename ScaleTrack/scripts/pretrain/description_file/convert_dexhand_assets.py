#!/usr/bin/env python3
"""Convert Inspire hands or object URDF trees to contact-ready USD assets."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=("hands", "objects", "humoto"),
    required=True,
    help="'humoto' recursively preserves every task-relative URDF path",
)
parser.add_argument("--input-root", type=str, required=True)
parser.add_argument("--output-root", type=str, required=True)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument(
    "--only-dotted-identifiers",
    action="store_true",
    help="Convert only URDFs whose stem contains a dot (repair mode)",
)
parser.add_argument("--report", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


@contextmanager
def sanitized_urdf_identifiers(input_path: Path):
    """Temporarily sanitize dotted URDF identifiers without touching meshes.

    Isaac Sim 5.1 may return success but emit an empty USD when robot/link
    identifiers contain dots (for example ``mug.000_link``). Mesh filenames are
    deliberately not modified.
    """

    if "." not in input_path.stem:
        yield input_path
        return
    tree = ET.parse(input_path)
    root = tree.getroot()
    for element in root.iter():
        for attribute in ("name", "link", "joint"):
            value = element.get(attribute)
            if value:
                element.set(attribute, value.replace(".", "_"))
    temporary_meshes = []
    mesh_aliases = {}
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename or "." not in Path(filename).stem:
            continue
        source_mesh = Path(filename)
        if not source_mesh.is_absolute():
            source_mesh = (input_path.parent / source_mesh).resolve()
        if not source_mesh.is_file():
            raise FileNotFoundError(
                f"{input_path}: referenced mesh does not exist: {source_mesh}"
            )
        if source_mesh not in mesh_aliases:
            descriptor, alias = tempfile.mkstemp(
                prefix="isaaclab_sanitized_mesh_",
                suffix=source_mesh.suffix,
                dir=input_path.parent,
            )
            os.close(descriptor)
            alias_path = Path(alias)
            alias_path.unlink()
            alias_path.symlink_to(source_mesh)
            mesh_aliases[source_mesh] = alias_path
            temporary_meshes.append(alias_path)
        mesh.set("filename", mesh_aliases[source_mesh].name)
    descriptor, temporary = tempfile.mkstemp(
        prefix="isaaclab_sanitized_",
        suffix=".urdf",
        dir=input_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        tree.write(temporary_path, encoding="utf-8", xml_declaration=True)
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)
        for mesh_path in temporary_meshes:
            mesh_path.unlink(missing_ok=True)


def convert(input_path: Path, output_path: Path, hands: bool) -> str:
    with sanitized_urdf_identifiers(input_path) as converter_input:
        cfg = UrdfConverterCfg(
            asset_path=str(converter_input.resolve()),
            usd_dir=str(output_path.parent.resolve()),
            usd_file_name=output_path.name,
            fix_base=hands,
            link_density=0.0 if hands else 200.0,
            merge_fixed_joints=False if hands else True,
            # The rollout q vectors contain all 12 revolute joints, including the
            # six URDF mimic joints.  Converting them to normal joints is required
            # for a 12-D action per hand.
            convert_mimic_joints_to_normal_joints=hands,
            joint_drive=(
                UrdfConverterCfg.JointDriveCfg(
                    drive_type="force",
                    target_type="position",
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=10.0, damping=1.0),
                )
                if hands
                else None
            ),
            collider_type="convex_decomposition",
            self_collision=hands,
            force_usd_conversion=args.overwrite or not output_path.exists(),
            make_instanceable=True,
        )
        usd_path = Path(UrdfConverter(cfg).usd_path).resolve()
        # Isaac Sim 5.1 can log a URDF import RuntimeError yet return normally,
        # leaving a tiny empty USDA wrapper. Treat that as conversion failure.
        if not usd_path.is_file() or usd_path.stat().st_size < 1000:
            raise RuntimeError(
                f"URDF importer produced an empty/invalid USD: {usd_path}"
            )
        return str(usd_path)


def main() -> None:
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if args.mode == "hands":
        inputs = [
            input_root / "inspire_hand_right.urdf",
            input_root / "inspire_hand_left.urdf",
        ]
    else:
        inputs = sorted(input_root.rglob("*.urdf"))
    if args.only_dotted_identifiers:
        inputs = [path for path in inputs if "." in path.stem]
    if not inputs:
        raise ValueError(f"no URDF files below {input_root}")

    report = {"mode": args.mode, "input_root": str(input_root), "converted": [], "failed": []}
    for input_path in inputs:
        relative = input_path.relative_to(input_root)
        output_path = (output_root / relative).with_suffix(".usd")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            usd_path = convert(input_path, output_path, hands=args.mode == "hands")
            report["converted"].append({"urdf": str(input_path), "usd": usd_path})
            print(f"[OK] {input_path} -> {usd_path}")
        except Exception as exc:
            report["failed"].append({"urdf": str(input_path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {input_path}: {exc}")
    report["coverage"] = {
        "urdf_count": len(inputs),
        "converted_count": len(report["converted"]),
        "failed_count": len(report["failed"]),
    }
    report_path = Path(args.report).resolve() if args.report else output_root / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
