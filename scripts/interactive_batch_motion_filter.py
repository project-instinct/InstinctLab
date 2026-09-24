"""Interactive Batch Motion Filter

## Task

Using a batch-wise viewer to show around 12 motions at a time, letting the user to click and decide which motion to select or discard. Then apply and save the motion files to the target directory in the same folder structure as the source directory.

If there is already a file with the exact same name in the target folder, just skip the display.

## Arguments

--urdf: The robot kinematics structure, prefer to use `pytorch_kinematics` to read it.

--from-dir: The source folder to load all motion files, but in a random order.

--ext: By default is `retargeted.npz` to read from.

--to-dir: The resultant folder, where all resultant files shall be placed in the same relative path of the source folder.

--resultant: Choose from "copy" or "symlink". Set to "copy" by default to copy all files exactly to the `--to-dir`

--speed-ratio: The Playback speed multiplier, adjustable in the window. skip frame to play if greater than 1.0

## GUI design

- A main window displaying around 12 motions per batch. recommend using PyVista to implement. Only display the robot skeleton is enough, no need to load and render all meshes mentioned in the urdf file.

    - In the main window grid, it should also display a z=0 plane, for a better reference.

    - With "all select" and "all discard" button to mark select or discard for all sub-windows.

    - In the main window, user can press S or D to choose select or discard mode, then left click a sub-window to mark the motion. And that window should be surounded by green or red box.

    - If the user put the mouse inside the window, it should perform normal 3D view-angle changing logic.

- Below these main window, it should show a progress bar revealing how many has been marked and how many to go throughout this entire file selection task.

- Below, is the "apply" button to apply the selection and to the file operations. And refresh the main-batch window.

    - At each "apply", the terminal should print those files, from where to where, or skipped.

- Pressing "=" or "-" button to speed up or down by 1.2 or 0.8 times. A.K.A override the `--speed-ratio` argument.

"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
import torch

import pytorch_kinematics as pk


BatchSize = 12
BatchRows = 3
BatchColumns = 4
ProgressRow = BatchRows
ControlRow = BatchRows + 1


@dataclass
class MotionData:
    framerate: float
    base_pos_w: np.ndarray
    base_quat_w: np.ndarray
    joint_pos: np.ndarray


@dataclass
class MotionPanel:
    rel_path: str
    source_path: str
    framerate: float
    link_pos_b_frames: np.ndarray
    base_pos_w_frames: np.ndarray
    base_quat_w_frames: np.ndarray
    skeleton: pv.PolyData
    last_frame: int = -1


def parse_urdf_skeleton(urdf_path: str) -> List[Tuple[str, str]]:
    root = ET.parse(urdf_path).getroot()
    bones = []
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        bones.append((parent, child))
    return bones


def load_motion_data(npz_path: str, chain_joint_names: List[str]) -> Optional[MotionData]:
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            framerate = float(data["framerate"])
            joint_names = list(data["joint_names"])
            base_pos_w = data["base_pos_w"].astype(np.float32)
            base_quat_w = data["base_quat_w"].astype(np.float32)
            joint_pos_raw = data["joint_pos"].astype(np.float32)

        joint_name_to_index = {name: index for index, name in enumerate(joint_names)}
        joint_pos = np.zeros((base_pos_w.shape[0], len(chain_joint_names)), dtype=np.float32)
        for chain_index, name in enumerate(chain_joint_names):
            data_index = joint_name_to_index.get(name)
            if data_index is not None:
                joint_pos[:, chain_index] = joint_pos_raw[:, data_index]

        return MotionData(
            framerate=framerate,
            base_pos_w=base_pos_w,
            base_quat_w=base_quat_w,
            joint_pos=joint_pos,
        )
    except Exception as error:
        print(f"  [WARN] Failed to load {npz_path}: {error}")
        return None


def compute_fk_all_frames(
    chain: pk.Chain, joint_pos: np.ndarray, link_names: List[str]
) -> np.ndarray:
    joint_pos_t = torch.from_numpy(joint_pos).to(torch.float32)
    link_positions = []
    for start in range(0, joint_pos.shape[0], 2048):
        fk_result = chain.forward_kinematics(joint_pos_t[start : start + 2048])
        link_positions.append(
            torch.stack(
                [fk_result[name].get_matrix()[:, :3, 3] for name in link_names], dim=1
            ).numpy()
        )
    return np.concatenate(link_positions, axis=0).astype(np.float32)


def quat_rotate(quat: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quat_w, quat_x, quat_y, quat_z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    cross = np.stack(
        [
            quat_y * vectors[..., 2] - quat_z * vectors[..., 1],
            quat_z * vectors[..., 0] - quat_x * vectors[..., 2],
            quat_x * vectors[..., 1] - quat_y * vectors[..., 0],
        ],
        axis=-1,
    )
    quat_vector = quat[..., 1:]
    return vectors + 2.0 * (quat_w[..., np.newaxis] * cross + np.cross(quat_vector, cross))


def world_link_positions(panel: MotionPanel, frame: int) -> np.ndarray:
    return quat_rotate(
        panel.base_quat_w_frames[frame][np.newaxis, :], panel.link_pos_b_frames[frame]
    ) + panel.base_pos_w_frames[frame][np.newaxis, :]


def collect_files(from_dir: str, extension: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(from_dir, followlinks=True):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.relpath(os.path.join(root, filename), from_dir))
    return files


class BatchMotionViewer:
    def __init__(
        self, speed_ratio: float, bone_indices: List[Tuple[int, int]], total_motion_files: int
    ):
        self.speed_ratio = speed_ratio
        self.bone_indices = bone_indices
        self.total_motion_files = total_motion_files
        self.completed_motion_files = 0
        self.plotter = pv.Plotter(
            shape=(BatchRows + 2, BatchColumns),
            groups=[(ProgressRow, np.s_[:])],
            row_weights=[1.0] * BatchRows + [0.25, 0.5],
            border=False,
            window_size=(1600, 1200),
            title="Interactive Batch Motion Filter",
        )
        self.panels: List[MotionPanel] = []
        self.decisions: List[Optional[bool]] = []
        self.apply_requested = False
        self.skip_confirmation_pending = False
        self.closed = False
        self.shown = False
        self.selection_mode = True
        self.mouse_down: Optional[Tuple[int, int]] = None
        self.playback_seconds = 0.0
        self.last_update_time = 0.0
        self.progress_fill: Optional[pv.PolyData] = None
        self._add_controls()
        self.plotter.add_key_event("a", self._request_apply)
        self.plotter.add_key_event("s", self._select_mode)
        self.plotter.add_key_event("d", self._discard_mode)
        self.plotter.add_key_event("equal", self._increase_speed)
        self.plotter.add_key_event("minus", self._decrease_speed)
        self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_left_press)
        self.plotter.iren.add_observer("LeftButtonReleaseEvent", self._on_left_release)
        self.plotter.iren.add_observer("ExitEvent", self._on_exit)

    def _add_controls(self) -> None:
        for column in range(BatchColumns):
            self.plotter.subplot(ControlRow, column)
            self.plotter.set_background("#202124")

        self.plotter.subplot(ControlRow, 0)
        self.plotter.add_checkbox_button_widget(
            self._on_apply_button,
            value=False,
            position=(25, 25),
            size=45,
            border_size=3,
            color_on="#2f80ed",
            color_off="#2f80ed",
            background_color="#202124",
        )
        self.plotter.subplot(ControlRow, 2)
        self.plotter.add_text(
            "ALL SELECT\nClick the green button\nS: select mode, D: discard mode",
            position="upper_left",
            font_size=12,
            color="white",
        )
        self.plotter.add_checkbox_button_widget(
            self._on_select_all_button,
            value=False,
            position=(825, 25),
            size=45,
            border_size=3,
            color_on="#27ae60",
            color_off="#27ae60",
            background_color="#202124",
        )
        self.plotter.subplot(ControlRow, 3)
        self.plotter.add_text(
            "ALL DISCARD\nClick the red button\nLeft-click: mark, Drag: change view",
            position="upper_left",
            font_size=12,
            color="white",
        )
        self.plotter.add_checkbox_button_widget(
            self._on_discard_all_button,
            value=False,
            position=(1225, 25),
            size=45,
            border_size=3,
            color_on="#e74c3c",
            color_off="#e74c3c",
            background_color="#202124",
        )
        self._add_progress_bar()
        self._update_apply_label(render=False)
        self._update_speed_label()

    def _add_progress_bar(self) -> None:
        self.plotter.subplot(ProgressRow, 0)
        self.plotter.set_background("#202124")
        self.plotter.add_mesh(
            pv.Plane(center=(6.0, 0.0, 0.0), i_size=12.0, j_size=0.6),
            color="#3c4043",
            lighting=False,
            reset_camera=False,
        )
        self.progress_fill = pv.Plane(center=(0.0005, 0.0, 0.0), i_size=0.001, j_size=0.6)
        self.plotter.add_mesh(
            self.progress_fill,
            color="#27ae60",
            lighting=False,
            reset_camera=False,
        )
        self.plotter.reset_camera()
        self.plotter.view_xy()
        self.plotter.camera.zoom(20.0)
        self._update_progress(render=False)

    def _update_progress(self, render: bool = True) -> None:
        if self.progress_fill is None:
            return
        total = self.total_motion_files
        completed = self.completed_motion_files
        width = max(12.0 * completed / total, 0.001) if total else 0.001
        self.progress_fill.points = pv.Plane(
            center=(width / 2.0, 0.0, 0.0), i_size=width, j_size=0.6
        ).points
        self.plotter.subplot(ProgressRow, 0)
        self.plotter.add_text(
            f"Completed: {completed}/{total}    To go: {total - completed}",
            position="upper_left",
            font_size=12,
            color="white",
            name="progress-label",
            render=False,
        )
        if render and self.shown:
            self.plotter.render()

    def set_completed_motion_files(self, completed_motion_files: int) -> None:
        self.completed_motion_files = min(max(completed_motion_files, 0), self.total_motion_files)
        self._update_progress()

    def _update_speed_label(self) -> None:
        self.plotter.subplot(ControlRow, 1)
        mode = "SELECT" if self.selection_mode else "DISCARD"
        self.plotter.add_text(
            f"Mode: {mode}\n[S] select  [D] discard\nSpeed: {self.speed_ratio:.2f}x\n[=] x1.2   [-] x0.8",
            position="upper_left",
            font_size=12,
            color="#27ae60" if self.selection_mode else "#e74c3c",
            name="mode-speed-label",
            render=False,
        )
        if self.shown:
            self.plotter.render()

    def _update_apply_label(self, render: bool = True) -> None:
        self.plotter.subplot(ControlRow, 0)
        unmarked = sum(decision is None for decision in self.decisions)
        if self.skip_confirmation_pending:
            text = f"WARNING: {unmarked} UNMARKED\nClick Apply again to skip them"
            color = "#e74c3c"
        else:
            text = "APPLY\nClick the blue button or press A"
            color = "white"
        self.plotter.add_text(
            text,
            position="upper_left",
            font_size=12,
            color=color,
            name="apply-label",
            render=False,
        )
        if render and self.shown:
            self.plotter.render()

    def _on_apply_button(self, _: bool) -> None:
        self._request_apply()

    def _on_select_all_button(self, _: bool) -> None:
        self._set_all_decisions(True)

    def _on_discard_all_button(self, _: bool) -> None:
        self._set_all_decisions(False)

    def _request_apply(self) -> None:
        unmarked = sum(decision is None for decision in self.decisions)
        if unmarked and not self.skip_confirmation_pending:
            self.skip_confirmation_pending = True
            self._update_apply_label()
            print(
                f"WARNING: {unmarked} motion(s) are unmarked. "
                "Click Apply again to confirm skipping them."
            )
            return
        self.apply_requested = True

    def _select_mode(self) -> None:
        self._set_selection_mode(True)

    def _discard_mode(self) -> None:
        self._set_selection_mode(False)

    def _set_selection_mode(self, selected: bool) -> None:
        self.selection_mode = selected
        self._update_speed_label()

    def _increase_speed(self) -> None:
        self.speed_ratio *= 1.2
        self._update_speed_label()

    def _decrease_speed(self) -> None:
        self.speed_ratio *= 0.8
        self._update_speed_label()

    def _on_exit(self, *_: object) -> None:
        self.closed = True

    def _record_mouse_down(self) -> None:
        self.mouse_down = self.plotter.iren.get_event_position()

    def _on_left_press(self, *_: object) -> None:
        self._record_mouse_down()

    def _on_left_release(self, *_: object) -> None:
        self._finish_click()

    def _finish_click(self) -> None:
        start = self.mouse_down
        self.mouse_down = None
        if start is None:
            return
        end = self.plotter.iren.get_event_position()
        if max(abs(end[0] - start[0]), abs(end[1] - start[1])) > 5:
            return
        try:
            row, column = self.plotter.iren.get_event_subplot_loc()
        except RuntimeError:
            return
        panel_index = row * BatchColumns + column
        if panel_index < len(self.panels):
            self._set_decision(panel_index, self.selection_mode)

    def _set_decision(self, panel_index: int, selected: bool, render: bool = True) -> None:
        self.decisions[panel_index] = selected
        if self.skip_confirmation_pending:
            self.skip_confirmation_pending = False
            self._update_apply_label(render=False)
        row, column = divmod(panel_index, BatchColumns)
        self.plotter.subplot(row, column)
        renderer = self.plotter.renderer
        self._set_border(renderer, "#27ae60" if selected else "#e74c3c", 5)
        self._add_panel_label(panel_index)
        if render:
            self.plotter.render()

    def _set_all_decisions(self, selected: bool) -> None:
        for panel_index in range(len(self.panels)):
            self._set_decision(panel_index, selected, render=False)
        self.plotter.render()

    @staticmethod
    def _set_border(renderer: pv.Renderer, color: str, width: float) -> None:
        border_actor = getattr(renderer, "_border_actor", None)
        if border_actor is not None:
            renderer.RemoveViewProp(border_actor)
        renderer.add_border(color=color, width=width)

    def _add_panel_label(self, panel_index: int) -> None:
        row, column = divmod(panel_index, BatchColumns)
        self.plotter.subplot(row, column)
        decision = self.decisions[panel_index]
        status = "SELECTED" if decision else "DISCARDED" if decision is not None else "UNDECIDED"
        color = "#27ae60" if decision else "#e74c3c" if decision is not None else "white"
        self.plotter.add_text(
            f"{panel_index + 1}. {Path(self.panels[panel_index].rel_path).name}\n{status}",
            position="upper_left",
            font_size=10,
            color=color,
            name=f"panel-label-{panel_index}",
            render=False,
        )

    def _make_skeleton(self, positions: np.ndarray) -> pv.PolyData:
        lines = np.empty((len(self.bone_indices), 3), dtype=np.int64)
        lines[:, 0] = 2
        lines[:, 1:] = self.bone_indices
        skeleton = pv.PolyData(positions.copy())
        skeleton.lines = lines.ravel()
        return skeleton

    def _populate(self, motions: List[MotionPanel]) -> None:
        self.panels = motions
        self.decisions = [None] * len(motions)
        self.skip_confirmation_pending = False
        self._update_apply_label(render=False)
        for panel_index in range(BatchSize):
            row, column = divmod(panel_index, BatchColumns)
            self.plotter.subplot(row, column)
            renderer = self.plotter.renderer
            renderer.clear()
            self.plotter.set_background("#202124")
            self._set_border(renderer, "#808080", 2)
            if panel_index >= len(motions):
                self.plotter.add_text("No motion", position="upper_left", font_size=12, color="white")
                continue

            panel = motions[panel_index]
            panel.skeleton = self._make_skeleton(world_link_positions(panel, 0))
            panel.last_frame = 0
            self.plotter.add_mesh(
                pv.Plane(center=(0.0, 0.0, 0.0), i_size=6.0, j_size=6.0),
                color="#64748b",
                opacity=0.25,
                lighting=False,
                reset_camera=False,
            )
            self.plotter.add_mesh(
                panel.skeleton,
                color="#f2f2f2",
                line_width=3,
                render_lines_as_tubes=True,
                lighting=False,
                reset_camera=False,
            )
            self.plotter.reset_camera()
            self.plotter.view_isometric()
            self._add_panel_label(panel_index)

    def _animate(self) -> None:
        now = time.perf_counter()
        self.playback_seconds += (now - self.last_update_time) * self.speed_ratio
        self.last_update_time = now
        changed = False
        for panel in self.panels:
            frame = int(self.playback_seconds * max(panel.framerate, 1.0)) % len(panel.link_pos_b_frames)
            if frame == panel.last_frame:
                continue
            panel.skeleton.points = world_link_positions(panel, frame)
            panel.last_frame = frame
            changed = True
        if changed:
            self.plotter.render()

    def run_batch(self, motions: List[MotionPanel]) -> Optional[List[Optional[bool]]]:
        self.apply_requested = False
        self.playback_seconds = 0.0
        self._populate(motions)
        if not self.shown:
            self.plotter.show(interactive_update=True, auto_close=False)
            self.shown = True
        else:
            self.plotter.render()
        self.last_update_time = time.perf_counter()

        while not self.apply_requested and not self.closed:
            self._animate()
            self.plotter.update(stime=1, force_redraw=False)

        if self.closed:
            return None
        return self.decisions.copy()

    def close(self) -> None:
        self.plotter.close()


def save_motion(source_path: str, destination_path: str, resultant: str) -> None:
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    if resultant == "copy":
        shutil.copy2(source_path, destination_path)
    else:
        os.symlink(source_path, destination_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive batch motion data filter.")
    parser.add_argument("--urdf", required=True, help="Robot URDF for skeleton visualization.")
    parser.add_argument("--from-dir", required=True, help="Directory containing source motion files.")
    parser.add_argument("--ext", default="retargeted.npz", help="Motion filename suffix.")
    parser.add_argument("--to-dir", required=True, help="Directory for selected motion files.")
    parser.add_argument(
        "--resultant", choices=("copy", "symlink"), default="copy", help="Selected-file output mode."
    )
    parser.add_argument("--speed-ratio", type=float, default=1.0, help="Initial playback speed multiplier.")
    args = parser.parse_args()

    if not args.speed_ratio > 0:
        parser.error("--speed-ratio must be greater than zero")

    from_dir = os.path.abspath(args.from_dir)
    to_dir = os.path.abspath(args.to_dir)
    if not os.path.isdir(from_dir):
        parser.error(f"--from-dir does not exist: {from_dir}")
    if not os.path.isfile(args.urdf):
        parser.error(f"--urdf does not exist: {args.urdf}")

    bones = parse_urdf_skeleton(args.urdf)
    with open(args.urdf, "rb") as urdf_file:
        chain = pk.build_chain_from_urdf(urdf_file.read())
    joint_names = [joint.name for joint in chain.get_joints()]
    fk_link_names = list(chain.forward_kinematics(torch.zeros(1, len(joint_names))).keys())
    link_name_to_index = {name: index for index, name in enumerate(fk_link_names)}
    bone_indices = [(link_name_to_index[parent], link_name_to_index[child]) for parent, child in bones]

    scanned_files = collect_files(from_dir, args.ext)
    files = [path for path in scanned_files if not os.path.lexists(os.path.join(to_dir, path))]
    if not files:
        print("No motion files need filtering.")
        return
    random.shuffle(files)

    stats = {
        "saved": 0,
        "discarded": 0,
        "failed": 0,
        "remaining": len(files),
        "total": len(files),
    }
    viewer = BatchMotionViewer(args.speed_ratio, bone_indices, stats["total"])
    try:
        for batch_start in range(0, len(files), BatchSize):
            batch_paths = files[batch_start : batch_start + BatchSize]
            panels = []
            for rel_path in batch_paths:
                source_path = os.path.join(from_dir, rel_path)
                motion_data = load_motion_data(source_path, joint_names)
                if motion_data is None:
                    stats["failed"] += 1
                    stats["remaining"] -= 1
                    continue
                try:
                    link_pos_b_frames = compute_fk_all_frames(chain, motion_data.joint_pos, fk_link_names)
                except Exception as error:
                    print(f"  [WARN] Failed FK for {source_path}: {error}")
                    stats["failed"] += 1
                    stats["remaining"] -= 1
                    continue
                panels.append(
                    MotionPanel(
                        rel_path=rel_path,
                        source_path=source_path,
                        framerate=motion_data.framerate,
                        link_pos_b_frames=link_pos_b_frames,
                        base_pos_w_frames=motion_data.base_pos_w,
                        base_quat_w_frames=motion_data.base_quat_w,
                        skeleton=pv.PolyData(),
                    )
                )

            viewer.set_completed_motion_files(stats["total"] - stats["remaining"])
            if not panels:
                continue
            decisions = viewer.run_batch(panels)
            if decisions is None:
                break
            print("APPLYING BATCH:")
            for panel, selected in zip(panels, decisions):
                if selected:
                    destination_path = os.path.join(to_dir, panel.rel_path)
                    print(f"  {panel.source_path} -> {destination_path}")
                    save_motion(panel.source_path, destination_path, args.resultant)
                    stats["saved"] += 1
                else:
                    print(f"  SKIPPED: {panel.source_path}")
                    stats["discarded"] += 1
                stats["remaining"] -= 1
            viewer.set_completed_motion_files(stats["total"] - stats["remaining"])
    finally:
        viewer.close()

    print("FILTERING COMPLETE")
    print(f"  Saved: {stats['saved']}")
    print(f"  Discarded: {stats['discarded']}")
    print(f"  Failed: {stats['failed']}")
    if stats["remaining"] > 0:
        print(f"  Remaining: {stats['remaining']}")


if __name__ == "__main__":
    main()
