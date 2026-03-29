"""
Render NFS predictions as videos, then hstack with existing GT videos from dataset_vis.

Step 1: Render NFS prediction npy → mp4
Step 2: ffmpeg hstack GT.mp4 | NFS.mp4 → combined.mp4

Usage:
    python vis_nfs_predictions.py --split test
    python vis_nfs_predictions.py --split train
"""
import os, sys, glob, subprocess, argparse
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NeuralFacialAnimation'))
from scripts.render_dataset import render_frame_to_array


def render_seq_array_to_video(verts_seq, faces, video_path, fps=30, size=3, dpi=100, label=''):
    """Render [T, V, 3] array directly to mp4."""
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    img0 = render_frame_to_array(verts_seq[0], faces, 0, frame_label=label, size=size, dpi=dpi)
    h, w = img0.shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    writer.write(cv2.cvtColor(img0, cv2.COLOR_RGB2BGR))
    for i in range(1, len(verts_seq)):
        img = render_frame_to_array(verts_seq[i], faces, i, frame_label=label, size=size, dpi=dpi)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()


def find_gt_video(gt_vis_dir, id_name, seq_name):
    """Find matching GT video from dataset_vis. Handles naming differences."""
    id_short = id_name.replace('--', '-')
    # Try exact match
    candidates = [
        f"{id_short}__{seq_name}.mp4",
        f"{id_short}__EXP_{seq_name}.mp4",
        f"{id_short}__{seq_name.replace('EXP_', '')}.mp4",
    ]
    for c in candidates:
        path = os.path.join(gt_vis_dir, c)
        if os.path.exists(path):
            return path
    # Fuzzy: search for seq_name in filename
    for f in os.listdir(gt_vis_dir):
        if seq_name in f and f.endswith('.mp4'):
            return os.path.join(gt_vis_dir, f)
    return None


def hstack_videos(gt_path, nfs_path, out_path):
    """Combine GT and NFS videos side by side using ffmpeg."""
    cmd = [
        'ffmpeg', '-y',
        '-i', gt_path, '-i', nfs_path,
        '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
        '-map', '[v]',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-shortest',
        out_path
    ]
    subprocess.run(cmd, capture_output=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default="/data2/inyup/nfs_predictions/mf_ROM")
    parser.add_argument("--gt_vis_dir", default="/source/inyup/NeuralFacialAnimation/dataset_vis/mf_ROM")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_dir", default="/data2/inyup/nfs_predictions/mf_ROM/vis")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # Faces
    std = np.load('/source/inyup/NeuralFacialAnimation/utils/mf/standardization.npy',
                  allow_pickle=True).item()
    faces = np.array(std['new_f'], dtype=np.int32)

    split_dir = os.path.join(args.pred_dir, args.split)
    gt_split_vis = os.path.join(args.gt_vis_dir, args.split)
    nfs_vid_dir = os.path.join(args.out_dir, args.split, 'nfs_only')
    combined_dir = os.path.join(args.out_dir, args.split, 'gt_vs_nfs')
    os.makedirs(nfs_vid_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    id_dirs = sorted([d for d in os.listdir(split_dir)
                      if os.path.isdir(os.path.join(split_dir, d))])

    for id_name in id_dirs:
        id_path = os.path.join(split_dir, id_name)
        seq_files = sorted([f for f in os.listdir(id_path)
                            if f.endswith('.npy') and f != 'rig'])

        for seq_file in tqdm(seq_files, desc=f'{id_name}'):
            seq_name = seq_file.replace('.npy', '')
            id_short = id_name.replace('--', '-')

            # Skip if combined already exists
            combined_path = os.path.join(combined_dir, f"{id_short}__{seq_name}.mp4")
            if os.path.exists(combined_path):
                continue

            # Step 1: Render NFS prediction video
            nfs_vid_path = os.path.join(nfs_vid_dir, f"{id_short}__{seq_name}.mp4")
            if not os.path.exists(nfs_vid_path):
                pred_verts = np.load(os.path.join(id_path, seq_file))
                render_seq_array_to_video(pred_verts, faces, nfs_vid_path,
                                          fps=args.fps, label=f'NFS {seq_name[:25]}')

            # Step 2: Find GT video and hstack
            gt_vid = find_gt_video(gt_split_vis, id_name, seq_name)
            if gt_vid:
                hstack_videos(gt_vid, nfs_vid_path, combined_path)
            else:
                print(f"  [SKIP] No GT video: {id_name}/{seq_name}")

    print("Done!")
