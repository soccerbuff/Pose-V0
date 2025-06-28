# -*- coding: utf-8 -*-
"""
Pose Animation Analyzer - Streamlit App (with Caching)
A web application for analyzing body pose movements from video files.
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import json
import tempfile
import os
from io import BytesIO
from typing import List, Dict, Tuple, Optional
import pandas as pd
from datetime import datetime
import shutil

# Try to import openai, but don't fail if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("OpenAI not available. AI Analysis features will be disabled.")

# Page configuration
st.set_page_config(
    page_title="Pose Animation Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.skeleton_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (19, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (20, 22),
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
        ]

    @staticmethod
    @st.cache_data(show_spinner=False)
    def extract_pose_data_cached(video_bytes, model_complexity, detection_confidence, tracking_confidence):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            temp_video_path = tmp_file.name
        mp_pose_local = mp.solutions.pose
        pose = mp_pose_local.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        cap = cv2.VideoCapture(temp_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_data = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            frame_data = {
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'landmarks': {},
                'valid': False
            }
            if results.pose_landmarks:
                frame_data['valid'] = True
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    frame_data['landmarks'][i] = {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    }
            pose_data.append(frame_data)
            frame_count += 1
        cap.release()
        os.unlink(temp_video_path)
        return pose_data, fps, total_frames

    @staticmethod
    @st.cache_data(show_spinner=False)
    def generate_annotated_video_cached(video_bytes, model_complexity, detection_confidence, tracking_confidence):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            temp_video_path = tmp_file.name
        annotated_video_path = temp_video_path.replace('.mp4', '_annotated.mp4')
        mp_pose_local = mp.solutions.pose
        pose = mp_pose_local.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))
        skeleton_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (19, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (20, 22),
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
        ]
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                for i, lm in enumerate(landmarks):
                    x_px = int(lm.x * width)
                    y_px = int(lm.y * height)
                    cv2.circle(frame, (x_px, y_px), 5, (0, 255, 0), -1)
                for start_idx, end_idx in skeleton_connections:
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        x1, y1 = int(landmarks[start_idx].x * width), int(landmarks[start_idx].y * height)
                        x2, y2 = int(landmarks[end_idx].x * width), int(landmarks[end_idx].y * height)
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            out.write(frame)
        cap.release()
        out.release()
        with open(annotated_video_path, 'rb') as f:
            video_bytes_out = f.read()
        os.remove(temp_video_path)
        os.remove(annotated_video_path)
        return video_bytes_out

    @staticmethod
    @st.cache_data(show_spinner=False)
    def calculate_statistics_cached(pose_data: List[Dict]):
        valid_frames = [frame for frame in pose_data if frame['valid']]
        if not valid_frames:
            return {}
        key_landmarks = [0, 15, 16, 27, 28, 11, 12]
        movement_data = {i: {'x': [], 'y': [], 'z': []} for i in key_landmarks}
        for frame_data in valid_frames:
            for landmark_idx in key_landmarks:
                if landmark_idx in frame_data['landmarks']:
                    landmark = frame_data['landmarks'][landmark_idx]
                    movement_data[landmark_idx]['x'].append(landmark['x'])
                    movement_data[landmark_idx]['y'].append(landmark['y'])
                    movement_data[landmark_idx]['z'].append(landmark['z'])
        stats = {
            'total_frames': len(pose_data),
            'valid_frames': len(valid_frames),
            'detection_rate': len(valid_frames) / len(pose_data) * 100,
            'video_duration': pose_data[-1]['timestamp'] if pose_data else 0,
            'landmark_movement': {}
        }
        landmark_names = ['Nose', 'Left Wrist', 'Right Wrist', 'Left Ankle', 'Right Ankle', 'Left Shoulder', 'Right Shoulder']
        for i, landmark_idx in enumerate(key_landmarks):
            if movement_data[landmark_idx]['x']:
                x_range = max(movement_data[landmark_idx]['x']) - min(movement_data[landmark_idx]['x'])
                y_range = max(movement_data[landmark_idx]['y']) - min(movement_data[landmark_idx]['y'])
                z_range = max(movement_data[landmark_idx]['z']) - min(movement_data[landmark_idx]['z'])
                stats['landmark_movement'][landmark_names[i]] = {
                    'x_range': x_range,
                    'y_range': y_range,
                    'z_range': z_range,
                    'total_movement': (x_range**2 + y_range**2 + z_range**2)**0.5
                }
        return stats

    @staticmethod
    @st.cache_data(show_spinner=False)
    def create_trajectory_plot_cached(pose_data: List[Dict], selected_landmarks: Optional[Dict[str, int]] = None) -> bytes:
        fig, ax = plt.subplots(figsize=(12, 8))
        all_landmarks = {
            'Nose': 0,
            'Left Wrist': 15,
            'Right Wrist': 16,
            'Left Ankle': 27,
            'Right Ankle': 28,
            'Left Shoulder': 11,
            'Right Shoulder': 12,
        }
        if selected_landmarks is None:
            selected_landmarks = all_landmarks
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        for i, (landmark_name, landmark_idx) in enumerate(selected_landmarks.items()):
            x_coords = []
            y_coords = []
            timestamps = []
            for frame_data in pose_data:
                if frame_data['valid'] and landmark_idx in frame_data['landmarks']:
                    landmark = frame_data['landmarks'][landmark_idx]
                    x_coords.append(landmark['x'])
                    y_coords.append(1 - landmark['y'])
                    timestamps.append(frame_data['timestamp'])
            if x_coords:
                ax.plot(x_coords, y_coords, 'o-', label=landmark_name, 
                       color=colors[i % len(colors)], markersize=4, alpha=0.7, linewidth=2)
        ax.set_xlabel('X Position (normalized)', fontsize=12)
        ax.set_ylabel('Y Position (normalized)', fontsize=12)
        ax.set_title('Landmark Trajectories Over Time', fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

def main():
    import tempfile
    import pandas as pd
    st.markdown('<h1 class="main-header">Pose Estimation and Analysis</h1>', unsafe_allow_html=True)
    
    # Fixed model parameters (no sidebar controls)
    model_complexity = 1
    detection_confidence = 0.60
    tracking_confidence = 0.70
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **Upload a video file to analyze body pose movements and generate animated skeleton plots.**
    This tool can be used to detect 33 body landmarks and creates:
    - Animated skeleton visualization
    - Movement trajectory analysis
    - Pose data statistics
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    if uploaded_file is not None:
        video_bytes = uploaded_file.getvalue()
        analyzer = PoseAnalyzer()
        # CACHED: Extract pose data
        pose_data, fps, total_frames = analyzer.extract_pose_data_cached(
            video_bytes, model_complexity, detection_confidence, tracking_confidence)
        valid_frames = [frame for frame in pose_data if frame['valid']]
        detection_rate = len(valid_frames) / len(pose_data) * 100
        tab4, tab1, tab2, tab3, tab_joint, tab_ai = st.tabs(["Detections", "Animation", "Trajectories", "Statistics", "Joint Analysis", "AI Analysis"])
        with tab4:
            st.header("Detections: Annotated Frames with Keypoints and Skeleton Overlay Video for Download")
            st.info("This tab shows the input video with pose keypoints and skeleton drawn on each frame.")
            # CACHED: Generate annotated video
            annotated_video_bytes = analyzer.generate_annotated_video_cached(
                video_bytes, model_complexity, detection_confidence, tracking_confidence)
            # Show frames with slider
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_vid:
                tmp_vid.write(annotated_video_bytes)
                temp_annotated_path = tmp_vid.name
            cap = cv2.VideoCapture(temp_annotated_path)
            total_annotated_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_slider = st.slider("Frame", min_value=0, max_value=max(0, total_annotated_frames-1), value=0, help="Step through annotated frames")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_height, frame_width = frame_rgb.shape[:2]
                if frame_width > 900:
                    scale = 900 / frame_width
                    new_width = int(frame_width * scale)
                    new_height = int(frame_height * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                st.image(frame_rgb, caption=f"Annotated Frame {frame_slider}")
            cap.release()
            st.download_button(
                label="Download Annotated Video",
                data=annotated_video_bytes,
                file_name="annotated_pose_video.mp4",
                mime="video/mp4"
            )
        with tab1:
            st.markdown("### Animated Skeleton Plot")
            st.info("This shows the skeleton movement. Use the slider to step through frames.")
            if valid_frames:
                frame_idx = st.slider(
                    "Frame", 
                    min_value=0, 
                    max_value=len(valid_frames)-1, 
                    value=0,
                    help="Drag to step through the animation frames"
                )
                current_frame = valid_frames[frame_idx]
                landmarks = current_frame['landmarks']
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_aspect('equal')
                ax.set_title(f'Body Pose - Frame {current_frame["frame"]} (Time: {current_frame["timestamp"]:.2f}s)', 
                           fontsize=16, fontweight='bold')
                ax.set_xlabel('X Position (normalized)', fontsize=12)
                ax.set_ylabel('Y Position (normalized)', fontsize=12)
                ax.grid(True, alpha=0.3)
                for i in range(33):
                    if i in landmarks:
                        x, y = landmarks[i]['x'], landmarks[i]['y']
                        y = 1 - y
                        ax.plot(x, y, 'o', color='#FFEAA7', markersize=8, alpha=0.8)
                for connection in analyzer.skeleton_connections:
                    start_idx, end_idx = connection
                    if start_idx in landmarks and end_idx in landmarks:
                        start_x, start_y = landmarks[start_idx]['x'], landmarks[start_idx]['y']
                        end_x, end_y = landmarks[end_idx]['x'], landmarks[end_idx]['y']
                        start_y = 1 - start_y
                        end_y = 1 - end_y
                        ax.plot([start_x, end_x], [start_y, end_y], 'k-', linewidth=2, alpha=0.7)
                st.pyplot(fig)
                plt.close(fig)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Frame", current_frame["frame"])
                with col2:
                    st.metric("Time", f"{current_frame['timestamp']:.2f}s")
                with col3:
                    st.metric("Progress", f"{frame_idx+1}/{len(valid_frames)}")
        with tab2:
            st.header("📈 Trajectory Analysis")
            st.markdown("View the movement paths of key body landmarks over time.")
            all_landmarks = {
                'Nose': 0,
                'Left Wrist': 15,
                'Right Wrist': 16,
                'Left Ankle': 27,
                'Right Ankle': 28,
                'Left Shoulder': 11,
                'Right Shoulder': 12,
            }
            selected = st.multiselect(
                "Select landmarks to show:",
                list(all_landmarks.keys()),
                default=list(all_landmarks.keys()),
                help="Choose which body parts' trajectories to display"
            )
            selected_landmarks = {k: all_landmarks[k] for k in selected}
            # CACHED: Trajectory plot
            trajectory_png = analyzer.create_trajectory_plot_cached(pose_data, selected_landmarks)
            st.image(trajectory_png, caption="Landmark Trajectories")
            if st.button("Save Trajectory Plot"):
                st.download_button(
                    label="Download Trajectory Plot (PNG)",
                    data=trajectory_png,
                    file_name=f"pose_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
        with tab3:
            st.markdown("### Pose Analysis Statistics")
            col1, col2 = st.columns(2)
            # CACHED: Statistics
            stats = analyzer.calculate_statistics_cached(pose_data)
            with col1:
                st.markdown("#### Detection Statistics")
                st.write(f"- Total frames processed: {len(pose_data)}")
                st.write(f"- Frames with pose detection: {len(valid_frames)}")
                st.write(f"- Detection rate: {detection_rate:.1f}%")
                st.write(f"- Video duration: {len(pose_data)/fps:.2f} seconds")
            with col2:
                st.markdown("#### Model Settings")
                st.write(f"- Model complexity: {model_complexity}")
                st.write(f"- Detection confidence: {detection_confidence}")
                st.write(f"- Tracking confidence: {tracking_confidence}")
                st.write(f"- Video FPS: {fps}")
            st.markdown("#### Export Data")
            if st.button("Export Pose Data (JSON)"):
                json_data = json.dumps(pose_data, indent=2)
                st.download_button(
                    label="Download Pose Data",
                    data=json_data,
                    file_name="pose_data.json",
                    mime="application/json"
                )
        with tab_joint:
            st.markdown("<h2>Joint Analysis</h2>", unsafe_allow_html=True)
            sport = st.selectbox("Select Sport", ["Archery"], key="joint_sport")
            if sport == "Archery":
                st.info("Archery joint metrics: right elbow angle, trunk lean, bow arm extension, bow arm elevation, stance width.")
                # Helper functions
                def angle_between_points(a, b, c):
                    # Angle at b (in degrees)
                    ba = np.array([a[0]-b[0], a[1]-b[1]])
                    bc = np.array([c[0]-b[0], c[1]-b[1]])
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
                    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                    return np.degrees(angle)
                def vector_angle_deg(v1, v2):
                    v1 = np.array(v1)
                    v2 = np.array(v2)
                    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                    return np.degrees(angle)
                # Landmark indices
                idx = {
                    'right_shoulder': 12, 'right_elbow': 14, 'right_wrist': 16,
                    'left_shoulder': 11, 'left_elbow': 13, 'left_wrist': 15,
                    'right_hip': 24, 'left_hip': 23,
                    'right_ankle': 28, 'left_ankle': 27
                }
                # Prepare lists
                right_elbow_angles = []
                trunk_leans = []
                left_elbow_angles = []
                bow_arm_elevations = []
                stance_widths = []
                times = []
                for frame in valid_frames:
                    lm = frame['landmarks']
                    t = frame['timestamp']
                    times.append(t)
                    # Right elbow angle
                    if all(i in lm for i in [idx['right_shoulder'], idx['right_elbow'], idx['right_wrist']]):
                        a = (lm[idx['right_shoulder']]['x'], lm[idx['right_shoulder']]['y'])
                        b = (lm[idx['right_elbow']]['x'], lm[idx['right_elbow']]['y'])
                        c = (lm[idx['right_wrist']]['x'], lm[idx['right_wrist']]['y'])
                        right_elbow_angles.append(angle_between_points(a, b, c))
                    else:
                        right_elbow_angles.append(np.nan)
                    # Trunk lean (angle between right shoulder-hip and vertical)
                    if all(i in lm for i in [idx['right_shoulder'], idx['right_hip']]):
                        shoulder = np.array([lm[idx['right_shoulder']]['x'], lm[idx['right_shoulder']]['y']])
                        hip = np.array([lm[idx['right_hip']]['x'], lm[idx['right_hip']]['y']])
                        vec = shoulder - hip
                        vertical = np.array([0, -1])
                        trunk_leans.append(vector_angle_deg(vec, vertical))
                    else:
                        trunk_leans.append(np.nan)
                    # Bow arm extension (left elbow angle)
                    if all(i in lm for i in [idx['left_shoulder'], idx['left_elbow'], idx['left_wrist']]):
                        a = (lm[idx['left_shoulder']]['x'], lm[idx['left_shoulder']]['y'])
                        b = (lm[idx['left_elbow']]['x'], lm[idx['left_elbow']]['y'])
                        c = (lm[idx['left_wrist']]['x'], lm[idx['left_wrist']]['y'])
                        left_elbow_angles.append(angle_between_points(a, b, c))
                    else:
                        left_elbow_angles.append(np.nan)
                    # Bow arm elevation (angle of left upper arm to horizontal)
                    if all(i in lm for i in [idx['left_shoulder'], idx['left_elbow']]):
                        shoulder = np.array([lm[idx['left_shoulder']]['x'], lm[idx['left_shoulder']]['y']])
                        elbow = np.array([lm[idx['left_elbow']]['x'], lm[idx['left_elbow']]['y']])
                        vec = elbow - shoulder
                        horizontal = np.array([1, 0])
                        bow_arm_elevations.append(vector_angle_deg(vec, horizontal))
                    else:
                        bow_arm_elevations.append(np.nan)
                    # Stance width (distance between ankles)
                    if all(i in lm for i in [idx['left_ankle'], idx['right_ankle']]):
                        la = np.array([lm[idx['left_ankle']]['x'], lm[idx['left_ankle']]['y']])
                        ra = np.array([lm[idx['right_ankle']]['x'], lm[idx['right_ankle']]['y']])
                        stance_widths.append(np.linalg.norm(la - ra))
                    else:
                        stance_widths.append(np.nan)
                # Show current values
                st.subheader("Current Frame Metrics")
                if len(times) > 0:
                    idx_cur = -1
                    st.metric("Right Elbow Angle (deg)", f"{right_elbow_angles[idx_cur]:.1f}")
                    st.metric("Trunk Lean (deg)", f"{trunk_leans[idx_cur]:.1f}")
                    st.metric("Bow Arm Extension (deg)", f"{left_elbow_angles[idx_cur]:.1f}")
                    st.metric("Bow Arm Elevation (deg)", f"{bow_arm_elevations[idx_cur]:.1f}")
                    st.metric("Stance Width (norm)", f"{stance_widths[idx_cur]:.3f}")
                # Plot progression
                st.subheader("Progression Over Time")
                fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
                axs[0].plot(times, right_elbow_angles, label="Right Elbow Angle")
                axs[0].set_ylabel("Elbow (deg)")
                axs[0].legend()
                axs[1].plot(times, trunk_leans, label="Trunk Lean")
                axs[1].set_ylabel("Trunk (deg)")
                axs[1].legend()
                axs[2].plot(times, left_elbow_angles, label="Bow Arm Extension")
                axs[2].set_ylabel("Bow Arm (deg)")
                axs[2].legend()
                axs[3].plot(times, bow_arm_elevations, label="Bow Arm Elevation")
                axs[3].set_ylabel("Elevation (deg)")
                axs[3].legend()
                axs[4].plot(times, stance_widths, label="Stance Width")
                axs[4].set_ylabel("Stance (norm)")
                axs[4].set_xlabel("Time (s)")
                axs[4].legend()
                st.pyplot(fig)
                plt.close(fig)
                # Download joint angles as CSV
                joint_df = pd.DataFrame({
                    "Time (s)": times,
                    "Right Elbow Angle (deg)": right_elbow_angles,
                    "Trunk Lean (deg)": trunk_leans,
                    "Bow Arm Extension (deg)": left_elbow_angles,
                    "Bow Arm Elevation (deg)": bow_arm_elevations,
                    "Stance Width (norm)": stance_widths
                })
                st.download_button("Download Joint Angles (CSV)", joint_df.to_csv(index=False), file_name="joint_angles.csv", mime="text/csv")
                # Download every 25th frame as JSON
                subsample = list(range(0, len(joint_df), 25))
                joint_json_25 = joint_df.iloc[subsample].to_dict(orient='records')
                st.download_button("Download Joint Angles (JSON, every 25th frame)", json.dumps(joint_json_25, indent=2), file_name="joint_angles_25.json", mime="application/json")
        with tab_ai:
            st.markdown("<h2>AI-Powered Biomechanics Analysis</h2>", unsafe_allow_html=True)
            if not OPENAI_AVAILABLE:
                st.error("OpenAI is not available. Please install the openai package to use this feature.")
                st.code("pip install openai")
            else:
                st.info("Upload your joint angle data as a JSON file and generate a biomechanics report using OpenAI.")
                uploaded_json = st.file_uploader("Upload Joint Angle Data (JSON only)", type=["json"], key="ai_json_upload")
                report_text = None
                if uploaded_json is not None:
                    try:
                        joint_data = json.load(uploaded_json)
                    except Exception as e:
                        st.error(f"Error reading JSON: {e}")
                        st.stop()
                    if st.button("Generate AI Report"):
                        with st.spinner("Generating report with OpenAI..."):
                            # Get API key from environment or user input
                            api_key = os.getenv("OPENAI_API_KEY")
                            if not api_key:
                                api_key = st.secrets.get("OPENAI_API_KEY", "")
                            if not api_key:
                                st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or add it to Streamlit secrets.")
                                st.stop()
                            
                            openai.api_key = api_key
                            prompt = f"""
You are an expert biomechanics coach specializing in archery. You have access to detailed joint angle data (right elbow angle, trunk lean, bow arm extension, bow arm elevation, stance width) for an athlete's shooting session, measured frame-by-frame.

Your tasks:
1. Analyze the provided joint angle data for patterns, consistency, and deviations.
2. Correlate the joint angle metrics with known biomechanical principles and research on archery performance.
3. Generate a detailed, athlete-friendly SWOT analysis:
    - Strengths: What is the athlete doing well biomechanically? (e.g., consistent elbow extension, stable stance, optimal trunk lean, etc.)
    - Weaknesses: Identify any inconsistencies, suboptimal angles, or deviations from elite technique.
    - Opportunities: Suggest specific, actionable routines, drills, or exercises to address weaknesses and improve performance (e.g., flexibility, strength, proprioception, or technical drills).
    - Threats: Highlight any biomechanical patterns that could increase injury risk or limit performance if not addressed.
4. Make the analysis engaging and motivating for the athlete, using positive and constructive language.
5. Include a summary table of key joint angle metrics (mean, std, min, max) and how they compare to typical values for elite archers (use web knowledge).
6. Conclude with a set of 3-5 personalized training routines or drills to address the athlete's main weaknesses.

Here is the joint angle data (in JSON):
{json.dumps(joint_data, indent=2)}

Please generate a detailed, structured PDF report with clear sections for each part of the analysis.
"""
                            try:
                                response = openai.ChatCompletion.create(
                                    model="gpt-4o",
                                    messages=[{"role": "user", "content": prompt}]
                                )
                                report_text = response.choices[0].message['content']
                            except Exception as e:
                                st.error(f"OpenAI API error: {e}")
                                st.stop()
                        if report_text:
                            st.markdown("### AI-Generated Report")
                            st.markdown(report_text)
                            # PDF download
                            try:
                                from fpdf import FPDF
                                import tempfile
                                pdf = FPDF()
                                pdf.add_page()
                                pdf.set_auto_page_break(auto=True, margin=15)
                                pdf.set_font("Arial", size=12)
                                for line in report_text.split('\n'):
                                    pdf.multi_cell(0, 10, line)
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                                    pdf.output(tmp_pdf.name)
                                    tmp_pdf.seek(0)
                                    st.download_button("Download PDF Report", tmp_pdf.read(), file_name="ai_biomechanics_report.pdf", mime="application/pdf")
                            except Exception as e:
                                st.error(f"PDF generation error: {e}")
    else:
        st.markdown("### Instructions")
        st.markdown("""
        1. **Upload a video file** using the file uploader above
        2. **Wait for processing** - the app will analyze each frame
        3. **View results** in the tabs
        """)
        st.markdown("### Tips for Best Results")
        st.markdown("""
        - Good lighting: Ensure the person is well-lit
        - Clear view: Person should be fully visible in the frame
        - Appropriate distance: Person should not be too far from camera
        - Stable camera: Avoid excessive camera movement
        - Single person: Works best with one person in frame
        """)

if __name__ == "__main__":
    main() 