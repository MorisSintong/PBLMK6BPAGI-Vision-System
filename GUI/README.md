# GUI Module

This document explains the structure and responsibilities of components in the `GUI/` folder.

## Module Purpose

The GUI module is responsible for:
- Displaying the camera stream (RGB/Depth with auto-switch based on lighting)
- Receiving operator interaction (start/stop camera, select display mode)
- Displaying status/alert information to the operator
- Rendering a 90° FOV radar with real-time obstacle positions

## Folder Structure

- `src/` — main GUI widgets and logic
- `inc/` — UI constants and supporting styles

## Main Components (`src`)

| File | Function |
|---|---|
| `main_window.py` | Assembles the main layout, connects type-safe signals between panels, routes GUI configuration to the core pipeline, and assembles the 5-stage pipeline. |
| `depth_view.py` | Display area with 2 modes: RGB and Depth (Overlay removed). `setScaledContents` set once at init. Only updates labels for the currently visible page. Handles empty fallback frames via `QImage.isNull()`. |
| `controls_panel.py` | Main camera controls, dynamic alert distance settings, and view mode selection (Auto/RGB/Depth). Auto mode follows the `light_mode_changed` signal from CameraThread. |
| `alert_panel.py` | Displays object/distance info with color changes based on threshold status (DANGER/WARN/SAFE). Stylesheets only updated when status changes (pre-computed style dicts). |
| `radar_view.py` | 90° FOV radar widget with cached static background pixmap. Only sweep line and obstacle blips are redrawn per frame. Connected to data via `obstacles_ready` signal. |

## Configuration (`inc`)

| File | Function |
|---|---|
| `ui_config.py` | Global UI constants (app name, minimum window size, default thresholds, radar dimensions, zone labels, action labels). |
| `styles.py` | Global stylesheet + color constants (status colors, radar colors, infobox styles, text colors). |

## GUI Data Flow

1. User interacts with **Start/Stop** or **Threshold Sliders** in `ControlsPanel`.
2. `main_window.py` connects these inputs and routes them to `CameraThread` and `FrameProcessor`.
3. Memory-safe frames (`QImage`) and status notifications from the Vision pipeline are sent via emitted signals.
4. `DepthView` checks image buffer integrity and renders visual overlay (HUD bounding box) to the screen. Only labels for the visible page are updated.
5. `RadarView` and `AlertPanel` update the UI in real-time from spatial obstacle data. Radar uses cached background, AlertPanel only updates stylesheets when status changes.

## Performance Optimizations

| Optimization | File | Impact |
|---|---|---|
| Cached static background pixmap | `radar_view.py` | ~80% less paint work (rings, labels, FOV lines pre-rendered) |
| Status-change-only stylesheets | `alert_panel.py` | 6→0 style recalcs/frame in steady state |
| setScaledContents once in init | `depth_view.py` | 6 fewer layout passes/frame |
| Visible-only label updates | `depth_view.py` | 2 fewer pixmap sets/frame |
