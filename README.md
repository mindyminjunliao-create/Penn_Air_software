# PennAir Vision — Aerial Robotics Software Challenge

This repository contains my submission for the PennAir Aerial Robotics software
challenge: shape detection on a static image (Part 1), on video (Part 2), made
background-agnostic (Part 3), extended to 3D using a pinhole camera model
(Part 4), and wrapped into a ROS 2 package (Part 5).

- **Author:** Mindy
- **Dev environment:** macOS host → Ubuntu VM (UTM) → Docker container (ROS 2 Humble)

---

## Part A — How to Run This Project

### Repo layout

```
.
├── part1_static.py            # Part 1: shape detection on the static image
├── part2_video.py             # Part 2: same algorithm applied frame-by-frame to video
├── part3_hard_video.py        # Part 3: background-agnostic version, RGB + noise based
├── part4_3d.py                # Part 4: adds depth (Z) / 3D coordinates
├── src/
│   ├── detector.py            # Core detection algorithms shared by part1-4 scripts
│   └── utils.py
├── assets/                    # Input test files provided by PennAir
│   ├── PennAir 2024 App Static.png
│   ├── PennAir 2024 App Dynamic.mp4
│   └── PennAir 2024 App Dynamic Hard.mp4
├── outputs_hsv/               # Part 1 result + mask debug images
├── outputs_hsv_process/       # Part 1/3 step-by-step HSV pipeline debug images
├── outputs_rgb_part3/         # Part 3 RGB + noise-mask debug images
├── outputs/                   # Part 4 3D result images
├── pennair_vision/            # Part 5: ROS 2 package
│   ├── pennair_vision/
│   │   ├── detector.py            # Core detection algorithms (ROS-side copy, used by Parts 1-4 logic)
│   │   ├── camera_node.py         # Publishes video frames as a ROS 2 topic
│   │   └── detection_node.py      # Subscribes to frames, runs detection + 3D math, publishes pose
│   ├── launch/pennair_vision.launch.py
│   ├── package.xml / setup.py / setup.cfg
│   └── test/
└── README.md
```

### Part 1 — Shape detection on a static image

```bash
python3 part1_static.py
```

Reads `assets/PennAir 2024 App Static.png`, detects the shapes, draws their
outlines, marks their centers, and saves the annotated result + debug masks
to `outputs_hsv/`.

### Part 2 — Shape detection on video

```bash
python3 part2_video.py
```

Reads `assets/PennAir 2024 App Dynamic.mp4` and processes it **frame by
frame** (as a stream, not all at once), drawing outlines/centers on every
frame and writing an annotated output video. See it in action below
(demo recording also saved at `demo_videos/part2_video_demo.mp4`):

https://github.com/user-attachments/assets/f8aced29-44c7-43ea-9d74-3595fb826997


### Part 3 — Background-agnostic detection

```bash
python3 part3_hard_video.py
```

Same idea as Part 2, but works regardless of background color/texture. Tested
on `assets/PennAir 2024 App Dynamic Hard.mp4`; debug output for frame 0 is
saved to `outputs_rgb_part3/`. See the result on the hard video below
(demo recording also saved at `demo_videos/part3_hard_video_demo.mp4`):

https://github.com/user-attachments/assets/b1c32a32-a691-41c7-867c-0faed6bb084e

After using the small_area filter, raising the min_area parameter from 400 to 1200, some background noises are erased, though not completely, as seen bellow:


https://github.com/user-attachments/assets/3440a3f8-aeb3-482a-b95e-b6a53ff2d6df

### Part 4 — 3D coordinates

```bash
python3 part4_3d.py
```

Takes the Part 3 output, uses the known reference circle (radius = 10 in) plus
the given camera intrinsic matrix to compute depth Z and the 3D (X, Y, Z) of
every detected object's center, using the circle's pixel size as the depth
reference. Result saved to `outputs/part4_3d_result.png`.

### Part 5 — ROS 2 integration

1. Copy the package into a ROS 2 workspace:
   ```bash
   cp -r pennair_vision ~/ros2_ws/src/
   ```
2. **Before building**, open `pennair_vision/camera_node.py` and set
   `self.video_path` to wherever you placed the test video on your machine
   (it currently points at `/ros2_ws/assets/PennAir 2024 App Dynamic Hard.mp4`
   inside my own container).
3. Build the workspace:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select pennair_vision --symlink-install
   source install/setup.bash
   ```
4. Launch both nodes with one command:
   ```bash
   ros2 launch pennair_vision pennair_vision.launch.py
   ```
   This starts:
   - `camera_node` — streams the video file as `sensor_msgs/Image` on `/camera/image_raw`
   - `detection_node` — runs the Part 3 detection algorithm on each frame, does
     the Part 4 pinhole-camera depth calculation, and publishes a
     `geometry_msgs/PointStamped` on `/target_3d_pose`
5. In a second terminal, watch the live 3D output:
   ```bash
   source ~/ros2_ws/install/setup.bash
   ros2 topic echo /target_3d_pose
   ```

---

## Part B — Development Journey & Key Techniques

### Part 1 & 2 — from grayscale to a clean HSV mask

- **First attempt: plain grayscale threshold.** The lighting/contrast between
  the shapes and the grass background wasn't consistent enough — some
  shapes blended into the background and were missed or mis-detected.
  ![Early grayscale-based result](outputs_hsv/part1_result.png)
- **Second attempt: hand-picked HSV range on a `hsv` branch.** Worked better,
  but broke down whenever a shape's color was close to the background's.
- **Fix, with help from office hours:** used an interactive HSV-filter tool to
  find a clean HSV range that isolates the background well, producing a much
  better mask.
- **Contour extraction:** `cv2.inRange()` was tuned to match the *background*,
  so the raw mask is white where the background is and black where the shapes
  are. The mask is inverted (`cv2.bitwise_not`) before `findContours`, so the
  **shapes** become the white foreground blobs that `RETR_EXTERNAL` picks up.
- **Cleanup:** the initial mask was still noisy (small speckles from grass
  texture), which threw off the centroid calculation. The two images below
  show the mask before and after cleanup:

  | Mask before cleanup (noisy) | Mask after cleanup (clean contour) |
  |---|---|
  | ![Raw noisy mask](outputs_hsv/part1_result_mask.png) | ![Cleaned mask](outputs_hsv/part1_result_mask_2.png) |

  Morphological cleanup (open/close, see below) plus a minimum-contour-area
  filter removed the noise and gave clean, stable outlines and centers.

### Part 3 — background-agnostic detection (RGB + noise, not HSV)

Reusing the Part 1/2 HSV approach failed on the harder video for two reasons:
1. Gradient-shaded shapes sometimes got split into two separate blobs where
   the gradient crossed a hue boundary:
   ![HSV approach splits a gradient shape into two detections](outputs_hsv_process/4_final_overlay.png)
2. A fixed HSV range obviously can't generalize to an *arbitrary* background —
   hand-tuning a mask from a website isn't a real solution if the background
   color is unknown ahead of time.

New approach, on a `rgb_strategy` branch:
- **Background sampling:** on the first frame, sample a fixed-size patch and
  take the most common RGB combination as "the background" — this makes the
  algorithm work with *any* background, not just the black one in the test
  video.
- **Color mask:** per-pixel Euclidean distance from that background RGB;
  anything far enough away is foreground. The image below is the best color
  mask I was able to get on frame 0 after tuning the morphological kernel
  size to `(3, 3)` — beyond this point, further kernel tuning stopped making
  a difference, which is a limitation of the current approach that I'd want
  to revisit:
  ![Best achievable color mask, kernel (3,3)](outputs_rgb_part3/frame0_1_color_mask.png)
- **Noise mask:** compare each pixel's local high-frequency noise level
  against the background's noise profile — a shape with a *different* texture
  frequency than the background gets flagged as foreground even when its
  color is close to the background color, and this also helps stop a single
  gradient shape from being split into two.
- **Combine + clean:** the color mask and noise mask are OR'd together, then
  cleaned with morphological open/close before finding contours. This is the
  best result I was able to achieve on frame 0 with this approach:
  ![Best final outline/center result on frame 0](outputs_rgb_part3/frame0_4_result.png)
- **Known remaining issues (flagged for future work):** the outlines aren't
  perfectly tight to the object boundary, background noise isn't 100%
  eliminated, and two overlapping objects aren't yet separated correctly.
- **Follow-up fix:** even after morphological cleanup, some small background
  noise specks survived `MORPH_CLOSE` and grew just large enough to pass the
  old `min_area = 400` contour-area filter. Raised `min_area` to a tunable
  function parameter and set it to `1200` for this video, which removed most
  of the small false-positive contours. This only helps with noise that's
  smaller than real objects, though — noise blobs that happen to be similar
  in size to an actual object still pass the filter and can't be
  distinguished by area alone, which is why background noise isn't 100%
  eliminated yet.

### Part 4 — making it 3D

Part 3's detections feed directly into Part 4: the known reference circle
(10 in radius) gives a pixel-radius-to-real-world-size ratio, which — combined
with the camera's intrinsic matrix — is used to solve for depth (Z) via the
pinhole camera model, and then back-project each detected center's (u, v)
pixel coordinates into (X, Y, Z) in camera-frame coordinates.

### Part 5 — ROS 2 integration

- **Environment:** Mac → UTM (Ubuntu VM) → Docker (ROS 2 Humble), to keep the
  ROS 2 dependency stack clean and reproducible.
- **`detector.py`** holds the reusable detection functions so both the
  standalone scripts and the ROS 2 nodes call the same, single implementation.
- **`camera_node`** re-publishes the test video as a live `Image` topic (as if
  it were a real camera feed); **`detection_node`** subscribes, runs the Part
  3 detector, does the Part 4 3D math, and publishes `/target_3d_pose`.

**Problems hit along the way:**

| Problem | Symptom | Fix |
|---|---|---|
| `cv_bridge` encoding mismatch | `CvBridgeError: [8UC3] is not a color format` when converting frames between nodes | Standardized both publisher and subscriber on `passthrough` encoding instead of relying on an RGB/BGR conversion, which avoided the incompatibility entirely |
| Host vs. container confusion | `ros2: command not found` | The terminal was actually on the Ubuntu VM host, not inside the Docker container — fixed by `sudo docker exec -it <container_id> bash` and re-sourcing `/opt/ros/humble/setup.bash` |
| Launch file "missing" | `ros2 launch` couldn't find the launch file in the install `share` directory | `setup.py`'s `data_files` wasn't copying the `launch/` folder — added the missing `data_files` entry, cleared `build/`/`install/`, and rebuilt with `--symlink-install` |

### Key techniques used, end to end

- HSV color-space thresholding + mask inversion + contour extraction (Parts 1–2)
- Morphological cleanup: `MORPH_CLOSE` (fill gaps within a blob),
  `MORPH_OPEN` (strip small noise blobs), and `dilate` (merge sparse noise
  points) used at different stages depending on what needed fixing
- RGB color-distance masking + local noise-frequency comparison for
  background-agnostic segmentation (Part 3)
- Convex hull + image moments for robust outline/centroid extraction
- Pinhole camera model (intrinsic matrix + known reference object size) for
  monocular depth estimation (Part 4)
- ROS 2 nodes/topics/launch files, Docker-based reproducible environment (Part 5)

### Known limitations / future work

- Overlapping objects are not yet separated in Part 3.
- Contour edges aren't perfectly tight to object boundaries under heavy noise.
- Detection could be made more efficient for true real-time use (e.g.
  avoiding recomputation of the noise mask over the whole frame every frame).

### Part 6 — An idea I considered but haven't implemented: persistent homology for overlap handling

**Status: this is an idea I looked into conceptually, not something I've
implemented or tested for this project — I'm including it here to be
transparent about where my thinking is, not to claim results I don't have.**
My familiarity with the tool itself comes from a previous, unrelated project
where I used persistent homology to explore the topological structure of the
photon ring around a black hole — I haven't yet applied it to a real-time
object-detection pipeline like this one, which is a different enough setting
(static topological structure vs. frame-by-frame video) that I'd expect new
problems to show up, as outlined below.

The current Part 3 pipeline treats each frame's foreground mask as an
independent set of blobs: `findContours` gives you back whatever connected
components exist *in that single frame*, with no memory of what happened a
moment earlier. That's exactly why two overlapping objects merge into one
blob and get reported as a single detection — the algorithm has no concept
of "this blob used to be two things."

**What persistent homology is:** it's a tool from algebraic topology that
tracks how the topological features of a shape (mainly: the number of
separate connected pieces, called the 0th Betti number, and the number of
loops/holes, the 1st Betti number) appear and disappear as you gradually
change a threshold — this sequence of threshold levels is called a
*filtration*. Instead of looking at the mask at one fixed threshold, you
look at it across a whole range of thresholds and record, for every
topological feature, the threshold at which it is "born" and the threshold
at which it "dies" (merges with another feature or disappears). Plotting
these birth/death pairs gives a *persistence diagram*: features with a long
birth-death gap ("high persistence") are considered real structure, while
short-lived ones are treated as noise.

**Why I think it's relevant here:** if you build a filtration out of the
Part 3 distance-transform of the foreground mask (i.e., grow the mask
outward from its local peaks, the way `cv2.watershed` seeding does), two
overlapping objects would show up as **two separate connected components
that exist for a range of thresholds before merging into one** — persistent
homology would let me detect that merge event and recover the fact that
there were two distinct objects, even after they've become one connected
blob in the final binary mask. In principle this could give a more
principled alternative to just running a watershed split after the fact,
because the persistence diagram itself tells you *how confident* to be that
two components should really be treated as separate (via how persistent
each one is), rather than just guessing a peak-count.

**Problems I anticipate running into, if I were to actually build this**
(again — these are predictions based on how the math and the tooling work,
not things I've hit in practice):
- **Cost:** computing persistent homology (e.g. with libraries like GUDHI or
  Ripser) is significantly more expensive than a single `findContours` call,
  and Part 2/3 are supposed to process video frame-by-frame — this would
  likely need to run at a lower rate than every frame, or on a
  downsampled/cropped region only, to stay usable.
- **Choosing what to filter on:** the result depends heavily on whether the
  filtration is built from the raw mask, a distance transform, or grayscale
  intensity — a poor choice would likely surface topological "features" that
  are really just noise texture rather than real object boundaries, which is
  the same noise problem Part 3 already struggles with, just moved into a
  different math framework.
- **Getting back to pixels:** a persistence diagram tells you *how many*
  components existed and *when* they merged, but it doesn't directly hand
  you a pixel mask for "object A" vs. "object B" — I'd likely need to pair
  it with something like a seeded watershed at the pre-merge threshold to
  turn the topological answer back into an actual segmentation, which is an
  extra non-trivial step on top of the topology itself.
- **Tooling mismatch:** most persistent-homology libraries are built around
  point clouds / simplicial complexes rather than 2D image masks directly;
  I'd need to go through a cubical-complex representation (which some
  libraries, like GUDHI, do support) rather than the point-cloud APIs most
  tutorials use.

If I get time to actually try this, I'd want to prototype it on a single
static frame with two known overlapping shapes first, before trying to make
it work frame-by-frame on video.
