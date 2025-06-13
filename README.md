# analysis-Video
This Python code was developed as part of a collaborative thesis project with POC Sports, aiming to improve helmet safety for youth alpine ski racers by analyzing real-world crash footage.

The workflow starts with 2D position data extracted from race videos using Kinovea, where the skier’s head is manually tracked frame-by-frame. Each video is paired with metadata: frame rate, scale (pixel-to-meter), and slope angle.

To reduce noise and improve accuracy, the position data is smoothed using a Savitzky-Golay filter, which preserves the trajectory while minimizing tracking errors. Velocity is then computed as the derivative of the smoothed positions with respect to time, providing frame-by-frame velocity vectors.

Since the skier moves on a slope, the code decomposes the velocity into:

A parallel component, representing forward motion along the piste.

A perpendicular component, representing the direction of impact during a crash.

This decomposition uses trigonometric corrections based on the known slope angle of each piste.

The moment of impact is identified by a sudden drop in velocity, and the corresponding frame is marked as the impact frame. The estimated impact velocity is taken from the last frame before this drop.

The code outputs:

Plots of position and velocity over time.

The estimated impact velocity in m/s.

The impact frame.

Integration with head impact zone classification (e.g., front, back, lateral, top).

This method provides a robust way to quantify impact severity in uncontrolled outdoor conditions, where traditional sensor-based measurements are not feasible. The approach was validated against GPS data and found to be more accurate than simplified geometric models used in previous literature.

By applying this method to a large dataset of youth ski crashes, we could extract meaningful patterns in impact velocity and location, supporting the development of better, age-specific helmet designs.
