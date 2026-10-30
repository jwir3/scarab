import numpy as np
import matplotlib.pyplot as plt
from corrugated_root_solver import Waveguide # type: ignore

def plot_waveguide_boundary(guide):
    # This runs from 0 - 4pi
    x_coords = np.linspace(0, 2.0 * guide.period(), 500)

    # 3. Compute the boundary using the compiled Rust method
    y_coords = [guide.boundary(x) for x in x_coords]

    # 4. Render the plot
    plt.plot(x_coords, y_coords, label="Corrugated Ceiling", color='blue')
    plt.axhline(0, color='black', linewidth=2, label="Flat Floor")
    plt.xlabel("Horizontal Position (x)")
    plt.ylabel("Height (y)")
    plt.title("Dynamical Waveguide Geometry")
    plt.legend()
    plt.grid(True)

def plot_trajectory(guide, x_start, theta_n_deg=45.0, iters=1):
    # Generate the intersection between the floor ray and the waveguide boundary
    theta_start = theta_n_deg * (np.pi / 180.0)

    (s_np1, x_np1_raw, x_np1, theta_np1, y_np1) = guide.iterate_map(False, x_start, theta_start)

    if x_np1_raw > guide.right_boundary():
        # Compute the y-coordinate where the ray crosses the right boundary
        slope = np.tan(theta_start)
        y_cross = slope * (guide.right_boundary() - x_start)

        # Draw segment 1: Floor to right boundary
        plt.plot([x_start, guide.right_boundary()], [0, y_cross], 'b-')

        # Draw segment 2: Left boundary to wrapped landing point on ceiling
        plt.plot([0, x_np1], [y_cross, y_np1], 'b-', label="Wrapped Ray")

    print(f"Intersection point: x = {x_np1}, y = {guide.boundary(x_np1)}")
    print(f"theta_np1: {theta_np1}")
    plt.plot(x_start, 0, 'ro', label="Floor Starting Point")

    plt.plot(x_np1, y_np1, 'go', label="Intersection with Waveguide")

    # Now, iterate again
    # (s_np2, x_np2, theta_np2, y_np2) = guide.iterate_map(s_np1, x_np1, theta_np1)
    # if not s_np2:
    #     y_np2 = 0.0

    # print(f"y_np2: {y_np2}")
    # plt.plot(x_np2, y_np2, 'ro', label="Intersection with Floor")
    # plt.plot([x_np1, x_np2], [y_np1, y_np2], 'b-', label="Ray from Waveguide to Floor")

    # custom_dtype = np.dtype([('col1', '?'), ('col2', 'f8'), ('col3', 'f8'), ('col4', 'f8')])

    # # 2. Create the empty structured array
    # points = np.empty(iters, dtype=custom_dtype)

    # # How to populate it:
    # points[0] = (False, x_start, theta_start, 0.0)
    # i = 0
    # for n in points:
    #     i = i+1
    #     if i < iters:
    #         points[i] = guide.iterate_map(bool(n[0]), n[1], n[2])

    # # print(points)
    # plt.plot(points[0][1], points[0][3], 'ro')
    # plt.plot(points[1][1], points[1][3], 'ro')

if __name__ == "__main__":
    y_0 = 2.0
    d = 0.1
    k = np.pi / 2.0
    guide = Waveguide(y_0, d, k)

    plot_waveguide_boundary(guide)
    plot_trajectory(guide, x_start=7.0, theta_n_deg=45.0, iters=3)

    plt.xlim(0, 2.0 * guide.period())

    plt.show()