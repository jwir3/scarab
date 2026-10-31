import numpy as np
import matplotlib.pyplot as plt
from corrugated_root_solver import Waveguide # type: ignore

# The number of periods of guide.period() that will be graphically displayed in the
# output graph. This is important because it is used for the wrapping code.
GRAPHICAL_PERIODS = 2

def plot_waveguide_boundary(guide):
    # This runs from 0 - 4pi
    x_coords = np.linspace(0, GRAPHICAL_PERIODS * guide.period(), 500)

    # 3. Compute the boundary using the compiled Rust method
    y_coords = [guide.boundary(x) for x in x_coords]

    # 4. Render the plot
    plt.plot(x_coords, y_coords, label="Corrugated Ceiling", color='blue')
    plt.axhline(0, color='black', linewidth=2, label="Flat Floor")
    plt.xlabel("Horizontal Position (x)")
    plt.ylabel("Height (y)")
    plt.title("Dynamical Waveguide Geometry")
    # plt.legend()
    plt.grid(True)

def plot_wrapped_ray(guide, x_start, x_end, theta, y_end):
    left = guide.left_boundary()
    right = guide.right_boundary()
    width = right - left
    if width <= 0:
        raise ValueError("The plotted waveguide must have a positive width")

    start_tile = int(np.floor((x_start - left) / width))
    start_x = x_start - start_tile * width
    end_x = x_end - start_tile * width
    slope = np.tan(theta)
    current_x = start_x
    current_y = 0.0
    label = "Wrapped Ray"

    if end_x > start_x:
        boundary = left + (start_tile + 1) * width
        while boundary < end_x:
            crossing_y = slope * (boundary - start_x)
            plt.plot([current_x, right], [current_y, crossing_y], 'b-', label=label)
            label = None
            start_tile += 1
            current_x = left
            current_y = crossing_y
            boundary += width
    elif end_x < start_x:
        boundary = left + start_tile * width
        if boundary == start_x:
            start_tile -= 1
            current_x = right
            boundary -= width
        while boundary > end_x:
            crossing_y = slope * (boundary - start_x)
            plt.plot([current_x, left], [current_y, crossing_y], 'b-', label=label)
            label = None
            start_tile -= 1
            current_x = right
            current_y = crossing_y
            boundary -= width

    final_x = end_x - start_tile * width
    plt.plot([current_x, final_x], [current_y, y_end], 'b-', label=label)

def plot_trajectory(guide, x_start, theta_n_deg=45.0, iters=1):
    # Generate the intersection between the floor ray and the waveguide boundary
    theta_start = theta_n_deg * (np.pi / 180.0)

    left = guide.left_boundary()
    right = guide.right_boundary()
    width = right - left
    if width <= 0:
        raise ValueError("The plotted waveguide must have a positive width")
    start_wrapped = left + (x_start - left) % width
    plt.plot(start_wrapped, 0, 'ro', label="Floor Starting Point")

    (s_np1, x_np1_raw, x_np1, theta_np1, y_np1) = guide.iterate_map(False, x_start, theta_start)
    plot_wrapped_ray(guide, x_start, x_np1_raw, theta_start, y_np1)
    plt.plot(x_np1, y_np1, 'go', label="Intersection with Waveguide")

    print(f"Intersection point: x = {x_np1}, y = {guide.boundary(x_np1)}")
    print(f"theta_np1: {theta_np1}")

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
    guide = Waveguide(y_0, d, k, GRAPHICAL_PERIODS)

    plot_waveguide_boundary(guide)
    plot_trajectory(guide, x_start=1.5, theta_n_deg=10, iters=3)

    plt.xlim(0, GRAPHICAL_PERIODS * guide.period())

    plt.show()