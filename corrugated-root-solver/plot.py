import numpy as np
import matplotlib.pyplot as plt
from corrugated_root_solver import Waveguide # type: ignore

# Debugging flag to toggle the display of the normal vectors at specific x-coordinates on the
# waveguide boundary. When set to True, the normal vectors will be plotted for visual inspection.
DEBUG_NORMALS = True

# The number of periods of guide.period() that will be graphically displayed in the
# output graph. This is important because it is used for the wrapping code.
GRAPHICAL_PERIODS = 2

def plot_normal_at_x(guide, x_normal=0.0):
    y_normal = guide.boundary(x_normal)
    n = np.asarray(guide.get_normal_at_x(x_normal), dtype=float) # get the normal vector at x_normal
    n /= np.linalg.norm(n) # normalize
    normal_half_length = 0.5

    plt.plot(
        [x_normal, x_normal + normal_half_length * n[0]],
        [y_normal, y_normal + normal_half_length * n[1]],
        "y--",
        label="Normal",
    )
    plt.plot(x_normal, y_normal, "yo")

def plot_waveguide_boundary(guide):
    x_coords = np.linspace(0, GRAPHICAL_PERIODS * guide.period(), 500)

    # 3. Compute the boundary using the compiled Rust method
    y_coords = [guide.boundary(x) for x in x_coords]

    # 4. Render the plot
    plt.plot(x_coords, y_coords, label="Corrugated Ceiling", color='black')
    plt.axhline(0, color='black', linewidth=2, label="Flat Floor")
    plt.xlabel("Horizontal Position (x)")
    plt.ylabel("Height (y)")
    plt.title("Dynamical Waveguide Geometry")
    # plt.legend()
    plt.grid(True)

def plot_wrapped_ray(guide, x_start, y_start, x_end, theta, y_end):
    # Ensure wrapping maps perfectly to your defined graphical bounds
    width = GRAPHICAL_PERIODS * guide.period()

    slope = np.tan(theta)
    step_dir = 1 if x_end > x_start else -1
    label = "Wrapped Ray"

    # Calculate which periodic 'tile' the start and end points fall into
    start_tile = int(np.floor(x_start / width))
    end_tile = int(np.floor(x_end / width))

    # 0. Base Case: The ray does not cross a visual boundary
    if start_tile == end_tile:
        plt.plot([x_start % width, x_end % width], [y_start, y_end], 'b-', label=label)
        return

    current_x_global = x_start
    current_y = y_start

    # 1. First Segment: Start point to the first boundary exit
    first_boundary_global = (start_tile + 1) * width if step_dir == 1 else start_tile * width
    y_cross = y_start + slope * (first_boundary_global - x_start)

    exit_x_plot = width if step_dir == 1 else 0.0
    plt.plot([x_start % width, exit_x_plot], [current_y, y_cross], 'b-', label=label)

    current_x_global = first_boundary_global
    current_y = y_cross

    # 2. Intermediate Segments: Ray fully crosses the unit cell from one side to the other
    for _ in range(abs(end_tile - start_tile) - 1):
        next_boundary_global = current_x_global + step_dir * width
        y_cross = y_start + slope * (next_boundary_global - x_start)

        enter_x_plot = 0.0 if step_dir == 1 else width
        exit_x_plot = width if step_dir == 1 else 0.0

        plt.plot([enter_x_plot, exit_x_plot], [current_y, y_cross], 'b-')

        current_x_global = next_boundary_global
        current_y = y_cross

    # 3. Final Segment: Re-entry to the wrapped landing point
    enter_x_plot = 0.0 if step_dir == 1 else width
    plt.plot([enter_x_plot, x_end % width], [current_y, y_end], 'b-')

def plot_trajectory(guide, x_start, theta_n_deg=45.0, iters=1):
    theta_start = theta_n_deg * (np.pi / 180.0)

    width = GRAPHICAL_PERIODS * guide.period()
    start_wrapped = x_start % width
    plt.plot(start_wrapped, 0, 'ro')

    s_start = False
    y_start = 0.0

    for i in range(0, iters):
        print(f"Initiating iteration: {i}")
        (s_np1, x_np1_raw, x_np1, theta_np1, y_np1) = guide.iterate_map(s_start, x_start, theta_start)

        # Pass y_start along with the raw, unwrapped coordinates to process the intersection heights
        plot_wrapped_ray(guide, x_start, y_start, x_np1_raw, theta_start, y_np1)

        # Plot the final intersection point at its wrapped location
        plt.plot(x_np1_raw % width, y_np1, 'go')

        print(f"Intersection point: raw_x = {x_np1_raw}, wrapped_x = {x_np1}, y = {y_np1}")
        print(f"theta_np1: {theta_np1}")

        s_start = s_np1
        x_start = x_np1
        theta_start = theta_np1
        y_start = y_np1

if __name__ == "__main__":
    y_0 = 2.0
    d = 0.73
    k = np.pi / 2.0
    guide = Waveguide(y_0, d, k, GRAPHICAL_PERIODS)

    plot_waveguide_boundary(guide)

    if DEBUG_NORMALS:
        for x in np.linspace(0.0, GRAPHICAL_PERIODS * guide.period(), 20):
            plot_normal_at_x(guide, x_normal=x)

    # plot_trajectory(guide, x_start=1.0, theta_n_deg=89, iters=2)
    # plot_trajectory(guide, x_start=3.0, theta_n_deg=45, iters=2)

    plt.xlim(0, GRAPHICAL_PERIODS * guide.period())

    plt.show()