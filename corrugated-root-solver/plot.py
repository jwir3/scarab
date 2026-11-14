import numpy as np
import matplotlib.pyplot as plt
from corrugated_root_solver import Waveguide, PhaseSpace # type: ignore

Y_0 = 1.0
D = 0.005
# K = np.pi / 2.0
K = 1.0

# Debugging flag to toggle the display of the normal vectors at specific x-coordinates on the
# waveguide boundary. When set to True, the normal vectors will be plotted for visual inspection.
DEBUG_NORMALS = True

# The number of periods of guide.period() that will be graphically displayed in the
# output graph. This is important because it is used for the wrapping code.
GRAPHICAL_PERIODS = 2

# The number of segments within a range over theta and x
SEGMENTS=20

# The number of iterations that should be computed for each trajectory.
ITERATIONS=10_000

X_MIN=0
X_MAX=6.0

# Leonel 2007 uses a "shortcut" where negative gamma indicates an upward pointing ray. However, we
# can't directly put negative theta into our simulation because we're not using this shortcut.
# Instead, we break it into two sections:
# 171.9 deg - 179.9 deg
# 0.1 deg - 8.1 deg
LEONEL_THETA_MIN_1=0.1
LEONEL_THETA_MAX_1=8.1
LEONEL_THETA_MIN_2=171.9
LEONEL_THETA_MAX_2=179.9

LEONEL_REPRODUCTION=(D/Y_0) <= 0.001

THETA_MIN = 0.1
THETA_MAX = 179.9

DEBUG=False

## Some Utility Functions
def get_gamma_for_theta(theta, guide):
    # If the physical ray is traveling left (angle > 90 degrees / pi/2 radians)
    if theta > (np.pi / 2.0):
        # Translate back to the negative theoretical gamma
        gamma = (theta - np.pi) / guide.get_wavenumber()
    else:
        # Positive theoretical gamma
        gamma = theta / guide.get_wavenumber()

    return gamma

def get_Xn_for_x(x, guide):
    Xn = (x * guide.get_wavenumber()) % (2 * np.pi)
    return Xn

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
    y_coords = [guide.boundary(x) for x in x_coords]

    # Set up the figure with a custom width and height
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.canvas.manager.set_window_title("Trajectory Diagram") # type: ignore

    x_coords = np.linspace(0, GRAPHICAL_PERIODS * guide.period(), 500)
    y_coords = [guide.boundary(x) for x in x_coords]

    ax.plot(x_coords, y_coords, label="Corrugated Ceiling", color='black')
    ax.axhline(0, color='black', linewidth=2, label="Flat Floor")
    ax.set_xlim(0, GRAPHICAL_PERIODS * guide.period())
    ax.set_xlabel("Horizontal Position (x)")
    ax.set_ylabel("Height (y)")
    ax.set_title("Dynamical Waveguide Geometry")
    ax.grid(True)

    # Enforce true geometry without squashing the window
    ax.set_aspect('equal')

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


def compute_trajectory(guide, x_start, theta_n_deg=45.0, iters=1, should_plot=True):
    theta_start = theta_n_deg * (np.pi / 180.0)

    width = GRAPHICAL_PERIODS * guide.period()
    start_wrapped = x_start % width
    if should_plot:
        plt.plot(start_wrapped, 0, 'ro')

    s_start = False
    y_start = 0.0

    for i in range(0, iters):
        if DEBUG:
            print(f"Initiating iteration: {i}")

        iteration_result = guide.iterate_map(s_start, x_start, theta_start)
        if iteration_result == None:
            break

        (s_np1, x_np1_raw, x_np1, theta_np1, y_np1) = iteration_result

        # Pass y_start along with the raw, unwrapped coordinates to process the intersection heights
        if should_plot:
            plot_wrapped_ray(guide, x_start, y_start, x_np1_raw, theta_start, y_np1)

        # Plot the final intersection point at its wrapped location
        if should_plot:
            plt.plot(x_np1_raw % width, y_np1, 'go')

        # Plot the intersection point normal
        if DEBUG_NORMALS and y_np1 > 0 and should_plot:
            n = x_np1_raw % width
            plot_normal_at_x(guide, x_normal=n)

        if DEBUG:
            print(f"Intersection point: raw_x = {x_np1_raw}, wrapped_x = {x_np1}, y = {y_np1}")
            print(f"theta_np1: {theta_np1}")

        s_start = s_np1
        x_start = x_np1
        theta_start = theta_np1
        y_start = y_np1

def plot_trajectory_space(guide):
    plot_waveguide_boundary(guide)
    compute_trajectory(guide, x_start=1.3, theta_n_deg=45, iters=20, should_plot=True)
    # compute_trajectory(guide, x_start=6.33571276243665, theta_n_deg=-4.088981432330988 * 180/np.pi, iters=3)
    # compute_trajectory(guide, x_start=1.0, theta_n_deg=30, iters=2)
    # compute_trajectory(guide, x_start=3.0, theta_n_deg=25, iters=2)

def plot_phase_space(guide):
    # When we are reproducing Leonel's results, we need to do two separate things: First, we need to
    # encode the inputs as gamma and Xn. Because Leonel uses a "shortcut" where left-pointing rays
    # have a negative gamma, we also want to split this into two separate segments that are
    # evaluated, since our implementation doesn't allow for negative theta values (it considers them
    # to be going into the flat surface).
    #
    # Second, we need to only visualize the small space where the mixing occurs. As such, we
    # hard-code the dimensions of the graph to match Leonel's original specification.
    if LEONEL_REPRODUCTION:
        gamma_graph_min = -0.09
        gamma_graph_max = 0.09
        x_graph_min = X_MIN
        x_graph_max = X_MAX
        x_starts = np.linspace(X_MIN, X_MAX, SEGMENTS)
        theta_starts_deg_1 = np.linspace(LEONEL_THETA_MIN_1, LEONEL_THETA_MAX_1, SEGMENTS)
        theta_starts_deg_2 = np.linspace(LEONEL_THETA_MIN_2, LEONEL_THETA_MAX_2, SEGMENTS)

        for x_start in x_starts:
            for theta_start in theta_starts_deg_1:
                compute_trajectory(guide, x_start=x_start, theta_n_deg=theta_start, iters=ITERATIONS, should_plot=False)
            for theta_start in theta_starts_deg_2:
                compute_trajectory(guide, x_start=x_start, theta_n_deg=theta_start, iters=ITERATIONS, should_plot=False)
    else:
        gamma_range = guide.get_gamma_range()
        gamma_graph_min = gamma_range[0]
        gamma_graph_max = gamma_range[1]
        x_graph_min = X_MIN
        x_graph_max = X_MAX

        # Convert the dynamic theoretical gamma limits into physical degrees
        # (Using K to map gamma back to theta in radians, then converting to degrees)
        theta_max_degs = abs(gamma_graph_max) * guide.get_wavenumber() * (180.0 / np.pi)

        # Create concentrated seeds for the right and left halves of the visible phase space
        theta_starts_right = np.linspace(0.1, theta_max_degs, SEGMENTS)
        theta_starts_left = np.linspace(180.0 - theta_max_degs, 179.9, SEGMENTS)

        x_starts = np.linspace(X_MIN, X_MAX, SEGMENTS)

        for x_start in x_starts:
            for theta_start in theta_starts_right:
                compute_trajectory(guide, x_start=x_start, theta_n_deg=theta_start, iters=ITERATIONS, should_plot=False)
            for theta_start in theta_starts_left:
                compute_trajectory(guide, x_start=x_start, theta_n_deg=theta_start, iters=ITERATIONS, should_plot=False)
    phase_space = guide.get_phase_space()

    # We only want the x and theta coordinates for when we're intersecting the floor
    x_coords = phase_space.get_x_coordinates_for_slice(False)
    theta_coords = phase_space.get_theta_coordinates_for_slice(False)
    gammas = []
    Xs = []
    for x in x_coords:
        Xn = get_Xn_for_x(x, guide)
        Xs.append(Xn)

    for theta in theta_coords:
        gamma = get_gamma_for_theta(theta, guide)
        gammas.append(gamma)

    fig, ax = plt.subplots(figsize=(6,6))
    fig.canvas.manager.set_window_title("Phase Space Diagram") # type: ignore
    ax.set_title(f"Phase Space for " + "$\\delta$" + f"={D/Y_0}")
    ax.set_xlabel("$X_{n}$")
    ax.set_ylabel("$\\gamma$")

    # Compute the limits of what we're looking at
    ax.set_xlim(x_graph_min, x_graph_max)
    ax.set_ylim(gamma_graph_min, gamma_graph_max)
    ax.plot(Xs, gammas, "b.", markersize=.1)
    plt.show()

if __name__ == "__main__":
    #### START MAIN CODE ###
    guide = Waveguide(Y_0, D, K, GRAPHICAL_PERIODS)

    # plot_trajectory_space(guide)
    plot_phase_space(guide)

    ### END MAIN CODE ###

    plt.show()