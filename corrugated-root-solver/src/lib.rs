#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_local_definitions)]
use pyo3::prelude::*;
use roots::{SearchError, SimpleConvergency, find_root_brent};
use std::f64::consts::PI;
use vecmath::{Vector2, vec2_dot, vec2_normalized};

use log::{debug, info};

pub mod logger;

#[pyclass]
struct Waveguide {
    y0: f64,
    d: f64,
    k: f64,

    /// The number of periods in the graphical version of the waveguide.
    graphical_periods: u32,
}

#[pymethods]
impl Waveguide {
    #[new]
    pub fn new(y0: f64, d: f64, k: f64, graphical_periods: u32) -> Self {
        Self {
            y0,
            d,
            k,
            graphical_periods,
        }
    }

    pub fn graphical_periods(&self) -> f64 {
        self.graphical_periods as f64
    }

    pub fn boundary(&self, x: f64) -> f64 {
        self.y0 + self.d * (self.k * x).cos()
    }

    pub fn derivative(&self, x: f64) -> f64 {
        -self.d * self.k * (self.k * x).sin()
    }

    pub fn period(&self) -> f64 {
        2.0 * PI / self.k
    }

    pub fn left_boundary(&self) -> f64 {
        0.0
    }

    pub fn right_boundary(&self) -> f64 {
        2.0 * self.period()
    }

    pub fn get_normal_at_x(&self, x: f64) -> Vector2<f64> {
        // let y = self.boundary(x);
        let slope = self.derivative(x);
        let v: Vector2<f64> = [slope, -1.0];

        vec2_normalized(v)
    }

    pub fn get_root_finding_step_size(&self, theta: f64) -> f64 {
        // XXX_jwir3: In theory, I thought that a search space of width period/4.0 should be
        //            sufficient for all possible values of d, but for some reason, there exists
        //            large values of d such that a search space of period/16 width is required.
        //            This is interesting and should be investigated further.
        if theta.cos() > 0.0 {
            self.period() / 32.0
        } else {
            -self.period() / 32.0
        }
    }

    // Corresponds to the function I in the paper, which checks for self-intersection of the ray
    // with the waveguide boundary, but only with the current period of the wave (i.e. the current
    // wave crest).
    fn local_shadowing_indicator(&self, psi: f64, theta: f64) -> bool {
        // If the reflected ray points upwards into the ceiling, it is guaranteed
        // to cause local shadowing regardless of the tangent slope.
        if theta.sin() > 0.0 {
            return true;
        }

        if psi.tan() > theta.tan() && theta.cos() < 0.0 {
            true
        } else if psi.tan() < theta.tan() && theta.cos() > 0.0 {
            true
        } else {
            false
        }
    }

    // This is a ray from the floor, with an angle of theta_n, that intersects the waveguide at
    // some x point.
    pub fn intersect_from_floor(&self, x_a: f64, theta_n: f64) -> f64 {
        let root_seeking_closure =
            build_root_seeking_closure(0.0, theta_n, x_a, self.y0, self.d, self.k);

        let step_size = self.get_root_finding_step_size(theta_n);

        // Define the absolute physical limits of the collision zone
        let x_enter = x_a + ((self.y0 - self.d) / theta_n.tan());
        let x_exit = x_a + ((self.y0 + self.d) / theta_n.tan());

        // debug!("(x_enter, x_exit): ({}, {})", x_enter, x_exit);

        let mut x_curr = x_enter;
        let mut valid_bracket = None;
        let is_forward = theta_n.cos() > 0.0;

        // March the ray forward (or backward) until we pass the physical ceiling limit
        while if is_forward {
            x_curr < x_exit
        } else {
            x_curr > x_exit
        } {
            let mut x_next = x_curr + step_size;

            // Clamp the final step exactly to the exit boundary
            if is_forward {
                x_next = x_next.min(x_exit);
            } else {
                x_next = x_next.max(x_exit);
            }

            let r_curr = root_seeking_closure(x_curr);
            let r_next = root_seeking_closure(x_next);

            // A sign change indicates we have perfectly bracketed the first intersection
            if r_curr * r_next <= 0.0 {
                // Brent's method requires the first argument to be the smaller value
                let bound_lower = x_curr.min(x_next);
                let bound_upper = x_curr.max(x_next);
                valid_bracket = Some((bound_lower, bound_upper));
                break;
            }

            x_curr = x_next;
        }

        // Extract the bracket, panicking safely if the physics fail
        let (bound_lower, bound_upper) =
            valid_bracket.expect("Failed to bracket root within physical limits");

        debug!("[{}, {}]", bound_lower, bound_upper);

        let mut convergency = SimpleConvergency {
            eps: 1e-10,
            max_iter: 100,
        };

        find_root_brent(
            bound_lower,
            bound_upper,
            &root_seeking_closure,
            &mut convergency,
        )
        .unwrap()
    }

    pub fn intersect_from_ceiling(&self, x_a: f64, theta_n: f64, x_floor: f64) -> Option<f64> {
        let root_seeking_closure =
            build_root_seeking_closure(self.boundary(x_a), theta_n, x_a, self.y0, self.d, self.k);

        let step_size = self.get_root_finding_step_size(theta_n);

        let is_forward = theta_n.cos() > 0.0;

        // Step microscopically away from the starting coordinate to avoid bracketing the trivial root
        let epsilon = if is_forward { 1e-8 } else { -1e-8 };
        let mut x_curr = x_a + epsilon;

        // If the ray points upwards, it will never hit the floor. Bound the search
        // physically to one period in the direction of travel.
        let x_exit = if theta_n.sin() > 0.0 {
            if is_forward {
                x_a + self.period()
            } else {
                x_a - self.period()
            }
        } else {
            x_floor
        };

        let mut valid_bracket = None;

        // March the ray forward (or backward) until we pass the physical limit
        while if is_forward {
            x_curr < x_exit
        } else {
            x_curr > x_exit
        } {
            let mut x_next = x_curr + step_size;

            // Clamp the final step exactly to the boundary
            if is_forward {
                x_next = x_next.min(x_exit);
            } else {
                x_next = x_next.max(x_exit);
            }

            let r_curr = root_seeking_closure(x_curr);
            let r_next = root_seeking_closure(x_next);

            // A sign change indicates we have perfectly bracketed the shadowing intersection
            if r_curr * r_next <= 0.0 {
                let bound_lower = x_curr.min(x_next);
                let bound_upper = x_curr.max(x_next);
                valid_bracket = Some((bound_lower, bound_upper));
                break;
            }

            x_curr = x_next;
        }
        if let Some((bound_lower, bound_upper)) = valid_bracket {
            let mut convergency = SimpleConvergency {
                eps: 1e-10,
                max_iter: 100,
            };
            Some(
                find_root_brent(
                    bound_lower,
                    bound_upper,
                    &root_seeking_closure,
                    &mut convergency,
                )
                .unwrap(),
            )
        } else {
            None
        }
    }

    pub fn get_angle_from_slope(&self, slope: f64) -> f64 {
        slope.atan()
    }

    pub fn get_slope_from_angle(&self, angle: f64) -> f64 {
        angle.tan()
    }

    fn psi_n(&self, x_n: f64) -> f64 {
        (-self.d * self.k * (self.k * x_n).sin()).atan()
    }

    // Retrieves the direction vector pointing in the same direction as the rotation angle theta.
    //
    // NOTE: theta is a LOCAL direction from the normal!
    pub fn get_vector_from_theta(&self, n: Vector2<f64>, theta: f64) -> Vector2<f64> {
        // let n = self.get_normal_at_x(x_np1_raw);
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // 2D Rotation matrix applied to the normal
        let v1 = [n[0] * cos_t - n[1] * sin_t, n[0] * sin_t + n[1] * cos_t];

        vec2_normalized(v1)
    }

    /**
     * The main workhorse of the simulation. This function takes the current state of the ray,
     * (x_n, theta_n), and computes the next state of the ray, (S_n+1, x_n+1_raw, x_n+1, theta_n+1).
     */
    pub fn iterate_map(&self, s_n: bool, x_n: f64, theta_n: f64) -> (bool, f64, f64, f64, f64) {
        let x_floor = x_n - (self.boundary(x_n) / theta_n.tan());

        let i_n = self.local_shadowing_indicator(self.psi_n(x_n), theta_n);

        debug!("Local shadowing indicator shows: {:?}", i_n);

        // We determine the next state and raw x coordinate together
        let (s_np1, x_np1_raw) = match s_n {
            false => {
                info!("Floor start");
                (true, self.intersect_from_floor(x_n, theta_n))
            }
            true => {
                info!("Ceiling start -- Checking for downstream collisions");
                if let Some(ceiling_root) = self.intersect_from_ceiling(x_n, theta_n, x_floor) {
                    (true, ceiling_root)
                } else {
                    (false, x_floor)
                }
            }
        };

        debug!("Root found was: {:?}", x_np1_raw);

        let theta_np1 = match s_np1 {
            false => -theta_n,
            true => (2.0 * self.psi_n(x_np1_raw)) - theta_n,
        };

        // Wrap the coordinate using Euclidean remainder
        let x_np1 = x_np1_raw.rem_euclid(self.period() * self.graphical_periods());

        debug!("Resulting modulo root was: {:?}", x_np1);

        let y_np1 = match s_np1 {
            true => self.boundary(x_np1),
            false => 0.0,
        };

        (s_np1, x_np1_raw, x_np1, theta_np1, y_np1)
    }
}

// The module name must match the `name` defined in the [lib] section of Cargo.toml
#[pymodule]
fn corrugated_root_solver(_py: Python, m: &PyModule) -> PyResult<()> {
    // Initialize the logger when the module is loaded
    logger::setup_logger().expect("Failed to initialize logger");

    m.add_class::<Waveguide>()?;
    Ok(())
}

pub fn build_root_seeking_closure(
    y_a: f64,
    theta: f64,
    x_a: f64,
    y_0: f64,
    d: f64,
    k: f64,
) -> impl Fn(f64) -> f64 {
    // The move keyword forces the closure to take ownership of the captured variables
    move |x: f64| y_a + theta.tan() * (x - x_a) - y_0 - d * (k * x).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
    use roots::{SimpleConvergency, find_root_brent};
    use std::f64::consts::PI;

    #[test]
    fn test_period_is_4() {
        let waveguide = Waveguide::new(1.0, 1.0, PI / 2.0, 1);
        assert_abs_diff_eq!(waveguide.period(), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_boundary_at_0() {
        let waveguide = Waveguide::new(1.0, 1.0, PI / 2.0, 1);
        assert_abs_diff_eq!(waveguide.boundary(0.0), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_boundary_at_1() {
        let waveguide = Waveguide::new(1.0, 1.0, PI / 2.0, 1);
        assert_abs_diff_eq!(waveguide.boundary(1.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_boundary_at_2_425172() {
        let waveguide = Waveguide::new(1.0, 1.0, PI / 2.0, 1);
        assert_abs_diff_eq!(waveguide.boundary(2.425172), 0.2148503627, epsilon = 1e-10);
    }

    #[test]
    fn test_boundary_at_4() {
        // Our period is 4, so this should be the same as 0.
        let waveguide = Waveguide::new(1.0, 1.0, PI / 2.0, 1);
        assert_abs_diff_eq!(
            waveguide.boundary(4.0),
            waveguide.boundary(0.0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_derivative_at_0() {
        let waveguide = Waveguide::new(1.0, 1.0, PI / 2.0, 1);
        assert_abs_diff_eq!(waveguide.derivative(0.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_at_1() {
        let waveguide = Waveguide::new(1.0, 1.0, PI / 2.0, 1);
        assert_abs_diff_eq!(waveguide.derivative(1.0), -1.57079632679, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_at_1_4761528() {
        let waveguide = Waveguide::new(1.0, 1.0, PI / 2.0, 1);
        assert_abs_diff_eq!(
            waveguide.derivative(1.4761528),
            -1.15153841601,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_derivative_at_3_3() {
        let waveguide = Waveguide::new(1.0, 1.0, PI / 2.0, 1);
        assert_abs_diff_eq!(waveguide.derivative(3.3), 1.39958977535, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_at_4() {
        let waveguide = Waveguide::new(1.0, 1.0, PI / 2.0, 1);
        assert_abs_diff_eq!(
            waveguide.derivative(4.0),
            waveguide.derivative(0.0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_corrugated_root_solver_closure_can_be_solved_using_brents_method() {
        // 1. Define waveguide parameters and current state
        let y_a = 3.0;
        let theta = -1.0f64.atan();
        let x_a = 0.0;
        let y_0 = 2.0;
        let d = 1.0;
        let k = std::f64::consts::PI / 2.0;

        // 2. Generate the single-variable closure
        let root_function = build_root_seeking_closure(y_a, theta, x_a, y_0, d, k);

        // 3. Define strict spatial bounds (e.g., between x_n and x_floor)
        let bound_lower = 0.0;
        let bound_upper = 1.0;

        // 4. Define convergency criteria
        let mut convergency = SimpleConvergency {
            eps: 1e-8f64,
            max_iter: 50,
        };

        // 5. Execute Brent's method
        let root = find_root_brent(bound_lower, bound_upper, &root_function, &mut convergency);

        assert!(root.is_ok(), "Root finding failed: {:?}", root);
        assert_abs_diff_eq!(root.unwrap(), 1.0, epsilon = 1e-8f64);
    }

    #[test]
    fn test_get_angle_from_slope_and_get_slope_from_angle_are_inverses() {
        let waveguide = Waveguide::new(1.0, 1.0, std::f64::consts::PI / 2.0, 1);
        let slope = 1.0;
        let angle = waveguide.get_angle_from_slope(slope);
        let computed_slope = waveguide.get_slope_from_angle(angle);
        assert_abs_diff_eq!(slope, computed_slope, epsilon = 1e-10);
    }

    #[test]
    fn test_iterate_map_from_known_x_and_45_deg_theta_should_not_self_shadow() {
        let theta_start = 45.0 * (std::f64::consts::PI / 180.0);
        let x_start = 1.1;
        let y_0 = 2.0;
        let d = 0.1;
        let k = std::f64::consts::PI / 2.0;

        let waveguide = Waveguide::new(y_0, d, k, 1);

        let (s_np1, _, x_np1, theta_np1, y_np1) =
            waveguide.iterate_map(false, x_start, theta_start);
        assert_eq!(s_np1, true);

        // Note that this is the intersection with the corrugated ceiling.
        assert_abs_diff_eq!(x_np1, 3.118507844621508, epsilon = 1e-10);
        assert_abs_diff_eq!(theta_np1, -0.47908415619699385, epsilon = 1e-10);
        assert_abs_diff_eq!(y_np1, waveguide.boundary(x_np1), epsilon = 1e-10);
    }

    #[test]
    fn test_iterate_map_floor_departure_1_0_x_89_deg() {
        let theta_start = 89.0 * (std::f64::consts::PI / 180.0);
        let x_start = 1.0;
        let y_0 = 2.0;
        let d = 0.73;
        let k = std::f64::consts::PI / 2.0;

        let waveguide = Waveguide::new(y_0, d, k, 2);

        let (s_np1, x_np1_raw, x_np1, theta_np1, y_np1) =
            waveguide.iterate_map(false, x_start, theta_start);

        // 1. Ray must hit the ceiling
        assert_eq!(s_np1, true);

        // 2. The raw x coordinate must be a valid root of the geometric intersection
        let root_closure = build_root_seeking_closure(0.0, theta_start, x_start, y_0, d, k);
        assert_abs_diff_eq!(root_closure(x_np1_raw), 0.0, epsilon = 1e-8);

        // 3. The wrapped y_np1 coordinate must perfectly match the boundary at x_np1
        assert_abs_diff_eq!(y_np1, waveguide.boundary(x_np1), epsilon = 1e-8);

        // 4. The specular reflection angle must match the analytical formula
        let expected_theta_np1 = (2.0 * waveguide.psi_n(x_np1_raw)) - theta_start;
        assert_abs_diff_eq!(theta_np1, expected_theta_np1, epsilon = 1e-8);

        let (s_np2, x_np2_raw, x_np2, theta_np2, y_np2) =
            waveguide.iterate_map(false, x_np1, theta_np1);
    }

    #[test]
    fn test_iterate_map_floor_departure_1_3_x_45_deg() {
        let theta_start = 45.0 * (std::f64::consts::PI / 180.0);
        let x_start = 1.3;
        let y_0 = 2.0;
        let d = 0.73;
        let k = std::f64::consts::PI / 2.0;

        let waveguide = Waveguide::new(y_0, d, k, 2);

        let (s_np1, x_np1_raw, x_np1, theta_np1, y_np1) =
            waveguide.iterate_map(false, x_start, theta_start);

        assert_eq!(s_np1, true);

        let root_closure = build_root_seeking_closure(0.0, theta_start, x_start, y_0, d, k);
        assert_abs_diff_eq!(root_closure(x_np1_raw), 0.0, epsilon = 1e-8);

        assert_abs_diff_eq!(y_np1, waveguide.boundary(x_np1), epsilon = 1e-8);

        let expected_theta_np1 = (2.0 * waveguide.psi_n(x_np1_raw)) - theta_start;
        assert_abs_diff_eq!(theta_np1, expected_theta_np1, epsilon = 1e-8);
    }

    #[test]
    fn test_iterate_map_floor_departure_3_0_x_25_deg() {
        let theta_start = 25.0 * (std::f64::consts::PI / 180.0);
        let x_start = 3.0;
        let y_0 = 2.0;
        let d = 0.73;
        let k = std::f64::consts::PI / 2.0;

        let waveguide = Waveguide::new(y_0, d, k, 2);

        let (s_np1, x_np1_raw, x_np1, theta_np1, y_np1) =
            waveguide.iterate_map(false, x_start, theta_start);

        assert_eq!(s_np1, true);

        let root_closure = build_root_seeking_closure(0.0, theta_start, x_start, y_0, d, k);
        assert_abs_diff_eq!(root_closure(x_np1_raw), 0.0, epsilon = 1e-8);

        assert_abs_diff_eq!(y_np1, waveguide.boundary(x_np1), epsilon = 1e-8);

        let expected_theta_np1 = (2.0 * waveguide.psi_n(x_np1_raw)) - theta_start;
        assert_abs_diff_eq!(theta_np1, expected_theta_np1, epsilon = 1e-8);
    }

    #[test]
    fn test_iterate_map_ceiling_departure_with_downstream_global_shadowing() {
        // Setup waveguide exactly as it was during Iteration 7
        let y_0 = 2.0;
        let d = 0.73;
        let k = std::f64::consts::PI / 2.0;
        let waveguide = Waveguide::new(y_0, d, k, 2);

        // Ray state at the start of Iteration 7
        let s_n = true; // Starting on the ceiling
        let x_start = 4.097784350252917;
        let theta_start = -2.9785129236040566; // Extremely shallow angle pointing left and down

        let (s_np1, x_np1_raw, _x_np1, _theta_np1, _y_np1) =
            waveguide.iterate_map(s_n, x_start, theta_start);

        // 1. Calculate the theoretical floor intersection (~ -12.44)
        let x_floor = x_start - (waveguide.boundary(x_start) / theta_start.tan());

        // 2. The ray must hit the ceiling (global shadowing), NOT the floor.
        // On your current code, this assertion will fail because s_np1 will be false.
        assert_eq!(
            s_np1, true,
            "Ray should suffer a downstream ceiling collision, but it escaped to the floor."
        );

        // 3. Ensure the coordinate returned is not the floor coordinate.
        assert!(
            (x_np1_raw - x_floor).abs() > 1.0,
            "Engine blindly returned the floor intersection instead of the downstream ceiling collision."
        );

        // 4. The raw intersection coordinate must be a valid root of the geometric intersection.
        let root_closure = build_root_seeking_closure(
            waveguide.boundary(x_start),
            theta_start,
            x_start,
            y_0,
            d,
            k,
        );
        assert_abs_diff_eq!(root_closure(x_np1_raw), 0.0, epsilon = 1e-8);
    }
}
