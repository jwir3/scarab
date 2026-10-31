#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_local_definitions)]
use pyo3::prelude::*;
use roots::{find_root_brent, SimpleConvergency};
use std::f64::consts::PI;

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
            y0, d, k, graphical_periods
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

    // This is a ray from the floor, with an angle of theta_n, that intersects the waveguide at
    // some x point.
    pub fn intersect_from_floor(&self, x_a: f64, theta_n: f64) -> f64 {
        let root_seeking_closure = build_root_seeking_closure(
            0.0,
            theta_n,
            x_a,
            self.y0,
            self.d,
            self.k
        );


        // In this case, it's either X_3 or X_4, depending on whether cos(theta_n) is positive or
        // negative.
        let bound_lower = if theta_n.cos() > 0.0 { x_a } else { -10.0 };

        // XXX_jwir3: I don't think this is right - the bounds should be relative.
        let bound_upper = if theta_n.cos() > 0.0 { 10.0 } else { x_a };

        // XXX_jwir3: We can advance the bracket if we receive an error 'NoBracketing', because
        //            this indicates that there was no root in the bracketed region.

        println!("[{}, {}]", bound_lower, bound_upper);

        let mut convergency = SimpleConvergency { eps: 1e-10, max_iter: 100 };

        find_root_brent(
            bound_lower,
            bound_upper,
            &root_seeking_closure,
            &mut convergency,
        ).unwrap()
    }

    // pub fn intersect_x4(&self, theta_n: f64, x_n: f64) -> f64 {
    //     let root_seeking_closure = build_root_seeking_closure(
    //         0.0,
    //         theta_n,
    //         x_n,
    //         self.y0,
    //         self.d,
    //         self.k
    //     );

    //     let bound_lower = x_n;
    //     let bound_upper = 10.0;

    //     let mut convergency = SimpleConvergency { eps: 1e-10, max_iter: 100 };

    //     find_root_brent(
    //         bound_lower,
    //         bound_upper,
    //         &root_seeking_closure,
    //         &mut convergency,
    //     ).unwrap()
    // }

    pub fn compute_specular_reflection_angle_ceiling(&self, x_n: f64, theta_n: f64) -> f64 {
        let derivative_at_x_n = self.derivative(x_n);
        let theta_t = self.get_angle_from_slope(derivative_at_x_n);
        let theta_r = 2.0 * theta_t - theta_n;
        theta_r
    }

    pub fn get_angle_from_slope(&self, slope: f64) -> f64 {
        slope.atan()
    }

    pub fn get_slope_from_angle(&self, angle: f64) -> f64 {
        angle.tan()
    }

    pub fn intersect_from_ceiling_to_floor(&self, x_n: f64, theta_n: f64) -> (f64, f64) {
        let specular_reflection_angle = self.compute_specular_reflection_angle_ceiling(x_n, theta_n);
        let slope = self.get_slope_from_angle(specular_reflection_angle);
        let y_n = self.boundary(x_n);
        (x_n - (y_n / slope), specular_reflection_angle)
    }

    fn psi_n(&self, x_n:f64) -> f64 {
        (-self.d * self.k * (self.k * x_n).sin()).atan()
    }

    // Corresponds to the function I in the paper, which checks for self-intersection of the ray
    // with the waveguide boundary.
    fn shadowing_indicator(&self, psi: f64, theta: f64) -> bool {
        if psi.tan() > theta.tan() && theta.cos() < 0.0 {
            true
        } else if psi.tan() < theta.tan() && theta.cos() > 0.0 {
            true
        } else {
            false
        }
    }

    /**
     * The main workhorse of the simulation. This function takes the current state of the ray,
     * (x_n, theta_n), and computes the next state of the ray, (S_n+1, x_n+1_raw, x_n+1, theta_n+1).
     */
    pub fn iterate_map(&self, s_n: bool, x_n: f64, theta_n: f64) -> (bool, f64, f64, f64, f64) {
        let x_floor = x_n - (self.boundary(x_n) / theta_n.tan());

        let i_n = self.shadowing_indicator(self.psi_n(x_n), theta_n);

        let s_np1 = match (s_n, i_n) {
            (false, _) => true,
            (true, true) => true,
            _ => false
        };

        let cos_theta_n_gt_zero = theta_n.cos() > 0.0;
        let x_np1_raw = match (s_n, i_n, cos_theta_n_gt_zero) {
            (true, false, _) => x_floor,
            (false, _, true) => {
                self.intersect_from_floor(x_n, theta_n)
            },
            (false, _, false) => {
                self.intersect_from_floor(x_n, theta_n)
            }
            _ => panic!("Undefined possible combination of s_n, i_n, cos(theta) > 0: {}, {}, {}", s_n, i_n, cos_theta_n_gt_zero)
        };

        println!("Root found was: {:?}", x_np1_raw);

        let theta_np1 = match s_np1 {
            false => -theta_n,
            true => (2.0 * self.psi_n(x_np1_raw)) - theta_n
        };

        // Calculate the spatial period T = 2pi / k
        // Wrap the coordinate using Euclidean remainder to handle negative x bounds safely
        // Note that we multiply this by the number of periods shown, if graphing.
        let x_np1 = x_np1_raw.rem_euclid(self.period() * self.graphical_periods());

        println!("Resulting modulo root was: {:?}", x_np1);

        let y_np1 = match s_np1 {
            true => self.boundary(x_np1),
            false => 0.0
        };

        (s_np1, x_np1_raw, x_np1, theta_np1, y_np1)
    }
}

// The module name must match the `name` defined in the [lib] section of Cargo.toml
#[pymodule]
fn corrugated_root_solver(_py: Python, m: &PyModule) -> PyResult<()> {
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
    move |x: f64| {
        y_a + theta.tan() * (x - x_a) - y_0 - d * (k * x).cos()
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use roots::{find_root_brent, SimpleConvergency};
    use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
    use super::*;

    #[test]
    fn test_period_is_4() {
        let waveguide = Waveguide::new(1.0, 1.0, PI/2.0, 1);
        assert_abs_diff_eq!(waveguide.period(), 4.0, epsilon=1e-10);
    }

    #[test]
    fn test_boundary_at_0() {
        let waveguide = Waveguide::new(1.0, 1.0, PI/2.0, 1);
        assert_abs_diff_eq!(waveguide.boundary(0.0), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_boundary_at_1() {
        let waveguide = Waveguide::new(1.0, 1.0, PI/2.0, 1);
        assert_abs_diff_eq!(waveguide.boundary(1.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_boundary_at_2_425172() {
        let waveguide = Waveguide::new(1.0, 1.0, PI/2.0, 1);
        assert_abs_diff_eq!(waveguide.boundary(2.425172), 0.2148503627, epsilon = 1e-10);
    }

    #[test]
    fn test_boundary_at_4() {
        // Our period is 4, so this should be the same as 0.
        let waveguide = Waveguide::new(1.0, 1.0, PI/2.0, 1);
        assert_abs_diff_eq!(waveguide.boundary(4.0), waveguide.boundary(0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_at_0() {
        let waveguide = Waveguide::new(1.0, 1.0, PI/2.0, 1);
        assert_abs_diff_eq!(waveguide.derivative(0.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_at_1() {
        let waveguide = Waveguide::new(1.0, 1.0, PI/2.0, 1);
        assert_abs_diff_eq!(waveguide.derivative(1.0), -1.57079632679, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_at_1_4761528() {
        let waveguide = Waveguide::new(1.0, 1.0, PI/2.0, 1);
        assert_abs_diff_eq!(waveguide.derivative(1.4761528), -1.15153841601, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_at_3_3() {
        let waveguide = Waveguide::new(1.0, 1.0, PI/2.0, 1);
        assert_abs_diff_eq!(waveguide.derivative(3.3), 1.39958977535, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_at_4() {
        let waveguide = Waveguide::new(1.0, 1.0, PI/2.0, 1);
        assert_abs_diff_eq!(waveguide.derivative(4.0), waveguide.derivative(0.0), epsilon = 1e-10);
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
        let mut convergency = SimpleConvergency { eps: 1e-8f64, max_iter: 50 };

        // 5. Execute Brent's method
        let root = find_root_brent(bound_lower, bound_upper, &root_function, &mut convergency);

        assert!(root.is_ok(), "Root finding failed: {:?}", root);
        assert_abs_diff_eq!(root.unwrap(), 1.0, epsilon=1e-8f64);
    }

    #[test]
    fn test_get_angle_from_slope_and_get_slope_from_angle_are_inverses() {
        let waveguide = Waveguide::new(1.0, 1.0, std::f64::consts::PI / 2.0, 1);
        let slope = 1.0;
        let angle = waveguide.get_angle_from_slope(slope);
        let computed_slope = waveguide.get_slope_from_angle(angle);
        assert_abs_diff_eq!(slope, computed_slope, epsilon=1e-10);
    }

    #[test]
    fn test_iterate_map_from_known_x_and_45_deg_theta_should_not_self_shadow() {
        let theta_start = 45.0 * (std::f64::consts::PI / 180.0);
        let x_start = 1.1;
        let y_0 = 2.0;
        let d = 0.1;
        let k = std::f64::consts::PI / 2.0;

        let waveguide = Waveguide::new(y_0, d, k, 1);

        let (s_np1, x_np1, theta_np1, y_np1) = waveguide.iterate_map(false, x_start, theta_start);
        assert_eq!(s_np1, true);

        // Note that this is the intersection with the corrugated ceiling.
        assert_abs_diff_eq!(x_np1, 3.118507844621508, epsilon=1e-10);
        assert_abs_diff_eq!(theta_np1, 0.47908415619699385, epsilon=1e-10);
        assert_abs_diff_eq!(y_np1, waveguide.boundary(x_np1), epsilon=1e-10);

        // let (s_np2, x_np2, theta_np2, y_np2) = waveguide.iterate_map(true, x_np1, theta_np1);
        // assert_abs_diff_eq!(, , epsilon=1e-10);

    }
}