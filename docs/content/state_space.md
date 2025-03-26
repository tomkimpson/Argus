# State-Space Methods and Kalman Filtering

The **Argus** project is built around a state-space formulation of stochastic processes, making it well-suited for applications like pulsar timing and gravitational wave detection.

---

## What Are State-Space Models?

A **state-space model** represents a dynamical system using a pair of equations:

1. **State Equation (Evolution/Transition)**
   \\[ x_{t} = F x_{t-1} + w_{t} \\]
   - Describes how the hidden state evolves over time.
   - \( F \) is the transition matrix, and \( w_t \) is process noise.

2. **Observation Equation**
   \\[ y_{t} = H x_{t} + v_{t} \\]
   - Describes how the observed data \( y_t \) relates to the hidden state.
   - \( H \) is the observation matrix, and \( v_t \) is observation noise.

State-space models are powerful for time series where you want to estimate latent (unobserved) variables over time in a principled way.

---

## Kalman Filter

The **Kalman filter** is a recursive algorithm that estimates the state of a linear dynamical system from noisy observations. It is optimal under the assumption of linear-Gaussian models.

### Key Steps

1. **Predict**
   - Estimate the next state and its uncertainty based on the current estimate.
2. **Update**
   - Incorporate the new observation to refine the prediction.

This makes the Kalman filter especially useful in domains like signal processing, control systems, and astrophysics.

---

## Why Use This for PTAs?

Pulsar timing arrays (PTAs) observe timing residuals from pulsars over long periods. The underlying signals (e.g., gravitational waves) are:

- Weak
- Stochastic
- Embedded in noisy observations

By framing PTA analysis as a **latent stochastic process** in a state-space form, we can:

- Efficiently model signal evolution over time
- Incorporate uncertainty and measurement noise
- Leverage fast recursive estimation with Kalman filtering

---

## Extensions

Argus can be extended to handle:

- Nonlinear systems (e.g., Extended or Unscented Kalman Filters)
- Correlated noise models
- Time-varying dynamics

---

For a deeper mathematical treatment, see the publications linked in the [main documentation](index.md).

