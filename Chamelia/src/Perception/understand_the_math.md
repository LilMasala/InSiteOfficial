At every timestep you have a belief — a distribution over possible patient states. When a new observation arrives you do two things:
**Prediction step** — "what do I expect the state to be now, before seeing the new observation?"

bˉt(x)=∫p(x∣x′,at−1,θpost)  bt−1(x′)  dx′\bar{b}_t(x) = \int p(x \mid x', a_{t-1}, \theta^{\text{post}}) \; b_{t-1}(x') \; dx'bˉt​(x)=∫p(x∣x′,at−1​,θpost)bt−1​(x′)dx′
You're essentially asking — given where I thought the patient was yesterday, and what action was taken, where do I think they are today? You integrate over all possible previous states.
Update step — "now I have a new observation, how do I revise my belief?"

bt(x)∝p(ot∣x)⋅bˉt(x)b_t(x) \propto p(o_t \mid x) \cdot \bar{b}_t(x)bt​(x)∝p(ot​∣x)⋅bˉt​(x)
Multiply the predicted belief by how likely the observation would be if the patient were in each state. States that explain the observation well get higher probability.
The problem is that integral in the prediction step. For a high dimensional patient state you can't compute it exactly.

Kalman Filter (v1.1 POC)
The Kalman filter solves the intractability by making two assumptions:

Dynamics are linear: xt=F⋅xt−1+noisex_t = F \cdot x_{t-1} + \text{noise}
xt​=F⋅xt−1​+noise
Everything is Gaussian: belief stays Gaussian forever

Under these assumptions the integral has a closed form solution. You never need to integrate — you just update a mean vector and covariance matrix:
**Prediction:**

x^t∣t−1=F⋅x^t−1\hat{x}_{t|t-1} = F \cdot \hat{x}_{t-1}x^t∣t−1​=F⋅x^t−1​
Σt∣t−1=F⋅Σt−1⋅FT+Q\Sigma_{t|t-1} = F \cdot \Sigma_{t-1} \cdot F^T + QΣt∣t−1​=F⋅Σt−1​⋅FT+Q
**Update:**

K=Σt∣t−1⋅HT⋅(H⋅Σt∣t−1⋅HT+R)−1K = \Sigma_{t|t-1} \cdot H^T \cdot (H \cdot \Sigma_{t|t-1} \cdot H^T + R)^{-1}K=Σt∣t−1​⋅HT⋅(H⋅Σt∣t−1​⋅HT+R)−1
x^t=x^t∣t−1+K⋅(ot−H⋅x^t∣t−1)\hat{x}_t = \hat{x}_{t|t-1} + K \cdot (o_t - H \cdot \hat{x}_{t|t-1})x^t​=x^t∣t−1​+K⋅(ot​−H⋅x^t∣t−1​)
Σt=(I−K⋅H)⋅Σt∣t−1\Sigma_t = (I - K \cdot H) \cdot \Sigma_{t|t-1}Σt​=(I−K⋅H)⋅Σt∣t−1​
Where:

FF
F = state transition matrix (how state evolves)

QQ
Q = process noise covariance (how uncertain the dynamics are)

HH
H = observation matrix (how state maps to observations)

RR
R = observation noise covariance (how noisy the sensors are)

KK
K = Kalman gain (how much to trust new observation vs prediction)


This is fast, mathematically clean, and works well when the Gaussian assumption is roughly true. For patient state it's an approximation but a good one for v1.1.


Particle Filter (future v1.5)
Drops the Gaussian assumption entirely. Instead of tracking a mean and covariance, you track NN
N particles — each particle is one possible patient state:

particle 1: {bg=120, trust=0.6, burnout=0.2, isf=1.1, ...}
particle 2: {bg=115, trust=0.7, burnout=0.1, isf=0.9, ...}
particle 3: {bg=135, trust=0.5, burnout=0.3, isf=1.2, ...}
...
particle N: {bg=118, trust=0.65, burnout=0.15, isf=1.0, ...}
Prediction: move each particle forward through the dynamics with noise
Update: weight each particle by how well it explains the new observation
Resample: particles with high weight get duplicated, low weight get dropped
This can represent multimodal distributions — "the patient is either in state A or state B" — which Kalman can't. The formulation flags this as important for psychological state which can be genuinely multimodal.
The downside is computational cost — you need enough particles to cover the state space well. This is where GPU parallelism becomes useful — all particles can be updated simultaneously.

Particle Filter (future v1.5)
Drops the Gaussian assumption entirely. Instead of tracking a mean and covariance, you track NN
N particles — each particle is one possible patient state:

particle 1: {bg=120, trust=0.6, burnout=0.2, isf=1.1, ...}
particle 2: {bg=115, trust=0.7, burnout=0.1, isf=0.9, ...}
particle 3: {bg=135, trust=0.5, burnout=0.3, isf=1.2, ...}
...
particle N: {bg=118, trust=0.65, burnout=0.15, isf=1.0, ...}
Prediction: move each particle forward through the dynamics with noise
Update: weight each particle by how well it explains the new observation
Resample: particles with high weight get duplicated, low weight get dropped
This can represent multimodal distributions — "the patient is either in state A or state B" — which Kalman can't. The formulation flags this as important for psychological state which can be genuinely multimodal.
The downside is computational cost — you need enough particles to cover the state space well. This is where GPU parallelism becomes useful — all particles can be updated simultaneously.