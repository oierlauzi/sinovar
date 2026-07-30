from .diffusion_map import compute_diffusion_embedding
from .affinity import (
    knn_affinity_from_squared_distance_matrix,
    thresholded_affinity_from_squared_distance_matrix,
    adaptive_sigma2_median
)
from .mds import (
    compute_landmark_mds_embedding,
    select_landmarks
)
from .smacof import compute_stochastic_smacof_embedding
