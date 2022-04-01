
#ifdef BUILD_TRAINING_GENERATORS

#include <cmath>
#include <torch/extension.h>
#include <vector>

namespace trt_pose {
namespace train {

torch::Tensor generate_cmap(
    torch::Tensor counts, torch::Tensor peaks, int height, int width, float stdev, int window);

} // namespace train
} // namespace trt_pose

#endif