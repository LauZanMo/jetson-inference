
##ifdef BUILD_TRAINING_GENERATORS
#include <cmath>
#include <torch/extension.h>
#include <vector>

    namespace trt_pose {
    namespace train {

    torch::Tensor generate_paf(torch::Tensor connections,
                               torch::Tensor topology,
                               torch::Tensor counts,
                               torch::Tensor peaks,
                               int           height,
                               int           width,
                               float         stdev);

    } // namespace train
} // namespace trt_pose

#endif
