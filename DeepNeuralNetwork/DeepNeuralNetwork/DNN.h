#pragma once

#include <Eigen/Core>
#include "Config.h"
#include "RNG.h"

#include "Layer.h"
#include "Layer/FullyConnected.h"
#include "Layer/Convolutional.h"
#include "Layer/Pooling.h"

#include "Output.h"
#include "Output/MSE.h"
#include "Output/CrossEntropy.h"