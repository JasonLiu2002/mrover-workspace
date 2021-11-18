
// Python ICP Algorithm

// pc_source = utils.load_pc('cloud_icp_source.csv')

//     ###YOUR CODE HERE###
//     pc_target = utils.load_pc('cloud_icp_target0.csv') # Change this to load in a different target

//     # utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])

//     iters = []
//     errors = []
//     epsilon = 0.005
//     iterations = 40

//     for i in range(iterations):

//         # compute correspondences
//         cp_list = []
//         cq_list = []
//         for p in pc_source:
//             distances = [numpy.linalg.norm(p - q) for q in pc_target]
//             min_index = numpy.argmin(numpy.asarray(distances))
//             cp_list.append(p)
//             cq_list.append(pc_target[min_index])

//         # compute transform
//         C_p = numpy.concatenate((cp_list), axis=1)
//         C_q = numpy.concatenate((cq_list), axis=1)
//         p_mean = numpy.mean(C_p, axis=1)
//         q_mean = numpy.mean(C_q, axis=1)
//         X = C_p - p_mean
//         Y = C_q - q_mean
//         S = X.dot(Y.T)
//         u, s, vh = numpy.linalg.svd(S)
//         v = vh.T

//         R_init = numpy.eye(3)
//         R_init[2, 2] = numpy.linalg.det(v.dot(u.T))
//         R = v @ R_init @ u.T
//         t = q_mean - R.dot(p_mean)

//         # check if we're done
//         total_error = 0
//         for p, q in zip(cp_list, cq_list):
//             total_error += numpy.linalg.norm((R.dot(p) + t) - q)**2

//         print(f"total error: {total_error}")
//         errors.append(total_error)
//         iters.append(i)
//         if total_error < epsilon:
//             break

//         for id, p in enumerate(pc_source):
//             pc_source[id] = R.dot(p) + t

#include "common.hpp"

struct correspondence
{
    float source;
    float target;
};

// does ceiling division
// TODO: move to common?
__device__ int ceilDivGPU(int a, int b)
{
    return (a + b - 1) / b;
}

// compute correspondence between source and target PCs based on minimum euclidean distance
// ignores extra points in one point cloud if one is larger than the other
__global__ void icpKernel(GPU_Cloud source_pc, GPU_Cloud target_pc, correspondence *correspondents)
{
    __shared__ float *distances[target_pc.size];

    // assign a block to each point p in the source cloud
    int p_i = blockIdx.x;

    // divide points in the target cloud between threads
    int points_per_thread = ceilDivGPU(target_pc.size, MAX_THREADS);

    // compute the distances between p and all points q in the target cloud
    // each thread computes the distances for a few points,
    // since there are more points than threads
    for (int i = 0; i < points_per_thread; i++)
    {
        // get the absolute index of q and throw it out if it's invalid
        int q_i = (threadIdx.x * points_per_thread) + i;
        if (q_i > target_pc.size)
            continue;

        distances[q_i] = length(source_pc.data[p_i] - target_pc.data[q_i]);
    }

    // Parallel reduction to get the point q with minimum distance to p
    // This is all equivalent to min(distances), but it does it in parallel
    float min = distances[0];
    int min_i = 0;
    int aliveThreads = (blockDim.x) / 2;
    while (aliveThreads > 0)
    {
        if (threadIdx.x < aliveThreads)
        {
            // TODO: is there a better way to do this min? does this disrupt CUDA branching or something?
            // min = fminf(min, distances[aliveThreads + threadIdx.x]);
            if (distances[aliveThreads + threadIdx.x] < min)
            {
                min = distances[aliveThreads + threadIdx.x];
                min_i = aliveThreads + threadIdx.x;
            }

            if (threadIdx.x >= (aliveThreads) / 2)
                distances[threadIdx.x] = min;
        }
        __syncthreads();
        aliveThreads /= 2;
    }

    // At the final thread, write p index and the minimum q index to the correspondences array
    if (threadIdx.x == 0)
    {
        correspondents[p_i] = {p_i, q_i};
    }
}

GPU_Cloud icp(GPU_Cloud source_pc, GPU_Cloud target_pc)
{
    int iterations = 20;
    float epsilon = 0.01;

    int num_correspondents = min(source_pc.size, target_pc.size);

    for (int i = 0; i < iterations; i++)
    {
        correspondence *correspondents;
        cudaMalloc(&correspondents, sizeof(correspondence *) * num_correspondents);

        // compute correspondences
        int blocks = num_correspondents;
        int threads = MAX_THREADS;
        icpKernel<<<blocks, threads>>>(source_pc, target_pc, correspondents);

        // compute mean of source points in correspondence

        // compute mean of target points in correspondence

        // subtract means from each set of points

        // compute covariance matrix
        
        // perform SVD on covariance matrix (using cuSOLVER)

        // compute rotation and translation matrices

        // apply transformation to every point in source PC

        // compute the total error between source and target PCs

        // exit if error is less than epsilon
    }
}