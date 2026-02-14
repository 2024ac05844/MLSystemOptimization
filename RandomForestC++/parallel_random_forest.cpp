/*****************************************************************************************
 Parallel Random Forest – Strong Scaling + Efficiency Experiment
 ----------------------------------------------------------------------------------------
 This program demonstrates parallelization of Random Forest training using OpenMP.

 It performs:

 1) Dataset generation
 2) Sequential Random Forest training (true baseline – no OpenMP)
 3) Parallel Random Forest training
 4) Weak scaling experiment (Trees vs Execution Time)
 5) Strong scaling experiment (Threads vs Speedup)
 6) Efficiency computation

 Designed for:
 - Visual Studio (Release Mode)
 - OpenMP enabled
 - Multi-core CPU systems
******************************************************************************************/
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <cmath>

using namespace std;

/* =============================================================================
                                TREE STRUCTURE
   =============================================================================
   Each tree is a simplified decision stump:
   - Splits on one randomly selected feature
   - Fixed threshold (0.5)
   - Majority voting on left and right child
   ============================================================================= */

struct Tree {
    int feature;
    double threshold;
    int left_label;
    int right_label;
};

/* ================= Dataset Generator ================= */
/* =============================================================================
                            SYNTHETIC DATASET GENERATOR
   =============================================================================
   Generates:
   - n_samples samples
   - n_features per sample
   - Label = 1 if feature sum > half of max possible sum
   This creates a balanced, predictable classification problem.
   ============================================================================= */

void generate_dataset(vector<vector<double>>& X,
                      vector<int>& y,
                      int n_samples,
                      int n_features)
{
    mt19937 gen(42);
    uniform_real_distribution<> dis(0, 1);

    X.resize(n_samples, vector<double>(n_features));
    y.resize(n_samples);

    for (int i = 0; i < n_samples; i++) {
        double sum = 0;

        for (int j = 0; j < n_features; j++) {
            X[i][j] = dis(gen);
            sum += X[i][j];
        }

        // Label rule
        y[i] = (sum > n_features * 0.5) ? 1 : 0;
    }
}

/* ================= Train One Heavy Decision Tree ================= */
/* =============================================================================
                    TRAIN A SINGLE (HEAVY) DECISION TREE
   =============================================================================
   Purpose:
   - Simulates realistic CART training cost
   - Adds heavy compute loop to make parallelism measurable

   Heavy compute loop ensures:
   - CPU-bound workload
   - Better strong scaling behavior
   ============================================================================= */

Tree train_tree(const vector<vector<double>>& X,
                const vector<int>& y,
                int seed)
{
    int n_features = X[0].size();

    mt19937 gen(seed);
    uniform_int_distribution<> fdis(0, n_features - 1);

    Tree tree;
    tree.feature = fdis(gen);   // Random feature selection
    tree.threshold = 0.5;

    int l0 = 0, l1 = 0, r0 = 0, r1 = 0;

    // Count class distribution on both sides of split
    for (int i = 0; i < X.size(); i++) {
        if (X[i][tree.feature] < tree.threshold)
            (y[i] == 0) ? l0++ : l1++;
        else
            (y[i] == 0) ? r0++ : r1++;
    }

    // Majority voting
    tree.left_label  = (l1 > l0);
    tree.right_label = (r1 > r0);

    // ================= HEAVY COMPUTE SECTION =================
    // Simulates realistic split evaluation cost in CART
    for (int k = 0; k < 100; k++) {
        for (int i = 0; i < X.size(); i++) {
            double tmp = X[i][tree.feature] * 0.12345;
            tmp = sqrt(tmp);
        }
    }

    return tree;
}

/* =============================================================================
                        PARALLEL RANDOM FOREST TRAINING
   =============================================================================
   Each tree is trained independently → Embarrassingly Parallel

   OpenMP parallel for:
   - Static scheduling
   - Each thread trains separate trees
   - No shared state conflicts
   ============================================================================= */

vector<Tree> train_random_forest_parallel(
    const vector<vector<double>>& X,
    const vector<int>& y,
    int n_trees)
{
    vector<Tree> forest(n_trees);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_trees; i++) {
        int seed = 1234 + i + omp_get_thread_num();
        forest[i] = train_tree(X, y, seed);
    }
    return forest;
}

/* =============================================================================
                        TRUE SEQUENTIAL RANDOM FOREST
   =============================================================================
   No OpenMP used.
   Provides the TRUE baseline for speedup calculation.
   ============================================================================= */

vector<Tree> train_random_forest_sequential(
    const vector<vector<double>>& X,
    const vector<int>& y,
    int n_trees)
{
    vector<Tree> forest(n_trees);
    for (int i = 0; i < n_trees; i++) {
        forest[i] = train_tree(X, y, 1234 + i);
    }
    return forest;
}

/* =============================================================================
                                PREDICTION
   ============================================================================= */

int predict_tree(const Tree& t, const vector<double>& sample)
{
    return (sample[t.feature] < t.threshold)
            ? t.left_label
            : t.right_label;
}

int predict_forest(const vector<Tree>& forest,
                   const vector<double>& sample)
{
    int v0 = 0, v1 = 0;

    for (const auto& t : forest)
        (predict_tree(t, sample) == 0) ? v0++ : v1++;

    return (v1 > v0);
}

double accuracy(const vector<Tree>& forest,
                const vector<vector<double>>& X,
                const vector<int>& y)
{
    int correct = 0;
    for (int i = 0; i < X.size(); i++)
        if (predict_forest(forest, X[i]) == y[i])
            correct++;
    return (double)correct / X.size();
}


/* =============================================================================
                                CONFIGURATION
   ============================================================================= */
#define MAX_SAMPLES (500000)
#define PARALLEL_SCALING_EXPERIMENT_TREES_COUNT (2000)
#define TRUE_SEQUENTIAL_BASELINE_EXPERIMENT_TREES_COUNT (2000)


/* =============================================================================
                                    MAIN
   ============================================================================= */

int main()
{
    // Disable OpenMP dynamic adjustment
    omp_set_dynamic(0);
    omp_set_nested(0);

    int n_samples = MAX_SAMPLES;
    int n_features = 10;

    vector<vector<double>> X;
    vector<int> y;
    generate_dataset(X, y, n_samples, n_features);

    cout << "Dataset generated: " << n_samples << " samples\n";


    /* =========================================================================
                        Parallel scaling Experiment - 1
       =========================================================================
       Fixed:
           - Number of threads
           - Dataset size

       Vary:
           - Number of trees

       Measure:
           - Execution time
           - Accuracy
       ========================================================================= */



    cout << "\n ================================================================================================================\n";
    
    cout << "\n*** PARALLEL SCALING EXPERIMENT-1 Trees vs ExecutionTime: With ALL CORES ***\n";
    cout << "\nMax Cores = " << omp_get_num_procs() << " Samples = " << n_samples << "\n\n";

    ofstream scaling_file("scaling_trees.csv");
    scaling_file << "Trees,ExecutionTime,Accuracy\n";

    for (int n_trees = 200; n_trees <= PARALLEL_SCALING_EXPERIMENT_TREES_COUNT; n_trees += 200) {
        auto start = chrono::high_resolution_clock::now();
        vector<Tree> forest = train_random_forest_parallel(X, y, n_trees);
        auto end = chrono::high_resolution_clock::now();

        double t = chrono::duration<double>(end - start).count();
        double acc = accuracy(forest, X, y);

        cout << " Trees = " << n_trees << " ExecutionTime = " << t << " Accuracy = " << acc << endl;
        scaling_file << n_trees << "," << t << "," << acc << "\n";
    }
    scaling_file.close();

    cout << "\n ================================================================================================================\n";

    ofstream strong_file("strong_scaling.csv");
    strong_file << "Threads,ExecutionTime,Speedup,Accuracy,Efficiency%\n";

    int n_trees_fixed = TRUE_SEQUENTIAL_BASELINE_EXPERIMENT_TREES_COUNT;
    int runs = 3;

    
    // TRUE SEQUENTIAL BASELINE (NO OpenMP)
    double base_time = 0.0;
    double accuBaseline = 0.0;
    for (int r = 0; r < runs; r++) {
        auto start = chrono::high_resolution_clock::now();
        vector<Tree> forest = train_random_forest_sequential(X, y, n_trees_fixed);
        auto end = chrono::high_resolution_clock::now();
        base_time += chrono::duration<double>(end - start).count();
        accuBaseline += accuracy(forest, X, y);
    }
    base_time /= runs;
    accuBaseline /= runs;

    cout << "\n*** TRUE SEQUENTIAL BASELINE EXPERIMENT - No OpenMP *** \n";
    cout << " Trees = " << n_trees_fixed
         << " ExecutionTime = " << base_time
        << " Accuracy = " << accuBaseline <<"\n";

    cout << "\n ================================================================================================================\n";

    /* =========================================================================
                        Parallel scaling Experiment - 2
       =========================================================================
       Fixed:
           - Dataset size
           - Number of trees

       Vary:
           - Number of threads

       Measure:
           - Execution time
           - Speedup
           - Efficiency
       ========================================================================= */

    int max_threads = omp_get_num_procs();
    if (max_threads < 1) max_threads = 1;

    cout << "\n\n*** PARALLEL SCALING EXPERIMENT-2: Fixed number of Trees. Number of Threads vs Speedup ***\n\n";

    for (int threads = 1; threads <= max_threads; threads++) {
        omp_set_num_threads(threads); 

        double avg_time = 0.0;
        double avg_accu = 0.0;
        for (int r = 0; r < runs; r++) {
            auto start = chrono::high_resolution_clock::now();
            vector<Tree> forest = train_random_forest_parallel(X, y, n_trees_fixed);
            auto end = chrono::high_resolution_clock::now();
            avg_time += chrono::duration<double>(end - start).count();
            avg_accu += accuracy(forest, X, y);
        }
        avg_time /= runs;
        avg_accu /= runs;

        double speedup = base_time / avg_time;
        double efficiency = (speedup / threads) * 100;

        cout << " Threads = " << threads
             << " ExecutionTime = " << avg_time
             << " Speedup = " << speedup
             << " Accuracy = " << avg_accu 
             << " Efficiency% = " << efficiency << endl;

        strong_file << threads << "," << avg_time << "," << speedup << "," << avg_accu << "," << efficiency << "\n";
    }

    strong_file.close();
    cout << "\n ================================================================================================================\n";

    cout << "\n All experiments finished. CSV files generated.\n";

    cout << "\n ================================================================================================================\n";
    return 0;
}
