#include <iostream>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>
#include <stack>

#define GLOBAL_K 3
#define MAX_CLAUSES 128
#define MAX_VARIABLES 8
#define NOCLAUSE MAX_VARIABLES

using namespace std;

typedef int16_t literal_t;
typedef uint16_t length_t;

struct Index {
    const length_t num_clauses;
    const length_t num_variables;
    length_t *sizes; // Number of clauses that contain the literal  
    literal_t *clauses; // Actual list of clauses, 2 * V

    Index(length_t C = MAX_CLAUSES, length_t V = MAX_VARIABLES):
        num_clauses(C),
        num_variables(V)
    {
        cudaMallocManaged(&sizes, 2 * num_variables * sizeof(length_t));
        cudaMallocManaged(&clauses, 2 * num_variables * num_variables * sizeof(literal_t));
        memset(sizes, 0, 2 * num_variables * sizeof(length_t));
        memset(clauses, NOCLAUSE, 2 * num_variables * num_variables * sizeof(literal_t));
    }

    ~Index() {
        cudaFree(sizes);
        cudaFree(clauses);
    }

    __host__ __device__ void insert(literal_t literal, length_t clause) {
        literal_t index = literal > 0 ? literal : -literal + num_variables;
        index -= 1; // Convert to 0-index
        length_t size = sizes[index];
        clauses[index * num_clauses + size] = clause;
        sizes[index]++;
    }

    __host__ __device__ literal_t *get(literal_t literal) {
        literal_t index = literal > 0 ? literal  : -literal + num_variables;
        index -= 1; // Convert to 0-index
        return &clauses[index * num_clauses];
    }
};

struct Formula {
    length_t num_clauses;
    length_t num_variables;
    length_t num_literals_per_clause;
    literal_t *formula; // Size: C * K

    Formula(length_t C, length_t V, length_t K):
        num_clauses(C),
        num_variables(V),
        num_literals_per_clause(K)
    {
        cudaMallocManaged(&formula, C * K * sizeof(literal_t));
    }

    ~Formula() {
        cudaFree(formula);
    }

    __host__ __device__ literal_t *get_clause(length_t clause) {
        return &formula[clause * num_literals_per_clause];
    }
};

struct Decisions {
    length_t num_variables;
    length_t num_assigned;
    length_t num_processed;
    literal_t *stack; // Size: V
    literal_t *is_assigned; // Size: V

    Decisions(length_t V):
        num_variables(V),
        num_assigned(0),
        num_processed(0)
    {
        cudaMallocManaged(&stack, V * sizeof(literal_t));
        cudaMallocManaged(&is_assigned, V * sizeof(literal_t));
        memset(stack, 0, V * sizeof(literal_t));
        memset(is_assigned, 0, V * sizeof(literal_t));
    }

    ~Decisions() {
        cudaFree(stack);
        cudaFree(is_assigned);
    }

    __host__ __device__ void push(literal_t literal) {
        stack[num_assigned] = literal;
        is_assigned[abs(literal) - 1] = literal;
        num_assigned++;
    }

    __host__ __device__ literal_t pop() {
        num_assigned--;
        literal_t literal = stack[num_assigned];
        is_assigned[abs(literal) - 1] = literal;
        return literal;
    }
};

struct State {
    Formula *F;
    Index *I;
    Decisions *D;

    State(length_t C, length_t V, length_t K, length_t A, length_t P) {
        cudaMallocManaged(&F, sizeof(Formula));
        cudaMallocManaged(&I, sizeof(Index));
        cudaMallocManaged(&D, sizeof(Decisions));
        new(F) Formula(C, V, K);
        new(I) Index(C, V);
        new(D) Decisions(V);
    }
};

__global__ void propagate(
    State *state,
    bool *no_conflict // Determines if there is a conflict
) {
    // All threads in the blocks load index into shared memory
    __shared__ length_t sizes[2 * MAX_VARIABLES];
    __shared__ literal_t index[2 * MAX_VARIABLES * MAX_CLAUSES];
    // Coalesced access to index
    for(length_t i = threadIdx.x; i < 2 * state->I->num_variables; i += blockDim.x) {
        sizes[i] = state->I->sizes[i];
        for(length_t j = 0; j < sizes[i]; j++) {
            index[i * state->I->num_clauses + j] = state->I->clauses[i * state->I->num_clauses + j];
        }
    }
    __syncthreads();

    // Assume no conflict
    *no_conflict = true;

    // While there are unprocessed assignments
    while(state->D->num_processed < state->D->num_assigned) {
        // Get the next assignment
        literal_t assignment = state->D->stack[state->D->num_processed];
        state->D->num_processed++;
        // Get the clauses that contain the opposite literal
        literal_t *clauses;
        if(assignment < 0) { clauses = &index[-assignment - 1]; }
        else { clauses = &index[assignment + state->I->num_variables - 1]; }
        // Each thread processes as many clauses as possible
        unsigned clauses_per_thread = ceil((float) sizes[assignment - 1] / blockDim.x);
        unsigned start = threadIdx.x * clauses_per_thread;
        unsigned end = min(start + clauses_per_thread, sizes[assignment - 1]);

        // For reduction
        __shared__ literal_t unit_literals[MAX_CLAUSES]; // Stores the unit literals found by each thread

        for(unsigned i = start; i < end; i++) {
            // Get the clause
            literal_t *clause = state->F->get_clause(clauses[i]);
            // Check if the clause is satisfied or unit
            bool is_satisfied = false;
            unsigned num_unassigned = 0;
            literal_t last_unassigned_literal = 0;
            for(unsigned v = 0; v < state->F->num_literals_per_clause; v++) {
                literal_t literal = clause[v];
                if(state->D->is_assigned[abs(literal) - 1] == 0) { // unassigned
                    num_unassigned++;
                    last_unassigned_literal = literal;
                } else if(state->D->is_assigned[abs(literal) - 1] == literal) { // satisfied
                    is_satisfied = true;
                    break;
                }
            }

            // Analyze data
            if(!is_satisfied) {
                if(num_unassigned == 0) { // Conflict
                    *no_conflict = false;
                    return;
                }
                unit_literals[i] = last_unassigned_literal;
            }

            // Bad serial code
            __syncthreads();
            // First thread adds unit literals to the stack, if any and if not already assigned
            if(threadIdx.x == 0) {
                for(unsigned i = 0; i < end; i++) {
                    if(unit_literals[i] != 0) {
                        if(state->D->is_assigned[abs(unit_literals[i]) - 1] == 0) { // conflict
                            *no_conflict = false;
                            return;
                        }
                        state->D->push(unit_literals[i]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

int main() {
    State *state;
    cudaMallocManaged(&state, sizeof(state));
    new(state) State(5, 3, 3, 0, 0);
    
    // Input formula
    literal_t instance[] = {
        1, 2, 3,
        1, -2, -3,
        1, 2, -3,
        1, 2, 0, 
        -1, 0, 0,
        1, 0, 0
    };

    for(unsigned c = 0; c < 5; c++) {
        for(unsigned v = 0; v < 3; v++) {
            literal_t literal = instance[c * 3 + v];
            state->F->formula[c * 3 + v] = literal;
            state->I->insert(literal, c);
        }
    }

    bool *no_conflict;
    cudaMallocManaged(&no_conflict, sizeof(bool));

    propagate<<<1, 1>>>(state, no_conflict);
    cudaDeviceSynchronize();

    cout << *no_conflict << endl;
}
