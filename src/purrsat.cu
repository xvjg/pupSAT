#include <cstdint>
#include <cstring>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>
#include <stack>

using namespace std;

typedef int16_t literal_t;
typedef uint32_t bitmap_t;
typedef uint16_t length_t;

#define MAX_K 3
#define MAX_VARIABLES 512
#define MAX_CLAUSES 1024
#define END_CLAUSE_INDEX INT16_MAX

////////////////////////////////////////
// Solver implementation
////////////////////////////////////////

// Encodes the state of the solver
enum state_t { SAT = 0, UNKNOWN = 1, UNSAT = 2 };

struct KSatMatrix {
    literal_t *hdata; // Matrix data
    length_t num_clauses; // Number of clauses in the matrix
    length_t num_variables; // Number of variables in the matrix
    length_t K; // Number of literals per clause

    KSatMatrix() {}
    KSatMatrix(length_t num_clauses, length_t K) { init(num_clauses, K); }
    ~KSatMatrix() {
        if(hdata) cudaFree(hdata);
    }

    void init(length_t num_clauses, length_t K) {
        this->num_clauses = num_clauses;
        this->K = K;
        cudaMallocManaged(&hdata, sizeof(literal_t) * num_clauses * K);
        for(length_t i = 0; i < num_clauses * K; i++) {
            hdata[i] = 0;
        }
    }

    void set(length_t clause, literal_t literal) {
        for(uint16_t i = 0; i < K; i++) {
            if(hdata[clause * K + i] == 0) {
                hdata[clause * K + i] = literal;
                return;
            }
        }
        throw "Clause is full";
    }
};

struct Assignment {
    bitmap_t *hpos; // Set of positive literals in the assignment
    bitmap_t *hneg; // Set of negative literals in the assignment
    literal_t *asgns; // Array of assignments
    length_t num_words; // Number of words in the bitvectors

    Assignment() {}
    Assignment(length_t num_variables) { init(num_variables); }
    ~Assignment() {
        if(hpos) cudaFree(hpos);
        if(hneg) cudaFree(hneg);
    }

    void init(length_t num_variables) {
        this->num_words = ceil((num_variables + 1) / 32.0);
        cudaMallocManaged(&hpos, sizeof(bitmap_t) * num_words);
        cudaMallocManaged(&hneg, sizeof(bitmap_t) * num_words);
        memset(hpos, 0, sizeof(bitmap_t) * num_words);
        memset(hneg, 0, sizeof(bitmap_t) * num_words);
    }

    __host__ __device__ void set(literal_t literal) {
        length_t index = abs(literal);
        length_t word = index / 32;
        bitmap_t mask = 1 << (index % 32);
        if(literal > 0) {
            hpos[word] |= mask;
        } else {
            hneg[word] |= mask;
        }
    }

    __host__ __device__ void unset(literal_t literal) {
        length_t index = abs(literal);
        length_t word = index / 32;
        bitmap_t mask = ~(1 << (index % 32));
        if(literal > 0) {
            hpos[word] &= mask;
        } else {
            hneg[word] &= mask;
        }
    }

    __host__ __device__ bool conflict() {
        for(length_t i = 0; i < num_words; i++) {
            if(hpos[i] & hneg[i]) return true;
        }
        return false;
    }

    __host__ literal_t next_unassigned() {
        for(length_t i = 0; i < num_words; i++) {
            if(hpos[i] == 0 && hneg[i] == 0) return i * 32;
            if(hpos[i] != 0) {
                for(length_t j = 0; j < 32; j++) {
                    if((hpos[i] & (1 << j)) == 0) return i * 32 + j;
                }
            }
            if(hneg[i] != 0) {
                for(length_t j = 0; j < 32; j++) {
                    if((hneg[i] & (1 << j)) == 0) return -(i * 32 + j);
                }
            }
        }
        return 0;
    }
};

struct VariableIndex {
    length_t *pos; // Index which clauses contain positive literal, literal -> [clause, clause, ...]
    length_t *neg; // Index which clauses contain negative literal, literal -> [clause, clause, ...]
    length_t num_variables; // Number of variables
    length_t num_clauses; // Number of clauses

    VariableIndex() {}
    VariableIndex(KSatMatrix &matrix) { init(matrix); }
    ~VariableIndex() {
        if(pos) cudaFree(pos);
        if(neg) cudaFree(neg);
    }

    void init(KSatMatrix &matrix) {
        this->num_variables = matrix.num_variables;
        this->num_clauses = matrix.num_clauses;
        cudaMallocManaged(&pos, sizeof(length_t) * num_variables * num_clauses);
        cudaMallocManaged(&neg, sizeof(length_t) * num_variables * num_clauses);
        for(length_t i = 0; i < num_variables * num_clauses; i++) {
            pos[i] = END_CLAUSE_INDEX;
            neg[i] = END_CLAUSE_INDEX;
        }

        for(uint16_t c = 0; c < num_clauses; c++) {
            for(uint16_t v = 0; v < matrix.K; v++) {
                // Get the literal
                int16_t literal = matrix.hdata[c * matrix.K + v];
                // Select the index
                length_t *index = literal > 0 ? pos : neg;
                // Add to empty slot in the index
                for(uint16_t i = 0; i < num_variables; i++) {
                    if(index[c * num_variables + i] == 0) {
                        index[c * num_variables + i] = literal;
                        break;
                    }
                }
            }
        }
    }

    __host__ __device__ length_t *locations(literal_t literal) {
        return literal > 0 ? &pos[literal] : &neg[-literal];
    }
};

struct DecisionStack {
    literal_t *hdata; // Array of decisions
    length_t num_variables; // Size of the stack
    length_t top; // Index of the top of the stack
    length_t processed; // Number of decisions processed

    DecisionStack() {}
    DecisionStack(length_t num_variables) { init(num_variables); }
    ~DecisionStack() {
        if(hdata) cudaFree(hdata);
    }

    void init(length_t num_variables) {
        this->num_variables = num_variables;
        this->top = 0;
        this->processed = 0;
        cudaMallocManaged(&hdata, sizeof(literal_t) * num_variables);
    }

    __host__ __device__ void push(literal_t literal) {
        hdata[top++] = literal;
    }

    __host__ __device__ literal_t pop() {
        return hdata[--top];
    }

    __host__ __device__ literal_t peek() {
        return hdata[top - 1];
    }
};

////////////////////////////////////////
// Solver functions
////////////////////////////////////////

// Solver state
struct Solver {
    KSatMatrix *matrix; // Matrix representation of the problem
    Assignment *asgn; // Current assignment
    DecisionStack *stack; // Decision stack
    state_t state; // State of the solver

    Solver(): state(UNKNOWN) {}

    ~Solver() {
        if(matrix) delete matrix;
        if(asgn) delete asgn;
        if(stack) delete stack;
    }

    __device__ __host__ void init(uint16_t num_clauses, uint16_t num_variables) {
        cudaMallocManaged(&matrix, sizeof(KSatMatrix));
    }
};

__global__ void propagate(
    KSatMatrix *mat,
    VariableIndex *index,
    DecisionStack *stack,
    Assignment *asgn,
    bool *success,
) {
    // Given unprocessed decisions on the stack, propagate the assignments
    // Check if there are any conflicts. If yes, return false in success
    // If new assignments are made, add them to the stack

    while(stack->processed < stack->top) {
        // Get the next decision
        literal_t decision = stack->hdata[stack->processed++];
        // Get the index of the decision
        length_t *locations = index->locations(decision);
        // Retrieve the clauses containing the decision into the shared memory
        __shared__ length_t clauses[MAX_CLAUSES];
        // Coaleced read
        for(length_t i = threadIdx.x; i < index->num_clauses; i += blockDim.x) {
            clauses[i] = locations[i];
        }
        // Wait for all threads to finish
        __syncthreads();
        // Every thread checks if the decision made a clause unsatisfied
        for(length_t i = threadIdx.x; i < index->num_clauses; i += blockDim.x) {
            // Get the clause
            length_t clause = clauses[i];
            // If the clause is empty, it is satisfied
            if(clause == END_CLAUSE_INDEX) continue;
            // Check if the clause is satisfied or has unsatisfied literals
            bool satisfied = false;
            bool unassigned = false;
            for(length_t j = 0; j < mat->K; j++) {
                // Get the literal
                literal_t literal = mat->hdata[clause * mat->K + j];
                // Check if the literal is assigned
                if(asgn->hdata[abs(literal)] == UNASSIGNED) {
                    unassigned = true;
                    continue;
                }
                // Check if the literal is satisfied
                if(literal > 0 && asgn->hdata[literal] == TRUE) {
                    satisfied = true;
                    break;
                }
                if(literal < 0 && asgn->hdata[-literal] == FALSE) {
                    satisfied = true;
                    break;
                }
            }
            // If the clause is not satisfied, the decision is invalid
            if(!satisfied) {
                *success = false;
                return;
            }
        }
    }
}

bool dpll_solve(Solver &s) {
    if(s.state != UNKNOWN) return s.state == SAT; // Already solved
    auto& asgn = s.asgn; // current assignment
    auto& mat = s.matrix; // matrix representation of the problem
}

////////////////////////////////////////
// Main
////////////////////////////////////////

// Parse a DIMACS file
void parse(Solver &s, const char *filename) {
    ifstream file(filename);
    string line;
    uint16_t clause = 0;
    while(getline(file, line)) {
        if(line[0] == 'c') continue; // Comment
        if(line[0] == 'p') { // Problem line
            stringstream ss(line);
            string p, cnf;
            uint16_t num_variables, num_clauses;
            ss >> p >> cnf >> num_variables >> num_clauses;
            s.init(num_clauses, num_variables);
            continue;
        }

        stringstream ss(line); // Clause
        int16_t literal; // Literal
        int16_t num_literals = 0; // Number of literals in clause
        while(ss >> literal) { // Read literals
            if(literal == 0) break; // End of clause
            s.matrix->set(clause, literal); // Set literal in matrix
            num_literals++; } // Increment number of literals

        if(num_literals == 0) { // Empty clause, UNSAT
            s.state = UNSAT; return; } // File is invalid for simplicity

        if (num_literals == 1) { // Unit clause in formula (forced assignment)
            s.asgn->set(literal); } // Set literal

        clause++;
    }
}

// Command line interface
int main(int argc, const char** argv)
{
    if(argc == 1) { printf("Usage: %s <filename>\n", argv[0]); return 1; }
    Solver s;
    parse(s, argv[1]);
    
    // Test propagation
    propagate<<<1, 1>>>(s.matrix, s.asgn, NULL);
}
