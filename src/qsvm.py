from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

from feature_engineering import *


class Cqsvm:
    """Quantum Support Vector Machine with parameterized quantum circuit."""

    def __init__(self, qubits, name):
        self.qubits = qubits
        self.name = name
        self.number_of_lgates = 0
        self.number_of_cnot = 0
        self.kernel_instance = None
        self.circuit_solution = None

    def create_circuit_function(self, solution):
        """Create quantum circuit and count gates."""
        self.number_of_lgates = 0
        self.number_of_cnot = 0

        rep_bit = solution[-2:]
        repeats = int(''.join(str(x) for x in rep_bit), 2) + 1

        x_params = ParameterVector('x', length=self.qubits)
        var_custom = QuantumCircuit(self.qubits)

        for _ in range(repeats):
            for i in range(self.qubits):
                var_custom.h(i)
                self.number_of_lgates += 1

                if solution[i] == 1:
                    self.number_of_lgates += 1
                    if solution[self.qubits] == 1 and solution[self.qubits + 1] == 0:
                        var_custom.rx(x_params[i], i)
                    elif solution[self.qubits] == 0 and solution[self.qubits + 1] == 1:
                        var_custom.ry(x_params[i], i)
                    elif solution[self.qubits] == 1 and solution[self.qubits + 1] == 1:
                        var_custom.rz(x_params[i], i)
                    else:
                        var_custom.rx(0 * x_params[i], i)
                else:
                    var_custom.rx(0 * x_params[i], i)

            qub_comb = list(combinations(range(self.qubits), 2))

            for i in range(self.qubits):
                for j in range(i + 1, self.qubits):
                    k = qub_comb.index((i, j))
                    if solution[self.qubits + 2 + k] == 1:
                        var_custom.cx(i, j)
                        var_custom.rz(2 * x_params[i] * x_params[j], j)
                        var_custom.cx(i, j)

                        self.number_of_lgates += 1
                        self.number_of_cnot += 2

            var_custom.barrier()

        return var_custom

    def get_kernel(self, solution):
        """Get or create FidelityStatevectorKernel instance."""
        solution_tuple = tuple(solution)

        if self.kernel_instance is not None and self.circuit_solution == solution_tuple:
            return self.kernel_instance

        new_circuit = self.create_circuit_function(solution)

        self.kernel_instance = FidelityStatevectorKernel(
            feature_map=new_circuit,
            enforce_psd=True
        )
        self.circuit_solution = solution_tuple

        return self.kernel_instance


if __name__ == "__main__":
    print(1)
