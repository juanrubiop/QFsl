from lambeq import CircuitAnsatz

from discopy.quantum.circuit import Functor, Id
from discopy.quantum import Circuit, qubit
from discopy.quantum.gates import Ry, Rx, Rz, CRx

from discopy.grammar.pregroup import Box, Category, Diagram, Ty
from lambeq.ansatz import BaseAnsatz
from collections.abc import Callable, Mapping

import regex as re

from sklearn.manifold import TSNE

import numpy as np

from sympy import Symbol


import torch

class FslNN(CircuitAnsatz):
    def __init__(self,preq_embeddings, ob_map, n_layers, n_single_qubit_params = 2, discard = False):
        self.preq_embeddings=preq_embeddings
        self.n_layers=n_layers
        super().__init__(ob_map,
                         n_layers,
                         n_single_qubit_params,
                         self.circuito,
                         discard,
                         [Rx,Ry])

    def circuito(self,n_qubits,params):
        ic(n_qubits)
        pattern = '^\w+?(?=__)'
        word=params[0][0].name
        matches = re.search(pattern,word)
        word = matches.group(0)

        circuit=Id(n_qubits)

        if (n_qubits==1):
            circuit >>= Id().tensor(*[Rx(phase=Symbol(f'{word} 1'))])
            circuit >>= Id().tensor(*[Rz(phase=Symbol(f'{word} 2'))])

            return circuit

        #Pre Q embeddings
        circuit >>= Id().tensor(*[Rx(phase=phi) for phi in self.preq_embeddings[n_qubits][word][0:n_qubits]])
        circuit >>= Id().tensor(*[Ry(phase=phi) for phi in self.preq_embeddings[n_qubits][word][n_qubits:2*n_qubits]])
        for j in range(n_qubits - 1):
            circuit >>= Id(j) @ CRx(phase=self.preq_embeddings[n_qubits][word][2*n_qubits+j]) @ Id(n_qubits - j - 2)

        #W transform
        n_string=str(n_qubits)
        for m in range(self.n_layers):
            qubit_shift=m*(3*n_qubits-1)
            circuit >>= Id().tensor(*[Ry(Symbol(n_string+'_qubit_{}'.format(i+qubit_shift))) for i in range(n_qubits)])
            circuit >>= Id().tensor(*[Rz(Symbol(n_string+'_qubit_{}'.format(i+n_qubits+qubit_shift))) for i in range(n_qubits)])
            for j in range(n_qubits - 1):
                circuit >>= Id(j) @ CRx(Symbol(n_string+'_qubit_{}'.format(j+2*n_qubits+qubit_shift))) @ Id(n_qubits - j - 2)

        return circuit

    def params_shape(self, n_qubits):
        return (self.n_layers + 1, n_qubits)

class FslBaseAnsatz(CircuitAnsatz):
    def __init__(self,preq_embeddings, ob_map, n_layers, n_single_qubit_params = 2, discard = False):
        self.preq_embeddings=preq_embeddings
        super().__init__(ob_map,
                         n_layers,
                         n_single_qubit_params,
                         self.circuito,
                         discard,
                         [Rx])
        
    def lower_dimensional_preq_embeddings(self,n_qubits,params,preq_embeddings):

        pattern = '^\w+?(?=__)'
        word=params[0][0].name
        matches = re.search(pattern,word)


        words =  list(preq_embeddings.keys())
        vectors = [preq_embeddings[word] for word in words]
        
        tsne = TSNE(n_components=n_qubits, random_state=0,perplexity=9,method='exact')
        
        if matches:
            matched_key = matches.group(0)
            try:
                matched_embedding=[preq_embeddings[matched_key.lower()]]
            except KeyError:
                matched_embedding=np.random.random((1, 50))
            manifold=np.append(matched_embedding,vectors[:10],axis=0)
            Y = tsne.fit_transform(manifold)
            return Y[0]
        
    def polar_coordinates(self,vec):
        x=vec[0]
        y=vec[1]
        z=vec[2]
        theta=np.arctan2(y,x) #azimuth
        phi=np.arctan2(np.sqrt(x**2+y**2+z**2),z) #inclination
        return theta,phi

    def circuito(self,n_qubits,params):
        
        circuit=Id(n_qubits)
        n_layers = params.shape[0] - 1

        
        #Pre Quantum Embedding Parametrization
        if (n_qubits==1):
            rotations=self.lower_dimensional_preq_embeddings(n_qubits=3,params=params,preq_embeddings=self.preq_embeddings)
            theta, phi=self.polar_coordinates(rotations)
            circuit >>= Id().tensor(*[Rx(phase=phi)])
            circuit >>= Id().tensor(*[Rz(phase=theta)])
        else:
            rotations=self.lower_dimensional_preq_embeddings(n_qubits=3,params=params,preq_embeddings=self.preq_embeddings)
            theta, phi=self.polar_coordinates(rotations)
            circuit >>= Id().tensor(*[Rx(phase=phi) for i in range(n_qubits)])
            circuit >>= Id().tensor(*[Rz(phase=theta) for i in range(n_qubits)])

        #W transform
        n_string=str(n_qubits)
        for m in range(n_layers):
            qubit_shift=m*(3*n_qubits-1)
            circuit >>= Id().tensor(*[Ry(Symbol(n_string+'_qubit_{}'.format(i+qubit_shift))) for i in range(n_qubits)])
            circuit >>= Id().tensor(*[Rz(Symbol(n_string+'_qubit_{}'.format(i+n_qubits+qubit_shift))) for i in range(n_qubits)])
            for j in range(n_qubits - 1):
                circuit >>= Id(j) @ CRx(Symbol(n_string+'_qubit_{}'.format(j+2*n_qubits+qubit_shift))) @ Id(n_qubits - j - 2)


        return circuit

    def params_shape(self, n_qubits):
        return (self.n_layers + 1, n_qubits)
    

class FslSim15Ansatz(CircuitAnsatz):
    """Modification of circuit 15 from Sim et al.

    Replaces circuit-block construction with two rings of CNOT gates, in
    opposite orientation.

    Paper at: https://arxiv.org/pdf/1905.10876.pdf

    Code adapted from DisCoPy.

    """

    def __init__(self,
                 preq_embeddings,
                 ob_map: Mapping[Ty, int],
                 n_layers: int,
                 n_single_qubit_params: int = 3,
                 discard: bool = False) -> None:
        """Instantiate a Sim 15 ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.grammar.pregroup.Ty` to
            the number of qubits it uses in a circuit.
        n_layers : int
            The number of layers used by the ansatz.
        n_single_qubit_params : int, default: 3
            The number of single qubit rotations used by the ansatz.
        discard : bool, default: False
            Discard open wires instead of post-selecting.

        """
        self.preq_embeddings=preq_embeddings
        super().__init__(ob_map,
                         n_layers,
                         n_single_qubit_params,
                         self.circuit,
                         discard,
                         [Rx, Rz])
        
    def lower_dimensional_preq_embeddings(self,n_qubits,params,preq_embeddings):

        pattern = '^\w+?(?=__)'
        word=params[0][0].name
        matches = re.search(pattern,word)


        words =  list(preq_embeddings.keys())
        vectors = [preq_embeddings[word] for word in words]
        
        tsne = TSNE(n_components=n_qubits, random_state=0,perplexity=9,method='exact')
        
        if matches:
            matched_key = matches.group(0)
            try:
                matched_embedding=[preq_embeddings[matched_key.lower()]]
            except KeyError:
                matched_embedding=np.random.random((1, 50))
            manifold=np.append(matched_embedding,vectors[:10],axis=0)
            Y = tsne.fit_transform(manifold)
            return Y[0]

    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        return (self.n_layers, 2 * n_qubits)

    def polar_coordinates(self,vec):
        x=vec[0]
        y=vec[1]
        z=vec[2]
        theta=np.arctan2(y,x) #azimuth
        phi=np.arctan2(np.sqrt(x**2+y**2+z**2),z) #inclination
        return theta,phi
    
    def circuit(self, n_qubits: int, params: np.ndarray) -> Circuit:
        rotations=self.lower_dimensional_preq_embeddings(n_qubits=n_qubits,params=params,preq_embeddings=self.preq_embeddings)
        if n_qubits == 1:
            circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
        else:
            
            circuit = Id(n_qubits)

            #Pre Quantum Embedding Parametrization
            if (n_qubits==1):
                rotations=self.lower_dimensional_preq_embeddings(n_qubits=3,params=params,preq_embeddings=self.preq_embeddings)
                theta, phi=self.polar_coordinates(rotations)
                circuit >>= Id().tensor(*[Rx(phase=phi)])
                circuit >>= Id().tensor(*[Rz(phase=theta)])
            else:
                rotations=self.lower_dimensional_preq_embeddings(n_qubits=3,params=params,preq_embeddings=self.preq_embeddings)
                theta, phi=self.polar_coordinates(rotations)
                circuit >>= Id().tensor(*[Rx(phase=phi) for i in range(n_qubits)])
                circuit >>= Id().tensor(*[Rz(phase=theta) for i in range(n_qubits)])

            #W transform sim15
            n_string=str(n_qubits)
            params=[ [ Symbol(n_string+'_qubit_{}'.format(i)) for i in range(params[0].size) ]]
            

            for thetas in params:
                sublayer1 = Id().tensor(*map(Ry, thetas[:n_qubits]))

                for i in range(n_qubits):
                    tgt = (i - 1) % n_qubits
                    sublayer1 = sublayer1.CX(i, tgt)

                sublayer2 = Id().tensor(*map(Ry, thetas[n_qubits:]))

                for i in range(n_qubits, 0, -1):
                    src = i % n_qubits
                    tgt = (i + 1) % n_qubits
                    sublayer2 = sublayer2.CX(src, tgt)

                circuit >>= sublayer1 >> sublayer2

        return circuit


class FslStronglyEntanglingAnsatz(CircuitAnsatz):
    """Strongly entangling ansatz.

    Ansatz using three single qubit rotations (RzRyRz) followed by a
    ladder of CNOT gates with different ranges per layer.

    This is adapted from the PennyLane implementation of the
    :py:class:`pennylane.StronglyEntanglingLayers`, pursuant to `Apache
    2.0 licence <https://www.apache.org/licenses/LICENSE-2.0.html>`_.

    The original paper which introduces the architecture can be found
    `here <https://arxiv.org/abs/1804.00633>`_.

    """

    def __init__(self,
                 preq_embeddings,
                 ob_map: Mapping[Ty, int],
                 n_layers: int,
                 n_single_qubit_params: int = 3,
                 ranges: list[int] | None = None,
                 discard: bool = False) -> None:
        """Instantiate a strongly entangling ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.grammar.pregroup.Ty` to
            the number of qubits it uses in a circuit.
        n_layers : int
            The number of circuit layers used by the ansatz.
        n_single_qubit_params : int, default: 3
            The number of single qubit rotations used by the ansatz.
        ranges : list of int, optional
            The range of the CNOT gate between wires in each layer. By
            default, the range starts at one (i.e. adjacent wires) and
            increases by one for each subsequent layer.
        discard : bool, default: False
            Discard open wires instead of post-selecting.

        """
        self.preq_embeddings=preq_embeddings
        super().__init__(ob_map,
                         n_layers,
                         n_single_qubit_params,
                         self.circuit,
                         discard,
                         [Rz, Ry])
        self.ranges = ranges

        if self.ranges is not None and len(self.ranges) != self.n_layers:
            raise ValueError('The number of ranges must match the number of '
                             'layers.')
        
    def lower_dimensional_preq_embeddings(self,n_qubits,params,preq_embeddings):
        pattern = '^\w+?(?=__)'
        word=params[0][0].name
        matches = re.search(pattern,word)


        words =  list(preq_embeddings.keys())
        vectors = [preq_embeddings[word] for word in words]
        
        tsne = TSNE(n_components=n_qubits, random_state=0,perplexity=9,method='exact')
        
        if matches:
            matched_key = matches.group(0)
            try:
                matched_embedding=[preq_embeddings[matched_key.lower()]]
            except KeyError:
                matched_embedding=np.random.random((1, 50))
            manifold=np.append(matched_embedding,vectors[:10],axis=0)
            Y = tsne.fit_transform(manifold)
            return Y[0]

    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        return (self.n_layers, 3 * n_qubits)

    def circuit(self, n_qubits: int, params: np.ndarray) -> Circuit:
        rotations=self.lower_dimensional_preq_embeddings(n_qubits=n_qubits,params=params,preq_embeddings=self.preq_embeddings)
        circuit = Id(qubit**n_qubits)

        #Pre Quantum Embedding Parametrization
        circuit >>=Id().tensor(*[Ry(phase=i) for i in rotations])


        for layer in range(self.n_layers):
            for j in range(n_qubits):
                syms = params[layer][j*3:j*3+3]
                syms = [ Symbol('Theta_{}'.format(i)) for i in range(len(syms)) ]
                circuit = circuit.Rz(syms[0], j).Ry(syms[1], j).Rz(syms[2], j)
            if self.ranges is None:
                step = layer % (n_qubits - 1) + 1
            elif self.ranges[layer] >= n_qubits:
                raise ValueError('The maximum range must be smaller '
                                 'than the number of qubits.')
            else:
                step = self.ranges[layer]
            for j in range(n_qubits):
                circuit = circuit.CX(j, (j+step) % n_qubits)
        return circuit
