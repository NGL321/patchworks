# Patchworks

An embodied graph architecture in which many small predictors, each working in its own
metric space, are reconciled into a coherent model of a world none of them sees whole.

This glossary defines Patchworks' own vocabulary. It deliberately does not inherit terms
from the sibling project `NGL321/laminar`.

## Language

**Cell**:
One node of the graph: a predictor that experiences the world only through the features it
is given and whose job is to advance them one step in time.
_Avoid_: neuron, unit, agent, module

**Chart**:
The low-dimensional coordinates a cell computes in — the compressed feature set it derives
from its node stalk and advances in time. Its dimension is fixed by construction and below
that of the node stalk; its content is learned and need not correlate with any exposed
feature.
_Avoid_: latent, internal representation, hidden state, embedding

**Node stalk**:
A cell's public face — the feature vector it exposes to the graph. Distinct from the cell's
private internal state, which reconciliation never touches.
_Avoid_: node state, node embedding, activation

**Edge stalk**:
The space shared by two adjacent cells, carrying a belief about a latent variable both are
modelling in common. It carries belief only; error is never a channel in it.
_Avoid_: message, edge feature, edge embedding

**Restriction map**:
The map from a cell's node stalk into one incident edge stalk. Performs transport and change
of basis only; all inference happens inside the cell.
_Avoid_: projection, encoder, transport map

**Disagreement**:
The difference, measured in an edge stalk, between the two adjacent cells' restrictions of
their node stalks. Patchworks' only error signal: derived, never carried, and never fully
cleared.
_Avoid_: error, residual, loss, surprise

**Inference phase**:
The half of a tick in which every cell locally advances its own chart and decodes a
prediction, using only its own persisted chart and the node stalk the last message-passing
phase left behind. No cross-cell exchange happens here.
_Avoid_: prediction phase, phase one

**Message-passing phase**:
The other half of a tick: every cell simultaneously exchanges restricted beliefs across its
edges and runs one round of reconciliation. Exactly one simultaneous step per tick — not an
iterative solve run to convergence.
_Avoid_: reconciliation phase, phase two, communication phase

**Reconciliation**:
The disagreement-reducing computation a cell runs during the message-passing phase: a single
local descent step against a neighbour's restricted belief. Penalised rather than enforced —
cells are pulled toward agreement, never projected onto it.
_Avoid_: consensus, synchronisation, message passing (bare — reserve that word for the phase)

**Cell-local**:
The sense of "local" in which a cell's learning uses only quantities that cell can see — no
error signal propagated across the graph.
_Avoid_: local (unqualified)

**Graph-local**:
The sense of "local" in which a cell exchanges only with its adjacent cells — no global
aggregation step, no all-to-all read.
_Avoid_: local (unqualified)

**Relay cell**:
A cell whose inference is the identity: it holds stalks and restriction maps but performs no
prediction, existing to provide a shared metric space for distant cells. Costs one tick of
latency like any other cell.
_Avoid_: hub, router, passthrough node

**Cell body**:
The machinery a cell runs: one set of weights, shared by every cell and frozen. Distinct from the
cell, which is that shared body plus the cell's own adapting surface and its persisted chart.
_Avoid_: the MLP, the network, cell weights

**Adapting surface**:
The parameters that carry a cell's ongoing adaptation — its biases and its restriction maps. The
surface continual learning governs, and the only thing in a cell that ever changes by learning.
_Avoid_: trainable parameters, the readout, fine-tuning surface

**Cell contract**:
What is uniform across every cell: its interface and the algorithm it runs. Capacity and
schedule may vary per cell; the contract may not. A relay cell is the degenerate instance —
a cell whose inference is the identity.
_Avoid_: cell type, node class
