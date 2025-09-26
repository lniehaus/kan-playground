/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * A learnable univariate function represented as a spline.
 * This is the core component of a KAN (Kolmogorov-Arnold Network).
 */
export class LearnableFunction {
  id: string;
  /** Control points for the spline */
  controlPoints: number[] = [];
  /** Grid points for the spline domain */
  gridPoints: number[] = [];
  /** Cached spline coefficients */
  splineCoeffs: number[][] = [];
  /** Input range for normalization */
  inputRange: [number, number] = [-1, 1];
  
  constructor(id: string, gridSize: number = 5, range: [number, number] = [-1, 1]) {
    this.id = id;
    this.inputRange = range;
    this.initializeGrid(gridSize);
    this.initializeControlPoints();
  }

  private initializeGrid(gridSize: number): void {
    const [min, max] = this.inputRange;
    for (let i = 0; i < gridSize; i++) {
      this.gridPoints.push(min + (max - min) * i / (gridSize - 1));
    }
  }

  private initializeControlPoints(): void {
    // Initialize with small random values
    for (let i = 0; i < this.gridPoints.length; i++) {
      this.controlPoints.push((Math.random() - 0.5) * 0.3);
    }
  }

  /** Evaluate the learnable function at input x using spline interpolation */
  evaluate(x: number): number {
    // Clamp input to range
    x = Math.max(this.inputRange[0], Math.min(this.inputRange[1], x));
    
    // Find the interval
    let i = 0;
    while (i < this.gridPoints.length - 1 && x > this.gridPoints[i + 1]) {
      i++;
    }
    
    if (i >= this.gridPoints.length - 1) {
      return this.controlPoints[this.controlPoints.length - 1];
    }
    
    // Linear interpolation between control points
    const x0 = this.gridPoints[i];
    const x1 = this.gridPoints[i + 1];
    const y0 = this.controlPoints[i];
    const y1 = this.controlPoints[i + 1];
    
    const t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
  }

  /** Compute derivative for backpropagation */
  derivative(x: number): number {
    x = Math.max(this.inputRange[0], Math.min(this.inputRange[1], x));
    
    let i = 0;
    while (i < this.gridPoints.length - 1 && x > this.gridPoints[i + 1]) {
      i++;
    }
    
    if (i >= this.gridPoints.length - 1) {
      return 0;
    }
    
    const x0 = this.gridPoints[i];
    const x1 = this.gridPoints[i + 1];
    const y0 = this.controlPoints[i];
    const y1 = this.controlPoints[i + 1];
    
    return (y1 - y0) / (x1 - x0);
  }

  /** Update control points based on gradients */
  updateParameters(gradients: number[], learningRate: number): void {
    for (let i = 0; i < this.controlPoints.length; i++) {
      if (i < gradients.length) {
        this.controlPoints[i] -= learningRate * gradients[i];
      }
    }
  }
}

/**
 * A KAN edge that connects two nodes with a learnable function
 */
export class KANEdge {
  id: string;
  sourceNode: KANNode;
  destNode: KANNode;
  learnableFunction: LearnableFunction;
  /** Cached input value for backpropagation */
  lastInput: number = 0;
  /** Accumulated gradients */
  accGradients: number[] = [];
  numAccumulatedGrads: number = 0;

  constructor(source: KANNode, dest: KANNode, gridSize: number = 5) {
    this.id = source.id + "-" + dest.id;
    this.sourceNode = source;
    this.destNode = dest;
    this.learnableFunction = new LearnableFunction(this.id, gridSize);
    this.accGradients = [];
    for (let i = 0; i < gridSize; i++) {
      this.accGradients.push(0);
    }
  }

  /** Forward pass through the edge */
  forward(input: number): number {
    this.lastInput = input;
    return this.learnableFunction.evaluate(input);
  }

  /** Accumulate gradients for parameter updates */
  accumulateGradients(outputGradient: number): void {
    // Compute gradients with respect to control points
    const input = this.lastInput;
    const gridPoints = this.learnableFunction.gridPoints;
    
    // Find the active interval
    let i = 0;
    while (i < gridPoints.length - 1 && input > gridPoints[i + 1]) {
      i++;
    }
    
    if (i < gridPoints.length - 1) {
      const x0 = gridPoints[i];
      const x1 = gridPoints[i + 1];
      const t = (input - x0) / (x1 - x0);
      
      // Gradients for the two active control points
      this.accGradients[i] += outputGradient * (1 - t);
      this.accGradients[i + 1] += outputGradient * t;
    }
    
    this.numAccumulatedGrads++;
  }

  /** Update parameters using accumulated gradients */
  updateParameters(learningRate: number): void {
    if (this.numAccumulatedGrads > 0) {
      const avgGradients = this.accGradients.map(g => g / this.numAccumulatedGrads);
      this.learnableFunction.updateParameters(avgGradients, learningRate);
      
      // Reset accumulators
      for (let i = 0; i < this.accGradients.length; i++) {
        this.accGradients[i] = 0;
      }
      this.numAccumulatedGrads = 0;
    }
  }
}

/**
 * A node in a Kolmogorov-Arnold Network
 */
export class KANNode {
  id: string;
  /** Input edges */
  inputEdges: KANEdge[] = [];
  /** Output edges */
  outputEdges: KANEdge[] = [];
  /** Cached output value */
  output: number = 0;
  /** Error derivative with respect to this node's output */
  outputDer: number = 0;
  /** Bias term (optional, can be disabled for pure KAN) */
  bias: number = 0;

  constructor(id: string, useBias: boolean = false) {
    this.id = id;
    if (useBias) {
      this.bias = (Math.random() - 0.5) * 0.1;
    }
  }

  /** Forward pass: sum all edge outputs */
  forward(): number {
    this.output = this.bias;
    for (const edge of this.inputEdges) {
      this.output += edge.forward(edge.sourceNode.output);
    }
    return this.output;
  }

  /** Backward pass: distribute gradients to input edges */
  backward(): void {
    for (const edge of this.inputEdges) {
      const inputGrad = this.outputDer * edge.learnableFunction.derivative(edge.lastInput);
      edge.accumulateGradients(this.outputDer);
      edge.sourceNode.outputDer += inputGrad;
    }
  }
}

/**
 * Build a Kolmogorov-Arnold Network
 */
export function buildKANNetwork(
  networkShape: number[],
  inputIds: string[],
  gridSize: number = 5,
  useBias: boolean = false
): KANNode[][] {
  const numLayers = networkShape.length;
  let nodeId = 1;
  const network: KANNode[][] = [];

  // Create nodes for each layer
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    const isInputLayer = layerIdx === 0;
    const currentLayer: KANNode[] = [];
    network.push(currentLayer);
    
    const numNodes = networkShape[layerIdx];
    for (let i = 0; i < numNodes; i++) {
      const id = isInputLayer ? inputIds[i] : nodeId.toString();
      if (!isInputLayer) nodeId++;
      
      const node = new KANNode(id, useBias && !isInputLayer);
      currentLayer.push(node);
    }
  }

  // Create edges between layers
  for (let layerIdx = 1; layerIdx < numLayers; layerIdx++) {
    const prevLayer = network[layerIdx - 1];
    const currentLayer = network[layerIdx];
    
    for (const destNode of currentLayer) {
      for (const sourceNode of prevLayer) {
        const edge = new KANEdge(sourceNode, destNode, gridSize);
        sourceNode.outputEdges.push(edge);
        destNode.inputEdges.push(edge);
      }
    }
  }

  return network;
}

/**
 * Forward propagation through KAN network
 */
export function kanForwardProp(network: KANNode[][], inputs: number[]): number {
  const inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error("Number of inputs must match input layer size");
  }

  // Set input layer outputs
  for (let i = 0; i < inputLayer.length; i++) {
    inputLayer[i].output = inputs[i];
  }

  // Forward propagate through remaining layers
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const currentLayer = network[layerIdx];
    for (const node of currentLayer) {
      node.forward();
    }
  }

  return network[network.length - 1][0].output;
}

/**
 * Backward propagation through KAN network
 */
export function kanBackProp(
  network: KANNode[][],
  target: number,
  errorFunc: { der: (output: number, target: number) => number }
): void {
  // Initialize output gradient
  const outputNode = network[network.length - 1][0];
  outputNode.outputDer = errorFunc.der(outputNode.output, target);

  // Backward propagate through layers
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    const currentLayer = network[layerIdx];
    
    // Reset input node gradients for this layer
    if (layerIdx > 1) {
      const prevLayer = network[layerIdx - 1];
      for (const node of prevLayer) {
        node.outputDer = 0;
      }
    }
    
    // Backward pass for current layer
    for (const node of currentLayer) {
      node.backward();
    }
  }
}

/**
 * Update KAN network parameters
 */
export function updateKANWeights(network: KANNode[][], learningRate: number): void {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const currentLayer = network[layerIdx];
    for (const node of currentLayer) {
      // Update bias if used
      if (node.bias !== 0) {
        node.bias -= learningRate * node.outputDer;
      }
      
      // Update edge parameters
      for (const edge of node.inputEdges) {
        edge.updateParameters(learningRate);
      }
    }
  }
}

/** Get output node from KAN network */
export function getKANOutputNode(network: KANNode[][]): KANNode {
  return network[network.length - 1][0];
}

/** Iterate over all nodes in KAN network */
export function forEachKANNode(
  network: KANNode[][],
  ignoreInputs: boolean,
  accessor: (node: KANNode) => any
): void {
  for (let layerIdx = ignoreInputs ? 1 : 0; layerIdx < network.length; layerIdx++) {
    const currentLayer = network[layerIdx];
    for (const node of currentLayer) {
      accessor(node);
    }
  }
}