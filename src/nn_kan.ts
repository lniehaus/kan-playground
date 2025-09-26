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
 * A learnable univariate function represented as a B-spline.
 * This is the core component of a KAN (Kolmogorov-Arnold Network).
 */
export class LearnableFunction {
  id: string;
  /** Control points for the B-spline */
  controlPoints: number[] = [];
  /** Knot vector for the B-spline */
  knotVector: number[] = [];
  /** Grid size (number of intervals) */
  gridSize: number;
  /** B-spline degree */
  degree: number;
  /** Input range for normalization */
  inputRange: [number, number] = [-1, 1];
  
  constructor(id: string, gridSize: number = 5, range: [number, number] = [-1, 1], degree: number = 3) {
    this.id = id;
    this.gridSize = gridSize;
    this.degree = Math.min(degree, gridSize - 1); // Degree cannot exceed gridSize - 1
    this.inputRange = range;
    this.initializeKnotVector();
    this.initializeControlPoints();
  }

  private initializeKnotVector(): void {
    const [min, max] = this.inputRange;
    const numControlPoints = this.gridSize + 1;
    const numKnots = numControlPoints + this.degree + 1;
    
    this.knotVector = [];
    
    // Create clamped knot vector
    // First degree+1 knots are min
    for (let i = 0; i <= this.degree; i++) {
      this.knotVector.push(min);
    }
    
    // Internal knots are uniformly distributed
    const numInternalKnots = numKnots - 2 * (this.degree + 1);
    for (let i = 1; i <= numInternalKnots; i++) {
      this.knotVector.push(min + (max - min) * i / (numInternalKnots + 1));
    }
    
    // Last degree+1 knots are max
    for (let i = 0; i <= this.degree; i++) {
      this.knotVector.push(max);
    }
  }

  private initializeControlPoints(): void {
    const numControlPoints = this.gridSize + 1;
    this.controlPoints = [];
    for (let i = 0; i < numControlPoints; i++) {
      this.controlPoints.push((Math.random() - 0.5) * 0.3);
    }
  }

  /** Evaluate the B-spline at input x using de Boor's algorithm */
  evaluate(x: number): number {
    // Clamp input to range
    x = Math.max(this.inputRange[0], Math.min(this.inputRange[1], x));
    
    // Find the knot span
    const span = this.findKnotSpan(x);
    
    // Evaluate using de Boor's algorithm
    return this.deBoor(span, x);
  }

  /** Find the knot span index for parameter x */
  private findKnotSpan(x: number): number {
    const n = this.controlPoints.length - 1; // Number of control points - 1
    const p = this.degree;
    
    if (x >= this.knotVector[n + 1]) {
      return n;
    }
    if (x <= this.knotVector[p]) {
      return p;
    }
    
    // Binary search
    let low = p;
    let high = n + 1;
    let mid = Math.floor((low + high) / 2);
    
    while (x < this.knotVector[mid] || x >= this.knotVector[mid + 1]) {
      if (x < this.knotVector[mid]) {
        high = mid;
      } else {
        low = mid;
      }
      mid = Math.floor((low + high) / 2);
    }
    
    return mid;
  }

  /** de Boor's algorithm for B-spline evaluation */
  private deBoor(span: number, x: number): number {
    const p = this.degree;
    
    // Initialize with control points
    let d: number[] = [];
    for (let j = 0; j <= p; j++) {
      d[j] = this.controlPoints[span - p + j];
    }
    
    // Apply de Boor's algorithm
    for (let r = 1; r <= p; r++) {
      for (let j = p; j >= r; j--) {
        const knotLeft = this.knotVector[span - p + j];
        const knotRight = this.knotVector[span + j - r + 1];
        const alpha = (x - knotLeft) / (knotRight - knotLeft);
        d[j] = (1 - alpha) * d[j - 1] + alpha * d[j];
      }
    }
    
    return d[p];
  }

  /** Compute derivative using finite differences */
  derivative(x: number): number {
    const h = 1e-6;
    const x1 = Math.max(this.inputRange[0], x - h);
    const x2 = Math.min(this.inputRange[1], x + h);
    
    if (x2 - x1 < 1e-10) {
      return 0;
    }
    
    return (this.evaluate(x2) - this.evaluate(x1)) / (x2 - x1);
  }

  /** Update control points based on gradients */
  updateParameters(gradients: number[], learningRate: number): void {
    for (let i = 0; i < this.controlPoints.length && i < gradients.length; i++) {
      this.controlPoints[i] -= learningRate * gradients[i];
    }
  }

  /** Get gradients with respect to control points for given input */
  getControlPointGradients(x: number): number[] {
    x = Math.max(this.inputRange[0], Math.min(this.inputRange[1], x));
    
    const span = this.findKnotSpan(x);
    const p = this.degree;
    const gradients: number[] = [];
    
    // Initialize gradients array - Fix: Use loop instead of fill()
    for (let i = 0; i < this.controlPoints.length; i++) {
      gradients.push(0);
    }
    
    // Compute basis functions using de Boor's algorithm
    const basisFunctions = this.computeBasisFunctions(span, x);
    
    // Set gradients for active control points
    for (let j = 0; j <= p; j++) {
      const controlPointIndex = span - p + j;
      if (controlPointIndex >= 0 && controlPointIndex < gradients.length) {
        gradients[controlPointIndex] = basisFunctions[j];
      }
    }
    
    return gradients;
  }

  /** Compute basis functions for given span and parameter */
  private computeBasisFunctions(span: number, x: number): number[] {
    const p = this.degree;
    const basisFunctions = new Array(p + 1);
    
    // Initialize
    let left = new Array(p + 1);
    let right = new Array(p + 1);
    basisFunctions[0] = 1.0;
    
    for (let j = 1; j <= p; j++) {
      left[j] = x - this.knotVector[span + 1 - j];
      right[j] = this.knotVector[span + j] - x;
      let saved = 0.0;
      
      for (let r = 0; r < j; r++) {
        const temp = basisFunctions[r] / (right[r + 1] + left[j - r]);
        basisFunctions[r] = saved + right[r + 1] * temp;
        saved = left[j - r] * temp;
      }
      basisFunctions[j] = saved;
    }
    
    return basisFunctions;
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

  constructor(source: KANNode, dest: KANNode, gridSize: number = 5, degree: number = 3) {
    this.id = source.id + "-" + dest.id;
    this.sourceNode = source;
    this.destNode = dest;
    this.learnableFunction = new LearnableFunction(this.id, gridSize, [-1, 1], degree);
    
    // Initialize accumulated gradients array
    const numControlPoints = gridSize + 1;
    this.accGradients = [];
    for (let i = 0; i < numControlPoints; i++) {
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
    // Get gradients with respect to control points
    const controlPointGradients = this.learnableFunction.getControlPointGradients(this.lastInput);
    
    // Accumulate gradients
    for (let i = 0; i < this.accGradients.length && i < controlPointGradients.length; i++) {
      this.accGradients[i] += outputGradient * controlPointGradients[i];
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
  degree: number = 3,
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
        const edge = new KANEdge(sourceNode, destNode, gridSize, degree);
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