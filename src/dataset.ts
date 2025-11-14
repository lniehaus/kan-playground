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

import * as d3 from 'd3';

/**
 * A two dimensional example: x and y coordinates with the label.
 */
export type Example2D = {
  x: number,
  y: number,
  label: number
};

type Point = {
  x: number,
  y: number
};

/**
 * Shuffles the array using Fisher-Yates algorithm. Uses the seedrandom
 * library as the random generator.
 */
export function shuffle(array: any[]): void {
  let counter = array.length;
  let temp = 0;
  let index = 0;
  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = Math.floor(Math.random() * counter);
    // Decrease counter by 1
    counter--;
    // And swap the last element with it
    temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

export type DataGenerator = (numSamples: number, noise: number) => Example2D[];

export function classifyTwoGaussData(numSamples: number, noise: number):
  Example2D[] {
  let points: Example2D[] = [];

  let varianceScale = d3.scale.linear().domain([0, .5]).range([0.1, 0.4]);
  let variance = varianceScale(noise);

  function genGauss(cx: number, cy: number, label: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      let x = normalRandom(cx, variance);
      let y = normalRandom(cy, variance);
      points.push({ x, y, label });
    }
  }

  genGauss(0.5, 0.5, 1); // Gaussian with positive examples.
  genGauss(-0.5, -0.5, -1); // Gaussian with negative examples.
  return points;
}

export function regressPlane(numSamples: number, noise: number):
  Example2D[] {
  let points: Example2D[] = [];

  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-1, 1);
    let y = randUniform(-1, 1);
    let noiseX = randUniform(-0.1, 0.1) * noise;
    let noiseY = randUniform(-0.1, 0.1) * noise;
    // Simple plane: z = (x + y) / 2, normalized to [-1, 1]
    let label = Math.max(-1, Math.min(1, (x + noiseX + y + noiseY) / 2));
    points.push({ x, y, label });
  }
  return points;
}

export function regressGaussian(numSamples: number, noise: number):
  Example2D[] {
  let points: Example2D[] = [];

  // Normalized gaussian centers within [-1, 1] range
  let gaussians = [
    [-0.6, 0.4, 1],
    [0, 0.4, -1],
    [0.6, 0.4, 1],
    [-0.6, -0.4, -1],
    [0, -0.4, 1],
    [0.6, -0.4, -1]
  ];

  function getLabel(x, y) {
    // Choose the one that is maximum in abs value.
    let label = 0;
    gaussians.forEach(([cx, cy, sign]) => {
      let distance = dist({ x, y }, { x: cx, y: cy });
      let newLabel = sign * Math.exp(-distance * distance / 0.2);
      if (Math.abs(newLabel) > Math.abs(label)) {
        label = newLabel;
      }
    });
    return Math.max(-1, Math.min(1, label));
  }

  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-1, 1);
    let y = randUniform(-1, 1);
    let noiseX = randUniform(-0.1, 0.1) * noise;
    let noiseY = randUniform(-0.1, 0.1) * noise;
    let label = getLabel(x + noiseX, y + noiseY);
    points.push({ x, y, label });
  }
  return points;
}

export function classifySpiralData(numSamples: number, noise: number):
  Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;

  function genSpiral(deltaT: number, label: number) {
    for (let i = 0; i < n; i++) {
      let r = (i / n) * 0.8; // Scale radius to stay within [-1, 1]
      let t = 1.75 * i / n * 2 * Math.PI + deltaT;
      let x = r * Math.sin(t) + randUniform(-0.1, 0.1) * noise;
      let y = r * Math.cos(t) + randUniform(-0.1, 0.1) * noise;
      // Ensure points stay within bounds
      x = Math.max(-1, Math.min(1, x));
      y = Math.max(-1, Math.min(1, y));
      points.push({ x, y, label });
    }
  }

  genSpiral(0, 1); // Positive examples.
  genSpiral(Math.PI, -1); // Negative examples.
  return points;
}

export function classifyCircleData(numSamples: number, noise: number):
  Example2D[] {
  let points: Example2D[] = [];
  let radius = 0.8; // Scale to fit in [-1, 1] range

  function getCircleLabel(p: Point, center: Point) {
    return (dist(p, center) < (radius * 0.5)) ? 1 : -1;
  }

  // Generate positive points inside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(0, radius * 0.5);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-0.1, 0.1) * noise;
    let noiseY = randUniform(-0.1, 0.1) * noise;
    x = Math.max(-1, Math.min(1, x + noiseX));
    y = Math.max(-1, Math.min(1, y + noiseY));
    let label = getCircleLabel({ x, y }, { x: 0, y: 0 });
    points.push({ x, y, label });
  }

  // Generate negative points outside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(radius * 0.7, radius);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-0.1, 0.1) * noise;
    let noiseY = randUniform(-0.1, 0.1) * noise;
    x = Math.max(-1, Math.min(1, x + noiseX));
    y = Math.max(-1, Math.min(1, y + noiseY));
    let label = getCircleLabel({ x, y }, { x: 0, y: 0 });
    points.push({ x, y, label });
  }
  return points;
}

export function classifyXORData(numSamples: number, noise: number):
  Example2D[] {
  function getXORLabel(p: Point) { return p.x * p.y >= 0 ? 1 : -1; }

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-1, 1);
    let padding = 0.1;
    x += x > 0 ? padding : -padding;  // Padding.
    let y = randUniform(-1, 1);
    y += y > 0 ? padding : -padding;
    let noiseX = randUniform(-0.1, 0.1) * noise;
    let noiseY = randUniform(-0.1, 0.1) * noise;
    x = Math.max(-1, Math.min(1, x + noiseX));
    y = Math.max(-1, Math.min(1, y + noiseY));
    let label = getXORLabel({ x, y });
    points.push({ x, y, label });
  }
  return points;
}

/**
 * Returns a sample from a uniform [a, b] distribution.
 * Uses the seedrandom library as the random generator.
 */
function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}

/**
 * Samples from a normal distribution. Uses the seedrandom library as the
 * random generator.
 *
 * @param mean The mean. Default is 0.
 * @param variance The variance. Default is 1.
 */
function normalRandom(mean = 0, variance = 1): number {
  let v1: number, v2: number, s: number;
  do {
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s > 1);

  let result = Math.sqrt(-2 * Math.log(s) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}

/** Returns the eucledian distance between two points in space. */
function dist(a: Point, b: Point): number {
  let dx = a.x - b.x;
  let dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

const SYMBOLIC_FUNCTIONS = [
  "sin", "cos", "tan", "asin", "acos", "atan",
  "sinh", "cosh", "tanh", "exp", "log", "sqrt",
  "abs", "ceil", "floor", "round", "sign", "pow"
];

const SYMBOLIC_IDENTIFIERS: { [key: string]: boolean } = {
  "Math": true,
  "x": true
};

const SYMBOLIC_VARIABLE_ALIASES: Array<[RegExp, string]> = [
  [/\bx1\b/gi, "x"],
  [/\bx2\b/gi, "x"],
  [/\by\b/gi, "x"],
  [/\bx_1\b/gi, "x"],
  [/\bx_2\b/gi, "x"]
];

const FALLBACK_SYMBOLIC_EXPRESSION = "Math.sin(Math.PI * x)";

type RawSymbolicEvaluator = (x: number, y?: number) => number;

export interface SymbolicDatasetResult {
  data: Example2D[];
  evaluator: (x: number) => number;
  normalizedEvaluator: (x: number) => number;
  normalizationScale: number;
  compileError?: string;
  sanitizedExpression: string;
  usedFallback: boolean;
  isValid: boolean;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function safeEvaluate(fn: (x: number) => number, x: number): number {
  try {
    let value = fn(x);
    if (!isFinite(value) || isNaN(value)) {
      return 0;
    }
    return value;
  } catch (err) {
    return 0;
  }
}

function sanitizeSymbolicExpression(expression: string): {
  sanitized: string,
  isValid: boolean,
  error?: string
} {
  if (!expression || expression.trim() === "") {
    return { sanitized: "0", isValid: true };
  }

  let sanitized = expression.trim();
  sanitized = sanitized.replace(/\^/g, "**");

  SYMBOLIC_VARIABLE_ALIASES.forEach(([pattern, replacement]) => {
    sanitized = sanitized.replace(pattern, replacement);
  });

  SYMBOLIC_FUNCTIONS.forEach(fn => {
    let regex = new RegExp(`\\b${fn}\\s*\\(`, "gi");
    sanitized = sanitized.replace(regex, `Math.${fn}(`);
  });

  sanitized = sanitized.replace(/\bpi\b/gi, "Math.PI");
  sanitized = sanitized.replace(/\be\b/gi, "Math.E");

  if (/[;={}\[\]]/.test(sanitized)) {
    return {
      sanitized,
      isValid: false,
      error: "Unsupported characters found in expression."
    };
  }

  const identifierRegex = /([A-Za-z_][A-Za-z0-9_]*)/g;
  let match: RegExpExecArray | null;
  while ((match = identifierRegex.exec(sanitized)) != null) {
    let identifier = match[1];
    const previousChar = match.index > 0 ? sanitized.charAt(match.index - 1) : "";
    if (previousChar === ".") {
      continue;
    }
    if (!SYMBOLIC_IDENTIFIERS[identifier]) {
      return {
        sanitized,
        isValid: false,
        error: `Unknown identifier: ${identifier}`
      };
    }
  }

  return { sanitized, isValid: true };
}

export function compileSymbolicExpression(expression: string): {
  evaluator: RawSymbolicEvaluator | null,
  sanitizedExpression: string,
  error?: string,
  isValid: boolean
} {
  const { sanitized, isValid, error } = sanitizeSymbolicExpression(expression);

  if (!isValid) {
    return {
      evaluator: null,
      sanitizedExpression: sanitized,
      error,
      isValid: false
    };
  }

  try {
    const compiled = new Function("x", "y", `"use strict"; return (${sanitized});`) as RawSymbolicEvaluator;
    const evaluator: RawSymbolicEvaluator = (x: number, y?: number) => {
      const yVal = typeof y === "number" ? y : x;
      return compiled(x, yVal);
    };
    return {
      evaluator,
      sanitizedExpression: sanitized,
      error,
      isValid: true
    };
  } catch (e) {
    const message = e instanceof Error ? e.message : "Failed to compile expression.";
    return {
      evaluator: null,
      sanitizedExpression: sanitized,
      error: message,
      isValid: false
    };
  }
}

function defaultSymbolicEvaluator(): RawSymbolicEvaluator {
  return (x: number) => Math.sin(Math.PI * x);
}

export function generateSymbolicDataset(
  numSamples: number,
  noise: number,
  expression: string,
  allowFallback = false): SymbolicDatasetResult {
  const compiled = compileSymbolicExpression(expression);
  let rawEvaluator = compiled.evaluator;
  let usedFallback = false;
  let sanitizedExpression = compiled.sanitizedExpression;
  let compileError = compiled.error;
  let isValid = compiled.isValid && !!rawEvaluator;

  if (!rawEvaluator) {
    if (allowFallback) {
      rawEvaluator = defaultSymbolicEvaluator();
      usedFallback = true;
      isValid = true;
      if (!compileError) {
        compileError = "Falling back to default symbolic function.";
      }
      sanitizedExpression = FALLBACK_SYMBOLIC_EXPRESSION;
    } else {
      const zeroFn = (x: number) => 0;
      return {
        data: [],
        evaluator: zeroFn,
        normalizedEvaluator: zeroFn,
        normalizationScale: 1,
        compileError: compileError || "Unable to parse expression.",
        sanitizedExpression,
        usedFallback: false,
        isValid: false
      };
    }
  }

  const baseEvaluator = (x: number) => rawEvaluator!(x, x);

  const xValues: number[] = [];
  const pristineValues: number[] = [];

  for (let i = 0; i < numSamples; i++) {
    const x = -1 + (2 * i) / Math.max(1, (numSamples - 1));
    xValues.push(x);
    pristineValues.push(safeEvaluate(baseEvaluator, x));
  }

  let maxAbs = pristineValues.reduce((max, val) => Math.max(max, Math.abs(val)), 0);
  if (!isFinite(maxAbs) || maxAbs < 1e-6) {
    maxAbs = 1;
  }

  const noisyValues: number[] = [];
  for (let i = 0; i < pristineValues.length; i++) {
    const cleanVal = pristineValues[i];
    const noiseStd = noise > 0 ? noise * maxAbs : 0;
    const noisyVal = cleanVal + (noiseStd > 0 ? normalRandom(0, noiseStd) : 0);
    noisyValues.push(noisyVal);
  }

  let finalScale = noisyValues.reduce((max, val) => Math.max(max, Math.abs(val)), maxAbs);
  if (!isFinite(finalScale) || finalScale < 1e-6) {
    finalScale = maxAbs > 0 ? maxAbs : 1;
  }

  const data: Example2D[] = noisyValues.map((value, idx) => {
    const normalized = clamp(value / finalScale, -1, 1);
    return {
      x: xValues[idx],
      y: normalized,
      label: normalized
    };
  });

  const normalizedEvaluator = (x: number) => clamp(baseEvaluator(x) / finalScale, -1, 1);

  return {
    data,
    evaluator: baseEvaluator,
    normalizedEvaluator,
    normalizationScale: finalScale,
    compileError,
    sanitizedExpression,
    usedFallback,
    isValid
  };
}
