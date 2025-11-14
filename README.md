# Deep playground

Deep playground is an interactive visualization of neural networks, written in
TypeScript using d3.js. We use GitHub issues for tracking new requests and bugs.
Your feedback is highly appreciated!

## What's new

- Added a **Symbolic Regression** problem type. Enter analytic expressions such as `sin(3*x) + 5*x` or `cos(x2) * exp(x1*4)` (aliases `x`, `x1`, `x2`, `y` all map to the single input) to auto-generate 1D datasets and watch each spline adapt in real time. The output panel now plots the learned curve against the ground-truth function for quick visual inspection.

**If you'd like to contribute, be sure to review the [contribution guidelines](CONTRIBUTING.md).**

## Development

To run the visualization locally, run:
- `npm i` to install dependencies
- `npm run build` to compile the app and place it in the `dist/` directory
- `npm run serve` to serve from the `dist/` directory and open a page on your browser.

For a fast edit-refresh cycle when developing run `npm run serve-watch`.
This will start an http server and automatically re-compile the TypeScript,
HTML and CSS files whenever they change.

## For owners
To push to production: `git subtree push --prefix dist origin gh-pages`.

This is not an official Google product.
