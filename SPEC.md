# AI Vision Playground — Technical Specification

## 1. Overview & Architecture
**AI Vision Playground** is a zero-build, zero-backend browser learning tool that shows beginners how a computer represents an image as a 2D numeric grid and applies 3x3 convolution kernels to detect features and edges.

- **Runtime:** Opens directly in Chrome via `file://` or a static HTTP server (`npx serve .`).
- **Styling & Scripts:** Tailwind CSS via CDN and plain ES/UMD JavaScript—no bundler or backend required.
- **Folder Isolation Rule:** Each folder owns a distinct responsibility and exposes explicit globals (`window.*`) for the browser while supporting CommonJS (`module.exports`) for Node.js tests where applicable. Agents must not edit files outside their assigned folder unless integrating in `ui/index.html`.

```text
ai-vision-playground/
├── SPEC.md
├── presets/
│   └── shapes.js              # 12x12 pixel grids & 3x3 kernels
├── filter-engine/
│   └── convolution.js         # 2D convolution & per-pixel step breakdown
├── ui/
│   └── index.html             # Interactive UI, canvas/grid editor & inspector
└── tests/
    └── test_convolution.js    # Node test runner verifying math & presets