/*
 * AI Vision Playground - convolution engine tests.
 * Run from the project root:  node tests/test_convolution.js
 *
 * Written in strict ES5 with no dependencies. All expected values are
 * hand-computed literals (see comments) - never derived from the code
 * under test.
 *
 * Contract (filter-engine/convolution.js):
 *   applyKernel(grid, kernel) -> new grid of raw correlation sums
 *       out[r][c] = sum kernel[i][j] * grid[r-1+i][c-1+j], zero padding,
 *       no flip, no clamping, rounded to <= 4 decimals, inputs untouched.
 *   clampPixel(v) -> Math.max(0, Math.min(255, Math.round(v)))
 *   explainCell(grid, kernel, row, col) -> { row, col, steps[9], sum,
 *       clamped, text }
 */

var passed = 0;
var failed = 0;
var EPS = 1e-6;

function log(s) {
  console.log(s);
}

function fail(msg) {
  throw new Error(msg);
}

function test(name, fn) {
  try {
    fn();
    passed++;
    log("✓ " + name);
  } catch (e) {
    failed++;
    log("✗ " + name);
    log("    " + (e && e.message ? e.message : String(e)));
  }
}

function fmt(v) {
  var i, s;
  if (v === null) { return "null"; }
  if (v === undefined) { return "undefined"; }
  if (typeof v === "object" && typeof v.length === "number") {
    s = "[";
    for (i = 0; i < v.length; i++) {
      if (i > 0) { s += ","; }
      s += fmt(v[i]);
    }
    return s + "]";
  }
  return String(v);
}

function assert(cond, msg) {
  if (!cond) { fail(msg || "assertion failed"); }
}

function assertEqual(actual, expected, msg) {
  if (actual !== expected) {
    fail((msg ? msg + ": " : "") + "expected " + fmt(expected) + ", got " + fmt(actual));
  }
}

function assertClose(actual, expected, eps, msg) {
  if (typeof actual !== "number" || !isFinite(actual) || Math.abs(actual - expected) > eps) {
    fail((msg ? msg + ": " : "") + "expected ~" + expected + ", got " + fmt(actual));
  }
}

function isArr(a) {
  return a !== null && typeof a === "object" && typeof a.length === "number";
}

/* Compare two 2D matrices; eps = 0 means exact equality. */
function assertMatrix(actual, expected, eps, msg) {
  var r, c, a, e;
  var p = msg ? msg + ": " : "";
  assert(isArr(actual), p + "result is not an array: " + fmt(actual));
  assertEqual(actual.length, expected.length, p + "row count");
  for (r = 0; r < expected.length; r++) {
    assert(isArr(actual[r]), p + "row " + r + " is not an array");
    assertEqual(actual[r].length, expected[r].length, p + "column count of row " + r);
    for (c = 0; c < expected[r].length; c++) {
      a = actual[r][c];
      e = expected[r][c];
      if (eps > 0) {
        if (typeof a !== "number" || Math.abs(a - e) > eps) {
          fail(p + "cell [" + r + "][" + c + "] expected ~" + e + ", got " + fmt(a) +
            "\n    actual:   " + fmt(actual) + "\n    expected: " + fmt(expected));
        }
      } else if (a !== e) {
        fail(p + "cell [" + r + "][" + c + "] expected " + e + ", got " + fmt(a) +
          "\n    actual:   " + fmt(actual) + "\n    expected: " + fmt(expected));
      }
    }
  }
}

function assertThrows(fn, msg) {
  var threw = false;
  try { fn(); } catch (e) { threw = true; }
  if (!threw) { fail((msg || "function") + " did not throw"); }
}

function copyMatrix(m) {
  var out = [];
  var r, c, row;
  for (r = 0; r < m.length; r++) {
    row = [];
    for (c = 0; c < m[r].length; c++) { row.push(m[r][c]); }
    out.push(row);
  }
  return out;
}

function makeGrid(rows, cols, value) {
  var out = [];
  var r, c, row;
  for (r = 0; r < rows; r++) {
    row = [];
    for (c = 0; c < cols; c++) { row.push(value); }
    out.push(row);
  }
  return out;
}

function exitWith(code) {
  if (typeof process !== "undefined" && process) {
    process.exitCode = code;
    if (code !== 0 && typeof process.exit === "function") { process.exit(code); }
  }
}

/* ------------------------------------------------------------------ */
/* Kernels and sample data                                             */
/* ------------------------------------------------------------------ */

var IDENTITY = [[0, 0, 0], [0, 1, 0], [0, 0, 0]];
var LAPLACIAN = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]];
var SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
var SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
var NINTH = 1 / 9;
var BOX = [[NINTH, NINTH, NINTH], [NINTH, NINTH, NINTH], [NINTH, NINTH, NINTH]];

/* Non-symmetric kernel used in the sample-matrix and explainCell tests:
 *   out[r][c] = 1*g[r-1][c-1] + 2*g[r-1][c] + 1*g[r][c] - 1*g[r+1][c+1]  */
var ASYM = [[1, 2, 0], [0, 1, 0], [0, 0, -1]];

var SAMPLE = [
  [10, 0, 5, 3],
  [2, 8, 1, 7],
  [4, 6, 9, 0],
  [1, 3, 2, 5]
];

/* Hand-computed ASYM over SAMPLE (A=g[r-1][c-1], B=g[r-1][c], C=g[r][c],
 * D=g[r+1][c+1], out = A + 2B + C - D, out-of-bounds = 0):
 * r0: A=B=0 -> C-D: 10-8=2, 0-1=-1, 5-7=-2, 3-0=3
 * r1: (0+20+2-6)=16, (10+0+8-9)=9, (0+10+1-0)=11, (5+6+7-0)=18
 * r2: (0+4+4-3)=5, (2+16+6-2)=22, (8+2+9-5)=14, (1+14+0-0)=15
 * r3: D=0 -> (0+8+1)=9, (4+12+3)=19, (6+18+2)=26, (9+0+5)=14          */
var SAMPLE_ASYM_EXPECTED = [
  [2, -1, -2, 3],
  [16, 9, 11, 18],
  [5, 22, 14, 15],
  [9, 19, 26, 14]
];

/* Vertical step edge, 5 rows x 6 cols: cols 0-2 = 0, cols 3-5 = 255. */
var STEP = [
  [0, 0, 0, 255, 255, 255],
  [0, 0, 0, 255, 255, 255],
  [0, 0, 0, 255, 255, 255],
  [0, 0, 0, 255, 255, 255],
  [0, 0, 0, 255, 255, 255]
];

/* Laplacian over STEP (out = 8*center - sum of in-bounds neighbours):
 * interior rows 1-3:
 *   c0,c1: all zeros -> 0
 *   c2: 0 - 3*255 (col 3 above/mid/below) = -765
 *   c3: 8*255 - (two in col 3 + three in col 4)*255 = 2040-1275 = 765
 *   c4: 2040 - 8*255 = 0
 *   c5: 2040 - (3 in col 4 + 2 in col 5; col 6 padded 0)*255 = 765
 * top/bottom rows (one neighbour row is padding):
 *   c2: 0 - 2*255 = -510
 *   c3: 2040 - 3*255 (c4 same row, c3/c4 adjacent row) = 1275
 *   c4: 2040 - 5*255 = 765
 *   c5 (corner): 2040 - 3*255 = 1275                                    */
var STEP_LAPLACIAN_EXPECTED = [
  [0, 0, -510, 1275, 765, 1275],
  [0, 0, -765, 765, 0, 765],
  [0, 0, -765, 765, 0, 765],
  [0, 0, -765, 765, 0, 765],
  [0, 0, -510, 1275, 765, 1275]
];

/* Sobel X over STEP (out = (right column weighted 1,2,1) - (left column)):
 * interior rows: c2,c3: right col is 255 -> (1+2+1)*255 = 1020, left 0
 *   c4: 1020 - 1020 = 0; c5: right is padding -> 0 - 1020 = -1020
 *   c0,c1: both sides 0 -> 0
 * top/bottom rows: one row is padding -> weights 2+1 = 3 -> +/-765       */
var STEP_SOBELX_EXPECTED = [
  [0, 0, 765, 765, 0, -765],
  [0, 0, 1020, 1020, 0, -1020],
  [0, 0, 1020, 1020, 0, -1020],
  [0, 0, 1020, 1020, 0, -1020],
  [0, 0, 765, 765, 0, -765]
];

/* ------------------------------------------------------------------ */
/* Load modules                                                        */
/* ------------------------------------------------------------------ */

function main() {
  var conv, applyKernel, explainCell, clampPixel;
  var PRESET_GRIDS = null;
  var presetError = null;

  try {
    conv = require(__dirname + "/../filter-engine/convolution.js");
  } catch (e) {
    log("✗ Could not load filter-engine/convolution.js: " + (e && e.message ? e.message : String(e)));
    log("  Make sure filter-engine/convolution.js exists and exports { applyKernel, explainCell, clampPixel }.");
    log("0 passed, 1 failed");
    exitWith(1);
    return;
  }

  applyKernel = conv && conv.applyKernel;
  explainCell = conv && conv.explainCell;
  clampPixel = conv && conv.clampPixel;

  try {
    PRESET_GRIDS = require(__dirname + "/../presets/shapes.js").PRESET_GRIDS;
  } catch (e2) {
    presetError = e2 && e2.message ? e2.message : String(e2);
  }

  test("module exports applyKernel, explainCell and clampPixel as functions", function () {
    assertEqual(typeof applyKernel, "function", "applyKernel");
    assertEqual(typeof explainCell, "function", "explainCell");
    assertEqual(typeof clampPixel, "function", "clampPixel");
  });

  /* 1. Identity ----------------------------------------------------- */
  test("identity kernel returns a same-size grid equal to the input", function () {
    var out = applyKernel(SAMPLE, IDENTITY);
    assertMatrix(out, [
      [10, 0, 5, 3],
      [2, 8, 1, 7],
      [4, 6, 9, 0],
      [1, 3, 2, 5]
    ], 0, "identity");
  });

  test("identity on a non-square grid keeps dimensions (2x5)", function () {
    var g = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]];
    var out = applyKernel(g, IDENTITY);
    assertMatrix(out, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], 0, "identity 2x5");
  });

  test("applyKernel returns a new grid and does not mutate grid or kernel", function () {
    var g = copyMatrix(SAMPLE);
    var k = copyMatrix(LAPLACIAN);
    var out = applyKernel(g, k);
    assert(out !== g, "output must be a new array, not the input grid");
    assert(out[0] !== g[0], "output rows must be new arrays");
    assertMatrix(g, [
      [10, 0, 5, 3],
      [2, 8, 1, 7],
      [4, 6, 9, 0],
      [1, 3, 2, 5]
    ], 0, "grid after call");
    assertMatrix(k, [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 0, "kernel after call");
  });

  /* 2. Correlation and zero padding --------------------------------- */
  test("non-symmetric kernel on 4x4 sample matches hand-computed matrix (correlation, zero padding)", function () {
    var out = applyKernel(SAMPLE, ASYM);
    assertMatrix(out, SAMPLE_ASYM_EXPECTED, 0, "ASYM over SAMPLE");
  });

  test("impulse response is the kernel rotated 180 degrees (proves no kernel flip)", function () {
    /* out[r][c] = K[2-r][2-c] when the only 1 is at the centre of a 3x3 */
    var g = [[0, 0, 0], [0, 1, 0], [0, 0, 0]];
    var k = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    assertMatrix(applyKernel(g, k), [[9, 8, 7], [6, 5, 4], [3, 2, 1]], 0, "impulse");
  });

  /* 3. Edge detection ----------------------------------------------- */
  test("Laplacian on uniform 255 grid: interior 0, corners 1275, edges 765", function () {
    /* corner: 8*255 - 3*255 = 1275; edge: 8*255 - 5*255 = 765 */
    var out = applyKernel(makeGrid(5, 5, 255), LAPLACIAN);
    assertMatrix(out, [
      [1275, 765, 765, 765, 1275],
      [765, 0, 0, 0, 765],
      [765, 0, 0, 0, 765],
      [765, 0, 0, 0, 765],
      [1275, 765, 765, 765, 1275]
    ], 0, "Laplacian uniform");
  });

  test("Laplacian on vertical 0->255 step edge: -765 / +765 across the edge, border values", function () {
    var out = applyKernel(STEP, LAPLACIAN);
    assertMatrix(out, STEP_LAPLACIAN_EXPECTED, 0, "Laplacian step");
  });

  test("Sobel X on vertical step edge: +1020 at the edge, 0 in flat regions", function () {
    var out = applyKernel(STEP, SOBEL_X);
    assertMatrix(out, STEP_SOBELX_EXPECTED, 0, "Sobel X step");
    assertEqual(out[2][2], 1020, "left of edge");
    assertEqual(out[2][3], 1020, "right of edge");
    assertEqual(out[2][0], 0, "flat dark region");
    assertEqual(out[2][4], 0, "flat bright region");
  });

  test("Sobel X on reversed 255->0 step edge gives -1020 at the edge", function () {
    var g = [
      [255, 255, 255, 0, 0, 0],
      [255, 255, 255, 0, 0, 0],
      [255, 255, 255, 0, 0, 0]
    ];
    var out = applyKernel(g, SOBEL_X);
    /* middle row: c1: 255-255 = 0; c2,c3: 0 - 4*255 = -1020; c4: 0 */
    assertEqual(out[1][1], 0, "flat bright");
    assertEqual(out[1][2], -1020, "left of edge");
    assertEqual(out[1][3], -1020, "right of edge");
    assertEqual(out[1][4], 0, "flat dark");
  });

  test("Sobel Y gives 0 for a vertical edge in interior rows", function () {
    var out = applyKernel(STEP, SOBEL_Y);
    var r, c;
    assertEqual(out.length, 5, "rows");
    for (r = 1; r <= 3; r++) {
      assertEqual(out[r].length, 6, "cols of row " + r);
      for (c = 0; c < 6; c++) {
        assertEqual(out[r][c], 0, "Sobel Y [" + r + "][" + c + "]");
      }
    }
  });

  /* 4. Box blur and rounding ---------------------------------------- */
  test("box blur (1/9) on uniform 9 patch: interior 9, edges 6, corners 4 (epsilon)", function () {
    /* interior: 9*9/9 = 9; edge: 6*9/9 = 6; corner: 4*9/9 = 4 */
    var out = applyKernel(makeGrid(5, 5, 9), BOX);
    assertMatrix(out, [
      [4, 6, 6, 6, 4],
      [6, 9, 9, 9, 6],
      [6, 9, 9, 9, 6],
      [6, 9, 9, 9, 6],
      [4, 6, 6, 6, 4]
    ], EPS, "box blur");
  });

  test("raw sums are rounded to at most 4 decimal places", function () {
    /* centre weight 1/3: 1/3=0.3333, 2/3=0.6667, 3/3=1, 4/3=1.3333 */
    var k = [[0, 0, 0], [0, 1 / 3, 0], [0, 0, 0]];
    var out = applyKernel([[1, 2], [3, 4]], k);
    assertMatrix(out, [[0.3333, 0.6667], [1, 1.3333]], 1e-9, "1/3 kernel");
  });

  /* 5. clampPixel --------------------------------------------------- */
  test("clampPixel clamps and rounds", function () {
    assertEqual(clampPixel(-50), 0, "-50");
    assertEqual(clampPixel(300), 255, "300");
    assertEqual(clampPixel(127.6), 128, "127.6");
    assertEqual(clampPixel(0), 0, "0");
    assertEqual(clampPixel(255), 255, "255");
    assertEqual(clampPixel(1020), 255, "1020");
    assertEqual(clampPixel(-765), 0, "-765");
  });

  /* 6. explainCell -------------------------------------------------- */
  test("explainCell at corner (0,0): 9 row-major steps, out-of-bounds flags, pixel 0 when OOB", function () {
    var ex = explainCell(SAMPLE, ASYM, 0, 0);
    /* step k -> kernelRow = floor(k/3), kernelCol = k%3,
     * gridRow = kernelRow-1, gridCol = kernelCol-1.
     * OOB: steps 0,1,2 (gridRow -1) and 3,6 (gridCol -1).
     * In bounds: 4 -> g[0][0]=10, 5 -> g[0][1]=0, 7 -> g[1][0]=2, 8 -> g[1][1]=8 */
    var expOOB = [true, true, true, true, false, false, true, false, false];
    var expPixel = [0, 0, 0, 0, 10, 0, 0, 2, 8];
    var expWeight = [1, 2, 0, 0, 1, 0, 0, 0, -1];
    var expProduct = [0, 0, 0, 0, 10, 0, 0, 0, -8];
    var i, s;
    assertEqual(ex.row, 0, "row");
    assertEqual(ex.col, 0, "col");
    assert(isArr(ex.steps), "steps is an array");
    assertEqual(ex.steps.length, 9, "step count");
    for (i = 0; i < 9; i++) {
      s = ex.steps[i];
      assertEqual(s.kernelRow, Math.floor(i / 3), "step " + i + " kernelRow");
      assertEqual(s.kernelCol, i % 3, "step " + i + " kernelCol");
      assertEqual(s.gridRow, Math.floor(i / 3) - 1, "step " + i + " gridRow");
      assertEqual(s.gridCol, (i % 3) - 1, "step " + i + " gridCol");
      assertEqual(s.outOfBounds, expOOB[i], "step " + i + " outOfBounds");
      assertEqual(s.pixel, expPixel[i], "step " + i + " pixel");
      assertEqual(s.weight, expWeight[i], "step " + i + " weight");
      assertEqual(s.product, expProduct[i], "step " + i + " product");
      assertEqual(s.product, s.pixel * s.weight, "step " + i + " product = pixel*weight");
    }
    assertEqual(ex.sum, 2, "sum (10 - 8)");
    assertEqual(ex.clamped, 2, "clamped");
  });

  test("explainCell at interior (1,1): products, sum and clamped", function () {
    var ex = explainCell(SAMPLE, ASYM, 1, 1);
    /* 3x3 window rows 0-2, cols 0-2: 10,0,5 / 2,8,1 / 4,6,9
     * products with ASYM: 10,0,0, 0,8,0, 0,0,-9 -> sum 9 */
    var expPixel = [10, 0, 5, 2, 8, 1, 4, 6, 9];
    var expProduct = [10, 0, 0, 0, 8, 0, 0, 0, -9];
    var i, s;
    assertEqual(ex.steps.length, 9, "step count");
    for (i = 0; i < 9; i++) {
      s = ex.steps[i];
      assertEqual(s.outOfBounds, false, "step " + i + " outOfBounds");
      assertEqual(s.gridRow, Math.floor(i / 3), "step " + i + " gridRow");
      assertEqual(s.gridCol, i % 3, "step " + i + " gridCol");
      assertEqual(s.pixel, expPixel[i], "step " + i + " pixel");
      assertEqual(s.product, expProduct[i], "step " + i + " product");
    }
    assertEqual(ex.sum, 9, "sum");
    assertEqual(ex.clamped, 9, "clamped");
  });

  test("explainCell sum equals applyKernel output (interior, corner) and clamped = clampPixel(sum)", function () {
    var out = applyKernel(SAMPLE, ASYM);
    var cells = [[1, 1], [2, 1], [0, 0], [3, 3], [0, 2]];
    /* hand values: (1,1)=9, (2,1)=22, (0,0)=2, (3,3)=14, (0,2)=-2 -> clamped 0 */
    var expSum = [9, 22, 2, 14, -2];
    var expClamped = [9, 22, 2, 14, 0];
    var i, ex, r, c;
    for (i = 0; i < cells.length; i++) {
      r = cells[i][0];
      c = cells[i][1];
      ex = explainCell(SAMPLE, ASYM, r, c);
      assertEqual(ex.sum, expSum[i], "sum at (" + r + "," + c + ")");
      assertEqual(ex.sum, out[r][c], "sum vs applyKernel at (" + r + "," + c + ")");
      assertEqual(ex.clamped, expClamped[i], "clamped at (" + r + "," + c + ")");
      assertEqual(ex.clamped, clampPixel(ex.sum), "clamped = clampPixel(sum) at (" + r + "," + c + ")");
    }
    /* Laplacian on the step edge: (2,3)=765 -> clamped 255; corner (0,5)=1275 */
    ex = explainCell(STEP, LAPLACIAN, 2, 3);
    assertEqual(ex.sum, 765, "Laplacian step (2,3)");
    assertEqual(ex.clamped, 255, "Laplacian step (2,3) clamped");
    ex = explainCell(STEP, LAPLACIAN, 0, 5);
    assertEqual(ex.sum, 1275, "Laplacian step corner (0,5)");
  });

  test("explainCell text is a non-empty string", function () {
    var ex = explainCell(SAMPLE, ASYM, 0, 0);
    assertEqual(typeof ex.text, "string", "text type");
    assert(ex.text.length > 0, "text must not be empty");
  });

  test("explainCell throws for a 2x2 kernel and out-of-range row/col", function () {
    var g = makeGrid(12, 12, 0);
    assertThrows(function () { explainCell(g, [[1, 0], [0, 1]], 5, 5); }, "2x2 kernel");
    assertThrows(function () { explainCell(g, LAPLACIAN, -1, 5); }, "row = -1");
    assertThrows(function () { explainCell(g, LAPLACIAN, 5, 12); }, "col = 12");
  });

  /* 7. Presets ------------------------------------------------------ */
  var presetNames = ["smiley", "house", "seven"];
  var p;
  for (p = 0; p < presetNames.length; p++) {
    (function (name) {
      test("preset '" + name + "' is 12x12 with only 0/255 and Laplacian gives a finite 12x12 grid", function () {
        var preset, g, out, r, c, v;
        assert(PRESET_GRIDS, "presets/shapes.js could not be loaded: " + presetError);
        preset = PRESET_GRIDS[name];
        assert(preset, "missing preset " + name);
        g = preset.grid;
        assert(isArr(g), "grid is not an array");
        assertEqual(g.length, 12, "row count");
        for (r = 0; r < 12; r++) {
          assertEqual(g[r].length, 12, "column count of row " + r);
          for (c = 0; c < 12; c++) {
            v = g[r][c];
            assert(v === 0 || v === 255, "value at [" + r + "][" + c + "] is " + v + ", expected 0 or 255");
          }
        }
        out = applyKernel(g, LAPLACIAN);
        assert(isArr(out), "Laplacian output is not an array");
        assertEqual(out.length, 12, "output row count");
        for (r = 0; r < 12; r++) {
          assertEqual(out[r].length, 12, "output column count of row " + r);
          for (c = 0; c < 12; c++) {
            v = out[r][c];
            assert(typeof v === "number" && isFinite(v), "output [" + r + "][" + c + "] is not finite: " + v);
          }
        }
      });
    })(presetNames[p]);
  }

  test("Laplacian on preset 'seven' at corner (0,0) is -255 (only neighbour [1][1] is 255)", function () {
    /* 8*0 - (g[0][1]=0 + g[1][0]=0 + g[1][1]=255) = -255 */
    assert(PRESET_GRIDS, "presets/shapes.js could not be loaded: " + presetError);
    assertEqual(applyKernel(PRESET_GRIDS.seven.grid, LAPLACIAN)[0][0], -255, "seven (0,0)");
  });

  log("");
  log(passed + " passed, " + failed + " failed");
  exitWith(failed > 0 ? 1 : 0);
}

main();
