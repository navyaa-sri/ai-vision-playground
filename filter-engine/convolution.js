/*
 * AI Vision Playground - filter-engine/convolution.js
 *
 * 3x3 convolution (technically cross-correlation: the kernel is NOT flipped)
 * with zero padding, plus a per-pixel step-by-step explanation for beginners.
 *
 *   output[r][c] = sum over i,j in 0..2 of kernel[i][j] * grid[r-1+i][c-1+j]
 *
 * Exports { applyKernel, explainCell, clampPixel }:
 *   - browser: window.Convolution
 *   - Node:    module.exports
 *
 * Written in strict ES5 (no Array.prototype.map/forEach, no JSON) so it also
 * runs under old engines such as Windows JScript.
 */
(function (root) {
  'use strict';

  var TIMES = '×'; // multiplication sign

  // ---------------------------------------------------------------- helpers

  function isArray(x) {
    return Object.prototype.toString.call(x) === '[object Array]';
  }

  function isFiniteNumber(x) {
    return typeof x === 'number' && isFinite(x);
  }

  // Round to at most 4 decimals and normalise -0 to 0.
  function roundSum(v) {
    var r = Math.round(v * 10000) / 10000;
    if (r === 0) {
      r = 0;
    }
    return r;
  }

  function validateGrid(grid) {
    var r, c, width;
    if (!isArray(grid) || grid.length === 0) {
      throw new Error('Convolution: grid must be a non-empty array of rows.');
    }
    if (!isArray(grid[0]) || grid[0].length === 0) {
      throw new Error('Convolution: grid rows must be non-empty arrays of numbers.');
    }
    width = grid[0].length;
    for (r = 0; r < grid.length; r++) {
      if (!isArray(grid[r])) {
        throw new Error('Convolution: grid row ' + r + ' is not an array.');
      }
      if (grid[r].length !== width) {
        throw new Error('Convolution: grid row ' + r + ' has length ' + grid[r].length +
          ' but row 0 has length ' + width + '; all rows must be the same width.');
      }
      for (c = 0; c < width; c++) {
        if (!isFiniteNumber(grid[r][c])) {
          throw new Error('Convolution: grid[' + r + '][' + c + '] is not a finite number.');
        }
      }
    }
  }

  function validateKernel(kernel) {
    var i, j;
    if (!isArray(kernel) || kernel.length !== 3) {
      throw new Error('Convolution: kernel must be a 3x3 array (3 rows of 3 numbers).');
    }
    for (i = 0; i < 3; i++) {
      if (!isArray(kernel[i]) || kernel[i].length !== 3) {
        throw new Error('Convolution: kernel row ' + i + ' must contain exactly 3 numbers (kernel must be 3x3).');
      }
      for (j = 0; j < 3; j++) {
        if (!isFiniteNumber(kernel[i][j])) {
          throw new Error('Convolution: kernel[' + i + '][' + j + '] is not a finite number.');
        }
      }
    }
  }

  function pixelAt(grid, r, c) {
    if (r < 0 || c < 0 || r >= grid.length || c >= grid[0].length) {
      return 0;
    }
    return grid[r][c];
  }

  // Raw (unrounded) weighted sum at one position. Assumes validated inputs.
  function rawSumAt(grid, kernel, row, col) {
    var i, j, s = 0;
    for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; j++) {
        s += kernel[i][j] * pixelAt(grid, row - 1 + i, col - 1 + j);
      }
    }
    return s;
  }

  function padLeft(str, width) {
    str = String(str);
    while (str.length < width) {
      str = ' ' + str;
    }
    return str;
  }

  function fmt(n) {
    return String(roundSum(n));
  }

  // Wrap negative numbers in parentheses so "a + (-b)" reads clearly.
  function fmtTerm(n) {
    var s = fmt(n);
    return (roundSum(n) < 0) ? '(' + s + ')' : s;
  }

  // ---------------------------------------------------------------- public

  function clampPixel(value) {
    return Math.max(0, Math.min(255, Math.round(value)));
  }

  function applyKernel(grid, kernel) {
    var r, c, row, out = [];
    validateGrid(grid);
    validateKernel(kernel);
    for (r = 0; r < grid.length; r++) {
      row = [];
      for (c = 0; c < grid[0].length; c++) {
        row.push(roundSum(rawSumAt(grid, kernel, r, c)));
      }
      out.push(row);
    }
    return out;
  }

  function explainCell(grid, kernel, row, col) {
    var i, j, gr, gc, oob, pixel, weight, product;
    var steps = [];
    var total = 0;
    var sum, clamped, lines, cells, w, k, terms, anyPadded = false;

    validateGrid(grid);
    validateKernel(kernel);
    if (!isFiniteNumber(row) || !isFiniteNumber(col) ||
        Math.floor(row) !== row || Math.floor(col) !== col) {
      throw new Error('Convolution: row and col must be whole numbers.');
    }
    if (row < 0 || row >= grid.length || col < 0 || col >= grid[0].length) {
      throw new Error('Convolution: cell (' + row + ', ' + col + ') is outside the ' +
        grid.length + 'x' + grid[0].length + ' grid (valid rows 0-' + (grid.length - 1) +
        ', cols 0-' + (grid[0].length - 1) + ').');
    }

    for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; j++) {
        gr = row - 1 + i;
        gc = col - 1 + j;
        oob = gr < 0 || gc < 0 || gr >= grid.length || gc >= grid[0].length;
        pixel = oob ? 0 : grid[gr][gc];
        weight = kernel[i][j];
        product = weight * pixel;
        total += product;
        if (oob) {
          anyPadded = true;
        }
        steps.push({
          kernelRow: i,
          kernelCol: j,
          gridRow: gr,
          gridCol: gc,
          pixel: pixel,
          weight: weight,
          product: roundSum(product),
          outOfBounds: oob
        });
      }
    }

    sum = roundSum(total);
    clamped = clampPixel(sum);

    // ---- build the explanation text
    lines = [];
    lines.push('Output pixel at row ' + row + ', column ' + col + ':');
    lines.push('');
    lines.push('1) Take the 3x3 neighbourhood centred on the pixel' +
      (anyPadded ? ' (cells shown as "pad" are outside the image and count as 0):' : ':'));

    // neighbourhood table
    cells = [];
    w = 3;
    for (k = 0; k < 9; k++) {
      cells.push(steps[k].outOfBounds ? 'pad' : fmt(steps[k].pixel));
      if (cells[k].length > w) {
        w = cells[k].length;
      }
    }
    for (i = 0; i < 3; i++) {
      lines.push('     [ ' + padLeft(cells[i * 3], w) + '  ' + padLeft(cells[i * 3 + 1], w) +
        '  ' + padLeft(cells[i * 3 + 2], w) + ' ]');
    }
    lines.push('');

    // kernel table
    lines.push('2) Line it up with the kernel (the weights):');
    cells = [];
    w = 1;
    for (k = 0; k < 9; k++) {
      cells.push(fmt(steps[k].weight));
      if (cells[k].length > w) {
        w = cells[k].length;
      }
    }
    for (i = 0; i < 3; i++) {
      lines.push('     [ ' + padLeft(cells[i * 3], w) + '  ' + padLeft(cells[i * 3 + 1], w) +
        '  ' + padLeft(cells[i * 3 + 2], w) + ' ]');
    }
    lines.push('');

    // multiplications
    lines.push('3) Multiply each pixel by its matching weight:');
    for (k = 0; k < 9; k++) {
      lines.push('     ' + fmt(steps[k].pixel) + ' ' + TIMES + ' ' + fmtTerm(steps[k].weight) +
        ' = ' + fmt(steps[k].product) +
        (steps[k].outOfBounds ? '   (padded cell outside the image, counts as 0)' : ''));
    }
    lines.push('');

    // addition
    terms = [];
    for (k = 0; k < 9; k++) {
      terms.push(fmtTerm(steps[k].product));
    }
    lines.push('4) Add the 9 products together:');
    lines.push('     ' + terms.join(' + ') + ' = ' + fmt(sum));
    lines.push('');

    lines.push('5) Raw sum = ' + fmt(sum) + '  (this is the filter\'s real output; it can be negative or above 255)');
    if (clamped !== sum) {
      lines.push('6) Display brightness = ' + clamped + '  (rounded and clamped into the 0-255 range, ' +
        (sum < 0 ? 'negatives become 0 = black)' : (sum > 255 ? 'anything above 255 becomes 255 = white)' : 'rounded to a whole number)')));
    } else {
      lines.push('6) Display brightness = ' + clamped + '  (already inside the 0-255 range, so it is shown as-is)');
    }

    return {
      row: row,
      col: col,
      steps: steps,
      sum: sum,
      clamped: clamped,
      text: lines.join('\n')
    };
  }

  var api = {
    applyKernel: applyKernel,
    explainCell: explainCell,
    clampPixel: clampPixel
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
  if (typeof window !== 'undefined') {
    window.Convolution = api;
  }
  if (root && typeof root === 'object' && typeof window === 'undefined' &&
      (typeof module === 'undefined' || !module.exports)) {
    root.Convolution = api;
  }
})(this);
