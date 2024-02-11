import { useRef, useEffect, MutableRefObject } from 'react';

function getPairOfGridCoordinates(canvas: HTMLCanvasElement, event: MouseEvent) {
  const DOMRectangle = canvas.getBoundingClientRect();
  const canvasxCoordinate = event.clientX - DOMRectangle.left;
  const canvasyCoordinate = event.clientY - DOMRectangle.top;
  const pairOfGridCoordinates = getIsometricCoordinatePairGiven(canvasxCoordinate, canvasyCoordinate, canvas.width);
  return pairOfGridCoordinates;
}

const tangentOf30Degrees = 0.5773502691896257645091487805019574556476017512701268760186023264;

function getCanvasCoordinatePairGiven(isometricxCoordinate: number, isometricyCoordinate: number, lengthOfSideOfCanvas: number) {
  const canvasxCoordinate = lengthOfSideOfCanvas / 2 + (isometricxCoordinate + isometricyCoordinate) * lengthOfSideOfCanvas / 22;
  const canvasyCoordinate = (lengthOfSideOfCanvas / 2 + 3 * lengthOfSideOfCanvas / 11 + isometricxCoordinate * lengthOfSideOfCanvas / 22 - isometricyCoordinate * lengthOfSideOfCanvas / 22) * tangentOf30Degrees;
  return {x: canvasxCoordinate, y: canvasyCoordinate};
}

function getIsometricCoordinatePairGiven(canvasxCoordinate: number, canvasyCoordinate: number, lengthOfSideOfCanvas: number) {
  const isometricxCoordinate = Math.round( 11 / lengthOfSideOfCanvas * canvasxCoordinate + 11 / lengthOfSideOfCanvas * canvasyCoordinate / tangentOf30Degrees - 14 );
  const isometricyCoordinate = Math.round( 3 + 11 / lengthOfSideOfCanvas * (canvasxCoordinate - canvasyCoordinate / tangentOf30Degrees) );
  return {x: isometricxCoordinate, y: isometricyCoordinate};
}

const colorOf = {
  Brick: '#AA4A44',
  Desert: '#F5D5A1',
  Grain: '#FADB5E',
  Ore: 'rgb(128,128,128)',
  Sea: '#00c5ff',
  Token: '#e3c08d',
  Wood: '#228B22',
  Wool: '#79D021'
}

const drawPath = function(ctx: CanvasRenderingContext2D, array_of_vertices: Pair[], color: string) {
  ctx.beginPath();
  const vertex_0 = array_of_vertices[0];
  ctx.moveTo(vertex_0.x, vertex_0.y);
  for (let i = 1; i < array_of_vertices.length; i++) {
    const vertex_i = array_of_vertices[i];
    ctx.lineTo(vertex_i.x, vertex_i.y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.stroke();
}

type Pair = {
  x: number,
  y: number
}

const drawTile = function(array_of_pairs: Pair[], color: string, context: CanvasRenderingContext2D) {
  const region = new Path2D();
  const pair_0 = array_of_pairs[0]
  region.moveTo(pair_0.x, pair_0.y);
  for (let i = 1; i < array_of_pairs.length; i++) {
    const pair_i = array_of_pairs[i];
    region.lineTo(pair_i.x, pair_i.y);
  }
  region.closePath();
  context.fillStyle = color;
  context.fill(region);
  array_of_pairs.push(pair_0);
  drawPath(context, array_of_pairs, '#000000');
}

const drawToken = function(ctx: CanvasRenderingContext2D, isometricxCoordinate: number, isometricyCoordinate: number, the_number: number, widthOfCanvas: number) {
  ctx.beginPath();
  const canvasCoordinatePair = getCanvasCoordinatePairGiven(isometricxCoordinate, isometricyCoordinate, widthOfCanvas);
  const pair_per_0_1 = getCanvasCoordinatePairGiven(0, 1, widthOfCanvas);
  const pair_per_0_0 = getCanvasCoordinatePairGiven(0, 0, widthOfCanvas);
  const radius = Math.sqrt(Math.pow(pair_per_0_1.x - pair_per_0_0.x, 2) + Math.pow(pair_per_0_1.y - pair_per_0_0.y, 2)) * Math.cos(Math.PI / 6);
  ctx.arc(canvasCoordinatePair.x, canvasCoordinatePair.y, radius, 0, Math.PI * 2, true);
  ctx.fillStyle = colorOf.Token;
  ctx.fill();
  ctx.strokeStyle = 'black';
  ctx.stroke();
  ctx.fillStyle = 'black';
  ctx.font = '40px Times New Roman';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(the_number.toString(), canvasCoordinatePair.x, canvasCoordinatePair.y);
}

const drawBoard = function(ctx: CanvasRenderingContext2D, widthOfCanvas: number) {

    // Tile 0: Desert
    const pair_per_2_0 = getCanvasCoordinatePairGiven(2, 0, widthOfCanvas);
    const pair_per_2_neg_2 = getCanvasCoordinatePairGiven(2, -2, widthOfCanvas);
    const pair_per_0_neg_2 = getCanvasCoordinatePairGiven(0, -2, widthOfCanvas);
    const pair_per_neg_2_0 = getCanvasCoordinatePairGiven(-2, 0, widthOfCanvas);
    const pair_per_neg_2_2 = getCanvasCoordinatePairGiven(-2, 2, widthOfCanvas);
    const pair_per_0_2 = getCanvasCoordinatePairGiven(0, 2, widthOfCanvas);
    let array_of_pairs = [pair_per_2_0, pair_per_2_neg_2, pair_per_0_neg_2, pair_per_neg_2_0, pair_per_neg_2_2, pair_per_0_2];
    drawTile(array_of_pairs, colorOf.Desert, ctx);

    // Tile 1: Grain
    const pair_per_6_neg_2 = getCanvasCoordinatePairGiven(6, -2, widthOfCanvas);
    const pair_per_6_neg_4 = getCanvasCoordinatePairGiven(6, -4, widthOfCanvas);
    const pair_per_4_neg_4 = getCanvasCoordinatePairGiven(4, -4, widthOfCanvas);
    const pair_per_4_0 = getCanvasCoordinatePairGiven(4, 0, widthOfCanvas);
    array_of_pairs = [pair_per_6_neg_2, pair_per_6_neg_4, pair_per_4_neg_4, pair_per_2_neg_2, pair_per_2_0, pair_per_4_0];
    drawTile(array_of_pairs, colorOf.Grain, ctx);

    // Tile 2: Ore
    const pair_per_4_neg_6 = getCanvasCoordinatePairGiven(4, -6, widthOfCanvas);
    const pair_per_2_neg_6 = getCanvasCoordinatePairGiven(2, -6, widthOfCanvas);
    const pair_per_0_neg_4 = getCanvasCoordinatePairGiven(0, -4, widthOfCanvas);
    array_of_pairs = [pair_per_4_neg_4, pair_per_4_neg_6, pair_per_2_neg_6, pair_per_0_neg_4, pair_per_0_neg_2, pair_per_2_neg_2];
    drawTile(array_of_pairs, colorOf.Ore, ctx);

    // Tile 3: Wood
    const pair_per_neg_2_neg_4 = getCanvasCoordinatePairGiven(-2, -4, widthOfCanvas);
    const pair_per_neg_4_neg_2 = getCanvasCoordinatePairGiven(-4, -2, widthOfCanvas);
    const pair_per_neg_4_0 = getCanvasCoordinatePairGiven(-4, 0, widthOfCanvas);
    array_of_pairs = [pair_per_0_neg_2, pair_per_0_neg_4, pair_per_neg_2_neg_4, pair_per_neg_4_neg_2, pair_per_neg_4_0, pair_per_neg_2_0];
    drawTile(array_of_pairs, colorOf.Wood, ctx);

    // Tile 4: Brick
    const pair_per_neg_6_2 = getCanvasCoordinatePairGiven(-6, 2, widthOfCanvas);
    const pair_per_neg_6_4 = getCanvasCoordinatePairGiven(-6, 4, widthOfCanvas);
    const pair_per_neg_4_4 = getCanvasCoordinatePairGiven(-4, 4, widthOfCanvas);
    array_of_pairs = [pair_per_neg_2_2, pair_per_neg_2_0, pair_per_neg_4_0, pair_per_neg_6_2, pair_per_neg_6_4, pair_per_neg_4_4];
    drawTile(array_of_pairs, colorOf.Brick, ctx);

    // Tile 5: Wool
    const pair_per_0_4 = getCanvasCoordinatePairGiven(0, 4, widthOfCanvas);
    //0,2
    //-2,2
    //-4,4
    const pair_per_neg_4_6 = getCanvasCoordinatePairGiven(-4, 6, widthOfCanvas);
    const pair_per_neg_2_6 = getCanvasCoordinatePairGiven(-2, 6, widthOfCanvas);
    array_of_pairs = [pair_per_0_4, pair_per_0_2, pair_per_neg_2_2, pair_per_neg_4_4, pair_per_neg_4_6, pair_per_neg_2_6];
    drawTile(array_of_pairs, colorOf.Wool, ctx);

    // Tile 6: Wood
    const pair_per_4_2 = getCanvasCoordinatePairGiven(4, 2, widthOfCanvas);
    const pair_per_2_4 = getCanvasCoordinatePairGiven(2, 4, widthOfCanvas);
    array_of_pairs = [pair_per_4_2, pair_per_4_0, pair_per_2_0, pair_per_0_2, pair_per_0_4, pair_per_2_4];
    drawTile(array_of_pairs, colorOf.Wood, ctx);

    // Tile 7: Wool
    const pair_per_8_0 = getCanvasCoordinatePairGiven(8, 0, widthOfCanvas);
    const pair_per_8_neg_2 = getCanvasCoordinatePairGiven(8, -2, widthOfCanvas);
    const pair_per_6_2 = getCanvasCoordinatePairGiven(6, 2, widthOfCanvas);
    array_of_pairs = [pair_per_8_0, pair_per_8_neg_2, pair_per_6_neg_2, pair_per_4_0, pair_per_4_2, pair_per_6_2];
    drawTile(array_of_pairs, colorOf.Wool, ctx);

    // Tile 8: Wool
    const pair_per_10_neg_4 = getCanvasCoordinatePairGiven(10, -4, widthOfCanvas);
    const pair_per_10_neg_6 = getCanvasCoordinatePairGiven(10, -6, widthOfCanvas);
    const pair_per_8_neg_6 = getCanvasCoordinatePairGiven(8, -6, widthOfCanvas);
    array_of_pairs = [pair_per_10_neg_4, pair_per_10_neg_6, pair_per_8_neg_6, pair_per_6_neg_4, pair_per_6_neg_2, pair_per_8_neg_2];
    drawTile(array_of_pairs, colorOf.Wool, ctx);

    // Tile 9: Grain
    const pair_per_8_neg_8 = getCanvasCoordinatePairGiven(8, -8, widthOfCanvas);
    const pair_per_6_neg_8 = getCanvasCoordinatePairGiven(6, -8, widthOfCanvas);
    array_of_pairs = [pair_per_8_neg_6, pair_per_8_neg_8, pair_per_6_neg_8, pair_per_4_neg_6, pair_per_4_neg_4, pair_per_6_neg_4];
    drawTile(array_of_pairs, colorOf.Grain, ctx);

    // Tile 10: Brick
    const pair_per_6_neg_10 = getCanvasCoordinatePairGiven(6, -10, widthOfCanvas);
    const pair_per_4_neg_10 = getCanvasCoordinatePairGiven(4, -10, widthOfCanvas);
    const pair_per_2_neg_8 = getCanvasCoordinatePairGiven(2, -8, widthOfCanvas);
    array_of_pairs = [pair_per_6_neg_8, pair_per_6_neg_10, pair_per_4_neg_10, pair_per_2_neg_8, pair_per_2_neg_6, pair_per_4_neg_6];
    drawTile(array_of_pairs, colorOf.Brick, ctx);

    // Tile 11: Wood
    const pair_per_0_neg_8 = getCanvasCoordinatePairGiven(0, -8, widthOfCanvas);
    const pair_per_neg_2_neg_6 = getCanvasCoordinatePairGiven(-2, -6, widthOfCanvas);
    array_of_pairs = [pair_per_2_neg_6, pair_per_2_neg_8, pair_per_0_neg_8, pair_per_neg_2_neg_6, pair_per_neg_2_neg_4, pair_per_0_neg_4];
    drawTile(array_of_pairs, colorOf.Wood, ctx);

    // Tile 12: Grain
    const pair_per_neg_4_neg_6 = getCanvasCoordinatePairGiven(-4, -6, widthOfCanvas);
    const pair_per_neg_6_neg_4 = getCanvasCoordinatePairGiven(-6, -4, widthOfCanvas);
    const pair_per_neg_6_neg_2 = getCanvasCoordinatePairGiven(-6, -2, widthOfCanvas);
    array_of_pairs = [pair_per_neg_2_neg_4, pair_per_neg_2_neg_6, pair_per_neg_4_neg_6, pair_per_neg_6_neg_4, pair_per_neg_6_neg_2, pair_per_neg_4_neg_2];
    drawTile(array_of_pairs, colorOf.Grain, ctx);

    // Tile 13: Grain
    const pair_per_neg_8_0 = getCanvasCoordinatePairGiven(-8, 0, widthOfCanvas);
    const pair_per_neg_8_2 = getCanvasCoordinatePairGiven(-8, 2, widthOfCanvas);
    array_of_pairs = [pair_per_neg_4_0, pair_per_neg_4_neg_2, pair_per_neg_6_neg_2, pair_per_neg_8_0, pair_per_neg_8_2, pair_per_neg_6_2];
    drawTile(array_of_pairs, colorOf.Grain, ctx);

    // Tile 14: Ore
    const pair_per_neg_10_4 = getCanvasCoordinatePairGiven(-10, 4, widthOfCanvas);
    const pair_per_neg_10_6 = getCanvasCoordinatePairGiven(-10, 6, widthOfCanvas);
    const pair_per_neg_8_6 = getCanvasCoordinatePairGiven(-8, 6, widthOfCanvas);
    array_of_pairs = [pair_per_neg_6_4, pair_per_neg_6_2, pair_per_neg_8_2, pair_per_neg_10_4, pair_per_neg_10_6, pair_per_neg_8_6];
    drawTile(array_of_pairs, colorOf.Ore, ctx);

    // Tile 15: Wool
    const pair_per_neg_8_8 = getCanvasCoordinatePairGiven(-8, 8, widthOfCanvas);
    const pair_per_neg_6_8 = getCanvasCoordinatePairGiven(-6, 8, widthOfCanvas);
    array_of_pairs = [pair_per_neg_4_6, pair_per_neg_4_4, pair_per_neg_6_4, pair_per_neg_8_6, pair_per_neg_8_8, pair_per_neg_6_8];
    drawTile(array_of_pairs, colorOf.Wool, ctx);

    // Tile 16: Wood
    const pair_per_neg_2_8 = getCanvasCoordinatePairGiven(-2, 8, widthOfCanvas);
    const pair_per_neg_6_10 = getCanvasCoordinatePairGiven(-6, 10, widthOfCanvas);
    const pair_per_neg_4_10 = getCanvasCoordinatePairGiven(-4, 10, widthOfCanvas);
    array_of_pairs = [pair_per_neg_2_8, pair_per_neg_2_6, pair_per_neg_4_6, pair_per_neg_6_8, pair_per_neg_6_10, pair_per_neg_4_10];
    drawTile(array_of_pairs, colorOf.Wood, ctx);    

    // Tile 17: Brick
    const pair_per_2_6 = getCanvasCoordinatePairGiven(2, 6, widthOfCanvas);
    const pair_per_0_8 = getCanvasCoordinatePairGiven(0, 8, widthOfCanvas);
    array_of_pairs = [pair_per_2_6, pair_per_2_4, pair_per_0_4, pair_per_neg_2_6, pair_per_neg_2_8, pair_per_0_8];
    drawTile(array_of_pairs, colorOf.Brick, ctx);

    // Tile 18: Ore
    const pair_per_6_4 = getCanvasCoordinatePairGiven(6, 4, widthOfCanvas);
    const pair_per_4_6 = getCanvasCoordinatePairGiven(4, 6, widthOfCanvas);
    array_of_pairs = [pair_per_6_4, pair_per_6_2, pair_per_4_2, pair_per_2_4, pair_per_2_6, pair_per_4_6];
    drawTile(array_of_pairs, colorOf.Ore, ctx);

    drawToken(ctx, 4, -2, 4, widthOfCanvas);
    drawToken(ctx, 2, -4, 3, widthOfCanvas);
    drawToken(ctx, -2, -2, 11, widthOfCanvas);
    drawToken(ctx, -4, 2, 6, widthOfCanvas);
    drawToken(ctx, -2, 4, 4, widthOfCanvas);
    drawToken(ctx, 2, 2, 3, widthOfCanvas);
    drawToken(ctx, 6, 0, 5, widthOfCanvas);
    drawToken(ctx, 8, -4, 11, widthOfCanvas);
    drawToken(ctx, 6, -6, 6, widthOfCanvas);
    drawToken(ctx, 4, -8, 5, widthOfCanvas);
    drawToken(ctx, 0, -6, 8, widthOfCanvas);
    drawToken(ctx, -4, -4, 9, widthOfCanvas);
    drawToken(ctx, -6, 0, 12, widthOfCanvas);
    drawToken(ctx, -8, 4, 10, widthOfCanvas);
    drawToken(ctx, -6, 6, 2, widthOfCanvas);
    drawToken(ctx, -4, 8, 9, widthOfCanvas);
    drawToken(ctx, 0, 6, 10, widthOfCanvas);
    drawToken(ctx, 4, 4, 8, widthOfCanvas);

  // y = -15
  const pair_per_4_neg_15 = getCanvasCoordinatePairGiven(4, -15, widthOfCanvas);
  const pair_per_6_neg_15 = getCanvasCoordinatePairGiven(6, -15, widthOfCanvas);
  drawPath(ctx, [pair_per_4_neg_15, pair_per_6_neg_15], 'rgb(64,64,64)');

  // y = -14
  const pair_per_3_neg_14 = getCanvasCoordinatePairGiven(3, -14, widthOfCanvas);
  const pair_per_7_neg_14 = getCanvasCoordinatePairGiven(7, -14, widthOfCanvas);
  drawPath(ctx, [pair_per_3_neg_14, pair_per_7_neg_14], 'rgb(64,64,64)');

  // y = -13
  const pair_per_2_neg_13 = getCanvasCoordinatePairGiven(2, -13, widthOfCanvas);
  const pair_per_8_neg_13 = getCanvasCoordinatePairGiven(8, -13, widthOfCanvas);
  drawPath(ctx, [pair_per_2_neg_13, pair_per_8_neg_13], 'rgb(64,64,64)');

  // y = -12
  const pair_per_1_neg_12 = getCanvasCoordinatePairGiven(1, -12, widthOfCanvas);
  const pair_per_9_neg_12 = getCanvasCoordinatePairGiven(9, -12, widthOfCanvas);
  drawPath(ctx, [pair_per_1_neg_12, pair_per_9_neg_12], 'rgb(64,64,64)');

  // y = -11
  const pair_per_0_neg_11 = getCanvasCoordinatePairGiven(0, -11, widthOfCanvas);
  const pair_per_10_neg_11 = getCanvasCoordinatePairGiven(10, -11, widthOfCanvas);
  drawPath(ctx, [pair_per_0_neg_11, pair_per_10_neg_11], 'rgb(64,64,64)');

  // y = -10
  const pair_per_neg_1_neg_10 = getCanvasCoordinatePairGiven(-1, -10, widthOfCanvas);
  const pair_per_11_neg_10 = getCanvasCoordinatePairGiven(11, -10, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_1_neg_10, pair_per_11_neg_10], 'rgb(64,64,64)');

  // y = -9
  const pair_per_neg_2_neg_9 = getCanvasCoordinatePairGiven(-2, -9, widthOfCanvas);
  const pair_per_12_neg_9 = getCanvasCoordinatePairGiven(12, -9, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_2_neg_9, pair_per_12_neg_9], 'rgb(64,64,64)');

  // y = -8 
  const pair_per_neg_3_neg_8 = getCanvasCoordinatePairGiven(-3, -8, widthOfCanvas);
  const pair_per_13_neg_8 = getCanvasCoordinatePairGiven(13, -8, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_3_neg_8, pair_per_13_neg_8], 'rgb(64,64,64)');

  // y = -7
  const pair_per_neg_4_neg_7 = getCanvasCoordinatePairGiven(-4, -7, widthOfCanvas);
  const pair_per_14_neg_7 = getCanvasCoordinatePairGiven(14, -7, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_4_neg_7, pair_per_14_neg_7], 'rgb(64,64,64)');

  // y = -6
  const pair_per_neg_5_neg_6 = getCanvasCoordinatePairGiven(-5, -6, widthOfCanvas);
  const pair_per_15_neg_6 = getCanvasCoordinatePairGiven(15, -6, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_5_neg_6, pair_per_15_neg_6], 'rgb(64,64,64)');

  // y = -5
  const pair_per_neg_6_neg_5 = getCanvasCoordinatePairGiven(-6, -5, widthOfCanvas);
  const pair_per_16_neg_5 = getCanvasCoordinatePairGiven(16, -5, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_6_neg_5, pair_per_16_neg_5], 'rgb(64,64,64)');

  // y = -4
  const pair_per_neg_7_neg_4 = getCanvasCoordinatePairGiven(-7, -4, widthOfCanvas);
  const pair_per_15_neg_4 = getCanvasCoordinatePairGiven(15, -4, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_7_neg_4, pair_per_15_neg_4], 'rgb(64,64,64)');

  // y = -3
  const pair_per_neg_8_neg_3 = getCanvasCoordinatePairGiven(-8, -3, widthOfCanvas);
  const pair_per_14_neg_3 = getCanvasCoordinatePairGiven(14, -3, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_8_neg_3, pair_per_14_neg_3], 'rgb(64,64,64)');

  // y = -2
  const pair_per_neg_9_neg_2 = getCanvasCoordinatePairGiven(-9, -2, widthOfCanvas);
  const pair_per_13_neg_2 = getCanvasCoordinatePairGiven(13, -2, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_9_neg_2, pair_per_13_neg_2], 'rgb(64,64,64)');

  // y = -1
  const pair_per_neg_10_neg_1 = getCanvasCoordinatePairGiven(-10, -1, widthOfCanvas);
  const pair_per_12_neg_1 = getCanvasCoordinatePairGiven(12, -1, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_10_neg_1, pair_per_12_neg_1], 'rgb(64,64,64)');

  // x axis
  const pair_per_neg_11_0 = getCanvasCoordinatePairGiven(-11, 0, widthOfCanvas);
  const pair_per_11_0 = getCanvasCoordinatePairGiven(11, 0, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_11_0, pair_per_11_0], 'rgb(0,0,0)');

  // y = 1
  const pair_per_neg_12_1 = getCanvasCoordinatePairGiven(-12, 1, widthOfCanvas);
  const pair_per_10_1 = getCanvasCoordinatePairGiven(10, 1, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_12_1, pair_per_10_1], 'rgb(64,64,64)');

  // y = 2
  const pair_per_neg_13_2 = getCanvasCoordinatePairGiven(-13, 2, widthOfCanvas);
  const pair_per_9_2 = getCanvasCoordinatePairGiven(9, 2, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_13_2, pair_per_9_2], 'rgb(64,64,64)');

  // y = 3
  const pair_per_neg_14_3 = getCanvasCoordinatePairGiven(-14, 3, widthOfCanvas);
  const pair_per_8_3 = getCanvasCoordinatePairGiven(8, 3, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_14_3, pair_per_8_3], 'rgb(64,64,64)');

  // y = 4
  const pair_per_neg_13_4 = getCanvasCoordinatePairGiven(-13, 4, widthOfCanvas);
  const pair_per_7_4 = getCanvasCoordinatePairGiven(7, 4, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_13_4, pair_per_7_4], 'rgb(64,64,64)');

  // y = 5
  const pair_per_neg_12_5 = getCanvasCoordinatePairGiven(-12, 5, widthOfCanvas);
  const pair_per_6_5 = getCanvasCoordinatePairGiven(6, 5, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_12_5, pair_per_6_5], 'rgb(64,64,64)');

  // y = 6
  const pair_per_neg_11_6 = getCanvasCoordinatePairGiven(-11, 6, widthOfCanvas);
  const pair_per_5_6 = getCanvasCoordinatePairGiven(5, 6, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_11_6, pair_per_5_6], 'rgb(64,64,64)');

  // y = 7
  const pair_per_neg_10_7 = getCanvasCoordinatePairGiven(-10, 7, widthOfCanvas);
  const pair_per_4_7 = getCanvasCoordinatePairGiven(4, 7, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_10_7, pair_per_4_7], 'rgb(64,64,64)');

  // y = 8
  const pair_per_neg_9_8 = getCanvasCoordinatePairGiven(-9, 8, widthOfCanvas);
  const pair_per_3_8 = getCanvasCoordinatePairGiven(3, 8, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_9_8, pair_per_3_8], 'rgb(64,64,64)');

  // y = 9
  const pair_per_neg_8_9 = getCanvasCoordinatePairGiven(-8, 9, widthOfCanvas);
  const pair_per_2_9 = getCanvasCoordinatePairGiven(2, 9, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_8_9, pair_per_2_9], 'rgb(64,64,64)');

  // y = 10
  const pair_per_neg_7_10 = getCanvasCoordinatePairGiven(-7, 10, widthOfCanvas);
  const pair_per_1_10 = getCanvasCoordinatePairGiven(1, 10, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_7_10, pair_per_1_10], 'rgb(64,64,64)');

  // y = 11
  const pair_per_neg_6_11 = getCanvasCoordinatePairGiven(-6, 11, widthOfCanvas);
  const pair_per_0_11 = getCanvasCoordinatePairGiven(-0, 11, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_6_11, pair_per_0_11], 'rgb(64,64,64)');

  // y = 12
  const pair_per_neg_5_12 = getCanvasCoordinatePairGiven(-5, 12, widthOfCanvas);
  const pair_per_neg_1_12 = getCanvasCoordinatePairGiven(-1, 12, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_5_12, pair_per_neg_1_12], 'rgb(64,64,64)');

  // y = 13
  const pair_per_neg_4_13 = getCanvasCoordinatePairGiven(-4, 13, widthOfCanvas);
  const pair_per_neg_2_13 = getCanvasCoordinatePairGiven(-2, 13, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_4_13, pair_per_neg_2_13], 'rgb(64,64,64)');

  // x = -13
  const pair_per_neg_13_5 = getCanvasCoordinatePairGiven(-13, 5, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_13_2, pair_per_neg_13_5], 'rgb(64,64,64)');

  // x = -12
  drawPath(ctx, [pair_per_neg_12_1, pair_per_neg_12_5], 'rgb(64,64,64)');

  // x = -11
  drawPath(ctx, [pair_per_neg_11_0, pair_per_neg_11_6], 'rgb(64,64,64)');

  // x = -10
  drawPath(ctx, [pair_per_neg_10_neg_1, pair_per_neg_10_7], 'rgb(64,64,64)');

  // x = -9
  drawPath(ctx, [pair_per_neg_9_neg_2, pair_per_neg_9_8], 'rgb(64,64,64)');

  // x = -8
  drawPath(ctx, [pair_per_neg_8_neg_3, pair_per_neg_8_9], 'rgb(64,64,64)');

  // x = -7
  drawPath(ctx, [pair_per_neg_7_neg_4, pair_per_neg_7_10], 'rgb(64,64,64)');

  // x = -6
  drawPath(ctx, [pair_per_neg_6_neg_5, pair_per_neg_6_11], 'rgb(64,64,64)');

  // x = -5
  drawPath(ctx, [pair_per_neg_5_neg_6, pair_per_neg_5_12], 'rgb(64,64,64)');

  // x = -4
  drawPath(ctx, [pair_per_neg_4_neg_7, pair_per_neg_4_13], 'rgb(64,64,64)');

  // x = -3
  const pair_per_neg_3_14 = getCanvasCoordinatePairGiven(-3, 14, widthOfCanvas);
  drawPath(ctx, [pair_per_neg_3_neg_8, pair_per_neg_3_14], 'rgb(64,64,64)');

  // x = -2
  drawPath(ctx, [pair_per_neg_2_neg_9, pair_per_neg_2_13], 'rgb(64,64,64)');

  // x = -1
  drawPath(ctx, [pair_per_neg_1_neg_10, pair_per_neg_1_12], 'rgb(64,64,64)');

  // y axis
  drawPath(ctx, [pair_per_0_neg_11, pair_per_0_11], '#000000');

  // x = 1
  drawPath(ctx, [pair_per_1_neg_12, pair_per_1_10], 'rgb(64,64,64)');

  // x = 2
  drawPath(ctx, [pair_per_2_neg_13, pair_per_2_9], 'rgb(64,64,64)');

  // x = 3
  drawPath(ctx, [pair_per_3_neg_14, pair_per_3_8], 'rgb(64,64,64)');

  // x = 4
  drawPath(ctx, [pair_per_4_neg_15, pair_per_4_7], 'rgb(64,64,64)');

  // x = 5
  const pair_per_5_neg_16 = getCanvasCoordinatePairGiven(5, -16, widthOfCanvas);
  drawPath(ctx, [pair_per_5_neg_16, pair_per_5_6], 'rgb(64,64,64)');

  // x = 6
  drawPath(ctx, [pair_per_6_neg_15, pair_per_6_5], 'rgb(64,64,64)');

  // x = 7
  drawPath(ctx, [pair_per_7_neg_14, pair_per_7_4], 'rgb(64,64,64)');

  // x = 8
  drawPath(ctx, [pair_per_8_neg_13, pair_per_8_3], 'rgb(64,64,64)');

  // x = 9
  drawPath(ctx, [pair_per_9_neg_12, pair_per_9_2], 'rgb(64,64,64)');

  // x = 10
  drawPath(ctx, [pair_per_10_neg_11, pair_per_10_1], 'rgb(64,64,64)');

  // x = 11
  drawPath(ctx, [pair_per_11_neg_10, pair_per_11_0], 'rgb(64,64,64)');

  // x = 12
  drawPath(ctx, [pair_per_12_neg_9, pair_per_12_neg_1], 'rgb(64,64,64)');

  // x = 13
  drawPath(ctx, [pair_per_13_neg_8, pair_per_13_neg_2], 'rgb(64,64,64)');

  // x = 14
  drawPath(ctx, [pair_per_14_neg_7, pair_per_14_neg_3], 'rgb(64,64,64)');

  // x = 15
  drawPath(ctx, [pair_per_15_neg_6, pair_per_15_neg_4], 'rgb(64,64,64)');

  ctx.font = '10px Times New Roman';
  ctx.fillStyle = 'rgb(64,64,64)';
  for (let i = -14; i <= 15; i++) {
    for (let j = -16; j <= 14; j++) {
      const pair = getCanvasCoordinatePairGiven(i, j, widthOfCanvas);
      ctx.fillText('(' + i + ',' + j + ')', pair.x, pair.y);
    }
  }
};

type Props = {
  actionToComplete: string,
  respond: Function
}

function BaseBoardDisplayer(props: Props) {
  const mutableRefObject: MutableRefObject<HTMLCanvasElement | null> = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = mutableRefObject.current;
    if (canvas) {
      const handleMouseDown = function (e: MouseEvent) {
        console.log(props.actionToComplete);
        const match = /Player (.*), place your first settlement\./.exec(props.actionToComplete);
        if (match) {
          const mousePosition = getPairOfGridCoordinates(canvas, e);
          const action = 'When action to complete was \"' + props.actionToComplete + '\", player clicked base board at (' + mousePosition.x + ', ' + mousePosition.y + ') relative to grid.'
          props.respond(action);
        }
      }
      canvas.addEventListener("mousedown", handleMouseDown);
      const context = canvas.getContext('2d');
      if (context) {
        context.clearRect(0, 0, canvas.width, canvas.width);
        drawBoard(context, canvas.width);
      }
      return () => { canvas.removeEventListener("mousedown", handleMouseDown); };
    }
  }, [props.actionToComplete]);
  const height_of_webpage = window.innerHeight;

  return (
    <center>
      <canvas
        height = { height_of_webpage - 7 }
        ref = { mutableRefObject }
        style = { { 'backgroundColor': colorOf.Sea } }
        width = { height_of_webpage - 7 }
      />
    </center>
  );
}

export default BaseBoardDisplayer;