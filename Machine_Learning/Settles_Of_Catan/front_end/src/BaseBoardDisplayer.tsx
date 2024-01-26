import { useRef, useEffect, MutableRefObject } from 'react';
import TableDisplayer from './TableDisplayer';

function get_canvas_coordinate_pair_given(isometric_x_coordinate: number, isometric_y_coordinate: number, length_of_side_of_canvas: number) {
  const canvas_x_coordinate = length_of_side_of_canvas / 2 + (isometric_x_coordinate + isometric_y_coordinate) * length_of_side_of_canvas / 20;
  const tangent_of_30_degrees = 0.5773502691896257645091487805019574556476017512701268760186023264;
  const canvas_y_coordinate = (length_of_side_of_canvas / 2 + 3 * length_of_side_of_canvas / 10 + isometric_x_coordinate * length_of_side_of_canvas / 20 - isometric_y_coordinate * length_of_side_of_canvas / 20) * tangent_of_30_degrees;
  return {x: canvas_x_coordinate, y: canvas_y_coordinate};
}

const color_of = {
  Brick: '#AA4A44',
  Desert: '#F5D5A1',
  Grain: '#FADB5E',
  Ore: 'rgb(192,192,192)',
  Wood: '#014421',
  Wool: '#A6C964'
}

var drawBoard = function(ctx: CanvasRenderingContext2D, width_of_canvas: number) {

    // Tile 0: Desert
    let region = new Path2D();
    const pair_per_2_0 = get_canvas_coordinate_pair_given(2, 0, width_of_canvas);
    region.moveTo(pair_per_2_0.x, pair_per_2_0.y);
    const pair_per_2_neg_2 = get_canvas_coordinate_pair_given(2, -2, width_of_canvas);
    region.lineTo(pair_per_2_neg_2.x, pair_per_2_neg_2.y);
    const pair_per_0_neg_2 = get_canvas_coordinate_pair_given(0, -2, width_of_canvas);
    region.lineTo(pair_per_0_neg_2.x, pair_per_0_neg_2.y);
    const pair_per_neg_2_0 = get_canvas_coordinate_pair_given(-2, 0, width_of_canvas);
    region.lineTo(pair_per_neg_2_0.x, pair_per_neg_2_0.y);
    const pair_per_neg_2_2 = get_canvas_coordinate_pair_given(-2, 2, width_of_canvas);
    region.lineTo(pair_per_neg_2_2.x, pair_per_neg_2_2.y);
    const pair_per_0_2 = get_canvas_coordinate_pair_given(0, 2, width_of_canvas);
    region.lineTo(pair_per_0_2.x, pair_per_0_2.y);
    region.lineTo(pair_per_2_0.x, pair_per_2_0.y);
    region.closePath();
    ctx.fillStyle = color_of.Desert;
    ctx.fill(region);

    // Tile 1: Grain
    region = new Path2D();
    const pair_per_6_neg_2 = get_canvas_coordinate_pair_given(6, -2, width_of_canvas);
    region.moveTo(pair_per_6_neg_2.x, pair_per_6_neg_2.y);
    const pair_per_6_neg_4 = get_canvas_coordinate_pair_given(6, -4, width_of_canvas);
    region.lineTo(pair_per_6_neg_4.x, pair_per_6_neg_4.y);
    const pair_per_4_neg_4 = get_canvas_coordinate_pair_given(4, -4, width_of_canvas);
    region.lineTo(pair_per_4_neg_4.x, pair_per_4_neg_4.y);
    region.lineTo(pair_per_2_neg_2.x, pair_per_2_neg_2.y);
    region.lineTo(pair_per_2_0.x, pair_per_2_0.y);
    const pair_per_4_0 = get_canvas_coordinate_pair_given(4, 0, width_of_canvas);
    region.lineTo(pair_per_4_0.x, pair_per_4_0.y);
    region.lineTo(pair_per_6_neg_2.x, pair_per_6_neg_2.y);
    region.closePath();
    ctx.fillStyle = color_of.Grain;
    ctx.fill(region);

    // Tile 2: Ore
    region = new Path2D();
    region.moveTo(pair_per_4_neg_4.x, pair_per_4_neg_4.y);
    const pair_per_4_neg_6 = get_canvas_coordinate_pair_given(4, -6, width_of_canvas);
    region.lineTo(pair_per_4_neg_6.x, pair_per_4_neg_6.y);
    const pair_per_2_neg_6 = get_canvas_coordinate_pair_given(2, -6, width_of_canvas);
    region.lineTo(pair_per_2_neg_6.x, pair_per_2_neg_6.y);
    const pair_per_0_neg_4 = get_canvas_coordinate_pair_given(0, -4, width_of_canvas);
    region.lineTo(pair_per_0_neg_4.x, pair_per_0_neg_4.y);
    region.lineTo(pair_per_0_neg_2.x, pair_per_0_neg_2.y);
    region.lineTo(pair_per_2_neg_2.x, pair_per_2_neg_2.y);
    region.lineTo(pair_per_4_neg_4.x, pair_per_4_neg_4.y);
    region.closePath();
    ctx.fillStyle = color_of.Ore;
    ctx.fill(region);

    // Tile 4: Brick
    region = new Path2D();
    region.moveTo(pair_per_neg_2_2.x, pair_per_neg_2_2.y);
    region.lineTo(pair_per_neg_2_0.x, pair_per_neg_2_0.y);
    const pair_per_neg_4_0 = get_canvas_coordinate_pair_given(-4, 0, width_of_canvas);
    region.lineTo(pair_per_neg_4_0.x, pair_per_neg_4_0.y);
    const pair_per_neg_6_2 = get_canvas_coordinate_pair_given(-6, 2, width_of_canvas);
    region.lineTo(pair_per_neg_6_2.x, pair_per_neg_6_2.y);
    const pair_per_neg_6_4 = get_canvas_coordinate_pair_given(-6, 4, width_of_canvas);
    region.lineTo(pair_per_neg_6_4.x, pair_per_neg_6_4.y);
    const pair_per_neg_4_4 = get_canvas_coordinate_pair_given(-4, 4, width_of_canvas);
    region.lineTo(pair_per_neg_4_4.x, pair_per_neg_4_4.y);
    region.lineTo(pair_per_neg_2_2.x, pair_per_neg_2_2.y);
    region.closePath();
    ctx.fillStyle = color_of.Brick;
    ctx.fill(region);

  // Tile 14: Ore
  region = new Path2D();
  region.moveTo(pair_per_neg_6_4.x, pair_per_neg_6_4.y);
  region.lineTo(pair_per_neg_6_2.x, pair_per_neg_6_2.y);
  const pair_per_neg_8_2 = get_canvas_coordinate_pair_given(-8, 2, width_of_canvas);
  region.lineTo(pair_per_neg_8_2.x, pair_per_neg_8_2.y);
  const pair_per_neg_10_4 = get_canvas_coordinate_pair_given(-10, 4, width_of_canvas);
  region.lineTo(pair_per_neg_10_4.x, pair_per_neg_10_4.y);
  const pair_per_neg_10_6 = get_canvas_coordinate_pair_given(-10, 6, width_of_canvas);
  region.lineTo(pair_per_neg_10_6.x, pair_per_neg_10_6.y);
  const pair_per_neg_8_6 = get_canvas_coordinate_pair_given(-8, 6, width_of_canvas);
  region.lineTo(pair_per_neg_8_6.x, pair_per_neg_8_6.y);
  region.lineTo(pair_per_neg_6_4.x, pair_per_neg_6_4.y);
  region.closePath();
  ctx.fillStyle = color_of.Ore;
  ctx.fill(region);

  // Tile 18: Ore
  region = new Path2D();
  const pair_per_6_4 = get_canvas_coordinate_pair_given(6, 4, width_of_canvas);
  region.moveTo(pair_per_6_4.x, pair_per_6_4.y);
  const pair_per_6_2 = get_canvas_coordinate_pair_given(6, 2, width_of_canvas);
  region.lineTo(pair_per_6_2.x, pair_per_6_2.y);
  const pair_per_4_2 = get_canvas_coordinate_pair_given(4, 2, width_of_canvas);
  region.lineTo(pair_per_4_2.x, pair_per_4_2.y);
  const pair_per_2_4 = get_canvas_coordinate_pair_given(2, 4, width_of_canvas);
  region.lineTo(pair_per_2_4.x, pair_per_2_4.y);
  const pair_per_2_6 = get_canvas_coordinate_pair_given(2, 6, width_of_canvas);
  region.lineTo(pair_per_2_6.x, pair_per_2_6.y);
  const pair_per_4_6 = get_canvas_coordinate_pair_given(4, 6, width_of_canvas);
  region.lineTo(pair_per_4_6.x, pair_per_4_6.y);
  region.lineTo(pair_per_6_4.x, pair_per_6_4.y);
  region.closePath();
  ctx.fillStyle = color_of.Ore;
  ctx.fill(region);

  // y = -14
  ctx.beginPath();
  const pair_per_4_neg_14 = get_canvas_coordinate_pair_given(4, -14, width_of_canvas);
  ctx.moveTo(pair_per_4_neg_14.x, pair_per_4_neg_14.y);
  const pair_per_5_neg_14 = get_canvas_coordinate_pair_given(5, -14, width_of_canvas);
  ctx.lineTo(pair_per_5_neg_14.x, pair_per_5_neg_14.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = -13
  ctx.beginPath();
  const pair_per_3_neg_13 = get_canvas_coordinate_pair_given(3, -13, width_of_canvas);
  ctx.moveTo(pair_per_3_neg_13.x, pair_per_3_neg_13.y);
  const pair_per_6_neg_13 = get_canvas_coordinate_pair_given(6, -13, width_of_canvas);
  ctx.lineTo(pair_per_6_neg_13.x, pair_per_6_neg_13.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = -12
  ctx.beginPath();
  const pair_per_2_neg_12 = get_canvas_coordinate_pair_given(2, -12, width_of_canvas);
  ctx.moveTo(pair_per_2_neg_12.x, pair_per_2_neg_12.y);
  const pair_per_7_neg_12 = get_canvas_coordinate_pair_given(7, -12, width_of_canvas);
  ctx.lineTo(pair_per_7_neg_12.x, pair_per_7_neg_12.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = -11
  ctx.beginPath();
  const pair_per_1_neg_11 = get_canvas_coordinate_pair_given(1, -11, width_of_canvas);
  ctx.moveTo(pair_per_1_neg_11.x, pair_per_1_neg_11.y);
  const pair_per_8_neg_11 = get_canvas_coordinate_pair_given(8, -11, width_of_canvas);
  ctx.lineTo(pair_per_8_neg_11.x, pair_per_8_neg_11.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = -10
  ctx.beginPath();
  const pair_per_0_neg_10 = get_canvas_coordinate_pair_given(0, -10, width_of_canvas);
  ctx.moveTo(pair_per_0_neg_10.x, pair_per_0_neg_10.y);
  const pair_per_9_neg_10 = get_canvas_coordinate_pair_given(9, -10, width_of_canvas);
  ctx.lineTo(pair_per_9_neg_10.x, pair_per_9_neg_10.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = -9
  ctx.beginPath();
  const pair_per_neg_1_neg_9 = get_canvas_coordinate_pair_given(-1, -9, width_of_canvas);
  ctx.moveTo(pair_per_neg_1_neg_9.x, pair_per_neg_1_neg_9.y);
  const pair_per_10_neg_9 = get_canvas_coordinate_pair_given(10, -9, width_of_canvas);
  ctx.lineTo(pair_per_10_neg_9.x, pair_per_10_neg_9.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = -8
  ctx.beginPath();
  const pair_per_neg_2_neg_8 = get_canvas_coordinate_pair_given(-2, -8, width_of_canvas);
  ctx.moveTo(pair_per_neg_2_neg_8.x, pair_per_neg_2_neg_8.y);
  const pair_per_11_neg_8 = get_canvas_coordinate_pair_given(11, -8, width_of_canvas);
  ctx.lineTo(pair_per_11_neg_8.x, pair_per_11_neg_8.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = -7
  ctx.beginPath();
  const pair_per_neg_3_neg_7 = get_canvas_coordinate_pair_given(-3, -7, width_of_canvas);
  ctx.moveTo(pair_per_neg_3_neg_7.x, pair_per_neg_3_neg_7.y);
  const pair_per_12_neg_7 = get_canvas_coordinate_pair_given(12, -7, width_of_canvas);
  ctx.lineTo(pair_per_12_neg_7.x, pair_per_12_neg_7.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = -6
  ctx.beginPath();
  const pair_per_neg_4_neg_6 = get_canvas_coordinate_pair_given(-4, -6, width_of_canvas);
  ctx.moveTo(pair_per_neg_4_neg_6.x, pair_per_neg_4_neg_6.y);
  const pair_per_13_neg_6 = get_canvas_coordinate_pair_given(13, -6, width_of_canvas);
  ctx.lineTo(pair_per_13_neg_6.x, pair_per_13_neg_6.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = -5
  ctx.beginPath();
  const pair_per_neg_5_neg_5 = get_canvas_coordinate_pair_given(-5, -5, width_of_canvas);
  ctx.moveTo(pair_per_neg_5_neg_5.x, pair_per_neg_5_neg_5.y);
  const pair_per_14_neg_5 = get_canvas_coordinate_pair_given(14, -5, width_of_canvas);
  ctx.lineTo(pair_per_14_neg_5.x, pair_per_14_neg_5.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = -4
  ctx.beginPath();
  const pair_per_neg_6_neg_4 = get_canvas_coordinate_pair_given(-6, -4, width_of_canvas);
  ctx.moveTo(pair_per_neg_6_neg_4.x, pair_per_neg_6_neg_4.y);
  const pair_per_14_neg_4 = get_canvas_coordinate_pair_given(14, -4, width_of_canvas);
  ctx.lineTo(pair_per_14_neg_4.x, pair_per_14_neg_4.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = -3
  ctx.beginPath();
  const pair_per_neg_7_neg_3 = get_canvas_coordinate_pair_given(-7, -3, width_of_canvas);
  ctx.moveTo(pair_per_neg_7_neg_3.x, pair_per_neg_7_neg_3.y);
  const pair_per_13_neg_3 = get_canvas_coordinate_pair_given(13, -3, width_of_canvas);
  ctx.lineTo(pair_per_13_neg_3.x, pair_per_13_neg_3.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = -2
  ctx.beginPath();
  const pair_per_neg_8_neg_2 = get_canvas_coordinate_pair_given(-8, -2, width_of_canvas);
  ctx.moveTo(pair_per_neg_8_neg_2.x, pair_per_neg_8_neg_2.y);
  const pair_per_12_neg_2 = get_canvas_coordinate_pair_given(12, -2, width_of_canvas);
  ctx.lineTo(pair_per_12_neg_2.x, pair_per_12_neg_2.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = -1
  ctx.beginPath();
  const pair_per_neg_9_neg_1 = get_canvas_coordinate_pair_given(-9, -1, width_of_canvas);
  ctx.moveTo(pair_per_neg_9_neg_1.x, pair_per_neg_9_neg_1.y);
  const pair_per_11_neg_1 = get_canvas_coordinate_pair_given(11, -1, width_of_canvas);
  ctx.lineTo(pair_per_11_neg_1.x, pair_per_11_neg_1.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x axis
  ctx.beginPath();
  const pair_per_neg_10_0 = get_canvas_coordinate_pair_given(-10, 0, width_of_canvas);
  ctx.moveTo(pair_per_neg_10_0.x, pair_per_neg_10_0.y);
  const pair_per_10_0 = get_canvas_coordinate_pair_given(10, 0, width_of_canvas);
  ctx.lineTo(pair_per_10_0.x, pair_per_10_0.y);
  ctx.strokeStyle = 'rgb(0,0,0)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = 1
  ctx.beginPath();
  const pair_per_neg_11_1 = get_canvas_coordinate_pair_given(-11, 1, width_of_canvas);
  ctx.moveTo(pair_per_neg_11_1.x, pair_per_neg_11_1.y);
  const pair_per_9_1 = get_canvas_coordinate_pair_given(9, 1, width_of_canvas);
  ctx.lineTo(pair_per_9_1.x, pair_per_9_1.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = 2
  ctx.beginPath();
  const pair_per_neg_12_2 = get_canvas_coordinate_pair_given(-12, 2, width_of_canvas);
  ctx.moveTo(pair_per_neg_12_2.x, pair_per_neg_12_2.y);
  const pair_per_8_2 = get_canvas_coordinate_pair_given(8, 2, width_of_canvas);
  ctx.lineTo(pair_per_8_2.x, pair_per_8_2.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = 3
  ctx.beginPath();
  const pair_per_neg_13_3 = get_canvas_coordinate_pair_given(-13, 3, width_of_canvas);
  ctx.moveTo(pair_per_neg_13_3.x, pair_per_neg_13_3.y);
  const pair_per_7_3 = get_canvas_coordinate_pair_given(7, 3, width_of_canvas);
  ctx.lineTo(pair_per_7_3.x, pair_per_7_3.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = 4
  ctx.beginPath();
  const pair_per_neg_12_4 = get_canvas_coordinate_pair_given(-12, 4, width_of_canvas);
  ctx.moveTo(pair_per_neg_12_4.x, pair_per_neg_12_4.y);
  ctx.lineTo(pair_per_6_4.x, pair_per_6_4.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = 5
  ctx.beginPath();
  const pair_per_neg_11_5 = get_canvas_coordinate_pair_given(-11, 5, width_of_canvas);
  ctx.moveTo(pair_per_neg_11_5.x, pair_per_neg_11_5.y);
  const pair_per_5_5 = get_canvas_coordinate_pair_given(5, 5, width_of_canvas);
  ctx.lineTo(pair_per_5_5.x, pair_per_5_5.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = 6
  ctx.beginPath();
  ctx.moveTo(pair_per_neg_10_6.x, pair_per_neg_10_6.y);
  ctx.lineTo(pair_per_4_6.x, pair_per_4_6.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = 7
  ctx.beginPath();
  const pair_per_neg_9_7 = get_canvas_coordinate_pair_given(-9, 7, width_of_canvas);
  ctx.moveTo(pair_per_neg_9_7.x, pair_per_neg_9_7.y);
  const pair_per_3_7 = get_canvas_coordinate_pair_given(3, 7, width_of_canvas);
  ctx.lineTo(pair_per_3_7.x, pair_per_3_7.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = 8
  ctx.beginPath();
  const pair_per_neg_8_8 = get_canvas_coordinate_pair_given(-8, 8, width_of_canvas);
  ctx.moveTo(pair_per_neg_8_8.x, pair_per_neg_8_8.y);
  const pair_per_2_8 = get_canvas_coordinate_pair_given(2, 8, width_of_canvas);
  ctx.lineTo(pair_per_2_8.x, pair_per_2_8.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = 9
  ctx.beginPath();
  const pair_per_neg_7_9 = get_canvas_coordinate_pair_given(-7, 9, width_of_canvas);
  ctx.moveTo(pair_per_neg_7_9.x, pair_per_neg_7_9.y);
  const pair_per_1_9 = get_canvas_coordinate_pair_given(1, 9, width_of_canvas);
  ctx.lineTo(pair_per_1_9.x, pair_per_1_9.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = 10
  ctx.beginPath();
  const pair_per_neg_6_10 = get_canvas_coordinate_pair_given(-6, 10, width_of_canvas);
  ctx.moveTo(pair_per_neg_6_10.x, pair_per_neg_6_10.y);
  const pair_per_0_10 = get_canvas_coordinate_pair_given(0, 10, width_of_canvas);
  ctx.lineTo(pair_per_0_10.x, pair_per_0_10.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = 11
  ctx.beginPath();
  const pair_per_neg_5_11 = get_canvas_coordinate_pair_given(-5, 11, width_of_canvas);
  ctx.moveTo(pair_per_neg_5_11.x, pair_per_neg_5_11.y);
  const pair_per_neg_1_11 = get_canvas_coordinate_pair_given(-1, 11, width_of_canvas);
  ctx.lineTo(pair_per_neg_1_11.x, pair_per_neg_1_11.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y = 12
  ctx.beginPath();
  const pair_per_neg_4_12 = get_canvas_coordinate_pair_given(-4, 12, width_of_canvas);
  ctx.moveTo(pair_per_neg_4_12.x, pair_per_neg_4_12.y);
  const pair_per_neg_2_12 = get_canvas_coordinate_pair_given(-2, 12, width_of_canvas);
  ctx.lineTo(pair_per_neg_2_12.x, pair_per_neg_2_12.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = -12
  ctx.beginPath();
  ctx.moveTo(pair_per_neg_12_2.x, pair_per_neg_12_2.y);
  ctx.lineTo(pair_per_neg_12_4.x, pair_per_neg_12_4.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = -11
  ctx.beginPath();
  ctx.moveTo(pair_per_neg_11_1.x, pair_per_neg_11_1.y);
  ctx.lineTo(pair_per_neg_11_5.x, pair_per_neg_11_5.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = -10
  ctx.beginPath();
  ctx.moveTo(pair_per_neg_10_0.x, pair_per_neg_10_0.y);
  ctx.lineTo(pair_per_neg_10_6.x, pair_per_neg_10_6.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = -9
  ctx.beginPath();
  ctx.moveTo(pair_per_neg_9_neg_1.x, pair_per_neg_9_neg_1.y);
  ctx.lineTo(pair_per_neg_9_7.x, pair_per_neg_9_7.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = -8
  ctx.beginPath();
  ctx.moveTo(pair_per_neg_8_neg_2.x, pair_per_neg_8_neg_2.y);
  ctx.lineTo(pair_per_neg_8_8.x, pair_per_neg_8_8.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = -7
  ctx.beginPath();
  ctx.moveTo(pair_per_neg_7_neg_3.x, pair_per_neg_7_neg_3.y);
  ctx.lineTo(pair_per_neg_7_9.x, pair_per_neg_7_9.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = -6
  ctx.beginPath();
  ctx.moveTo(pair_per_neg_6_neg_4.x, pair_per_neg_6_neg_4.y);
  ctx.lineTo(pair_per_neg_6_10.x, pair_per_neg_6_10.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = -5
  ctx.beginPath();
  ctx.moveTo(pair_per_neg_5_neg_5.x, pair_per_neg_5_neg_5.y);
  ctx.lineTo(pair_per_neg_5_11.x, pair_per_neg_5_11.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = -4
  ctx.beginPath();
  ctx.moveTo(pair_per_neg_4_neg_6.x, pair_per_neg_4_neg_6.y);
  ctx.lineTo(pair_per_neg_4_12.x, pair_per_neg_4_12.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = -3
  ctx.beginPath();
  ctx.moveTo(pair_per_neg_3_neg_7.x, pair_per_neg_3_neg_7.y);
  const pair_per_neg_3_13 = get_canvas_coordinate_pair_given(-3, 13, width_of_canvas);
  ctx.lineTo(pair_per_neg_3_13.x, pair_per_neg_3_13.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = -2
  ctx.beginPath();
  ctx.moveTo(pair_per_neg_2_neg_8.x, pair_per_neg_2_neg_8.y);
  ctx.lineTo(pair_per_neg_2_12.x, pair_per_neg_2_12.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = -1
  ctx.beginPath();
  ctx.moveTo(pair_per_neg_1_neg_9.x, pair_per_neg_1_neg_9.y);
  ctx.lineTo(pair_per_neg_1_11.x, pair_per_neg_1_11.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // y axis
  ctx.beginPath();
  ctx.moveTo(pair_per_0_neg_10.x, pair_per_0_neg_10.y);
  ctx.lineTo(pair_per_0_10.x, pair_per_0_10.y);
  ctx.strokeStyle = 'rgb(0,0,0)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 1
  ctx.beginPath();
  ctx.moveTo(pair_per_1_neg_11.x, pair_per_1_neg_11.y);
  ctx.lineTo(pair_per_1_9.x, pair_per_1_9.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 2
  ctx.beginPath();
  ctx.moveTo(pair_per_2_neg_12.x, pair_per_2_neg_12.y);
  ctx.lineTo(pair_per_2_8.x, pair_per_2_8.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 3
  ctx.beginPath();
  ctx.moveTo(pair_per_3_neg_13.x, pair_per_3_neg_13.y);
  ctx.lineTo(pair_per_3_7.x, pair_per_3_7.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 4
  ctx.beginPath();
  ctx.moveTo(pair_per_4_neg_14.x, pair_per_4_neg_14.y);
  ctx.lineTo(pair_per_4_6.x, pair_per_4_6.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 5
  ctx.beginPath();
  ctx.moveTo(pair_per_5_neg_14.x, pair_per_5_neg_14.y);
  ctx.lineTo(pair_per_5_5.x, pair_per_5_5.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 6
  ctx.beginPath();
  ctx.moveTo(pair_per_6_neg_13.x, pair_per_6_neg_13.y);
  ctx.lineTo(pair_per_6_4.x, pair_per_6_4.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 7
  ctx.beginPath();
  ctx.moveTo(pair_per_7_neg_12.x, pair_per_7_neg_12.y);
  ctx.lineTo(pair_per_7_3.x, pair_per_7_3.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 8
  ctx.beginPath();
  ctx.moveTo(pair_per_8_neg_11.x, pair_per_8_neg_11.y);
  ctx.lineTo(pair_per_8_2.x, pair_per_8_2.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 9
  ctx.beginPath();
  ctx.moveTo(pair_per_9_neg_10.x, pair_per_9_neg_10.y);
  ctx.lineTo(pair_per_9_1.x, pair_per_9_1.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 10
  ctx.beginPath();
  ctx.moveTo(pair_per_10_neg_9.x, pair_per_10_neg_9.y);
  ctx.lineTo(pair_per_10_0.x, pair_per_10_0.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 11
  ctx.beginPath();
  ctx.moveTo(pair_per_11_neg_8.x, pair_per_11_neg_8.y);
  ctx.lineTo(pair_per_11_neg_1.x, pair_per_11_neg_1.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 12
  ctx.beginPath();
  ctx.moveTo(pair_per_12_neg_7.x, pair_per_12_neg_7.y);
  ctx.lineTo(pair_per_12_neg_2.x, pair_per_12_neg_2.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 13
  ctx.beginPath();
  ctx.moveTo(pair_per_13_neg_6.x, pair_per_13_neg_6.y);
  ctx.lineTo(pair_per_13_neg_3.x, pair_per_13_neg_3.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // x = 14
  ctx.beginPath();
  ctx.moveTo(pair_per_14_neg_5.x, pair_per_14_neg_5.y);
  ctx.lineTo(pair_per_14_neg_4.x, pair_per_14_neg_4.y);
  ctx.strokeStyle = 'rgb(128,128,128)';
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.fillStyle = 'rgb(128,128,128)';
  for (let i = -13; i <= 14; i++) {
    for (let j = -14; j <= 13; j++) {
      const pair = get_canvas_coordinate_pair_given(i, j, width_of_canvas);
      ctx.fillText('(' + i + ',' + j + ')', pair.x, pair.y);
    }
  }
};

function BaseBoardDisplayer() {
  const mutableRefObject: MutableRefObject<HTMLCanvasElement | null> = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = mutableRefObject.current;
    if (canvas) {
      const context = canvas.getContext('2d');
      if (context) {
        drawBoard(context, canvas.width);
      }
    }
  }, []);
  const height_of_webpage = window.innerHeight;
  return (
    <center>
      <canvas
        ref = { mutableRefObject }
        width = { height_of_webpage - 4 }
        height = { height_of_webpage - 4 }
      />
    </center>
  );
}

export default BaseBoardDisplayer;