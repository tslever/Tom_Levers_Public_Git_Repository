import { useRef, useEffect, MutableRefObject } from 'react';
import ActionDisplayer from './ActionDisplayer';

function getMousePosition(canvas: HTMLCanvasElement, event: MouseEvent) {
  const DOM_rect = canvas.getBoundingClientRect();
  const canvas_x_coordinate = event.clientX - DOM_rect.left;
  const canvas_y_coordinate = event.clientY - DOM_rect.top;
  const pair_of_isometric_coordinates = get_isometric_coordinate_pair_given(canvas_x_coordinate, canvas_y_coordinate, canvas.width);
  return 'Mouse clicked at (' + pair_of_isometric_coordinates.x + ', ' + pair_of_isometric_coordinates.y + ') relative to canvas.';
}

const tangent_of_30_degrees = 0.5773502691896257645091487805019574556476017512701268760186023264;

function get_canvas_coordinate_pair_given(isometric_x_coordinate: number, isometric_y_coordinate: number, length_of_side_of_canvas: number) {
  const canvas_x_coordinate = length_of_side_of_canvas / 2 + (isometric_x_coordinate + isometric_y_coordinate) * length_of_side_of_canvas / 22;
  const canvas_y_coordinate = (length_of_side_of_canvas / 2 + 3 * length_of_side_of_canvas / 11 + isometric_x_coordinate * length_of_side_of_canvas / 22 - isometric_y_coordinate * length_of_side_of_canvas / 22) * tangent_of_30_degrees;
  return {x: canvas_x_coordinate, y: canvas_y_coordinate};
}

function get_isometric_coordinate_pair_given(canvas_x_coordinate: number, canvas_y_coordinate: number, length_of_side_of_canvas: number) {
  const isometric_x_coordinate = Math.round( 11 / length_of_side_of_canvas * canvas_x_coordinate + 11 / length_of_side_of_canvas * canvas_y_coordinate / tangent_of_30_degrees - 14 );
  const isometric_y_coordinate = Math.round( 3 + 11 / length_of_side_of_canvas * (canvas_x_coordinate - canvas_y_coordinate / tangent_of_30_degrees) );
  return {x: isometric_x_coordinate, y: isometric_y_coordinate};
}

const color_of = {
  Brick: '#AA4A44',
  Desert: '#F5D5A1',
  Grain: '#FADB5E',
  Ore: 'rgb(128,128,128)',
  Sea: '#00c5ff',
  Wood: '#014421',
  Wool: '#A6C964'
}

const drawGridLine = function(ctx: CanvasRenderingContext2D, left_endpoint: {x: number, y: number}, right_endpoint: {x: number, y: number}, color?: string) {
  ctx.beginPath();
  ctx.moveTo(left_endpoint.x, left_endpoint.y);
  ctx.lineTo(right_endpoint.x, right_endpoint.y);
  if (color) {
    ctx.strokeStyle = color;
  } else {
    ctx.strokeStyle = 'rgb(64,64,64)';
  }
  ctx.lineWidth = 1;
  ctx.stroke();
}

const drawBoard = function(ctx: CanvasRenderingContext2D, width_of_canvas: number) {

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

  // y = -15
  const pair_per_4_neg_15 = get_canvas_coordinate_pair_given(4, -15, width_of_canvas);
  const pair_per_6_neg_15 = get_canvas_coordinate_pair_given(6, -15, width_of_canvas);
  drawGridLine(ctx, pair_per_4_neg_15, pair_per_6_neg_15);

  // y = -14
  const pair_per_3_neg_14 = get_canvas_coordinate_pair_given(3, -14, width_of_canvas);
  const pair_per_7_neg_14 = get_canvas_coordinate_pair_given(7, -14, width_of_canvas);
  drawGridLine(ctx, pair_per_3_neg_14, pair_per_7_neg_14);

  // y = -13
  const pair_per_2_neg_13 = get_canvas_coordinate_pair_given(2, -13, width_of_canvas);
  const pair_per_8_neg_13 = get_canvas_coordinate_pair_given(8, -13, width_of_canvas);
  drawGridLine(ctx, pair_per_2_neg_13, pair_per_8_neg_13);

  // y = -12
  const pair_per_1_neg_12 = get_canvas_coordinate_pair_given(1, -12, width_of_canvas);
  const pair_per_9_neg_12 = get_canvas_coordinate_pair_given(9, -12, width_of_canvas);
  drawGridLine(ctx, pair_per_1_neg_12, pair_per_9_neg_12);

  // y = -11
  const pair_per_0_neg_11 = get_canvas_coordinate_pair_given(0, -11, width_of_canvas);
  const pair_per_10_neg_11 = get_canvas_coordinate_pair_given(10, -11, width_of_canvas);
  drawGridLine(ctx, pair_per_0_neg_11, pair_per_10_neg_11);

  // y = -10
  const pair_per_neg_1_neg_10 = get_canvas_coordinate_pair_given(-1, -10, width_of_canvas);
  const pair_per_11_neg_10 = get_canvas_coordinate_pair_given(11, -10, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_1_neg_10, pair_per_11_neg_10);

  // y = -9
  const pair_per_neg_2_neg_9 = get_canvas_coordinate_pair_given(-2, -9, width_of_canvas);
  const pair_per_12_neg_9 = get_canvas_coordinate_pair_given(12, -9, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_2_neg_9, pair_per_12_neg_9);

  // y = -8 
  const pair_per_neg_3_neg_8 = get_canvas_coordinate_pair_given(-3, -8, width_of_canvas);
  const pair_per_13_neg_8 = get_canvas_coordinate_pair_given(13, -8, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_3_neg_8, pair_per_13_neg_8);

  // y = -7
  const pair_per_neg_4_neg_7 = get_canvas_coordinate_pair_given(-4, -7, width_of_canvas);
  const pair_per_14_neg_7 = get_canvas_coordinate_pair_given(14, -7, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_4_neg_7, pair_per_14_neg_7);

  // y = -6
  const pair_per_neg_5_neg_6 = get_canvas_coordinate_pair_given(-5, -6, width_of_canvas);
  const pair_per_15_neg_6 = get_canvas_coordinate_pair_given(15, -6, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_5_neg_6, pair_per_15_neg_6);

  // y = -5
  const pair_per_neg_6_neg_5 = get_canvas_coordinate_pair_given(-6, -5, width_of_canvas);
  const pair_per_16_neg_5 = get_canvas_coordinate_pair_given(16, -5, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_6_neg_5, pair_per_16_neg_5);

  // y = -4
  const pair_per_neg_7_neg_4 = get_canvas_coordinate_pair_given(-7, -4, width_of_canvas);
  const pair_per_15_neg_4 = get_canvas_coordinate_pair_given(15, -4, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_7_neg_4, pair_per_15_neg_4);

  // y = -3
  const pair_per_neg_8_neg_3 = get_canvas_coordinate_pair_given(-8, -3, width_of_canvas);
  const pair_per_14_neg_3 = get_canvas_coordinate_pair_given(14, -3, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_8_neg_3, pair_per_14_neg_3);

  // y = -2
  const pair_per_neg_9_neg_2 = get_canvas_coordinate_pair_given(-9, -2, width_of_canvas);
  const pair_per_13_neg_2 = get_canvas_coordinate_pair_given(13, -2, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_9_neg_2, pair_per_13_neg_2);

  // y = -1
  const pair_per_neg_10_neg_1 = get_canvas_coordinate_pair_given(-10, -1, width_of_canvas);
  const pair_per_12_neg_1 = get_canvas_coordinate_pair_given(12, -1, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_10_neg_1, pair_per_12_neg_1);

  // x axis
  const pair_per_neg_11_0 = get_canvas_coordinate_pair_given(-11, 0, width_of_canvas);
  const pair_per_11_0 = get_canvas_coordinate_pair_given(11, 0, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_11_0, pair_per_11_0, '#000000');

  // y = 1
  const pair_per_neg_12_1 = get_canvas_coordinate_pair_given(-12, 1, width_of_canvas);
  const pair_per_10_1 = get_canvas_coordinate_pair_given(10, 1, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_12_1, pair_per_10_1);

  // y = 2
  const pair_per_neg_13_2 = get_canvas_coordinate_pair_given(-13, 2, width_of_canvas);
  const pair_per_9_2 = get_canvas_coordinate_pair_given(9, 2, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_13_2, pair_per_9_2);

  // y = 3
  const pair_per_neg_14_3 = get_canvas_coordinate_pair_given(-14, 3, width_of_canvas);
  const pair_per_8_3 = get_canvas_coordinate_pair_given(8, 3, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_14_3, pair_per_8_3);

  // y = 4
  const pair_per_neg_13_4 = get_canvas_coordinate_pair_given(-13, 4, width_of_canvas);
  const pair_per_7_4 = get_canvas_coordinate_pair_given(7, 4, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_13_4, pair_per_7_4);

  // y = 5
  const pair_per_neg_12_5 = get_canvas_coordinate_pair_given(-12, 5, width_of_canvas);
  const pair_per_6_5 = get_canvas_coordinate_pair_given(6, 5, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_12_5, pair_per_6_5);

  // y = 6
  const pair_per_neg_11_6 = get_canvas_coordinate_pair_given(-11, 6, width_of_canvas);
  const pair_per_5_6 = get_canvas_coordinate_pair_given(5, 6, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_11_6, pair_per_5_6);

  // y = 7
  const pair_per_neg_10_7 = get_canvas_coordinate_pair_given(-10, 7, width_of_canvas);
  const pair_per_4_7 = get_canvas_coordinate_pair_given(4, 7, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_10_7, pair_per_4_7);

  // y = 8
  const pair_per_neg_9_8 = get_canvas_coordinate_pair_given(-9, 8, width_of_canvas);
  const pair_per_3_8 = get_canvas_coordinate_pair_given(3, 8, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_9_8, pair_per_3_8);

  // y = 9
  const pair_per_neg_8_9 = get_canvas_coordinate_pair_given(-8, 9, width_of_canvas);
  const pair_per_2_9 = get_canvas_coordinate_pair_given(2, 9, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_8_9, pair_per_2_9);

  // y = 10
  const pair_per_neg_7_10 = get_canvas_coordinate_pair_given(-7, 10, width_of_canvas);
  const pair_per_1_10 = get_canvas_coordinate_pair_given(1, 10, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_7_10, pair_per_1_10);

  // y = 11
  const pair_per_neg_6_11 = get_canvas_coordinate_pair_given(-6, 11, width_of_canvas);
  const pair_per_0_11 = get_canvas_coordinate_pair_given(-0, 11, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_6_11, pair_per_0_11);

  // y = 12
  const pair_per_neg_5_12 = get_canvas_coordinate_pair_given(-5, 12, width_of_canvas);
  const pair_per_neg_1_12 = get_canvas_coordinate_pair_given(-1, 12, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_5_12, pair_per_neg_1_12);

  // y = 13
  const pair_per_neg_4_13 = get_canvas_coordinate_pair_given(-4, 13, width_of_canvas);
  const pair_per_neg_2_13 = get_canvas_coordinate_pair_given(-2, 13, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_4_13, pair_per_neg_2_13);

  // x = -13
  const pair_per_neg_13_5 = get_canvas_coordinate_pair_given(-13, 5, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_13_2, pair_per_neg_13_5);

  // x = -12
  drawGridLine(ctx, pair_per_neg_12_1, pair_per_neg_12_5);

  // x = -11
  drawGridLine(ctx, pair_per_neg_11_0, pair_per_neg_11_6);

  // x = -10
  drawGridLine(ctx, pair_per_neg_10_neg_1, pair_per_neg_10_7);

  // x = -9
  drawGridLine(ctx, pair_per_neg_9_neg_2, pair_per_neg_9_8);

  // x = -8
  drawGridLine(ctx, pair_per_neg_8_neg_3, pair_per_neg_8_9);

  // x = -7
  drawGridLine(ctx, pair_per_neg_7_neg_4, pair_per_neg_7_10);

  // x = -6
  drawGridLine(ctx, pair_per_neg_6_neg_5, pair_per_neg_6_11);

  // x = -5
  drawGridLine(ctx, pair_per_neg_5_neg_6, pair_per_neg_5_12);

  // x = -4
  drawGridLine(ctx, pair_per_neg_4_neg_7, pair_per_neg_4_13);

  // x = -3
  const pair_per_neg_3_14 = get_canvas_coordinate_pair_given(-3, 14, width_of_canvas);
  drawGridLine(ctx, pair_per_neg_3_neg_8, pair_per_neg_3_14);

  // x = -2
  drawGridLine(ctx, pair_per_neg_2_neg_9, pair_per_neg_2_13);

  // x = -1
  drawGridLine(ctx, pair_per_neg_1_neg_10, pair_per_neg_1_12);

  // y axis
  drawGridLine(ctx, pair_per_0_neg_11, pair_per_0_11, '#000000');

  // x = 1
  drawGridLine(ctx, pair_per_1_neg_12, pair_per_1_10);

  // x = 2
  drawGridLine(ctx, pair_per_2_neg_13, pair_per_2_9);

  // x = 3
  drawGridLine(ctx, pair_per_3_neg_14, pair_per_3_8);

  // x = 4
  drawGridLine(ctx, pair_per_4_neg_15, pair_per_4_7);

  // x = 5
  const pair_per_5_neg_16 = get_canvas_coordinate_pair_given(5, -16, width_of_canvas);
  drawGridLine(ctx, pair_per_5_neg_16, pair_per_5_6);

  // x = 6
  drawGridLine(ctx, pair_per_6_neg_15, pair_per_6_5);

  // x = 7
  drawGridLine(ctx, pair_per_7_neg_14, pair_per_7_4);

  // x = 8
  drawGridLine(ctx, pair_per_8_neg_13, pair_per_8_3);

  // x = 9
  drawGridLine(ctx, pair_per_9_neg_12, pair_per_9_2);

  // x = 10
  drawGridLine(ctx, pair_per_10_neg_11, pair_per_10_1);

  // x = 11
  drawGridLine(ctx, pair_per_11_neg_10, pair_per_11_0);

  // x = 12
  drawGridLine(ctx, pair_per_12_neg_9, pair_per_12_neg_1);

  // x = 13
  drawGridLine(ctx, pair_per_13_neg_8, pair_per_13_neg_2);

  // x = 14
  drawGridLine(ctx, pair_per_14_neg_7, pair_per_14_neg_3);

  // x = 15
  drawGridLine(ctx, pair_per_15_neg_6, pair_per_15_neg_4);

  ctx.fillStyle = 'rgb(64,64,64)';
  for (let i = -14; i <= 15; i++) {
    for (let j = -16; j <= 14; j++) {
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
      canvas.addEventListener(
        "mousedown",
        function (e: MouseEvent) {
          const message = getMousePosition(canvas, e);
          console.log(message);
        }
      );
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
        width = { height_of_webpage - 7 }
        height = { height_of_webpage - 7 }
        style = { { 'backgroundColor': color_of.Sea } }
      />
    </center>
  );
}

export default BaseBoardDisplayer;