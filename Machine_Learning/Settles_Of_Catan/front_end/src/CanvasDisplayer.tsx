// https://www.dhiwise.com/post/designing-stunning-artwork-with-react-canvas-draw

import { useRef, useEffect } from 'react';

type Props = {
  aspectRatio: number,
  backgroundColor: string,
  widthPercentage: number
};

function get_canvas_x_coordinate_given(width_of_canvas: number, isometric_x_coordinate: number) {
  return width_of_canvas / 2 + isometric_x_coordinate * width_of_canvas / 20;
}

var drawGrid = function(ctx: CanvasRenderingContext2D, w: number, h: number, step: number) {
  const tangent_of_30_degrees = 0.5773502691896257645091487805019574556476017512701268760186023264;
  
  // Vertical Lines
  ctx.beginPath(); 
  for (var x = -10; x <= 10; x += 1) {
    ctx.moveTo(get_canvas_x_coordinate_given(w, x), 0);
    ctx.lineTo(get_canvas_x_coordinate_given(w, x), h);
  }
  ctx.strokeStyle = 'rgb(0,0,0)';
  ctx.lineWidth = 1;
  ctx.stroke();
  
  // Positively Sloping Lines Beginning On Top Edge
  ctx.beginPath();
  for (var x = 0; x <= w; x += step) {
    ctx.moveTo(x, 0);
    ctx.lineTo(0, tangent_of_30_degrees * x);
  }
  ctx.moveTo(w, 0);
  ctx.lineTo(0, tangent_of_30_degrees * w);
  ctx.strokeStyle = 'rgb(255,0,0)';
  ctx.lineWidth = 1;
  ctx.stroke();
  
  // Negatively Sloping Lines Beginning On Top Edge
  ctx.beginPath(); 
  for (var x = 0; x <= w; x += step) {
    ctx.moveTo(x, 0);
    ctx.lineTo(w, tangent_of_30_degrees * (w - x));
  }
  ctx.strokeStyle = 'rgb(0,255,0)';
  ctx.lineWidth = 1;
  ctx.stroke();
  
  // Negatively Sloping Lines Beginning On Left Edge
  ctx.beginPath();
  for (var y = step * tangent_of_30_degrees; y <= h; y += step * tangent_of_30_degrees) {
    ctx.moveTo(0, y);
    ctx.lineTo(w, y + w * tangent_of_30_degrees);
  }
  ctx.strokeStyle = 'rgb(0,0,255)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Positively Sloping Lines Beginning On Right Edge
  ctx.beginPath();
  for (var y = step * tangent_of_30_degrees; y <= h; y += step * tangent_of_30_degrees) {
    ctx.moveTo(w, y);
    ctx.lineTo(0, y + w * tangent_of_30_degrees);
  }
  ctx.strokeStyle = 'rgb(255,0,255)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Tile 0: Desert
  let region = new Path2D();
  region.moveTo(5 * step, 6 * step * tangent_of_30_degrees);
  region.lineTo(6 * step, 7 * step * tangent_of_30_degrees);
  region.lineTo(6 * step, 9 * step * tangent_of_30_degrees);
  region.lineTo(5 * step, 6 * step * tangent_of_30_degrees);
  region.closePath();
  ctx.fillStyle = '#F5D5A1';
  ctx.fill(region);

  // Tile 4: Brick
  region = new Path2D();
  region.moveTo(4 * step, 3 * step * tangent_of_30_degrees);
  region.lineTo(5 * step, 4 * step * tangent_of_30_degrees);
  region.lineTo(5 * step, 6 * step * tangent_of_30_degrees);
  region.lineTo(4 * step, 3 * step * tangent_of_30_degrees);
  region.closePath();
  ctx.fillStyle = '#AA4A44';
  ctx.fill(region);

  // Tile 14: Ore
  region = new Path2D();
  region.moveTo(3 * step, 0);
  region.lineTo(4 * step, 1 * step * tangent_of_30_degrees);
  region.lineTo(4 * step, 3 * step * tangent_of_30_degrees);
  region.lineTo(3 * step, 4 * step * tangent_of_30_degrees);
  region.lineTo(2 * step, 3 * step * tangent_of_30_degrees);
  region.lineTo(2 * step, 1 * step * tangent_of_30_degrees);
  region.lineTo(3 * step, 0);
  region.closePath();
  ctx.fillStyle = '#52595D';
  ctx.fill(region);

  // Tile 18: Ore
  region = new Path2D();
  region.moveTo(9 * step, 6 * step * tangent_of_30_degrees);
  region.lineTo(10 * step, 7 * step * tangent_of_30_degrees);
  region.lineTo(10 * step, 9 * step * tangent_of_30_degrees);
  region.lineTo(9 * step, 6 * step * tangent_of_30_degrees);
  region.closePath();
  ctx.fillStyle = '#52595D';
  ctx.fill(region);
};

function CanvasDisplayer(props: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const context = canvas.getContext('2d');
      if (context) {
        drawGrid(context, canvas.width, canvas.height, canvas.width / 10);
      }
    }
  }, []);
  //const width_of_scroll_bar_in_Google_Chrome_for_Windows_10 = 17
  //const width_of_webpage = window.innerWidth - width_of_scroll_bar_in_Google_Chrome_for_Windows_10
  const height_of_webpage = window.innerHeight;
  return (
    <center>
      <canvas ref = { canvasRef } width = { height_of_webpage - 4 } height = { height_of_webpage - 4 } style = { { backgroundColor: '#ffffff' } }/>
    </center>
  );
}

export default CanvasDisplayer;