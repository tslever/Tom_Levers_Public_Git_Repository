// https://www.dhiwise.com/post/designing-stunning-artwork-with-react-canvas-draw

import { useRef, useEffect } from 'react';

type Props = {
  aspectRatio: number,
  backgroundColor: string,
  widthPercentage: number
};

var drawGrid = function(ctx: CanvasRenderingContext2D, w: number, h: number, step: number) {
  ctx.beginPath();
  for (var x = 0; x <= w; x += step) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
  }
  ctx.strokeStyle = 'rgb(255,0,0)';
  ctx.lineWidth = 1;
  ctx.stroke(); 
  ctx.beginPath(); 
  for (var y = 0; y <= h; y += step) {
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
  }
  ctx.strokeStyle = 'rgb(20,20,20)';
  ctx.lineWidth = 1;
  ctx.stroke(); 
};

function CanvasDisplayer(props: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const context = canvas.getContext('2d');
      if (context) {
        drawGrid(context, canvas.width, canvas.height, 20);
      }
    }
  }, []);
  return (
    <canvas ref = { canvasRef } width = { 1000 } height = { 1000 } style = { { backgroundColor: '#ffffff' } }/>
  );
}

export default CanvasDisplayer;