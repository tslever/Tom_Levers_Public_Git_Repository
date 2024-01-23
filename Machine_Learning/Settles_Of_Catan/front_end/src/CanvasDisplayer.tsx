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
        drawGrid(context, canvas.width, canvas.height, canvas.width / 10);
      }
    }
  }, []);
  const width_of_scroll_bar_in_Google_Chrome_for_Windows_10 = 17
  const width_of_webpage = window.innerWidth - width_of_scroll_bar_in_Google_Chrome_for_Windows_10
  return (
    <canvas ref = { canvasRef } width = { width_of_webpage } height = { width_of_webpage } style = { { backgroundColor: '#ffffff' } }/>
  );
}

export default CanvasDisplayer;