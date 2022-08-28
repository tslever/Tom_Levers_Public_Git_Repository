//*********************************************************************
//  TSquareGray.java         By Dale/Joyce/Weems              Chapter 3
//
//  Creates a jpg file containing a recursive TSquare.
//  Varies gray levels used to draw squares.
//  Run argument 1: full name of target jpg file
//  Run argument 2: int indicated inner threshold
//  Run argument 3: int indicating outer threshold
//  
//********************************************************************
package ch03.fractals;

import java.awt.image.*;
import java.awt.Color;
import java.io.*;
import javax.imageio.*;

public class TSquareGray
{
   static final int SIDE = 1000;    // image is SIDE X SIDE
   static BufferedImage image = new BufferedImage(SIDE, SIDE, BufferedImage.TYPE_INT_RGB);
   static final int WHITE = Color.WHITE.getRGB();
   static final int BLACK = Color.BLACK.getRGB();
   static int inThreshold, outThreshold;
   static int grayDecrement;
   
   private static void drawSquare(int x, int y, int s, int g)
   // center of square is x,y  length of side is s, gray level is g
   {
      if (s <= inThreshold)  // base case
         return;
      else
      { 
         // determine corners
         int left = x - s/2;
         int top  = y - s/2;
         int right = x + s/2;
         int bottom = y + s/2;

         // potentially paint the gray square
         if (s < outThreshold)
         {
            Color color = new Color(g, g, g);
            int colorCode = color.getRGB();

            for (int i = left; i < right; i++)
              for (int j = top; j < bottom; j++)
              {
                 image.setRGB(i, j, colorCode);
              }
            g = g - grayDecrement;
         }
         
         // recursively paint squares at the corners
         drawSquare(left, top, s/2, g);
         drawSquare(left, bottom, s/2, g);
         drawSquare(right, top, s/2, g);
         drawSquare(right, bottom, s/2, g);
      }
   }  

   public static void main (String[] args) throws IOException
   {
      String fileOut = args[0];
      inThreshold = Integer.parseInt(args[1]);
      outThreshold = Integer.parseInt(args[2]);
      
      // calculate amount to decrement gray level
      int countLevels = 0; // number of drawn levels
      int width = SIDE/2;  // size of square at current level
      while (width > inThreshold)
      {
        if (width < outThreshold) countLevels++;
        width = width / 2;
      }
      grayDecrement = 256 / countLevels;

      // make image black
      for (int i = 0; i < SIDE; i++)
        for (int j = 0; j < SIDE; j++)
        {
           image.setRGB(i, j, BLACK);
        }   

      // first square
      drawSquare(SIDE/2, SIDE/2, SIDE/2, 255);
      
      // save image
      File outputfile = new File(fileOut);
      ImageIO.write(image, "jpg", outputfile);
   }
}