//----------------------------------------------------------------------
// RandomFigs.java          by Dale/Joyce/Weems                Chapter 2
//
// Demonstrates polymorphism.
//----------------------------------------------------------------------
package ch02.apps;

import ch02.figures.*;
import java.util.Random;
import java.text.DecimalFormat;

public class RandomFigs
{
   public static void main(String[] args)
   {
      DecimalFormat df = new DecimalFormat("#.###");
      Random rand = new Random();
      final int COUNT = 5;
      double totalArea = 0;

      FigureInterface[] figures = new FigureInterface[COUNT];

      // generate figures
      for (int i = 0; i < COUNT; i++)
      {
         switch (rand.nextInt(2))
         {
            case 0:  figures[i] = new Circle(1.0); 
                     System.out.print("circle area     3.14\n"); break;
                                       
            case 1:  figures[i] = new Rectangle(1.0, 2.0); 
                     System.out.print("rectangle area  2.00\n"); break;
         }
      }
      
      // sum areas
      for (int i = 0; i < COUNT; i++)
         totalArea = totalArea + figures[i].area();
      
      System.out.println("\nTotal: " + df.format(totalArea));
   }
}