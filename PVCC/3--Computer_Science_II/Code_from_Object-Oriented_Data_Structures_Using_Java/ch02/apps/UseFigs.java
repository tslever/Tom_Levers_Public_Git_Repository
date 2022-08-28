package ch02.apps;

import ch02.figures.*;

public class UseFigs
{
   public static void main(String[] args)
   {
      Circle myCircle = new Circle(5);
      System.out.println(myCircle.perimeter());
      System.out.println(myCircle.area());
      Rectangle myRectangle = new Rectangle(7, 8);
      System.out.println(myRectangle.perimeter());
      System.out.println(myRectangle.area());
   }
}