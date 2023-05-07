package Com.TSL.BinaryExpressionTreeUtilities;


/**
* @author YINGJIN CUI
* @version 1.0
* since   2020-05
*
* BXTDriver.java: the driver program for BXT class.
* Input: a postfix string, each token separated by a space.
*/

import java.util.*;


public class BXTDriver
{
	
   /**
    * main is the entry point of this program, which iterates over postfix expressions. For each expression, a binary
    * expression tree is generated based on the postfix expression; a representation of the tree is displayed; an infix
    * version, a prefix version, and itself are displayed; and a floating-point number representing the result of
    * all the arithmetic operations represented by the tree is displayed.
    * 
    * @param args
    */
	
   public static void main(String[] args)
   {
	   
      ArrayList<String> postExp = new ArrayList<String>();
      postExp.add("14 -5 / ");
      postExp.add("20 3 -4 + * ");
      postExp.add("2 3 + 5 / 4 5 - *");
   
      for( String postfix : postExp )
      {
         System.out.println("Postfix Exp: "  + postfix);
         BXT tree = new BXT();
         tree.buildTree( postfix );
         System.out.println("BXT: "); 
         System.out.println( tree.display() );
         System.out.print("Infix order:  ");
         System.out.println( tree.infix() );
         System.out.print("Prefix order:  ");
         System.out.println( tree.prefix() );
         System.out.print("Postfix order:  ");
         System.out.println( tree.postfix() );
         System.out.print("Evaluates to " + tree.evaluateTree());
         System.out.println("\n------------------------");
      }
   }
}