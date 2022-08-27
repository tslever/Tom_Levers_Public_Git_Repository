package ch05.apps;
import support.*;

public class FPDriver 
{
  public static void main(String[] args)
  {
    FamousPerson p1, p2, p3;
    p1 = new FamousPerson("Herman", "Hollerith", 1860, "Developed mechanical counting machine based on punched cards.");
    System.out.println(p1);
    p2 = new FamousPerson("Herman", "Hollerith", 1860, "American scientist.");
    System.out.println("Should be true: " + p1.equals(p2));
    p2 = new FamousPerson("Herm", "Hollerith", 1860, "American scientist.");
    System.out.println("Should be false: " + p1.equals(p2));
    p2 = p1;
    System.out.println("Should be true: " + p1.equals(p2));

    System.out.println("Should be 0: " + p1.compareTo(p2));
    System.out.println("Should be 0: " + p2.compareTo(p2));

    p2 = new FamousPerson("Aaron", "Hollerith", 1860, "Developed mechanical counting machine based on punched cards.");
    System.out.println("Should be positive: " + p1.compareTo(p2));
    System.out.println("Should be negative: " + p2.compareTo(p1));

    p1 = new FamousPerson("Aaron", "Adams", 1860, "Developed mechanical counting machine based on punched cards.");
    System.out.println("Should be negative: " + p1.compareTo(p2));
    System.out.println("Should be positive: " + p2.compareTo(p1));
  }
}