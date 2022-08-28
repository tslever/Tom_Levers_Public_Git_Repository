package ch02.figures;
public class Rectangle implements FigureInterface
{
  protected double length, width;

  public Rectangle(double length, double width)
  {
    this.length = length;
    this.width = width;
  }    
       
  public double perimeter()
  // Returns perimeter of this figure.
  {
    return(2 * (length + width));
  }
   
  public double area()
  // Returns area of this figure.
  {
    return(length * width);
  }
}
