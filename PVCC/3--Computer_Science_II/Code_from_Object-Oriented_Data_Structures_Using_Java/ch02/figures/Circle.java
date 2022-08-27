package ch02.figures;
public class Circle implements FigureInterface
{
  protected double radius;
      
  public Circle(double radius)
  {
    this.radius = radius;
  }    
       
  public double perimeter()
  // Returns perimeter of this figure.
  {
    return(2 * PI * radius);
  }
   
  public double area()
  // Returns area of this figure.
  {
    return(PI * radius * radius);
  }
}
