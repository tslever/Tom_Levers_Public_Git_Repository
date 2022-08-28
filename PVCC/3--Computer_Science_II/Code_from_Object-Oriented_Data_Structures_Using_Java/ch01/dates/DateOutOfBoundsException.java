//----------------------------------------------------------------------
// DateOutOfBoundsException.java   by Dale/Joyce/Weems         Chapter 1
//
// Exception for use by dates related classes.
//----------------------------------------------------------------------

package ch01.dates;

public class DateOutOfBoundsException extends Exception
{
  public DateOutOfBoundsException()
  {
    super();
  }

  public DateOutOfBoundsException(String message)
  {
    super(message);
  }
}
