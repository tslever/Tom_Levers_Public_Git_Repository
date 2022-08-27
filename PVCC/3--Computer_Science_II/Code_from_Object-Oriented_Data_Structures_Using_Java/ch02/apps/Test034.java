//----------------------------------------------------------------------------
// Test034.java              by Dale/Joyce/Weems                     Chapter 2
//
// Batch test case example
//----------------------------------------------------------------------------
import ch02.stacks.*;

public class Test034
{
  public static void main(String[] args)
  { 
    StackInterface<String> test = new ArrayBoundedStack<String>( );
    test.push("trouble in the fields");
    test.push("love at the five and dime");
    test.push("once in a very blue moon");
    String s = test.top();
    if (s.equals("once in a very blue moon"))
      System.out.println("Test 34 passed");
    else
      System.out.println("Test 34 failed");
  }
}