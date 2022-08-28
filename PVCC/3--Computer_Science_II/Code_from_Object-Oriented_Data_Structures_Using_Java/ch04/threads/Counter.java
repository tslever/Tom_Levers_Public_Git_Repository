//---------------------------------------------------------------------------
// Counter.java              by Dale/Joyce/Weems                    Chapter 4
//
// Tracks the current value of a counter.
//---------------------------------------------------------------------------

package ch04.threads;

public class Counter
{
  private int count;
  
  public Counter()
  {
     count = 0;
  }

  public void increment()
  {
    count++;
  }

  public int getCount(){return count;}
}
